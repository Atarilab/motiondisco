"""Claude Max subscription backend.

Serves ``claude-max/<model>`` by shelling out to the ``claude`` CLI in
headless mode (``claude -p --output-format json``). The CLI authenticates
from the user's OAuth subscription credentials, so no ANTHROPIC_API_KEY
is involved — see ``shinka.claude_max_config`` for why the child
environment must be scrubbed of one.

Structurally mirrors ``local_openai.py``: module-level backoff constants,
a sync/async pair with the standard positional provider signature, and
defensive usage extraction that tolerates a missing ``usage`` block.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import backoff

from shinka.claude_max_config import (
    build_claude_cli_env,
    claude_max_zero_cost,
)
from shinka.llm.constants import (
    BACKOFF_MAX_TIME,
    BACKOFF_MAX_TRIES,
    BACKOFF_MAX_VALUE,
)

from .pricing import calculate_cost, model_exists
from .result import QueryResult

logger = logging.getLogger(__name__)

MAX_TRIES = BACKOFF_MAX_TRIES
MAX_VALUE = BACKOFF_MAX_VALUE
MAX_TIME = BACKOFF_MAX_TIME


class ClaudeMaxError(RuntimeError):
    """Base error for the Claude Max CLI backend."""


class ClaudeMaxTransientError(ClaudeMaxError):
    """Retryable: overload, connection blip, timeout."""


class ClaudeMaxFatalError(ClaudeMaxError):
    """Not retryable, and must not be swallowed by outer retry loops."""


class ClaudeMaxAuthError(ClaudeMaxFatalError):
    """Subscription credentials missing, expired, or overridden by a key."""


class ClaudeMaxRateLimitError(ClaudeMaxFatalError):
    """Subscription rate-limit window exhausted.

    Fatal by choice: a Max plan's window resets on the order of hours, so
    parking the run in exponential backoff wastes far more than it saves.
    Failing loudly lets the run be restarted after the reset.
    """


@dataclass(frozen=True)
class ClaudeMaxCliClient:
    """Stands in for an SDK client — carries everything a call needs."""

    binary: str
    cli_model_name: str
    pricing_model_name: str
    original_model_name: str
    cwd: str
    timeout: float
    extra_args: Tuple[str, ...] = ()
    zero_cost: bool = field(default_factory=claude_max_zero_cost)


# HTTP statuses in `api_error_status` that are worth another attempt.
_TRANSIENT_STATUSES = frozenset({408, 409, 500, 502, 503, 504, 529})
_AUTH_STATUSES = frozenset({401, 403})

_RATE_LIMIT_RE = re.compile(
    r"rate.?limit|usage limit reached|quota exceeded|too many requests",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"not logged in|please run /login|invalid api key|authentication_failed"
    r"|oauth_org_not_allowed|credit balance is too low",
    re.IGNORECASE,
)
_TRANSIENT_RE = re.compile(
    r"overloaded|econnreset|etimedout|socket hang up|internal server error",
    re.IGNORECASE,
)


def backoff_handler(details) -> None:
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"Claude Max CLI - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _estimate_tokens(text: str) -> int:
    """Crude fallback when the CLI reports no usage block."""
    return max(1, len(text) // 4)


def _render_prompt(msg: str, msg_history: List[Dict]) -> str:
    """Flatten prior turns into one prompt string.

    ``claude -p`` is single-turn, so a replayed ``msg_history`` (used by
    the patch-retry path in ``async_runner``) becomes a tagged
    transcript. Lower fidelity than real turns, but only reachable when
    ``max_patch_attempts > 1``.
    """
    if not msg_history:
        return msg

    parts: List[str] = []
    for turn in msg_history:
        role = str(turn.get("role", "user")).upper()
        content = turn.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        parts.append(f"[{role}]\n{content}")
    parts.append(f"[USER]\n{msg}")
    return "\n\n".join(parts)


def _build_command(
    client: ClaudeMaxCliClient,
    system_msg: str,
    effort: Optional[str],
    max_budget_usd: Optional[float],
) -> List[str]:
    """Assemble a hermetic headless invocation.

    Every flag here has a job:
      --output-format json   machine-readable envelope with usage + cost
      --model                explicit, so ~/.claude/settings.json cannot
                             silently change which model a run used
      --system-prompt        REPLACES Claude Code's agent prompt; without
                             it the model is told it is a coding agent
      --tools ""             no tool use — this is a text completion
      --safe-mode            no CLAUDE.md, skills, plugins, hooks, MCP,
                             custom commands. Auth and model selection
                             keep working (unlike --bare, which forces
                             ANTHROPIC_API_KEY auth and ignores OAuth)
      --strict-mcp-config    defence in depth alongside --safe-mode
      --no-session-persistence  no session files per generation
      --permission-mode dontAsk no chance of a headless prompt hang
    The prompt itself goes on stdin, never argv.
    """
    cmd = [
        client.binary,
        "-p",
        "--output-format",
        "json",
        "--model",
        client.cli_model_name,
        "--system-prompt",
        system_msg,
        "--tools",
        "",
        "--safe-mode",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
    ]
    if effort:
        cmd += ["--effort", effort]
    if max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    cmd += list(client.extra_args)
    return cmd


def _parse_envelope(stdout: str) -> Optional[Dict[str, Any]]:
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    return envelope if isinstance(envelope, dict) else None


def _classify_failure(
    returncode: int,
    stdout: str,
    stderr: str,
    envelope: Optional[Dict[str, Any]],
) -> ClaudeMaxError:
    """Map a failed invocation onto the retry/fail-fast taxonomy.

    Note the failure test cannot use ``subtype``: the CLI reports
    ``subtype="success"`` even for a 401 or 404, signalling the real
    outcome through ``is_error`` + ``api_error_status`` instead.
    """
    status = None
    message = stderr.strip()
    if envelope is not None:
        raw_status = envelope.get("api_error_status")
        if isinstance(raw_status, int):
            status = raw_status
        message = str(envelope.get("result") or "").strip() or message
    if not message:
        message = stdout.strip()[:400] or f"claude CLI exited {returncode}"

    if _RATE_LIMIT_RE.search(message) or status == 429:
        return ClaudeMaxRateLimitError(
            f"Claude Max subscription rate limit reached: {message}. "
            "Limits reset on a rolling window; restart the run once it does, "
            "or switch to an API-key model such as 'claude-opus-4-7'."
        )

    if _AUTH_RE.search(message) or status in _AUTH_STATUSES:
        return ClaudeMaxAuthError(
            f"Claude Max backend could not authenticate: {message}. "
            "Run `claude auth` (or `claude setup-token`) as the user running "
            "Shinka. Note that Shinka deliberately strips ANTHROPIC_API_KEY "
            "from the CLI subprocess so the subscription is used — if you "
            "meant to bill the API, use a plain model name like "
            "'claude-opus-4-7' instead of 'claude-max/...'."
        )

    if _TRANSIENT_RE.search(message) or status in _TRANSIENT_STATUSES:
        return ClaudeMaxTransientError(f"Claude Max CLI transient failure: {message}")

    return ClaudeMaxFatalError(f"Claude Max CLI failed: {message}")


def _extract_usage(
    envelope: Dict[str, Any],
    prompt: str,
    system_msg: str,
    content: str,
) -> Tuple[int, int]:
    """Token counts, preferring the CLI's own accounting.

    ``modelUsage`` is per-model and camelCase; the top-level ``usage`` is
    the raw snake_case Anthropic block for the primary model. Either may
    be absent, so fall back to a length estimate. Output tokens are
    clamped to >= 1 because ``QueryResult.__str__`` divides by them.
    """
    in_tokens = out_tokens = 0

    model_usage = envelope.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        for entry in model_usage.values():
            if not isinstance(entry, dict):
                continue
            in_tokens += int(entry.get("inputTokens", 0) or 0)
            in_tokens += int(entry.get("cacheReadInputTokens", 0) or 0)
            in_tokens += int(entry.get("cacheCreationInputTokens", 0) or 0)
            out_tokens += int(entry.get("outputTokens", 0) or 0)

    if not out_tokens:
        usage = envelope.get("usage")
        if isinstance(usage, dict):
            in_tokens = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
            )
            out_tokens = int(usage.get("output_tokens", 0) or 0)

    if not in_tokens:
        in_tokens = _estimate_tokens(system_msg + prompt)
    return in_tokens, max(out_tokens, 1)


def _extract_costs(
    envelope: Dict[str, Any],
    client: ClaudeMaxCliClient,
    in_tokens: int,
    out_tokens: int,
) -> Tuple[float, float, float]:
    """Return ``(cost, input_cost, output_cost)``.

    Nothing is actually billed on a Max plan; these are the notional
    API-equivalent figures the CLI reports, kept so that
    ``max_api_costs`` budgets and the cost-aware bandit keep working.

    ``total_cost_usd`` covers the whole invocation including the CLI's
    own internal model calls, so it can exceed the input/output split
    derived from pricing.csv for the primary model.
    """
    if client.zero_cost:
        return 0.0, 0.0, 0.0

    if model_exists(client.pricing_model_name):
        input_cost, output_cost = calculate_cost(
            client.pricing_model_name, in_tokens, out_tokens
        )
    else:
        input_cost, output_cost = 0.0, 0.0

    total = envelope.get("total_cost_usd")
    cost = (
        float(total)
        if isinstance(total, (int, float))
        else input_cost + output_cost
    )
    return cost, input_cost, output_cost


def _build_result(
    envelope: Dict[str, Any],
    client: ClaudeMaxCliClient,
    msg: str,
    system_msg: str,
    prompt: str,
    new_msg_history: List[Dict],
    kwargs: Dict,
    model_posteriors: Optional[Dict[str, float]],
) -> QueryResult:
    content = str(envelope.get("result") or "")
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    in_tokens, out_tokens = _extract_usage(envelope, prompt, system_msg, content)
    cost, input_cost, output_cost = _extract_costs(
        envelope, client, in_tokens, out_tokens
    )

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        # The original "claude-max/..." name, so results record which
        # backend produced the program (query.py rebinds its local
        # model_name to the pricing name before calling us).
        model_name=client.original_model_name,
        kwargs=kwargs,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        thinking_tokens=0,
        cost=cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )


def _prepare(
    client: Any,
    msg: str,
    system_msg: str,
    msg_history: List[Dict],
    output_model,
    kwargs: Dict,
) -> Tuple[List[str], str, List[Dict], Dict]:
    if output_model is not None:
        raise NotImplementedError(
            "Structured output is not supported for claude-max/ CLI models."
        )
    if not isinstance(client, ClaudeMaxCliClient):
        raise ClaudeMaxFatalError(
            f"Expected a ClaudeMaxCliClient, got {type(client).__name__}."
        )

    call_kwargs = dict(kwargs)
    effort = call_kwargs.pop("effort", None)
    max_budget_usd = call_kwargs.pop("max_budget_usd", None)
    if call_kwargs:
        logger.debug(f"Claude Max CLI ignoring unsupported kwargs: {sorted(call_kwargs)}")

    cmd = _build_command(client, system_msg, effort, max_budget_usd)
    prompt = _render_prompt(msg, msg_history)
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    return cmd, prompt, new_msg_history, dict(kwargs)


@backoff.on_exception(
    backoff.expo,
    (ClaudeMaxTransientError,),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_claude_max(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    cmd, prompt, new_msg_history, sent_kwargs = _prepare(
        client, msg, system_msg, msg_history, output_model, kwargs
    )

    try:
        proc = subprocess.run(
            cmd,
            cwd=client.cwd,
            env=build_claude_cli_env(),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=client.timeout,
        )
    except FileNotFoundError as exc:
        raise ClaudeMaxFatalError(
            f"claude CLI not found at '{client.binary}'."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClaudeMaxTransientError(
            f"claude CLI timed out after {client.timeout:.0f}s."
        ) from exc

    envelope = _parse_envelope(proc.stdout)
    if proc.returncode != 0 or envelope is None or envelope.get("is_error"):
        raise _classify_failure(proc.returncode, proc.stdout, proc.stderr, envelope)

    return _build_result(
        envelope, client, msg, system_msg, prompt, new_msg_history,
        sent_kwargs, model_posteriors,
    )


@backoff.on_exception(
    backoff.expo,
    (ClaudeMaxTransientError,),
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_claude_max_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    cmd, prompt, new_msg_history, sent_kwargs = _prepare(
        client, msg, system_msg, msg_history, output_model, kwargs
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=client.cwd,
            env=build_claude_cli_env(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ClaudeMaxFatalError(
            f"claude CLI not found at '{client.binary}'."
        ) from exc

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(prompt.encode()), timeout=client.timeout
        )
    except asyncio.TimeoutError as exc:
        with suppress(ProcessLookupError):
            proc.kill()
        with suppress(Exception):
            await proc.communicate()
        raise ClaudeMaxTransientError(
            f"claude CLI timed out after {client.timeout:.0f}s."
        ) from exc

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")

    envelope = _parse_envelope(stdout)
    if proc.returncode != 0 or envelope is None or envelope.get("is_error"):
        raise _classify_failure(proc.returncode or 1, stdout, stderr, envelope)

    return _build_result(
        envelope, client, msg, system_msg, prompt, new_msg_history,
        sent_kwargs, model_posteriors,
    )
