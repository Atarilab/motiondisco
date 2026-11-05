"""Configuration for the ``claude-max/`` subscription backend.

Model names prefixed ``claude-max/`` are served by shelling out to the
locally installed ``claude`` CLI in headless mode, which authenticates
from the user's Claude Max/Pro OAuth credentials rather than an API key.
This module owns the name parsing, binary discovery, and — critically —
the sanitized child environment.

Lives at package root (not under ``llm/providers/``) for the same reason
as ``local_openai_config``: ``model_resolver`` imports it without pulling
in the provider package.
"""

from __future__ import annotations

import atexit
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

CLAUDE_MAX_PREFIX = "claude-max/"

BIN_ENV = "SHINKA_CLAUDE_CLI_BIN"
TIMEOUT_ENV = "SHINKA_CLAUDE_CLI_TIMEOUT"
EXTRA_ARGS_ENV = "SHINKA_CLAUDE_CLI_EXTRA_ARGS"
ZERO_COST_ENV = "SHINKA_CLAUDE_MAX_ZERO_COST"

# CLI aliases accepted by `claude --model`, mapped to the pricing.csv row
# used for capability flags and the notional cost fallback. Aliases track
# "latest", so the mapping is a best-effort proxy, not a guarantee.
_ALIAS_PRICING_MODEL = {
    "opus": "claude-opus-4-7",
    "opus[1m]": "claude-opus-4-7",
    "opusplan": "claude-opus-4-7",
    "best": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "sonnet[1m]": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

# Environment variables that would hijack the CLI's authentication or
# routing and must never reach the subprocess.
#
# This is load-bearing, not defensive hygiene: `shinka/llm/client.py`
# calls `load_shinka_dotenv()` at import, and a repo `.env` carrying
# ANTHROPIC_API_KEY therefore puts that key in `os.environ` for every
# Shinka process. Verified behavior: with ANTHROPIC_API_KEY set, the CLI
# authenticates with the *key* and ignores OAuth entirely — so without
# this strip list the backend would silently bill the API while
# appearing to work, which is exactly what the subscription path exists
# to avoid.
_STRIP_ENV: Tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)

_scratch_dir: Optional[str] = None


@dataclass(frozen=True)
class ResolvedClaudeMaxModel:
    original_model_name: str
    cli_model_name: str
    pricing_model_name: str


def parse_claude_max_model(model_name: str) -> Optional[ResolvedClaudeMaxModel]:
    """Parse a ``claude-max/<model-or-alias>`` identifier."""
    if not model_name.startswith(CLAUDE_MAX_PREFIX):
        return None

    suffix = model_name[len(CLAUDE_MAX_PREFIX) :].strip()
    if not suffix:
        raise ValueError(
            "Claude Max model name is missing after 'claude-max/'. "
            "Use e.g. 'claude-max/claude-opus-4-7' or 'claude-max/opus'."
        )

    # Unknown names pass straight through: the CLI is the authority on
    # which models exist, and a name absent from pricing.csv simply
    # degrades to a zero notional cost.
    return ResolvedClaudeMaxModel(
        original_model_name=model_name,
        cli_model_name=suffix,
        pricing_model_name=_ALIAS_PRICING_MODEL.get(suffix, suffix),
    )


def resolve_claude_cli_binary() -> str:
    """Locate the ``claude`` executable, or explain how to get one."""
    override = os.getenv(BIN_ENV, "").strip()
    if override:
        if not (os.path.isfile(override) and os.access(override, os.X_OK)):
            raise ValueError(
                f"{BIN_ENV} points at '{override}', which is not an executable file."
            )
        return override

    binary = shutil.which("claude")
    if not binary:
        raise ValueError(
            "claude CLI not found on PATH. The 'claude-max/' backend shells out "
            "to Claude Code to use your subscription. Install it from "
            "https://claude.com/claude-code, or set "
            f"{BIN_ENV} to its absolute path. Use a plain model name such as "
            "'claude-opus-4-7' if you want the ANTHROPIC_API_KEY path instead."
        )
    return binary


def claude_cli_scratch_dir() -> str:
    """A stable, empty cwd for CLI invocations.

    Running from the Shinka checkout would let the CLI derive project
    settings and session paths from the repo. A scratch dir makes each
    call independent of where the run was launched.
    """
    global _scratch_dir
    if _scratch_dir is None or not os.path.isdir(_scratch_dir):
        _scratch_dir = tempfile.mkdtemp(prefix="shinka-claude-max-")
        atexit.register(shutil.rmtree, _scratch_dir, True)
    return _scratch_dir


def claude_cli_timeout(default: float) -> float:
    raw = os.getenv(TIMEOUT_ENV, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{TIMEOUT_ENV} must be a number, got '{raw}'.") from exc
    if value <= 0:
        raise ValueError(f"{TIMEOUT_ENV} must be positive, got '{raw}'.")
    return value


def claude_cli_extra_args() -> List[str]:
    """Escape hatch for flags Shinka does not model (e.g. --add-dir)."""
    return shlex.split(os.getenv(EXTRA_ARGS_ENV, ""))


def claude_max_zero_cost() -> bool:
    return bool(os.getenv(ZERO_COST_ENV, "").strip())


def build_claude_cli_env() -> Dict[str, str]:
    """Child environment with every auth-hijacking variable removed."""
    env = os.environ.copy()
    for key in _STRIP_ENV:
        env.pop(key, None)
    return env
