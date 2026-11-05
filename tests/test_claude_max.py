"""Tests for the claude-max/ subscription backend.

Every test is fully stubbed — none of them invoke the real `claude` CLI
or consume subscription quota.
"""

import asyncio
import json
import subprocess

import pytest
import shinka.llm.client as llm_client_module
import shinka.llm.query as query_module
from shinka import claude_max_config as cfg
from shinka.llm.kwargs import sample_model_kwargs
from shinka.llm.providers import claude_max
from shinka.llm.providers.model_resolver import resolve_model_backend


def _client(**overrides):
    defaults = dict(
        binary="/usr/bin/true",
        cli_model_name="opus",
        pricing_model_name="claude-opus-4-7",
        original_model_name="claude-max/opus",
        cwd="/tmp",
        timeout=30.0,
    )
    defaults.update(overrides)
    return claude_max.ClaudeMaxCliClient(**defaults)


def _envelope(**overrides):
    """Shape verified against claude CLI 2.1.222 --output-format json."""
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "num_turns": 1,
        "stop_reason": "end_turn",
        "result": "hello",
        "total_cost_usd": 0.00123,
        "usage": {
            "input_tokens": 184,
            "output_tokens": 6,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "modelUsage": {
            "claude-opus-4-7": {
                "inputTokens": 184,
                "outputTokens": 6,
                "cacheReadInputTokens": 0,
                "cacheCreationInputTokens": 0,
                "costUSD": 0.00107,
                "canonicalModel": "claude-opus-4-7",
            }
        },
        "terminal_reason": "completed",
    }
    envelope.update(overrides)
    return envelope


# --------------------------------------------------------------------------
# Name parsing / resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name,cli_model,pricing_model",
    [
        ("claude-max/claude-opus-4-7", "claude-opus-4-7", "claude-opus-4-7"),
        ("claude-max/opus", "opus", "claude-opus-4-7"),
        ("claude-max/sonnet", "sonnet", "claude-sonnet-4-6"),
        ("claude-max/haiku", "haiku", "claude-haiku-4-5-20251001"),
        # Unknown names pass through: the CLI is the authority on models.
        ("claude-max/claude-opus-9", "claude-opus-9", "claude-opus-9"),
    ],
)
def test_resolve_claude_max_models(model_name, cli_model, pricing_model):
    resolved = resolve_model_backend(model_name)
    assert resolved.provider == "claude_max"
    assert resolved.cli_model_name == cli_model
    assert resolved.api_model_name == pricing_model


def test_bare_anthropic_model_still_routes_to_api_key_backend():
    """Regression guard: the API path must be completely unaffected."""
    resolved = resolve_model_backend("claude-opus-4-7")
    assert resolved.provider == "anthropic"
    assert resolved.cli_model_name is None


def test_empty_claude_max_suffix_raises():
    with pytest.raises(ValueError, match="missing after 'claude-max/'"):
        resolve_model_backend("claude-max/")


def test_unknown_model_error_mentions_claude_max():
    with pytest.raises(ValueError, match="not supported"):
        resolve_model_backend("totally-unknown-model")


# --------------------------------------------------------------------------
# The env-sanitising hazard: an inherited ANTHROPIC_API_KEY would silently
# authenticate the CLI with the API instead of the Max subscription.
# --------------------------------------------------------------------------


def test_child_env_strips_anthropic_credentials(monkeypatch):
    for name in cfg._STRIP_ENV:
        monkeypatch.setenv(name, "should-not-survive")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = cfg.build_claude_cli_env()

    for name in cfg._STRIP_ENV:
        assert name not in env, f"{name} leaked into the CLI subprocess"
    assert env["PATH"] == "/usr/bin"


def test_missing_binary_error_is_actionable(monkeypatch):
    monkeypatch.delenv(cfg.BIN_ENV, raising=False)
    monkeypatch.setattr(cfg.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="claude CLI not found"):
        cfg.resolve_claude_cli_binary()


def test_scratch_dir_is_outside_the_repo():
    assert "ShinkaEvolve" not in cfg.claude_cli_scratch_dir()


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------


def test_command_is_hermetic_and_uses_oauth():
    cmd = claude_max._build_command(_client(), "SYS", "high", None)

    for flag in (
        "--safe-mode",              # no CLAUDE.md / skills / hooks / MCP
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",          # replaces the coding-agent prompt
        "--tools",                  # disabled
        "--output-format",
    ):
        assert flag in cmd

    assert "--effort" in cmd and cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == ""
    # --bare would force ANTHROPIC_API_KEY auth and never read OAuth.
    assert "--bare" not in cmd
    assert "--dangerously-skip-permissions" not in cmd


def test_effort_omitted_when_disabled():
    cmd = claude_max._build_command(_client(), "SYS", None, None)
    assert "--effort" not in cmd


def test_prompt_is_not_passed_on_argv(monkeypatch):
    """Guards E2BIG: proposal prompts routinely exceed the 128KB argv limit."""
    big_msg = "x" * 200_000
    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input")
        return subprocess.CompletedProcess(
            cmd, 0, json.dumps(_envelope()), ""
        )

    monkeypatch.setattr(claude_max.subprocess, "run", _fake_run)
    claude_max.query_claude_max(_client(), "m", big_msg, "sys", [], None)

    assert big_msg not in captured["cmd"]
    assert captured["input"] == big_msg


# --------------------------------------------------------------------------
# Envelope parsing
# --------------------------------------------------------------------------


def test_usage_prefers_model_usage():
    assert claude_max._extract_usage(_envelope(), "p", "s", "c") == (184, 6)


def test_usage_falls_back_to_snake_case():
    env = _envelope()
    del env["modelUsage"]
    assert claude_max._extract_usage(env, "p", "s", "c") == (184, 6)


def test_usage_estimates_when_absent():
    env = _envelope()
    del env["modelUsage"]
    del env["usage"]
    in_tok, out_tok = claude_max._extract_usage(env, "p" * 40, "s" * 40, "c")
    assert in_tok == 20
    assert out_tok >= 1


def test_output_tokens_never_zero_so_str_cannot_divide_by_zero(monkeypatch):
    """QueryResult.__str__ divides by output_tokens."""
    env = _envelope(
        usage={"input_tokens": 5, "output_tokens": 0}, modelUsage={}
    )

    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, json.dumps(env), ""
        ),
    )
    result = claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)

    assert result.output_tokens >= 1
    assert "Thinking tokens" in str(result)  # would raise ZeroDivisionError


def test_cost_prefers_reported_total(monkeypatch):
    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, json.dumps(_envelope()), ""
        ),
    )
    result = claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)
    assert result.cost == pytest.approx(0.00123)
    assert result.input_cost > 0  # split derived from pricing.csv


def test_zero_cost_env_forces_free(monkeypatch):
    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, json.dumps(_envelope()), ""
        ),
    )
    result = claude_max.query_claude_max(
        _client(zero_cost=True), "m", "msg", "sys", [], None
    )
    costs = (result.cost, result.input_cost, result.output_cost)
    assert costs == (0.0, 0.0, 0.0)


def test_result_keeps_claude_max_marker(monkeypatch):
    """So the archive records which backend produced each program."""
    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, json.dumps(_envelope()), ""
        ),
    )
    result = claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)
    assert result.model_name == "claude-max/opus"


def test_msg_history_round_trips(monkeypatch):
    """async_runner feeds new_msg_history back on patch retries."""
    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, json.dumps(_envelope()), ""
        ),
    )
    result = claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)
    assert result.new_msg_history == [
        {"role": "user", "content": "msg"},
        {"role": "assistant", "content": "hello"},
    ]


def test_structured_output_unsupported():
    with pytest.raises(NotImplementedError):
        claude_max.query_claude_max(_client(), "m", "msg", "sys", [], object())


# --------------------------------------------------------------------------
# Failure classification. The CLI reports subtype="success" even for a 401,
# so is_error/api_error_status are the real signal.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "envelope_overrides,expected",
    [
        (
            {"is_error": True, "api_error_status": 401,
             "result": "Invalid API key · Fix external API key"},
            claude_max.ClaudeMaxAuthError,
        ),
        (
            {"is_error": True, "api_error_status": None,
             "result": "Not logged in · Please run /login"},
            claude_max.ClaudeMaxAuthError,
        ),
        (
            {"is_error": True, "api_error_status": 429,
             "result": "usage limit reached"},
            claude_max.ClaudeMaxRateLimitError,
        ),
        (
            {"is_error": True, "api_error_status": 404,
             "result": "There's an issue with the selected model"},
            claude_max.ClaudeMaxFatalError,
        ),
        (
            {"is_error": True, "api_error_status": 529, "result": "Overloaded"},
            claude_max.ClaudeMaxTransientError,
        ),
    ],
)
def test_failure_classification(envelope_overrides, expected):
    env = _envelope(**envelope_overrides)
    # Exercise the classifier directly: going through query_claude_max would
    # make the transient case actually sleep through the backoff schedule.
    error = claude_max._classify_failure(1, json.dumps(env), "", env)
    assert isinstance(error, expected)


def test_rate_limit_is_fatal_not_retried():
    """Chosen behavior: a 5-hour window reset is not worth backing off for."""
    assert issubclass(
        claude_max.ClaudeMaxRateLimitError, claude_max.ClaudeMaxFatalError
    )
    assert not issubclass(
        claude_max.ClaudeMaxRateLimitError, claude_max.ClaudeMaxTransientError
    )


def test_fatal_errors_are_not_swallowed_by_llm_retry_loops():
    from shinka.llm.llm import FATAL_LLM_EXCEPTIONS

    assert claude_max.ClaudeMaxFatalError in FATAL_LLM_EXCEPTIONS


def test_permanent_failure_runs_the_cli_exactly_once(monkeypatch):
    """MAX_TRIES=20 over MAX_TIME=6000s must never apply to a fatal error."""
    calls = {"n": 0}
    env = _envelope(is_error=True, api_error_status=404, result="no such model")

    def _fake_run(cmd, **kw):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 1, json.dumps(env), "")

    monkeypatch.setattr(claude_max.subprocess, "run", _fake_run)
    with pytest.raises(claude_max.ClaudeMaxFatalError):
        claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)

    assert calls["n"] == 1


def test_unparseable_output_is_fatal(monkeypatch):
    monkeypatch.setattr(
        claude_max.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "not json", ""),
    )
    with pytest.raises(claude_max.ClaudeMaxFatalError):
        claude_max.query_claude_max(_client(), "m", "msg", "sys", [], None)


# --------------------------------------------------------------------------
# Client construction + dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("getter", ["get_client_llm", "get_async_client_llm"])
def test_client_construction(monkeypatch, getter):
    monkeypatch.setenv(cfg.BIN_ENV, "/usr/bin/true")

    client, api_model_name, provider = getattr(llm_client_module, getter)(
        "claude-max/opus"
    )

    assert provider == "claude_max"
    assert api_model_name == "claude-opus-4-7"
    assert client.cli_model_name == "opus"
    assert client.original_model_name == "claude-max/opus"
    assert client.timeout == llm_client_module.TIMEOUT


def test_query_routes_claude_max(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "get_client_llm",
        lambda model_name, structured_output=False: (
            "client", "claude-opus-4-7", "claude_max",
        ),
    )
    monkeypatch.setattr(
        query_module,
        "query_claude_max",
        lambda *a, **kw: "ok",
    )
    assert query_module.query("claude-max/opus", "hello", "sys") == "ok"


def test_query_async_routes_claude_max(monkeypatch):
    monkeypatch.setattr(
        query_module,
        "get_async_client_llm",
        lambda model_name, structured_output=False: (
            "client", "claude-opus-4-7", "claude_max",
        ),
    )

    async def _fake(*a, **kw):
        return "ok-async"

    monkeypatch.setattr(query_module, "query_claude_max_async", _fake)
    result = asyncio.run(
        query_module.query_async("claude-max/opus", "hello", "sys")
    )
    assert result == "ok-async"


# --------------------------------------------------------------------------
# kwargs translation
# --------------------------------------------------------------------------


def test_kwargs_emit_only_cli_consumable_fields():
    kwargs = sample_model_kwargs(
        model_names=["claude-max/claude-opus-4-7"],
        temperatures=[0.0, 0.5],
        max_tokens=[32768],
        reasoning_efforts=["high"],
    )
    assert kwargs == {
        "model_name": "claude-max/claude-opus-4-7",
        "effort": "high",
    }
    # The CLI has no flag for any of these.
    for absent in (
        "temperature", "max_tokens", "max_output_tokens", "thinking",
    ):
        assert absent not in kwargs


def test_kwargs_omit_temperature_for_alias_whose_row_allows_it():
    """claude-sonnet-4-6 has temperature_unsupported=0 in pricing.csv."""
    kwargs = sample_model_kwargs(
        model_names=["claude-max/sonnet"],
        temperatures=[0.7],
        max_tokens=[1024],
        reasoning_efforts=["low"],
    )
    assert "temperature" not in kwargs


def test_kwargs_effort_for_model_without_pricing_row():
    kwargs = sample_model_kwargs(
        model_names=["claude-max/claude-opus-9"],
        temperatures=[0.0],
        max_tokens=[1024],
        reasoning_efforts=["medium"],
    )
    assert kwargs["effort"] == "medium"


@pytest.mark.parametrize(
    "shinka_effort,cli_effort",
    [("min", "low"), ("low", "low"), ("medium", "medium"), ("max", "max")],
)
def test_effort_vocabulary_mapping(shinka_effort, cli_effort):
    kwargs = sample_model_kwargs(
        model_names=["claude-max/opus"],
        temperatures=[0.0],
        max_tokens=[1024],
        reasoning_efforts=[shinka_effort],
    )
    assert kwargs["effort"] == cli_effort


def test_effort_omitted_when_reasoning_disabled():
    kwargs = sample_model_kwargs(
        model_names=["claude-max/opus"],
        temperatures=[0.0],
        max_tokens=[1024],
        reasoning_efforts=["disabled"],
    )
    assert "effort" not in kwargs


def test_api_key_path_kwargs_unchanged():
    """The anthropic branch must keep its adaptive-thinking shape."""
    kwargs = sample_model_kwargs(
        model_names=["claude-opus-4-7"],
        temperatures=[0.0],
        max_tokens=[32768],
        reasoning_efforts=["high"],
    )
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["max_tokens"] == 32768


# --------------------------------------------------------------------------
# Retry constants convention
# --------------------------------------------------------------------------


def test_retry_constants_match_shared_values():
    from shinka.llm.constants import (
        BACKOFF_MAX_TIME,
        BACKOFF_MAX_TRIES,
        BACKOFF_MAX_VALUE,
    )

    assert claude_max.MAX_TRIES == BACKOFF_MAX_TRIES
    assert claude_max.MAX_VALUE == BACKOFF_MAX_VALUE
    assert claude_max.MAX_TIME == BACKOFF_MAX_TIME
