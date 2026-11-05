# Claude Max Subscription

Shinka can drive Claude through your **Claude Max/Pro subscription** instead of
metered API billing, by shelling out to the locally installed `claude` CLI in
headless mode. The CLI authenticates from your OAuth credentials, so no
`ANTHROPIC_API_KEY` is involved.

Both paths coexist. A model name decides which one a call takes:

| Model name | Backend | Billing |
|---|---|---|
| `claude-opus-4-7` | Anthropic API | `ANTHROPIC_API_KEY`, metered |
| `claude-max/claude-opus-4-7` | `claude` CLI | Your subscription |

---

## Prerequisites

1. Install Claude Code and make sure `claude` is on `PATH`.
2. Log in as the same OS user that runs Shinka:

```bash
claude auth               # interactive login
# or, for a long-lived token:
claude setup-token
```

Verify:

```bash
claude -p "say ok" --output-format json
```

If the binary lives somewhere unusual, point Shinka at it:

```bash
export SHINKA_CLAUDE_CLI_BIN=/opt/claude/bin/claude
```

---

## Usage

```yaml
evo_config:
  llm_models:
    - claude-max/claude-opus-4-7
```

```bash
motiondisco_run --task-dir examples/circle_packing \
  --set evo.llm_models='["claude-max/claude-opus-4-7"]'

motiondisco_launch task=loco_manipulation \
  'evo_config.llm_models=[claude-max/claude-opus-4-7]'
```

### Model names and aliases

Anything after `claude-max/` is handed to `claude --model` verbatim, so both
full ids and CLI aliases work:

```yaml
- claude-max/claude-opus-4-7   # full id
- claude-max/opus              # alias -> latest Opus
- claude-max/sonnet
- claude-max/haiku
```

Aliases resolve to a `pricing.csv` row for capability flags and cost reporting
(`opus` → `claude-opus-4-7`, `sonnet` → `claude-sonnet-4-6`, `haiku` →
`claude-haiku-4-5-20251001`). Unknown names still work — the CLI is the
authority on which models exist — they just report zero notional cost.

### Mixing backends in one run

Both may appear in the same `llm_models` list, in which case the UCB bandit
treats them as separate arms and compares them directly:

```yaml
evo_config:
  llm_models:
    - claude-max/claude-opus-4-7
    - claude-opus-4-7
```

Program metadata records the full name, so the archive shows which backend
produced each program.

---

## What is and isn't sent

`sample_model_kwargs` emits only what the CLI can consume. The `--effort` flag
is mapped from Shinka's `reasoning_efforts`:

| Shinka | CLI |
|---|---|
| `min`, `low` | `--effort low` |
| `medium` | `--effort medium` |
| `high` | `--effort high` |
| `max` | `--effort max` |
| `disabled` | flag omitted |

`temperature`, `max_tokens`, and `thinking` have **no CLI equivalent** and are
therefore not sampled for `claude-max/` models — `QueryResult.kwargs` records
exactly what was sent rather than parameters the call never carried. Output
length is bounded by the model's own limit (64k for Opus) rather than by
`evo_config.llm_kwargs.max_tokens`.

Structured output (`output_model`) is not supported, matching the API-key
Anthropic path.

---

## Hermetic invocation

Each call runs as:

```
claude -p --output-format json --model <model> --system-prompt <sys>
       --tools "" --safe-mode --strict-mcp-config
       --no-session-persistence --permission-mode dontAsk [--effort <level>]
```

with the prompt on **stdin** and the working directory set to a scratch temp
dir. This matters more than it looks:

- **`--safe-mode`** stops the CLI from loading `CLAUDE.md`, skills, plugins,
  hooks, MCP servers, and custom commands. Without it, a run launched from a
  repo would silently inject that repo's instructions into every evolution
  prompt. Auth and model selection keep working normally.
- **`--system-prompt`** replaces Claude Code's coding-agent prompt rather than
  appending to it, so the model receives Shinka's task prompt and nothing else.
- **`--tools ""`** disables all tools: this is a text completion, not an agent
  session.
- **`--no-session-persistence`** avoids writing a session file per generation.
- **stdin, not argv** — proposal prompts carry the parent program plus archive
  inspirations and routinely exceed Linux's 128 KB single-argument limit, which
  would otherwise fail with `E2BIG`.
- **scratch cwd** keeps the invocation independent of where the run started.

`--bare` is deliberately *not* used: it forces `ANTHROPIC_API_KEY` auth and
never reads OAuth — the opposite of what this backend is for.

### `ANTHROPIC_API_KEY` is stripped from the subprocess

If an `ANTHROPIC_API_KEY` is visible to the CLI, it authenticates with **the
API key and ignores OAuth entirely** — a run that appears to work while quietly
billing the API. Since `shinka.llm.client` calls `load_shinka_dotenv()` at
import, a repo `.env` puts exactly that key into `os.environ` for every Shinka
process.

Shinka therefore removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_BASE_URL`, the `ANTHROPIC_*_MODEL` overrides, and the Bedrock/Vertex
switches from the child environment. To bill the API on purpose, use a plain
model name instead of a `claude-max/` one.

---

## Cost accounting

Nothing is actually billed on a subscription, but the CLI reports what the call
*would* have cost. Shinka records that notional figure so `max_api_costs`
budgets and the cost-aware bandit keep working unchanged:

```yaml
evo_config:
  max_api_costs: 6.0    # still stops the run
```

Two caveats:

- `total_cost_usd` covers the whole invocation, including the CLI's own
  internal model calls, so `cost` can slightly exceed `input_cost + output_cost`
  (which are derived from `pricing.csv` for the primary model).
- The real constraint is your plan's **rate-limit window**, not dollars.

To report zero instead — for example to make the subscription arm look free to
the bandit against paid models:

```bash
export SHINKA_CLAUDE_MAX_ZERO_COST=1
```

Note this disables the `max_api_costs` guard, so pair it with an explicit
`num_generations`.

---

## Rate limits fail fast

Subscription limits reset on a rolling multi-hour window, so backing off is
close to useless. A rate-limit or auth failure raises immediately and
propagates through Shinka's retry loops (which normally swallow exceptions),
stopping the run with an actionable message. Restart once the window resets.

Transient failures — overload, connection resets, timeouts — still retry with
the standard exponential backoff.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `SHINKA_CLAUDE_CLI_BIN` | Absolute path to `claude` if not on `PATH` |
| `SHINKA_CLAUDE_CLI_TIMEOUT` | Per-call timeout in seconds (default 1200) |
| `SHINKA_CLAUDE_CLI_EXTRA_ARGS` | Extra CLI flags, shell-quoted |
| `SHINKA_CLAUDE_MAX_ZERO_COST` | Report cost as `0.0` |

---

## Trade-offs

- **Latency.** Every call spawns a process and pays CLI startup, so it is
  slower than an HTTP request. Keep `max_proposal_jobs` modest.
- **Multi-turn fidelity.** `claude -p` is single-turn, so a replayed
  `msg_history` is flattened into a tagged transcript in one user message.
  This only occurs when `max_patch_attempts > 1`; at the default of `1` every
  call is single-turn on both backends.
- **`motiondisco_models`** does not list `claude-max/` models, matching the
  `local/` backend's behavior.
