---
title: "fak related tools, daemons, and CLI workflows"
description: "Catalog of the fak binary verbs and companion tools, from serve, run, and preflight to bench, recall, and the CI test runners, and when to use each."
---

# Related Tools, Daemons, and Workflows

This document catalogs the tools, daemons, and workflows that accompany the fak server — what they do, how they integrate, and when to use them.

## Core fak Commands

The primary `fak` binary (`fak.exe` on Windows) provides several verbs for running, testing, and serving the kernel:

| Command | Purpose |
|---------|---------|
| `fak serve` | Start the OpenAI-compatible HTTP gateway with tool-call adjudication |
| `fak run --trace <file>` | Replay a frozen tool-call trace through the kernel (offline testing) |
| `fak preflight --tool <name> --args <json>` | Test a single tool call against the policy (rung-only check) |
| `fak bench --suite <name>` | Run the vDSO ablation benchmark (in-process vs spawned-hook comparison) |
| `fak turntax --suite <name>` | Price the extra error-code model turn the 1-shot kernel deletes |
| `fak agent --offline\|--base-url` | Run live turn-count A/B tests against real models |
| `fak recall --session <dir>` | Persist a finished session as a core dump (durable quarantine) |
| `fak dream --dir <dir>` | Offline cleanup pass over a session core image |
| `fak debug --session <dir>` | Attach to a session core dump and demand-page the working set |
| `fak policy --dump\|--check` | Author/validate the deployable capability floor |
| `fak hook < call.json` | Spawned-hook decide (A/B baseline for benchmarking) |

### `fak serve` — The Gateway Daemon

`fak serve` is the primary daemon for production use. It runs an OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/messages`) that adjudicates tool calls before they reach your client.

**Typical startup:**
```bash
# Front a local Ollama server
fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b

# With custom policy and auth
fak serve --addr 0.0.0.0:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b \
  --policy policy.json \
  --require-key-env FAK_GATEWAY_KEY
```

**Key routes:**
- `GET /healthz` — Unauthenticated liveness check
- `POST /v1/chat/completions` — OpenAI-compatible adjudication proxy
- `POST /v1/messages` — Anthropic Messages API (adjudicated)
- `POST /v1/fak/syscall` — Run one adjudicated tool call directly
- `POST /v1/fak/policy/reload` — Reload policy without restart
- `GET /metrics` — Prometheus metrics

See [`docs/fak/server-quickstart.md`](server-quickstart.md) for full scenarios.

## Testing and CI Tools

### `fak/test.ps1` / `fak/test.sh`

The canonical test runners for fak. On Windows, `test.ps1` runs the Go test suite inside WSL to avoid OS Application Control issues with unsigned test binaries.

**Usage:**
```powershell
# Run the whole suite
.\fak\test.ps1

# Run one package
.\fak\test.ps1 ./internal/ctxmmu/

# Force a clean run (no cache)
.\fak\test.ps1 -count=1 ./...
```

The underlying `fak/test.sh` can be called directly from WSL.

### `fak/scripts/ci.ps1`

The CI gate that runs build + vet + test + claims lint as one mechanical witness. This is what CI pipelines should invoke.

**Usage:**
```powershell
.\fak\scripts\ci.ps1
```

Exits non-zero on any failure.

## Demo and Example Scripts

### `fak/examples/adjudication-demo/run.sh`

Live demonstration of the kernel's capability gate. Drives a real local model behind `fak serve` and shows:
- **CONSTRUCTIVE** — The kernel allows safe tool calls that execute and clean up
- **ADVERSARIAL** — We instruct the model to propose dangerous calls; the kernel refuses every one

**Usage:**
```bash
./examples/adjudication-demo/run.sh            # Full demo
./examples/adjudication-demo/run.sh --dry-run  # Show verdicts without execution
```

See [`fak/examples/adjudication-demo/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/adjudication-demo/README.md) for details.

### `fak/cmd/simpledemo`

Friendliest way to run a local AI model on your own computer. Auto-finds `.gguf` models in common locations and provides an interactive chat interface.

**Usage:**
```bash
go run ./cmd/simpledemo

# With specific model
.\simpledemo.exe -gguf C:\path\to\model.gguf
```

See [`fak/cmd/simpledemo/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/cmd/simpledemo/README.md) for model recommendations and troubleshooting.

## Model Fetching Scripts

### `fak/scripts/fetch-model.sh` / `fetch-model.ps1`

Fetches SmolLM2 weights for the in-kernel model engine. Creates a Python venv, installs dependencies, downloads from HuggingFace, and exports to `internal/model/.cache/`.

**Usage:**
```bash
# Linux/macOS/WSL
./fak/scripts/fetch-model.sh

# Windows PowerShell
.\fak\scripts\fetch-model.ps1

# Check prerequisites only
./fak/scripts/fetch-model.sh --check
```

### `fak/scripts/fetch-gguf.sh` / `fetch-gguf.ps1`

Downloads GGUF model weights for local inference.

**Usage:**
```bash
./fak/scripts/fetch-gguf.sh qwen2.5:1.5b
```

## Policy Templates

The `fak/examples/` directory contains policy manifest templates for different agent use cases:

| File | Intended use | Main boundary |
|------|--------------|---------------|
| `policy.example.json` | General manifest shape | Explicit destructive denies + provenance/IFC |
| `dev-agent-policy.json` | Coding agent in this repo | No shared-history mutations without release discipline |
| `customer-support-readonly-policy.json` | Support lookup + ticket handoff | Read/customer-ticket workflow; no direct account action |
| `research-agent-policy.json` | Open-web research and note taking | Read/search/summarize; no posting, shell, upload |
| `devops-dryrun-policy.json` | Infra review without execution | Plan/diff/template only; no apply or delete |

**Usage:**
```bash
# Validate a policy
fak policy --check examples/customer-support-readonly-policy.json

# Use with any verb
fak serve --policy examples/dev-agent-policy.json ...
```

See [`fak/examples/README.md`](https://github.com/anthony-chaudhary/fak/blob/main/examples/README.md) for the full template catalog.

## Fleet Operations Tools

### `tools/fleet_sessions.py`

The cross-account "what stopped, why, and how to resume" index. Scans all Claude Code sessions on the host, categorizes dispositions (LIVE, DONE, DEAD_MIDTOOL, STOPPED_LIMIT, etc.), and produces account availability status.

**Modes:**
- `summary` (default) — Compact operator table grouped by disposition
- `json` — Full machine payload
- `resume` — Ready-to-run resume commands for genuinely-stopped sessions

**Usage:**
```bash
# Show status
python3 tools/fleet_sessions.py summary

# Get resume commands
python3 tools/fleet_sessions.py resume

# JSON output
python3 tools/fleet_sessions.py json --window 24
```

### `tools/fleet_resume_watchdog.py` / `.ps1`

The cross-account resume layer for autonomous Claude sessions. Runs on a ~5-minute cron schedule to automatically resume DEAD sessions under their correct accounts.

**Features:**
- DRY-RUN by default (set `FAK_LIVE=1` or pass `--live`)
- Resume-once enforcement via durable ledger
- Re-homes throttled sessions to healthy accounts
- Notifications for auth-blocked accounts

**Usage:**
```bash
# Dry run
python3 tools/fleet_resume_watchdog.py

# Live mode
python3 tools/fleet_resume_watchdog.py --live
```

### `tools/fleet_supervisor_watchdog.py` / `.ps1`

Keeps the job-fleet supervisor alive as a detached process. When the supervisor dies (crash, host sleep), this watchdog re-launches it.

**Usage:**
```bash
# Enable supervision
export FAK_SUPERVISOR_ENABLE=1
python3 tools/fleet_supervisor_watchdog.py
```

Exit codes: 0 = alive/disabled | 10 = respawned.

### `tools/fleet_status.ps1` / `tools/fleet_status.py`

Quick status overview of the fleet: which sessions are running, which accounts are throttled, and current supervisor state.

**Usage:**
```powershell
.\tools\fleet_status.ps1
```

## Release Tools

### `tools/release_bump.py`

Bumps the `VERSION` file for a new release based on semantic versioning.

### `tools/sync_memory.py`

Copies between the local auto-memory store (`~/.claude/projects/<slug>/memory/`) and the committed mirror (`.claude/memory/`).

**Usage:**
```bash
# Push home store to repo mirror before committing
python3 tools/sync_memory.py --push

# Pull repo mirror to home store when seeding a fresh node
python3 tools/sync_memory.py --pull
```

The memory store layout itself is operator-private; the `sync_memory.py` pull flow
above is the public-facing seam.

## Benchmarking Tools

### `fak/cmd/fanbench`

Benchmark for measuring fan-out performance with N sub-agents.

### `fak/cmd/sessionbench`

Session-based benchmarking tool.

### `tools/permission_system_benchmark.py`

Permission system benchmark methodology and execution.

### `tools/transcript_workload.py`

Derives realistic workload profiles from Claude Code transcripts for benchmarking.

## Development Scripts

### `fak/scripts/dogfood-claude.sh` / `.ps1`

One-command setup to run fak as a local model backend for the Claude Code CLI. Starts a local model behind `fak serve` and points Claude Code at it.

**Usage:**
```bash
# Linux/macOS
./fak/scripts/dogfood-claude.sh

# Windows PowerShell
.\fak\scripts\dogfood-claude.ps1
```

### `tools/agent_walltime.py`

Analyzes Claude Code session transcripts to measure where agent time goes (model vs tools vs idle).

**Usage:**
```bash
python3 tools/agent_walltime.py --since-hours 24
```

### `tools/session_audit.py`

Audits recent Claude Code sessions for token-weighted cost/efficiency metrics.

## Workflow Examples

### Local Development Workflow

```bash
# 1. Build and test
go build ./...
.\fak\test.ps1

# 2. Run the kernel in offline mode
./fak run --trace testdata/tau2/tau2-smoke.json

# 3. Test policy decisions
./fak preflight --tool delete_account --args '{}'

# 4. Start the gateway with local model
ollama serve &
./fak serve --addr 127.0.0.1:8080 \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:1.5b

# 5. Verify
curl http://127.0.0.1:8080/healthz
```

### Production Deployment Workflow

```bash
# 1. Author and validate policy
fak policy --dump > floor.json
# Edit floor.json
fak policy --check floor.json

# 2. Build production binary
go build -o fak ./cmd/fak

# 3. Start with auth and monitoring
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
export FAK_HTTP_WRITE_TIMEOUT_S=600

./fak serve --addr 0.0.0.0:8080 \
  --base-url https://api.openai.com/v1 \
  --provider openai \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY \
  --policy floor.json \
  --require-key-env FAK_GATEWAY_KEY

# 4. Monitor
curl -H "Authorization: Bearer $FAK_GATEWAY_KEY" \
  http://127.0.0.1:8080/metrics
```

### Fleet Operations Workflow

```bash
# 1. Check fleet status
python3 tools/fleet_sessions.py summary

# 2. Resume stopped sessions if needed
python3 tools/fleet_resume_watchdog.py --live

# 3. Verify supervisor is alive
python3 tools/fleet_supervisor_watchdog.py

# 4. Run overnight soak
.\tools\run_overnight_soak.ps1
```

## Integration Points

### For Claude Code Users

- **`fak serve`** as a local model backend: See [`fak/cmd/simpledemo/CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/cmd/simpledemo/CLAUDE.md)
- **`dogfood-claude.sh`**: One-command local model + kernel setup
- **`fleet_sessions.py`**: Track stopped sessions across accounts

### For Gateway Users

- **`/v1/fak/syscall`**: Direct adjudicated tool call execution
- **`/v1/fak/policy/reload`**: Hot-reload capability floor
- **`/metrics`**: Prometheus metrics for observability

### For Integrators

- **`fak run`**: Offline trace replay for testing
- **`fak preflight`**: Per-call policy oracle
- **`fak policy --check`**: Pre-deployment validation

## See Also

- [`docs/cli-reference.md`](https://github.com/anthony-chaudhary/fak/blob/main/README.md) — Main fak documentation
- [`fak/GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md) — Install and run guide
- [`fak/POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — Capability floor schema
- [`docs/fak/server-quickstart.md`](server-quickstart.md) — Server deployment scenarios
- [`CONTRIBUTING.md`](https://github.com/anthony-chaudhary/fak/blob/main/CONTRIBUTING.md) — Contributor guide & repo conventions
