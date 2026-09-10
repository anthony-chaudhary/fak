# Native Harness Orchestration & Management Architecture

> **Authoritative Platform Architecture Specification**  
> **Topic:** First-Class Native Orchestration and Management of External Agent Harnesses (OpenCode, OpenAI Codex, Claude Code, and Fak Native)  
> **Paradigm:** Transition from Wire Proxy Interception (`fak guard`) to Native Subprocess Governance & Deep Telemetry  
> **Classification:** Gate 1 (Factory Test & Autonomous Development Flywheel) / Gate 2 (Private Serving Infrastructure)  
> **Date:** 2026-09-09  

---

## 1. Executive Summary & The Paradigm Shift

### The Legacy 'guard' Mindset (Out-of-Process Wire Proxy Interception)
Historically, **`fak guard`** approached harness governance by interposing an external HTTP reverse-proxy between an agent runtime and the upstream LLM API. The operator or factory configured:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8080/v1"
export OPENAI_BASE_URL="http://localhost:8080/v1"
fak guard -- claude
```
This design was built around a specific assumption: *if you can intercept model completions and tool call proposals at the network wire, you can enforce security policies and cache read-only tools without modifying external agent runtimes.*

While effective as a zero-code wrapper for interactive desktop exploration, this wire-proxy model breaks down severely when orchestrating large-scale autonomous fleets and managing modern external coding agents:

1. **Protocol Rigidity & Streaming Mismatch:** Modern coding harnesses (OpenCode, Codex, Claude Code) rely increasingly on non-standard wire features: streaming delta tool calls, partial AST diffs, native thinking/reasoning blocks (`delta.reasoning_content`, `<think>`, encrypted chain-of-thought tokens), bidirectional WebSockets, and direct MCP process transports. A generic HTTP proxy parser either strips critical reasoning fields or corrupts chunk boundaries.
2. **Process Blindness & Orphaned Subprocesses:** The proxy daemon only monitors its loopback TCP socket. When an underlying agent CLI crashes, freezes on an interactive confirmation prompt (`Do you want to run this bash command? [y/N]`), or spawns un-tracked background subprocesses (e.g. `go test`, clang compilers, linters), the proxy hangs indefinitely. It cannot kill child process sub-trees, leading to runaway CPU, memory exhaustion, and zombie processes.
3. **Permission Layer Friction & Deadlocks:** Modern harnesses already ship with their own robust permission models, sandboxing options, and headless automation flags (e.g. OpenCode `--dangerously-skip-permissions`, Codex `--dangerously-bypass-permissions`). Superimposing a synthetic network-level permission gate caused persistent operational deadlocks in autonomous pipelines (witnessed in `agent-memory/fak/bg-worker-edit-deadlock-workaround.md` where background workers were paralyzed when attempting git worktree commits).
4. **Daemon Overhead & Ephemeral Socket Depletion:** Running 20–50 parallel worker subagents in autonomous waves (e.g. `platform/dispatch/`) required spawning 20–50 independent proxy daemon instances, each consuming 35–50 MB RSS and allocating TCP ports and file descriptors, quickly hitting OS ephemeral port limits.

---

### The Native Controls Paradigm: Manage the Harness *As Is*

The **Native Harness Orchestration** architecture discards the MITM proxy illusion. Instead, our native harness assumes the role of **First-Class Supervisor and Orchestrator**:

Rather than wrapping external agents with network interception, the native supervisor:
- Invokes external harnesses natively with first-party headless flags and parameters.
- Governs the OS process tree deterministically to prevent runaway background processes.
- Mounts isolated sandboxes and git worktrees directly into each harness instance.
- Observes execution through real-time log demuxing and direct queries against the harness's local persistence stores (e.g. SQLite databases).
- Communicates directly with upstream model serving (`fak serve` or cloud endpoints) without an intermediate proxy hop.

---

## 2. The Four Architectural Pillars

### Pillar 1: Unified Driver Abstraction (`HarnessDriver`)

Every external agent framework has its own CLI ergonomics, configuration schemas, and communication interfaces. The supervisor abstracts these differences through a clean, stateless driver interface:
- **OpenCode Driver (`HarnessDriverOpenCode`)**: `opencode run --print-logs --dangerously-skip-permissions --agent <agent> "<prompt>"` or `opencode serve --port <port> --headless`.
- **OpenAI Codex Driver (`HarnessDriverCodex`)**: `codex exec --dangerously-bypass-permissions "<prompt>"` or `codex-rs` RPC mode with discrete prompt-task admission and in-flight input queue steering.
- **Claude Code Driver (`HarnessDriverClaude`)**: `claude -p --permission-mode bypassPermissions "<prompt>"`.
- **Fak Native Driver (`HarnessDriverFakNative`)**: In-process execution via `pkg/harnesskit` or `fak dev / fak loop drive` binary execution.

### Pillar 2: Deterministic Process Tree Governance & Sandbox Isolation
- Cross-platform process groups (Windows Job Objects with `KillOnJobClose`, POSIX `Setpgid`).
- Complete elimination of orphan child processes (linters, tests, compilers).
- Execution timeouts, memory RSS quotas, and isolated git worktree sandboxing (`platform/worktree`).

### Pillar 3: Deep Native Observability & Forensics
- Real-time multiplexed stdout/stderr capture with structured event stream normalization (`HarnessEvent`).
- Direct read-only SQLite forensic tap (`opencode.db` via `platform/opencode`) extracting microsecond tool timings (`start_ms`/`end_ms`), token counts (input/output/reasoning/cache), and subagent hierarchy tree rollups.
- Heartbeat monitoring and stall detection (<30s) for prompt-deadlocks and tool-thrashing loops.
- Standardized execution receipt (`HarnessReceipt`) with cryptographic provenance.

### Pillar 4: Factory Integration & Two-Tier Pacing
- Two-tier governor coordination: yielding inference slots during heavy child harness I/O turns.
- Binding to active contract leases (`platform/leaseref`).
- Fleet metrics for active processes, durations, tool latencies, stalls, and token efficiency.
