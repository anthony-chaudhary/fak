---
loop: goal
goal_slug: native-harness-orchestration
witness: "go test -v ./platform/harness/... ./platform/dispatch/..."
budget: { max_iters: 20 }
lane: harness
---
# Objective
Deliver first-class native harness orchestration and lifecycle management in `fak` and `fak-private` for orchestrating and managing foreign agent harnesses (headless OpenCode, OpenAI Codex, Claude Code, Pi, and native Fak harnesses) directly via native process controls, headless flags, non-blocking event streams, and deep forensic data store taps. Transition decisively away from legacy out-of-process wire interception proxies (`fak guard`) toward native, first-party harness control and multi-harness fleet observability.

# Non-Goals
- Do not reintroduce MITM reverse-proxy wire interception or synthetic URL rewrites (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` redirection).
- Do not modify core frozen ABI (`internal/abi`).
- Do not author loose PowerShell (`.ps1`), Bash (`.sh`), or batch scripts (native Go programs only).
- Do not bypass or break existing 5-Gate boundary invariants (`cmd/fak-boundary`).
- Do not disrupt active contract lease arbitration (`platform/leaseref`, `platform/dispatch/contract_lease.go`).

# Plan
- [x] 1. **Milestone 1: Architectural Specification & Goal Registration**
  - Codify the paradigm shift from wire-proxy interception to native harness control in `docs/architecture/NATIVE-HARNESS-ORCHESTRATION-AND-MANAGEMENT.md`.
  - Register `goal_native_harness_orchestration` in canonical registry (`goals/fak/registry/goals.json`).
- [ ] 2. **Milestone 2: Unified Harness Driver & Supervisor Core (`platform/harness`)**
  - Implement `platform/harness/types.go` with `HarnessType`, `HarnessSpec`, `HarnessSession`, `SessionState`, `HarnessEvent`, `HarnessReceipt`, and `HarnessDriver` interface.
  - Implement `platform/harness/supervisor.go` with thread-safe session registry, capacity bounding, in-flight cancellation, and graceful drain.
- [ ] 3. **Milestone 3: Process Tree Governance & Sandbox Isolation**
  - Implement cross-platform process tree control (`process.go`, `process_windows.go`, `process_posix.go`) using Windows Job Objects and POSIX process groups to eliminate orphaned child processes (linters, compilers, tests).
  - Enforce timeout ceilings, memory RSS limits, and dedicated git worktree sandbox isolation (`platform/worktree`).
- [ ] 4. **Milestone 4: Native Harness Drivers (OpenCode, Codex, Claude Code, Fak Native)**
  - Implement `driver_opencode.go` configuring `opencode run --print-logs --dangerously-skip-permissions`, directory scoping, ndjson stream parsing, and read-only SQLite forensic extraction (`opencode.db` via `platform/opencode`).
  - Implement `driver_codex.go` configuring `codex exec --dangerously-bypass-permissions`, prompt-task admission, input queue steering (`send_message`), and JSON-RPC event parsing.
  - Implement `driver_claude.go` configuring `claude -p --permission-mode bypassPermissions` and non-interactive stream handling.
  - Implement `driver_fak.go` for in-process `pkg/harnesskit` or standalone binary execution.
- [ ] 5. **Milestone 5: Deep Observability, Stall Detection, and Receipt Emission**
  - Implement real-time multiplexed stdout/stderr capture without blocking or memory exhaustion (`observability.go`).
  - Implement stall detection: catch hung processes, confirmation prompt deadlocks, and repetitive tool-thrashing loops within a bounded time window.
  - Standardize durable execution receipts (`HarnessReceipt`) with microsecond tool latencies, token consumption breakdown, exit status, and witness verification hashes.
- [ ] 6. **Milestone 6: Factory Integration & Dispatch Wiring (`platform/dispatch`)**
  - Integrate `platform/harness` into `platform/dispatch/worker.go` and `runner.go`.
  - Provide complete test coverage (`supervisor_test.go`, `driver_test.go`, `observability_test.go`, `receipt_test.go`).
  - Verify 5-gate boundary checks (`cmd/fak-boundary check --staged`) and package tests.

# Scoreboard & Target Metrics
- Process cleanup guarantee: 100% of spawned child processes killed on cancel/timeout (zero zombie leakage).
- Stall detection latency: < 30s detection of hung/prompt-locked child processes.
- Memory overhead: < 5MB per supervised harness session (vs 35-50MB for legacy `fak guard` HTTP proxy daemon).
- Forensic precision: microsecond-level tool start/end timing extracted directly from native data stores.
