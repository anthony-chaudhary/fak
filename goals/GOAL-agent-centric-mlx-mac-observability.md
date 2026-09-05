---
loop: goal
goal_slug: agent-centric-mlx-mac-observability
witness: "go test -v ./internal/macobs/... && fak validate --mine internal/macobs/ cmd/fak/macobs.go"
budget: { max_iters: 15 }
lane: macobs
---
# Objective
Deliver 10x more agent-centric observability into MLX and Mac-specific performance metrics, unified memory dynamics, and actionable runtime steering for autonomous agents and subagents on Apple Silicon.

# Non-Goals
- Do not modify the frozen ABI (`internal/abi/`).
- Do not introduce non-Go external dependencies or new shell/Python scripts.
- Do not fabricate hardware or runtime counters when absent; maintain honest provenance (`WITNESSED`, `MODELED`, `UNAVAILABLE`).
- Do not break existing `internal/macbench`, `internal/macfit`, or `internal/engine/mlx.go` behaviors.

# Plan
- [x] 1. Audit existing MLX and Mac performance observability and define the 10x agent-centric contract.
- [x] 2. Implement `internal/macobs` leaf package with MLX runtime parsing, Mac unified memory telemetry, agent concurrency headroom modeling, and structured action verdicts.
- [x] 3. Implement `fak macobs` CLI verb with human and agent JSON envelopes (`fak.macobs.v1`).
- [x] 4. Add comprehensive unit, mock, and darwin-integration tests covering MLX metrics, Apple Silicon IORegistry/sysctl hardware counters, and agent recommendations.
- [x] 5. Execute final independent witness gate and seal goal evidence.

# Results and Verification Evidence

### Executive Summary
Delivered 10x more agent-centric observability into MLX and Apple Silicon Mac performance through the new `internal/macobs` leaf package and `fak macobs` CLI verb.

Key capabilities delivered:
1. **Apple Silicon Hardware Telemetry (`HardwareTelemetry`)**:
   - `IOAccelerator` via `/usr/sbin/ioreg` XML parsing: `Alloc system memory`, `In use system memory`, `Device Utilization %`, `Renderer Utilization %`, `recoveryCount`.
   - Kernel `sysctl`: `hw.memsize`, `iogpu.wired_limit_mb` (and `iogpu.wired_mem_limit`), `vm.swapusage`, `kern.memorystatus_level`.
   - `vm_stat` paging dynamics: pages compressed, wired, free, page-ins, page-outs.
   - Thermal and power management (`pmset`): thermal state (`NOMINAL`, `FAIR`, `SERIOUS`, `CRITICAL`), CPU/GPU thermal levels, power source (`AC` vs `BATTERY`), battery percentage.

2. **MLX Serving Telemetry (`MLXServingTelemetry`)**:
   - Scrapes and parses Prometheus endpoints for `vllm-mlx` and `mlx-lm`.
   - Captures active/queued requests, KV cache usage percentage, TTFT, ITL, prompt/decode tokens/sec, and prefix cache hit/miss ratio.

3. **Multi-Agent Concurrency Headroom Modeling (`HeadroomTelemetry`)**:
   - Calculates exact model KV bytes per token given layer count, KV heads, head dimension, and element bytes.
   - Computes available KV pool under physical unified memory, OS reserve, and wired memory limit.
   - Evaluates `MaxSharedAgents` (prefix reuse enabled) vs `MaxIsolatedAgents` (independent contexts) and computes the exact `ConcurrencyAdvantage` multiplier.

4. **Agent Action Verdicts & Bottleneck Diagnosis (`AnalysisReport`)**:
   - Closed actionable verdict vocabulary: `HEADROOM_OK`, `REDUCE_CONCURRENCY`, `EVICT_PREFIX_CACHE`, `PRESSURE_DEGRADE`, `THERMAL_THROTTLED`, `SWAP_CRITICAL`.
   - Closed bottleneck classifications: `BOTTLENECK_NONE`, `BOTTLENECK_MEMORY_BANDWIDTH`, `BOTTLENECK_PREFILL_COMPUTE`, `BOTTLENECK_QUEUE_CONTENTION`, `BOTTLENECK_SWAP`, `BOTTLENECK_THERMAL`.
   - Concrete remediation strings and one-line admission evaluation (`gate_passed: bool`).

5. **CLI Verb `fak macobs`**:
   - Supports `--json` emitting canonical `fak.macobs.v1` envelope.
   - Supports `--check-headroom` for fast subagent admission checks.
   - Supports `--agents`, `--prefix-tokens`, `--tail-tokens`, `--mlx-endpoint`, and `--watch`.

### Independent Witness Verification
- `go test -v ./internal/macobs/...`: 26 passed (0 failures)
- `go test -v ./cmd/fak -run TestRunMacObs`: 8 passed (0 failures)
- `go test -v ./internal/devindex -run 'TestVerbTierCoverageIsTotal|TestVerbManifestCoversEveryDispatcherVerb'`: passed
- `fak validate --mine internal/macobs/ cmd/fak/macobs.go cmd/fak/macobs_test.go internal/devindex/tiers.go internal/devindex/verbs.go internal/devindex/devreuse.go cmd/fak/main.go dos.toml`:
  `OK: committed tip 5ee8b5714ed2 + 20 owned path(s) importer build/vet and changed-package tests clean`

# Scratch / last-refusal
