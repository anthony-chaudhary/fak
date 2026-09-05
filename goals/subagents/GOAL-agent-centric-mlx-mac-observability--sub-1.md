---
parent_goal: goals/GOAL-agent-centric-mlx-mac-observability.md
sub_step: 1_audit_and_contract
witness: "code audit and schema contract definition for 10x agent-centric MLX & Mac observability"
target_files:
  - internal/engine/mlx.go
  - internal/macfit/macfit.go
  - internal/compute/gpustats_darwin.go
  - internal/modelperfobs/darwin_pressure.go
---
# Sub-Goal Objective
Audit current MLX and Mac-specific performance telemetry in `internal/engine/mlx.go`, `internal/macfit/macfit.go`, `internal/compute/gpustats_darwin.go`, and `internal/modelperfobs/darwin_pressure.go`.
Define the comprehensive schema contract for 10x agent-centric observability:
1. What does MLX (both mlx-lm and vllm-mlx or native ride) currently expose vs what is missing for agents?
2. What Mac hardware/OS signals (IORegistry IOAccelerator, sysctl wired_mem_limit, swap, memory pressure level, thermal state) must be captured?
3. How do we synthesize these into direct agent decisions:
   - Agent concurrency headroom (how many concurrent subagents can fit without swap)?
   - Prefix cache hit ratio & TTFT savings (is multi-turn preamble reused)?
   - Bottleneck diagnosis (memory-bound decode vs compute-bound prefill vs lock contention)?
   - Closed action verdict tokens (`HEADROOM_OK`, `PRESSURE_DEGRADE`, `EVICT_PREFIX_CACHE`, `REDUCE_CONCURRENCY`, `THERMAL_THROTTLED`)?
4. Produce a detailed architectural specification and schema for `internal/macobs`.
