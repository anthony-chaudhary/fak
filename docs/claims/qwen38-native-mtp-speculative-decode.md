---
title: "Qwen 3.8 native MTP speculative decode — shipped mechanisms, simulated performance"
description: "Shipped MTP rollback, cache-coexistence, and Metal verification mechanisms; comparative throughput and hardware scaling remain simulated pending physical evidence."
---

# Qwen 3.8 native MTP speculative decode — shipped mechanisms, simulated performance

[← Claims index](../../CLAIMS.md)

> **Mixed-status claim.** The implementation mechanisms below retain their `[SHIPPED]` status.
> The 15.22 tok/s 4-way comparison and hardware-speedup values are `[SIMULATED]` fixtures:
> their advertised packet and raw evidence files are missing. See the
> [simulated-results discipline](../standards/simulated-results-discipline.md) and open
> measurement issue [#12239](https://github.com/anthony-chaudhary/fak/issues/12239).

- [SHIPPED] On-device recurrent state rollback (`internal/compute/recurrent_rollback.go`): shadows live convolution and recurrent state slots directly in GPU VRAM and performs atomic rewind to the accepted token prefix with strictly zero device-to-host transfers (`D2HBytes == 0`, `D2HEvents == 0`, `ZeroD2HVerified == true`). Witness: `internal/compute/recurrent_rollback_test.go` `TestRecurrentRollbackAccuracy`, `TestRecurrentRollbackZeroD2H`, and `TestRecurrentRollbackLatencyImprovement`.
- [SHIPPED] MTP cache coexistence and graceful fallback downgrade (`internal/model/mtp_cache_coexist.go`): supports prompt prefix reuse with MTP draft sessions. When divergence $\le$ `MaxRecurrentRollbackDepth` (default $K=4$), executes on-device recurrent rollback; when divergence exceeds rollback depth, executes graceful cold prefill fallback while preserving MTP readiness for subsequent turns. Witness: `internal/model/mtp_cache_coexist_test.go` `TestMTPCacheCoexistPromptDivergenceRollbackAndFallback`.
- [SHIPPED] Apple Silicon Metal MTP draft-verify-rollback loop (`internal/model/metal_mtp.go`, `internal/agent/inkernel_planner.go`): resident MTP candidate proposal generation, wide-M Metal verification dispatch, atomic Context-MMU page commit/rollback (`ctxmmu.CheckpointManager`), and greedy temperature-zero tripwires into an autonomous speculative decode loop. Integrated into `InKernelPlanner.generateReusedRecovering` with verified bit-exact token sequence identity against baseline autoregressive decode and prompt prefix cache coexistence. Witness: `internal/model/metal_mtp_test.go` `TestMetalMTPDraftVerifyRollbackLoop` and `internal/agent/inkernel_planner_test.go` `TestInKernelPlannerMetalMTP`.
- [SHIPPED] Wide-M Tail-Causal SDPA Metal 4 TensorOps and batched recurrent GDN (`internal/metalgemm/sdpa_nax_tile.go`): Multi-token speculative verification kernel ($M=16..24$) reading K/V tiles once per simdgroup pair with bit-exact parity against serial execution, 16x-24x memory access reduction, zero hot-path allocations, and 1.07x latency ratio at $M=4$ achieving >3.2x arithmetic efficiency per streamed weight byte. Witness: `internal/metalgemm/sdpa_nax_tile_test.go` `TestMetalWideMSpeculativeVerification`.
- [SIMULATED] 4-way Apple Silicon MTP comparison fixture (`internal/macbench/mtp_comparison.go`, `cmd/fak/macbench.go`): the 15.22 tok/s fak-native value, 0.785 acceptance rate, and comparisons with llama.cpp, AX Engine, and MTPLX are constructor-generated fixture values, not observed Apple M3 Pro measurements. The advertised `experiments/benchmark/runs/by-machine/node-macos-a/20260908T160000Z-macbench-mtp/packet.json` and its digest-bound raw/quality files are not committed. Schema/unit validation proves internal arithmetic only. Physical promotion remains open in [#12239](https://github.com/anthony-chaudhary/fak/issues/12239).
- [SHIPPED] Multi-Token Prediction parameter tuning engine (`cmd/tunemtp`, `internal/mtptune`): automated search and profiling over draft depth $K \in \{1..8\}$ and acceptance probability curves. The engine and deterministic tests are shipped; the emitted $K=4$ throughput ranking remains simulated until a physical, source-bound sweep is captured. Witness: `cmd/tunemtp/main_test.go` and `internal/mtptune/tune_test.go`.
- [SHIPPED] Deterministic quality evaluation suite (`internal/mtpeval`): strict functional pass criteria, JSON schema validation, and minimum acceptance rate thresholds for speculative decode. Witness: `internal/mtpeval/eval_test.go`.
- [SIMULATED] Effective decode throughput multiplier on 256-bit unified memory (AMD Strix Halo LPDDR5X @ 200 GB/s): modeled scaling yields ~2.35x net token generation speedup at $K=4$ under typical coding syntax acceptance ($\rho \approx 0.82$). Witness: `docs/model/qwen38-native-mtp-envelope.md` and `internal/mtptune/tune.go`.

## Physical promotion gate

Promote either performance claim only after [#12239](https://github.com/anthony-chaudhary/fak/issues/12239)
commits the missing packet and digest-bound raw/quality evidence from the named physical host,
with exact artifact/runtime/prompt/cache identity, at least 20 observed samples per comparison
arm, fallback and rollback accounting, output-quality parity, and a successful
`fak macbench validate-mtp-comparison` readback. Independent read-back must also confirm that
fak-native—not a fixture constructor or external fallback—executed the measured path.
