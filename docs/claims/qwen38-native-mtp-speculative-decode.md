---
title: "Qwen 3.8 native MTP speculative decode"
description: "Native Multi-Token Prediction (MTP) draft head on Qwen 3.8 hybrid GDN architectures with on-device recurrent rollback and graceful cache coexistence fallback."
---

# Qwen 3.8 native MTP speculative decode

[← Claims index](../../CLAIMS.md)

- [SHIPPED] On-device recurrent state rollback (`internal/compute/recurrent_rollback.go`): shadows live convolution and recurrent state slots directly in GPU VRAM and performs atomic rewind to the accepted token prefix with strictly zero device-to-host transfers (`D2HBytes == 0`, `D2HEvents == 0`, `ZeroD2HVerified == true`). Witness: `internal/compute/recurrent_rollback_test.go` `TestRecurrentRollbackAccuracy`, `TestRecurrentRollbackZeroD2H`, and `TestRecurrentRollbackLatencyImprovement`.
- [SHIPPED] MTP cache coexistence and graceful fallback downgrade (`internal/model/mtp_cache_coexist.go`): supports prompt prefix reuse with MTP draft sessions. When divergence $\le$ `MaxRecurrentRollbackDepth` (default $K=4$), executes on-device recurrent rollback; when divergence exceeds rollback depth, executes graceful cold prefill fallback while preserving MTP readiness for subsequent turns. Witness: `internal/model/mtp_cache_coexist_test.go` `TestMTPCacheCoexistPromptDivergenceRollbackAndFallback`.
- [SHIPPED] Multi-Token Prediction parameter tuning engine (`cmd/tunemtp`, `internal/mtptune`): automated search and profiling over draft depth $K \in \{1..8\}$ and acceptance probability curves, establishing the $K=4$ sweet spot across Code, Math, and JSON tasks on unified-memory topologies. Witness: `cmd/tunemtp/main_test.go` and `internal/mtptune/tune_test.go`.
- [SHIPPED] Deterministic quality evaluation suite (`internal/mtpeval`): strict functional pass criteria, JSON schema validation, and minimum acceptance rate thresholds for speculative decode. Witness: `internal/mtpeval/eval_test.go`.
- [SIMULATED] Effective decode throughput multiplier on 256-bit unified memory (AMD Strix Halo LPDDR5X @ 200 GB/s): modeled scaling yields ~2.35x net token generation speedup at $K=4$ under typical coding syntax acceptance ($\rho \approx 0.82$). Witness: `docs/model/qwen38-native-mtp-envelope.md` and `internal/mtptune/tune.go`.
