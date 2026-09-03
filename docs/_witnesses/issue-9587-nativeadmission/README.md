# Issue #9587 — Apple Unified Memory Aggregate Reservation Witness

Verdict: **PASS**. The production integration of aggregate memory reservations, Darwin memory pressure sampling, model load plan derivation, and GPU lease coordination was executed and witnessed on Apple Silicon (M3 Pro, 36 GiB unified memory, `darwin/arm64`).

## Witness Summary

- **Issue:** #9587 (`feat(nativeadmission): reserve aggregate Apple unified memory by bytes and pressure`)
- **Parent:** #9577
- **Engine:** `fak-native`
- **Model:** `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- **Hardware:** Apple M3 Pro, 18 GPU cores, 38,654,705,664 bytes unified memory pool.
- **Topology:** Apple Unified Memory Architecture (UMA); device buffers share the physical memory pool with host processes and are not treated as host-addressable space.

## Verified Properties

1. **Pre-Allocation Refusal:** Overcommit requests and critical pressure refuse before the model loader is invoked.
2. **Phase Transition:** Reservations start at `StartupPeakBytes` during model loading and downshift to `SteadyBytes` once loaded into steady state.
3. **Safe Coexistence:** Multiple small models whose aggregate memory fits within allocatable capacity safely coexist and hold concurrent reservations.
4. **Exact-Once Teardown:** Teardown paths and panics release held reservations exactly once without leaking bytes in `reservations.json`.
5. **Conservative Exclusive Rollback:** Setting `FAK_NATIVE_ADMISSION=exclusive` bypasses aggregate reservation sharing and preserves the single-owner safety lease.
