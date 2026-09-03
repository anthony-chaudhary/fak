# Issue #9983 — Batch Qwen3.8 Target Verification for MTP Blocks Witness

Verdict: **PASS**. Evaluated proposed MTP draft blocks with a single target-side verification operation rather than N serial target decode steps, including net wall time with full accounting overhead.

## Witness Summary

- **Issue:** #9983 (`perf(model): batch Qwen3.8 target verification for MTP blocks`)
- **Parent:** #9819
- **Module:** `internal/model@r624+g1b3f14b96`
- **Engine:** `fak-native` (`fak-native/f32/qwen3.8-whole-sequence-target-verify-v1`)
- **Target Verification Operations per Block:** 1 (target decode steps: 0)

## Verified Properties

1. **One Operation per Block:** `VerifyForwardOneOperation` executes a single whole-sequence forward evaluating the complete proposed candidate block in one operation instead of N serial `Step` invocations.
2. **Net Accounting:** Complete accounting measures and charges setup, drafting, target verification, rejection, rollback, synchronization, and recovery costs.
3. **Lossless Downgrade:** Unsupported shapes (such as quantized target formats or unrecognized architectures) downgrade explicitly to ordinary fak-native target decode with typed error tokens (`TargetVerificationDowngradeError`), without any silent external runtime fallback.
