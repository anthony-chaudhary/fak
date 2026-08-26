---
title: "Qwen3.8-27B fak-native cache attribution — issue #8819"
description: "Verdict: HOLDCACHERESTOREREGRESSION; parity remains HOLDBELOWPARITY. A live fak-native CUDA run on the exact Qwen3.8-27B Q4KM artifact produced 5/5 exact..."
---
# Qwen3.8-27B fak-native cache attribution — issue #8819

**Verdict: `HOLD_CACHE_RESTORE_REGRESSION`; parity remains `HOLD_BELOW_PARITY`.** A live fak-native CUDA run on the exact Qwen3.8-27B Q4_K_M artifact produced 5/5 exact cold outputs at 11.8–12.1 decode tok/s. The identical-prompt arm produced 0/5 exact outputs; its four confirmed 24-token cache hits decoded at about 0.2 tok/s.

## Scope

- **For:** fak-native Qwen3.8 operators.
- **Problem:** prefix/session reuse is the selected bottleneck after the campaign's native/reference parity rejection.
- **Today:** cold unique requests are correct, while confirmed cache hits collapse decode throughput. The stable prompt's first uncached response was already wrong (`Stable` rather than `Q38`), and all four cache hits retained that wrong output.
- **Better because:** this matched run narrows the next measurement to recurrent/prefix state clone, staging, restore, and backend copy rather than authorizing a kernel rewrite or reference-runtime fallback.
- **Witness:** `go test .` checks ten unique rows, exact quality, cache-token accounting, native identity, source hashes, and both HOLD verdicts.

Centrality: **Core**. P1–P4: the work directly measures the native execution path; it does not substitute vLLM/llama.cpp, claim a gain from failed quality, broaden the operating envelope, or conceal setup/recovery overhead.

## Read the result precisely

| Arm | Rows | Exact quality | Reuse | Native decode observation |
|---|---:|---:|---:|---:|
| cold unique | 5 | 5/5 `Q38` | 0 tokens | 11.8–12.1 tok/s |
| cache identical | 5 | 0/5 (`Stable`) | reps 2–5: 24 tokens | cache hits: ~0.2 tok/s |

The quality failure cannot be assigned solely to restore because repetition 1—the uncached stable prompt—also failed. The throughput collapse is isolated to the confirmed reuse rows: each reports `reused=24tok`, `prefill=0tok`, and two-token decode in 9.51–9.75 seconds. Therefore cache reuse is not a quality-constrained improvement and remains held.

The public witness intentionally omits control-plane names, credentials, private paths, and raw internal logs. `summary.json` binds the retained rows and native identity to the independently read-back source archive and log hashes.

## Next experiment

Instrument bytes and elapsed time at `PrefixSnapshot.Clone/Restore`, Qwen recurrent-state cloning, host staging/restore, and the CUDA Qwen35 backend copy seam. Rerun five cold and five cache-hit requests in one process with an unambiguous exact-output prompt. Accept a candidate only if all outputs are exact and native identity remains CUDA/GDN/Q4K.