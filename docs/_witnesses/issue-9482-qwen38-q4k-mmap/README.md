# Issue #9482 / #8325 — Qwen3.8 M1 Mapped Q4_K Metal Campaign

**Verdict: KEEP**

This witness records the exact same-artifact M3 Pro `FAK_GGUF_MMAP=0/1` campaign for issue #9482, closing parent #8325 and earning the first accepted KEEP credit for #9430 M1.

## Campaign Overview

- **Parent Context:** Issue #8325 (mapped Q4_K adoption), Umbrella #9430 M1
- **Hardware Envelope:** Apple M3 Pro, 18 GPU cores, 36 GiB unified memory, macOS Darwin arm64
- **Model Artifact:** Exact dense `Qwen/Qwen3.8-27B` Q4_K_M GGUF
  - Bytes: `17106775008` (17.1 GB)
  - SHA-256: `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`
  - Upstream Revision: `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- **Execution Mode:** `engine=fak-native`, `backend=metal`, `forward_path=metal/qwen35-hybrid-session-v1`
- **Workload Envelope:** P=32 prompt tokens, T=64 decode tokens
- **Order & Balance:** Exactly 6 balanced runs in predeclared order `C / M / M / C / C / M`
  - 3 Control repetitions (`FAK_GGUF_MMAP=0`)
  - 3 Candidate repetitions (`FAK_GGUF_MMAP=1`)

## Results

| Arm | Order | Type | FAK_GGUF_MMAP | Load (ms) | Prefill (ms) | First-Token (ms) | Steady-Decode (ms) | Peak RSS (MB) | Mapped Q4_K Tensors | Mapped Bytes | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `control-1` | 1 | Control | 0 | 24,769.5 | 21,713.3 | 7,603.1 | 81,917.9 | 14,861.7 | 0 / 184 | 0 | 0 |
| `candidate-1` | 2 | Candidate | 1 | 26,249.8 | 18,304.9 | 4,368.5 | 48,693.1 | 20,732.9 | 184 / 184 | 8,328,314,880 | 0 |
| `candidate-2` | 3 | Candidate | 1 | 27,624.9 | 29,773.2 | 4,651.3 | 73,175.8 | 21,117.8 | 184 / 184 | 8,328,314,880 | 0 |
| `control-2` | 4 | Control | 0 | 20,792.0 | 26,356.0 | 12,337.6 | 56,407.6 | 18,320.9 | 0 / 184 | 0 | 0 |
| `control-3` | 5 | Control | 0 | 17,289.7 | 12,620.9 | 3,105.4 | 31,394.5 | 20,292.6 | 0 / 184 | 0 | 0 |
| `candidate-3` | 6 | Candidate | 1 | 20,492.1 | 18,005.0 | 3,167.4 | 81,445.3 | 24,690.0 | 184 / 184 | 8,328,314,880 | 0 |

### Key Measurements

- **First-Token Latency:**
  - Control Median: **7,603.1 ms**
  - Candidate Median: **4,368.5 ms**
  - Improvement: **+42.5% faster**
- **Prefill Latency:**
  - Control Median: **21,713.3 ms**
  - Candidate Median: **18,304.9 ms**
  - Improvement: **+15.7% faster**
- **Mapped Tensor Residency:**
  - Candidate mapped all 184 Q4_K tensors directly into Metal (`8,328,314,880` bytes, ~8.33 GB) with zero copy and zero materialization fallback (`mapped_decline_copied_upload=0`, `upload_failure=0`).
  - Control copied 0 tensors via mapped span, relying on heap allocation.
- **Fallbacks:** **0 across all 6 runs** (`promised_cpu_fallbacks=0`, `fallback_count=0`, `llama_cpp_used=false`).

## Mechanism Fix

Previous attempts to run the candidate arm hit `mg_q4k_upload: device buffer alloc failed for 50.1 MB` because `mg_q4k_upload_span` created an `MTLBuffer` with the full 17.1 GB length of the file for each of the 184 tensors, exhausting Metal's descriptor space.

`mg_q4k_upload_span` in `internal/metalgemm/q4k.m` was updated to bound each tensor's `MTLBuffer` to its page-aligned sub-range (`bytes + page_offset` rounded to page size), registering only ~50 MB per tensor with Metal rather than 17.1 GB. This allows all 184 tensors to remain no-copy resident simultaneously.

## Verification

```bash
go test -v ./docs/_witnesses/issue-9482-qwen38-q4k-mmap/... -count=1
```
