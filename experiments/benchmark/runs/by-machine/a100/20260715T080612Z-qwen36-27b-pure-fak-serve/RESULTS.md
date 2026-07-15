# Qwen3.6-27B pure-fak serve — A100 40 GB

**Status: OBSERVED, partial acceptance.** On 2026-07-15, the exact pinned Qwen3.6-27B Q4_K_M artifact was resident in the in-kernel service configured for CUDA required mode. One authenticated OpenAI-compatible request returned HTTP 200 and the exact requested text `7`.

## Observed request

| Measure | Observation |
|---|---:|
| Prompt tokens | 23 |
| Completion tokens | 1 |
| Wall time | 200 s |
| Prefill | 23 tokens / 192.63 s / 0.1 tok/s (runtime-rounded) |
| Decode | 1 token / 7.08 s / 0.1 tok/s (runtime-rounded) |
| Correctness | exact response text `7` |
| Finish reason | `stop` |

The service remained healthy after the request. Free device memory was 16,555 MiB before and 16,537 MiB afterward.

## Acceptance boundary

This is **not** the closing artifact for #4379:

- Full-logits CPU-vs-CUDA cosine is not yet captured.
- The request log's legacy `q8dec` field describes the host Q8 implementation and does not independently identify the selected device operation path; #4817 tracks making that witness unambiguous.
- Therefore the result proves a real exact-model serve response and request-scoped throughput, but does not yet prove the required CUDA full-logits parity or pure-device path identity.

Private raw transcripts retain endpoint/session details and are intentionally not copied here. The public manifest contains only scrubbed hardware class, artifact identity, request, and measurements.
