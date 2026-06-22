# DGX endpoint comparison

Metric used for peak selection: `completion_tokens_per_sec`

| Stack | Hardware | Model | Peak C | Req/s | Comp tok/s | Total tok/s | p95 ms | Errors | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| `raw-sglang` | gpu-server | `Qwen/Qwen3.6-27B` | 64 | 7.2 | 1451.6 | 24181.9 | 8134.7 | 5/64 | 1.00x |
| `fak-gateway` | gpu-server | `Qwen/Qwen3.6-27B` | 64 | 5.5 | 1085.6 | 17799.6 | 9979.9 | 9/64 | 0.75x |

