# Inkling-2-Mini fak-native CUDA decode on an RTX 5080 (2026-11-23/24)

**[HW-WITNESSED]** One RTX 5080 16 GB desktop (sm_120, Linux, CUDA 13.4), the GPU
held under an exclusive measure lease with the node's other GPU services
stopped for the window. This is the public summary of two maintainer receipts; the
raw logs stay with the lab records because they carry lab addresses.

Model: `unsloth/Inkling-2-Mini-GGUF`, file `Inkling-2-Mini-UD-Q3_K_XL.gguf`
(13,631,779,296 bytes, sha256 `e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639`).
Workload: one stream, greedy, the 19-token capitals prompt, then 128 tokens; decode
is the 127 steps after the first token, prefill excluded. Order A B B A, profiler off.

## Result

| engine | decode tok/s | ms/token |
|---|---|---|
| fak d09931ed9 (host-driven, one device MatVec per matrix) | 6.31 / 6.32 | 158.4 |
| fak a8030e394 (resident forward, CUDA graph replay), 2026-11-23 | 82.55 / 82.54 | 12.11 |
| llama.cpp ad50351f5 `llama-bench` tg128, same card and GGUF | 80.99 | 12.35 |
| fak 203f56bbc (trunk with the fused small-op and expert-GEMV kernels), 2026-11-24 | **104.65 / 104.64** | 9.56 |
| fastest llama.cpp fork measured on this card (realugbun) | 119.28 | 8.38 |

- The resident path is 13.1x the host-driven path and level with upstream llama.cpp
  tg128. The two use different harnesses (fak's Go bench against `llama-bench`), so
  read that as parity, not a lead.
- The 2026-11-24 trunk is 1.27x the resident path (2.57 ms less per token). Its A arm
  re-measured a8030e394 at 82.33 / 82.66, within 0.3% of the 2026-11-23 number.
- The fastest fork on the same card is still 1.18 ms/token ahead.

## Correctness

Greedy tokens are compared with the llama.cpp ad50351f5 CPU backend
(`TestInkling2CUDAOracleGreedyMatchesLlamaCpp`, `internal/ggufload/inkling2loop_test.go`).

| case | a8030e394 | 203f56bbc |
|---|---|---|
| capitals, 32 tokens | PASS | PASS |
| fibonacci, 32 tokens | PASS | PASS |
| chat-template, 32 tokens | PASS | PASS |
| passphrase-recall-1946 | diverges at decode token 10 | diverges at decode token 10 |

The passphrase case diverges at the same token on both builds (the older
host-driven device path diverges one token later). There, llama.cpp's own top two candidates are 0.21 nats apart and swap
with `-ub 512`, so it is a near-tie, not a cache defect: tokens 0-9 match after 1946
context tokens, past the 1024-token local window.

## What the resident path does

- `internal/compute/cuda/inkling2_loop.cuh` holds the small-op kernels (RMS and depth
  mix, the loop-controller GEMV, short conv, q/r/k/v post-processing with the KV ring
  append, sink attention, top-8 routing, the MoE combine, softcap and argmax) and the
  expert-indexed GEMVs for NVFP4, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and Q3_K.
- `internal/ggufload/inkling2loop_resident.go` drives the forward. A per-token arena
  keeps device addresses identical from token to token, so one captured CUDA graph
  replays every token. Each token costs one host-to-device copy and one 4-byte
  device-to-host argmax.
- Opt-outs: `FAK_INKLING2_RESIDENT=0` returns to the host-driven path,
  `FAK_INKLING2_GRAPH=0` runs eagerly, and `FAK_INKLING2_NVFP4_GEMV=0` keeps the
  earlier expert GEMV.

## Reproduce

```bash
go run ./cmd/fak-dev build-cuda build --arch sm_120 --cuda-home /opt/cuda
go test -c -tags cuda ./internal/ggufload
# Serving binary with the CUDA backend (GETTING-STARTED.md, "Inkling-2-Mini on NVIDIA CUDA"):
go run ./cmd/fak-dev build-cuda binary --arch sm_120 --cuda-home /opt/cuda ./cmd/fak ./fak
./fak run /var/lib/fak/models/inkling2/gguf-mainline/Inkling-2-Mini-UD-Q3_K_XL.gguf
FAK_INKLING2_GGUF=<gguf> FAK_INKLING2_BENCH=128 ./ggufload.test -test.run TestInkling2CUDADecodeBench -test.v
FAK_INKLING2_GGUF=<gguf> ./ggufload.test -test.run TestInkling2CUDAOracleGreedyMatchesLlamaCpp -test.v
```

## Not claimed

- No served-endpoint throughput: this is the in-process decode bench, one stream.
- No prefill claim: prefill still runs one token at a time by default. A batched
  prefill path exists behind `FAK_INKLING2_BATCHED_PREFILL=1` and is not the default.
- No output-quality claim beyond the greedy oracle cases above.
