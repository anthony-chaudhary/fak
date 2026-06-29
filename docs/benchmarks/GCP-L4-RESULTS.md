---
title: "fak-cuda on a real GCP L4, measured"
description: "The completed GCP L4 head-to-head for Qwen2.5-3B Q8_0: llama.cpp CUDA, fak-cpu, and fak-cuda on one NVIDIA L4 with enough host RAM for fak-cuda's f32 path."
---

# GCP-L4-RESULTS - fak-cuda on one real GCP L4

Issue #18 asked for the missing fak-cuda device number from the GCP L4
head-to-head. The earlier cheap L4 shape (`g2-standard-8`, 32 GB RAM) had already
proved the build bug was fixed, but `modelbench-cuda` was SIGKILLed while loading
the f32 device path. The successful rerun uses the same single NVIDIA L4 GPU class
on `g2-standard-32` (32 vCPU, 128 GB RAM), so the host-memory gate is removed
without switching to Hopper.

## Run

| | |
|---|---|
| GPU | NVIDIA L4, 23,034 MiB, `arch=sm_89` |
| Machine | GCP `g2-standard-32`, 1x L4, 32 vCPU, 128 GB RAM, `us-central1-b` |
| Model | Qwen2.5-3B-Instruct, `qwen2.5-3b-instruct-q8_0.gguf` |
| Harness | `python tools/gcp_bench.py --tier g2-l4-32 --zone us-central1-b --engine all --max-run-hours 2` |
| Artifacts | `experiments/benchmark/runs/by-machine/gcp-g2-l4-32/20260629T132955Z-gcp/` |

## Head-to-head (single-stream, pp512 / tg128)

| Engine | Backend | Precision | Prefill tok/s | Decode tok/s |
|---|---|---|--:|--:|
| llama.cpp | llama.cpp CUDA | Q8_0 | 6,638.3 | 70.8 |
| fak-cpu | fak-in-kernel pure-Go | Q8_0 | 78.2 | 8.62 |
| fak-cuda | fak-in-kernel via compute HAL `cuda` | f32 | 16.6 | 18.7 |

The fak-cuda row is a real device row: the harness runs `modelbench-cuda` with
`-backend cuda -require-non-reference`, and the result reports backend
`selected: "cuda"` with tier `sm_89`. The measured gap is not a win claim:
fak-cuda is slower than llama.cpp on this L4 and runs f32 weights while the
llama row is Q8_0.

## What changed from the failed L4 run

The failed `gcp-g2-l4/20260624T142454Z-gcp` rerun reached CUDA build success on
sm_89, then the VM killed `modelbench-cuda`. This rerun keeps one L4 but uses the
128 GB host-memory shape. That is why fak-cuda can load and complete; the original
32 GB proof tier remains useful for cheap plumbing checks, not for this f32
Qwen2.5-3B fak-cuda row.
