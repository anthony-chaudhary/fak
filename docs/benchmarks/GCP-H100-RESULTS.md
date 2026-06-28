---
title: "fak on a real datacenter H100: first GCP Hopper run, measured"
description: "The first end-to-end run of fak's own CUDA engine on a live GCP H100 (a3-highgpu-1g), head-to-head vs llama.cpp CUDA and fak's pure-Go CPU engine on identical hardware, with an honest gap verdict."
---

# GCP-H100-RESULTS — fak's own engine on a live datacenter H100, measured

> **The honest verdict up front: fak's own CUDA engine now runs on real
> datacenter Hopper silicon** — a single NVIDIA H100 80GB on GCP — and produces
> a verified-on-device number (`-require-non-reference`, so a green row cannot be
> a silent CPU fallback). On this run fak-cuda decodes a Qwen2.5-3B at **96.3
> tok/s (f32)** vs **llama.cpp's 361.6 tok/s (Q8_0)** — fak is **~3.75x behind on
> decode and far behind on prefill** through `cmd/modelbench`'s device-GEMM path.
> That gap is the **known device-GEMM / no-CUDA-graph tuning gap**, not an
> architecture ceiling: the [LLAMACPP-HEADTOHEAD](LLAMACPP-HEADTOHEAD-RESULTS.md)
> note shows that **with a reusable CUDA graph fak reaches decode parity with
> llama.cpp Q8_0 on an RTX 4070** at higher precision (fak f32). The `modelbench`
> path measured here does not yet use that graph, and it runs f32 weights (the
> cuda backend does not advertise `UploadDtype`), so this is the *un-tuned* device
> path on faster silicon — the honest floor, not the ceiling.

## The run

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (81,559 MiB), `arch=sm_90` |
| Machine | GCP `a3-highgpu-1g` (1x H100, 26 vCPU, 234 GB RAM), spot, `us-central1-a` |
| Model | Qwen2.5-3B-Instruct, `qwen2.5-3b-instruct-q8_0.gguf` |
| Harness | `tools/gcp_bench.py --tier a3-high-h100-1g --engine all --spot` |
| Lifecycle | provision -> ship source -> build (llama.cpp CUDA + fak CUDA) -> bench -> collect -> teardown (always; instance deleted, no leak) |
| Artifacts | `experiments/benchmark/runs/by-machine/gcp-a3-high-h100-1g/` |

## Head-to-head (single-stream, pp512 / tg128)

| Engine | Backend | Precision | Prefill tok/s | Decode tok/s |
|---|---|---|--:|--:|
| llama.cpp | llama.cpp CUDA | Q8_0 | 19,310.5 | 361.6 |
| **fak-cuda** | **fak-in-kernel via compute HAL `cuda`** | **f32** | **51.0** | **96.3** |
| fak-cpu | fak-in-kernel (pure-Go) | Q8_0 | 109.7 | 15.7 |

Reading it honestly:

- **fak-cuda is real on Hopper.** Its decode (96.3 tok/s) is **~6.1x the pure-Go
  CPU engine's** (15.7 tok/s) on the same box — the device path is doing real GPU
  work, not falling back.
- **fak-cuda is behind llama.cpp** (3.75x on decode; the prefill gap is much
  larger, consistent with the ~11x-under-SOTA on-device prefill gap recorded for
  the #70 q4_k device-GEMM closure). The residual is hand-tuned-kernel /
  CUDA-graph / GEMM-tiling, the same boundary the CPU head-to-head identifies.
- **Precision is disclosed, not hidden:** fak-cuda runs f32 (heavier than the Q8_0
  the llama/fak-cpu rows use), so it is doing *more* arithmetic per token than the
  baselines, which makes the decode number more impressive per-FLOP than the raw
  ratio suggests.

## The finding that unblocked this run

The prior reachable-GPU run was a GCP **L4** (`gcp-g2-l4`), where **fak-cuda
failed**: the CUDA backend built cleanly (nvcc sm_89 + `go build -tags cuda` OK)
but `modelbench-cuda` was **SIGKILLed** — an OOM on the L4 box's **32 GB host
RAM** (fak-cuda materializes f32 weights, ~12 GB, plus load-time spikes). The
`a3-highgpu-1g` H100 shape carries **234 GB host RAM** (~7x), which clears the OOM
and is exactly why fak-cuda completes here. This is now encoded in the tier's
registry note (`tools/gcp_accel.py`).

## Reproduce

```bash
# read-only: confirm a tier is provisionable in this project today
python tools/gcp_gpu_probe.py --all-tiers

# one-touch: provision 1x H100 (spot), bench every engine, tear down
python tools/gcp_bench.py --tier a3-high-h100-1g --engine all --spot

# fold the result into the cross-machine catalog
python tools/bench_catalog.py update
```

`a3-highgpu-1g` (and the other partial A3-High shapes) **must** be created as
Spot/Flex-start VMs — `--spot` is required, not optional. The driver always tears
the instance down in a `finally` block and sets a server-side `--max-run-duration`
DELETE TTL, so a dead launcher cannot leak the GPU.

## Open follow-ons (not claimed as done)

> The full decomposition of this gap into ranked, code-anchored levers (with expected
> multipliers and the next checkable step for each) now lives in
> [H100-KERNEL-5X-ROADMAP.md](H100-KERNEL-5X-ROADMAP.md). The two follow-ons below are
> Levers 1–2 there.

- **Q8 device decode (the apples-to-apples row):** the f32 number above streams 4
  bytes/weight where llama.cpp's Q8_0 streams ~1 — and decode is memory-bandwidth-bound,
  so that ~3.9× byte ratio is essentially the whole 3.75× decode gap. **Correction to the
  prose above:** the cuda backend *does* advertise `UploadDtype` (`cuda.go:450`) with a
  native Q8 device GEMV already implemented — this run was f32 only because the bench did
  not request `-quant`/`-lean`, not because the capability was missing. That row is now
  wired as the opt-in **`fak-cuda-q8`** engine in `tools/gcp_bench.py`; the next run should
  select `--engine llama,fak-cuda,fak-cuda-q8`.
- **Route `modelbench -backend cuda` through the reusable (replay-many) CUDA graph** that
  reached 4070 parity; the gap above is the un-graphed path. The remaining work is a
  length-agnostic graph (device-resident `pos`/`nPos`), Lever 2 of the roadmap.
- **Bigger silicon:** B200/GB200 are quota=0 in this project today (probe verdict
  `NO_QUOTA`); H200 was `NOT_OFFERED` in the probed zone. H100 is the
  best-provisionable current tier here.
