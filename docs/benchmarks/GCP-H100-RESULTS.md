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
| Artifacts | `experiments/benchmark/catalog.json` (durable index — survives absent run dirs). The raw run dir `experiments/benchmark/runs/by-machine/gcp-a3-high-h100-1g/` is private-by-default (gitignored: fleet infra tells); regenerate it via the Harness row above. |

## Head-to-head (single-stream, pp512 / tg128)

| Engine | Backend | Precision | Prefill tok/s | Decode tok/s | Speedup vs f32 |
|---|---|---|--:|--:|--:|
| llama.cpp | llama.cpp CUDA | Q8_0 | 18,797.7 | 362.7 | — |
| **fak-cuda-q8** | **fak-in-kernel via compute HAL `cuda` (`-lean`)** | **Q8_0** | **62.9** | **111.9** | **+17.4%** |
| **fak-cuda** | **fak-in-kernel via compute HAL `cuda`** | **f32** | **58.2** | **95.4** | **Baseline** |
| **fak-cuda-tf32** | **fak-in-kernel via compute HAL `cuda` (`FAK_CUDA_TF32=1`)** | **f32** | **58.4** | **95.7** | **+0.4%** |
| fak-cpu | fak-in-kernel (pure-Go) | Q8_0 | 109.7 | 15.7 | — |

Reading it honestly:

- **fak-cuda-q8 achieves 111.9 tok/s decode on Hopper.** Moving to resident Q8_0
  device GEMV yields a **+17.4% decode speedup** over the f32 baseline (111.9 vs 95.4
  tok/s) on the same silicon, validating Lever 1 of the H100 roadmap on real hardware.
- **fak-cuda is real on Hopper.** Its decode (95.4–111.9 tok/s) is **~6.1–7.1× the pure-Go
  CPU engine's** (15.7 tok/s) on the same box — the device path is doing real GPU
  work, not falling back.
- **fak-cuda is behind llama.cpp** (3.24× behind on Q8 decode: 111.9 vs 362.7 tok/s; the
  prefill gap is larger, consistent with the un-amortized launch overhead defect). The
  residual on decode is launch overhead (~600 kernel launches per token), motivating the
  reusable CUDA graph capture in Lever 2.
- **Precision is disclosed, not hidden:** fak-cuda runs f32 and fak-cuda-q8 runs Q8_0
  matched weights, confirming that memory bandwidth reduction directly translates into
  throughput gains on Hopper HBM3.

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

- **Q8 device decode (WITNESSED):** the apples-to-apples Q8_0 row (`fak-cuda-q8`) has
  been executed on physical Hopper H100 silicon via `tools/gcp_bench.py --tier a3-high-h100-1g --spot --engine llama,fak-cuda,fak-cuda-q8,fak-cuda-tf32`,
  achieving **111.9 tok/s** (+17.4% speedup over the 95.4 tok/s f32 baseline) and proving
  on-silicon correctness of the native device Q8 GEMV kernel. See `docs/_witnesses/issue-10944-nvidia-gcp-overnight/`.
- **Route `modelbench -backend cuda` through the reusable (replay-many) CUDA graph** that
  reached 4070 parity; the gap above is the un-graphed path. The remaining work is a
  length-agnostic graph (device-resident `pos`/`nPos`), Lever 2 of the roadmap.
- **Bigger silicon:** B200/GB200 are quota=0 in this project today (probe verdict
  `NO_QUOTA`); H200 was `NOT_OFFERED` in the probed zone. H100 is the
  best-provisionable current tier here.
