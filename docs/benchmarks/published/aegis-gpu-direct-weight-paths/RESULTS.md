---
title: "Campaign record: Host-to-GPU weight paths on the aegis RTX 5080, for Inkling-2-Mini expert offload"
description: "Campaign record, not a general claim: read the page's status and envelope before quoting it. fak aegis-gpu-direct-weight-paths (2026-11-24): Host-to-GPU weight paths on the aegis RTX 5080, for Inkling-2-Mini expert offload."
---

# Host-to-GPU weight paths on the aegis RTX 5080, for Inkling-2-Mini expert offload (2026-11-24)

Status: `[HW-WITNESSED]` on `aegis-fak`. Microbenchmark only. No engine change and no end-to-end decode run.

**Question.** Inkling-2-Mini UD-Q4_K_M (16.61 GB) does not fit in 16 GB of VRAM. UD-Q3_K_XL (13.63 GB) fits and
decodes at 119.9 tok/s. If some experts live in host RAM, which path gets their bytes into a kernel fastest, and what
does each path cost the resident compute running beside it?

**Box.** RTX 5080 16 GB on PCIe Gen4 x16. Driver 615.78.08 is the open kernel module, CUDA 13.4. The kernel is
7.2.11-arch1-1 with `CONFIG_PCI_P2PDMA=y` and `CONFIG_HMM_MIRROR=y`; the IOMMU is on or in pass-through. BAR1 is
16 GiB. The host has 30 GB of RAM, and the root and `/var/lib/fak` filesystems are btrfs (the model directory is
NOCOW). The NVMe is an AGI 2 TB drive.

**Lease.** Share lease `hwl-a416def7b06c94d3`, released afterwards. The measurement ran beside the resident
`fak-llama-qwen27b` llama-server, which held 15.1 GB, so the bench was limited to 192 MiB of VRAM. During runs 1 and 2
that server was decoding at 90-95% utilization. During runs 3 and 4 it was idle at 0-1%. The tables use runs 3, 4-1
and 4-2.

## Commands

```sh
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 hostpath_bw.cu -o hostpath_bw -lpthread
./hostpath_bw --file /var/lib/fak/models/inkling2/gguf-mainline/Inkling-2-Mini-UD-Q4_K_M.gguf --host-mb 1024 --dev-mb 192
nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 cufile_bw.cu -o cufile_bw -L/opt/cuda/lib64 -lcufile -lpthread
./cufile_bw <same gguf> 192        # also with CUFILE_ENV_PATH_JSON=<copy with use_pci_p2pdma=true>
/opt/cuda/gds/tools/gdscheck -p
```

One expert slab is 4,812,800 B. That is one layer's gate_up plus down for one expert in Q4_K_M, page-rounded. The
receipts are in [`receipts/`](receipts/).

## Results (GB/s, decimal)

| Path | Stream | One expert (4.8 MB) | Top-8 gather | Notes |
| --- | ---: | ---: | ---: | --- |
| `cudaMemcpyAsync` from pinned (`cudaHostAlloc`) | 28.9 | 27.8-28.7 (167-173 us) | - | PCIe ceiling |
| `cudaMemcpyAsync` from `cudaHostRegister`'d malloc | 28.7 | 28.4 | - | same as pinned |
| `cudaMemcpyAsync` from pageable | 12.9-14.6 | 14.1-15.6 | - | about half, because the driver bounces it |
| Zero-copy kernel read, mapped pinned | 26.5-26.8 | 26.5 (182 us) | 26.5-26.8 | needs a grid of at least 4xSM per chunk; 1xSM gave 20.9 |
| Zero-copy kernel read, `cudaHostRegister`'d malloc | 26.2-26.6 | - | - | |
| Raw pageable `malloc` pointer read by a kernel (HMM) | 26.8 steady | - | 26.5-26.8 | `pageableMemoryAccess=1`; the first touch runs at 2.1-3.9 GB/s; the data does not migrate (a CPU re-touch takes 0.6 ms) |
| Raw `mmap`'d GGUF pointer read by a kernel (HMM) | 16.7-20.2 | - | - | file-backed pages |
| `cudaHostRegister(mmap file, ReadOnly)` | **fails** | | | `operation not supported`; llama.cpp `GGML_CUDA_REGISTER_HOST` would fall back |
| Managed memory, `cudaMemPrefetchAsync` | 14.7-25.7 | - | - | noisy |
| Managed memory, fault on demand | 6.0-7.7 | - | - | |
| NVMe `O_DIRECT` pread into pinned, 8 threads, 4 MiB | 2.5-3.3 | - | - | 1 thread gives 0.3-1.6 |
| cuFile `cuFileRead` into VRAM (compat) | 2.0 / 2.5-2.8 with p2pdma json | - | - | `GDR=unsupported`, so it is never true GDS |

### Copy engine vs zero-copy beside resident compute (runs 4-1 and 4-2)

| Pair | VRAM-bound kernel | Host-side path |
| --- | ---: | ---: |
| Kernel alone | 901 GB/s | - |
| Kernel plus pinned `cudaMemcpyAsync` on a second stream | **864-865 (-4%)** | 28.0-28.7 (unchanged) |
| Kernel plus zero-copy read kernel (grid 2xSM) on a second stream | **380-402 (-56%)** | 25.0-26.5 |

Zero-copy holds SM slots while it waits on PCIe latency. The DMA engine does not.

## What is and is not available

- **GPUDirect Storage, full mode: not available.**
  - `gdscheck` reports `NO GDS SUPPORT - Reason: GDR=unsupported` for the GeForce card.
  - `/proc/driver/nvidia-fs` is absent.
  - The filesystem is btrfs. cuFile logs a udev failure for the btrfs virtual device, and the earlier research found
    btrfs is compat-only ([MBL-244](../../research/SSD-DIRECT-VS-RAM-MARKET-2026-10-29.md)).
  - Enabling `use_pci_p2pdma` in the JSON does not change this.
  - `gdsio` needs `libnuma.so.1`. numactl is missing and was not installed.
  - Compat mode is never faster than `O_DIRECT` into pinned memory followed by a memcpy.
- **HMM is available.** On this open driver a kernel can dereference plain `malloc` memory at full PCIe speed, so
  pinning is optional for zero-copy. This explains the memory note that `cudaHostGetDevicePointer` succeeds on pageable
  memory: it returns the same VA.

## What it buys Inkling-2-Mini (model, not measured)

Shape, from the GGUF headers:

- 24 blocks with 128 experts each, and 8 used per token.
- Loop layers 6-17 run 3 times, so a token makes 48 layer-evals and 384 expert uses.
- Q4_K_M expert tensors total 14.78 GB, or 4.81 MB per (layer, expert). Non-expert tensors total 1.82 GB.
- Q3_K_XL experts total 12.20 GB.

Roughly 1-2.5 GB of Q4_K_M experts must move to host RAM, which is 7-17% of the experts.

| Quantity | Value |
| --- | ---: |
| Resident step at 119.9 tok/s | 8.3 ms, about 173 us per layer-eval |
| One expert miss, either fast path | 167-182 us, about one whole layer-eval |
| Uniform routing, 10% cold, at most 1.85 GB/token of expert reads | about 185 MB/token from host |
| That miss traffic at 26.5-28.7 GB/s | about 7 ms/token |

Projections:

- **Serial misses with no cache:** about 15 ms/token, or about 60-65 tok/s.
- **Hot-expert cache:** misses fall to 2-3% of uses, about 50 MB/token. That costs about 2 ms of serial time, or about
  95-100 tok/s.
- **Copy-engine prefetch overlapped with compute:** approaches the resident bound. A miss in loop pass 1 stays in VRAM
  for passes 2 and 3.
- **NVMe on the miss path:** 185 MB/token at 3.3 GB/s is 56 ms/token, about 17 tok/s. NVMe is a load and warm tier
  only: all 14.8 GB of experts load in about 4.5 s.

Q3_K_XL resident stays the fastest configuration. Q4_K_M with offload trades speed for quality.

## Ranked recommendation

1. Keep the host expert pool pinned or `cudaHostRegister`'d. Fill a VRAM expert-slot ring with `cudaMemcpyAsync` on a
   dedicated copy stream, and prefetch with router prediction or loop-pass reuse. This gives 28.7 GB/s and costs the
   resident kernels only 4%.
2. Use zero-copy, from mapped pinned memory or a raw HMM pointer, only for a demand miss that nothing else is waiting
   on. It saves the staging slot and the extra launch. Never run it beside resident kernels, which lose 56%.
3. Never use the following on the decode path: managed memory, `cudaHostRegister` of an mmap'd file (not supported),
   raw pageable memcpy (half speed), or NVMe and cuFile (GDS full mode is not available).

Prior art:

- llama.cpp 29b52d39 (MIT) `ggml/src/ggml-cuda/ggml-cuda.cu`:
  - `:166` `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
  - `:196` `GGML_OP_OFFLOAD_MIN_BATCH`=32
  - `:6117` `cudaHostRegister(...ReadOnly)`
- project-maya 376b4b87 (MIT) `src/core/expert_source.cpp:602,1312,1793` uses `cudaHostGetDevicePointer`
  zero-copy experts.
- PHL-155: over x8 or USB4, decode loses to the CPU. Here x16 gives 28.7 GB/s and the CPU split is unmeasured (see
  `internal/model/moe_bandwidth_partition.go` in fak).
