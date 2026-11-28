---
title: "fak Benchmarks — Latest Witnessed Results"
description: "Latest witnessed hardware benchmarks for fak, highlighting recent runs (Nov 24–27, 2026) and the newest record per platform."
---

# Benchmarks

fak Benchmarks is the front door for witnessed hardware runs and platform records.

> **TL;DR:**
> - Active (2026-11-26): RTX 5080 16 GB native CUDA on Inkling-2-Mini hits 122.34 tok/s plain decode, 5,861 tok/s W4A4 prefill, and 1.84× faster C4 TTFT than `realugbun/llama.cpp`.
> - Prior-only / historical: Mac M3 Pro (2026-11-24 retired Splash port; pure `internal/metalgemm` pending) and AMD Strix Halo (2026-10-22).
> - Authority: [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) · Charts: [BENCHMARK-GALLERY.md](../../BENCHMARK-GALLERY.md).

Run Inkling-2-Mini locally on CUDA (`sm_120`):

```sh
fak serve --engine inkernel --backend cuda --gguf /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf
```

## Latest hardware results — 2026-11-26

One row per hardware family: the newest committed performance receipt and its claim boundary. Each row stays historical until a newer quality-complete run replaces it.

| Platform | Latest witnessed result | Status & Details |
| --- | --- | --- |
| Mac | Historical (2026-11-24): Inkling-2-Mini UD-Q3_K_XL on 36 GB M3 Pro hit 40.3 vs 37.5 tok/s (97.5–107.4% of Splash 1.7.2; llama.cpp 18.9 tok/s) on the retired vendored Splash Metal port. | Retracted as a fak-native Metal claim; pure internal/metalgemm resident receipts are pending. [M3 Pro historical record](published/m3pro-inkling2-mini-metal/RESULTS.md) |
| AMD | Historical (2026-10-22): DeepSeek-V4.1-Flash-Q2_K on 128 GB Strix Halo (Radeon 8060S, Vulkan) hit 4.61 tok/s warm decode (+29% vs 3.57 base), 32/32 greedy parity. | Hardware-only prior record (30 tok/s target unmet); stands until a newer AMD run exists. [Strix Halo results](STRIX-HALO-DEEPSEEK-V41-FLASH.md) |
| NVIDIA | Active (2026-11-26): Inkling-2-Mini UD-Q3_K_XL on RTX 5080 16 GB (sm_120) hit 122.34 tok/s native CUDA decode (kv=f16), 2,017.61 ms TTFT and 130.44 agg tok/s at HTTP C4_p2048_d256 vs realugbun (3,718.34 ms / 122.47) and stock llama.cpp (4,683.83 ms / 115.69), and 0.002912 mean KL at 16K (32/32 oracle). | HW-witnessed (C1–C4 <=16K). Leads TTFT and C4 agg e2e; trails single-stream HTTP decode; >16K needs fp8/nvfp4 KV. [RTX 5080 v0.60 receipt](published/rtx5080-inkling2-v060/RESULTS.md) |

## Recent published runs (Nov 24–27, 2026)

| Date | Hardware & Run | Key Result | Receipt |
| --- | --- | --- | --- |
| 2026-11-26 | RTX 5080 16 GB — Inkling-2-Mini v0.60.0 (`C1–C4 <= 16K`, `32K fp8`) | 122.34 tok/s decode; C4 TTFT 2,017.61 ms (1.84× vs realugbun); 32K `fp8` KL 0.004225 | [Receipt](published/rtx5080-inkling2-v060/RESULTS.md) |
| 2026-11-25 | RTX 5080 16 GB — NVFP4 W4A4 CUDA prefill | 5,861.0 tok/s pp512 HTTP vs 3,012.4 realugbun (1.95×); default on `sm_120f`; KL 0.003728 | [Receipt](published/aegis-inkling2-prefill-nvfp4-w4a4/RESULTS.md) |
| 2026-11-24 | M3 Pro 36 GB — Inkling-2-Mini (historical) | Retired vendored Splash port (40.3 tok/s); pure `internal/metalgemm` receipts pending | [Record](published/m3pro-inkling2-mini-metal/RESULTS.md) |

## Archive & governance

- Full sheet archive: [CATALOG.md](CATALOG.md) (older runs, fleet/safety sheets, pending plans, runbooks) · [Published campaigns](published/INDEX.md)
- Repro & models: [repro packet](../repro-packet.md#inkling-2-mini-reproducer-cli-only) · [pinned model sources](../supported/models.md#exact-hugging-face-pins-for-witnessed-models)
- Authority & claims: [BENCHMARK-AUTHORITY.md](../../BENCHMARK-AUTHORITY.md) · [BENCHMARK-GALLERY.md](../../BENCHMARK-GALLERY.md) · [CLAIMS.md](../../CLAIMS.md)
- Qwen model index: [QWEN-PERFORMANCE-INDEX.md](QWEN-PERFORMANCE-INDEX.md) · [QWEN38-27B-LATEST.md](QWEN38-27B-LATEST.md)
- Policies: [fak-native engine identity](../native-inference-goal.md) · [compute placement tax](../serving/compute-placement-tax.md) · update the latest table and [hardware-latest.json](hardware-latest.json) together.
