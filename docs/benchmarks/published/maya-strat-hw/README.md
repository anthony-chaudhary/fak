# maya + Strata on owned NVIDIA HW — campaign index (2026-10-30)

Goal: `GOAL-maya-strat-hw-config-sweep` (operator direction 2026-10-30: run
mw00/project-maya + Niko1221/Strata on owned hardware, prefer over llama.cpp where
possible, search each engine's config space). This directory holds the folded
campaign receipts; worker scratch lives under `_scratch/goals/GOAL-maya-strat-hw-config-sweep/`
(never committed).

## Status

| node | status | note |
| --- | --- | --- |
| raider-fak | **DONE** | DS4.1 ABBA complete (11 cells), lease released ~19:45Z, Xid count unchanged (12→12); see [RESULTS.md](RESULTS.md) §c–e, [raider-fak/RECEIPT.json](raider-fak/RECEIPT.json) |
| aegis-fak | **DONE (partial)** | Flash-Next Q2_0 ABBA-lite complete 2026-10-31T03:34–~04:20Z: strata 30.55 tok/s @ctx2048 warm = 3.6x same-day llama.cpp `--cpu-moe` baseline; maya ABSENT-IN-WINDOW (CYBER-FROST download 9.8%, resumed detached PID 31936); see [RESULTS.md](RESULTS.md) §g, [aegis-fak/RECEIPT.json](aegis-fak/RECEIPT.json) |

Original target model (Qwen3.8-27B-UD-Q2_K_XL, dense `qwen35`) is not servable by
either pinned engine — hard architecture refusals in committed code at both pins
(witnessed verbatim on both nodes; RESULTS.md §f). Re-ranked target:
DS4.1-Flash GSQ-RCO IQ2_XXS on raider-fak (both engines ship dedicated DS4.1
paths; local llama.cpp baseline receipts exist).

## Pinned artifacts

### Engines

| engine | repo @ sha | version | raider-fak binary (sha256) | aegis-fak binary (sha256) |
| --- | --- | --- | --- | --- |
| maya | `mw00/project-maya` @ `376b4b87402bc992af7b4a8dec089ef7cba7801c` | v1.1.39 | `/home/fak/oss-engines/maya-376b4b87/build/maya-ds4` `af05d0398b948cfa13691eaccb9d0385d0e316d41c69cf57a1ae39b0e7200e9a` (plain `build/maya` also built: `f6eab89d15475312e224b46d279843a1c70220f4e6d5db319b9dc0a608220774`) | `/home/fak/oss-engines/maya-376b4b87/build/maya` `5bf812e64cf4266a74cc5f28c23dfabd6ae0208a7103f840c799b3671c746b91` |
| strata | `Niko1221/Strata` @ `46c152a1310f77cd2f0ce2c7cd213962df14fe83` | 0.1.53 | `/home/fak/oss-engines/strata-46c152a1/build/strata` `f5d10380f777c45dfdbae65e7cbf927e1501b055d6bc763f114b00925dfc5e7e`; plus `build/strata-dsv41` (built `-DSTRATA_BUILD_DSV41=ON`) `0eb3f21e07dc4b507a5e49dc62782698109f4953028909eb4b2714ab718591bf` | `/home/fak/oss-engines/strata-46c152a1/build/strata` `7a1a910f1499180084b145ca851b8eed6c2f6be1a3fef6653f5296f5e3b5680c` |
| llama.cpp b11870 (baseline, DO-NOT-TOUCH tree) | build `b11870` (74bca395) | 1.12.x | `/home/fak/oss-engines/llama.cpp-b11870/build-cuda/bin/llama-server` `8453c97a0243d3feceea0340110b846c9516cd82f34ef16f553d668476080310` (re-verified preflight 2026-10-30T17:44Z) | same pinned binary `8453c97a…80310` (see `docs/benchmarks/aegis-5080-20261030/RESULTS.md` envelope) |

Both OSS engines pin their ggml source at llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (fetched to
`/home/fak/oss-engines/llama.cpp-pinned-3cf03257`, untouched). sm_120 CUDA builds
green on both nodes (toolkit 13.4 / nvcc V13.4.92, `CMAKE_CUDA_ARCHITECTURES=120`).

### Models

| model | location | pin |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XXS (sweep model, raider) | `/var/lib/fak/models/` | repo `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` @ `a419d74cd28d5b9f8c5e2b9077d11a609c17bc32`, manifest sha256 `d743507760fe3b906135a7f33c97eb3a1923dadfedc3bed42aed09997c56b07d`; shard oids below; all 6 files re-sha256-verified MODEL_OK 2026-10-30T17:47Z |
| Qwen3.8-27B-UD-Q2_K_XL (arch-refusal target) | `/var/lib/fak/models/Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9,828,981,664 B, sha256 `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` |

DS4.1 shard oids (from `docs/benchmarks/raider-5090/ds41-gsq-iq2xxs-20261028/download-receipt.json`):

| shard | size_bytes | oid (sha256) |
| --- | ---: | --- |
| DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf | 5,780,704 | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| …-00002-of-00006.gguf | 39,535,781,184 | `25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3` |
| …-00003-of-00006.gguf | 39,736,712,000 | `8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48` |
| …-00004-of-00006.gguf | 39,036,908,384 | `9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0` |
| …-00005-of-00006.gguf | 19,923,784,832 | `7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1` |
| DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (shard-6 symlink target) | 110,595,290,816 | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |

The maya/strata pack (`/home/fak/.fak/raider-sweep/ds41-model/maya-ds4`) is a
symlink farm over these pinned shards (47,284 tensors; Engram 2 layers × 24 hash
columns) — identical weights to the llama.cpp L-cells.

## Nodes

| item | raider-fak | aegis-fak |
| --- | --- | --- |
| boot_id | `d1026348-79cd-485e-a9b5-c8f5da69e578` | `31fc7046-7671-4813-8d33-7856b915dd9d` |
| GPU | RTX 5090 Laptop GPU, 24,463 MiB, sm_120 | RTX 5080, 16,303 MiB, sm_120 |
| driver | 615.78.08 | 615.78.08 |
| GPU UUID | `GPU-2a2a837f-c82d-0f22-4538-e3a063903380` | `GPU-3e9fda0a-061a-6eb7-26c3-541bf8489090` |
| CPU / RAM | Intel Core Ultra 9 290HX Plus, 24 cores, **AVX512F_ABSENT**; ~30 GB RAM | 24 cores; ~30 GB RAM, 1 GB swap |
| disk (/var/lib/fak) | 54 GB free at campaign start (ENVELOPE_ABSENT for any qwen4exp artifact) | 1,580 GB free |

## File map

| file | content |
| --- | --- |
| `README.md` | this index |
| `RESULTS.md` | TL;DR winner table, scope & pins, full 11-cell ABBA table, baseline anomaly attribution, config-space ablations, arch-refusal data points, aegis Flash-Next Q2_0 section (§g), cross-engine verdict |
| `raider-fak/RECEIPT.json` | schema `fak.maya-strat.run-receipt/1`: verdict, lease trail, binaries, model oids, oracle prompt, all 11 cell objects verbatim, typed failures |
| `aegis-fak/` | `RECEIPT.json` — HW-WITNESSED receipt: Q2_0 artifact sha pins, 4 cells (L1/L2/S1/S2), 4 typed failures incl. both OOM deviations, lease/service trail, host-stability incident note |
| `docs/lessons/perf-hillclimb/lessons.jsonl` | two appended entries: PHL-165 (arch-gate preflight), PHL-166 (baseline regime shift) |

Related prior receipts: `docs/benchmarks/raider-5090/ds41-gsq-iq2xxs-20261028/`
(historical llama.cpp arms 712.7 / 1160.1 / 1055.6 / 1014.5 tok/s @ 2026-10-29T04:42Z
and the 1.05 tok/s rerun @ 2026-10-29T20:25Z), `docs/benchmarks/aegis-5080-20261030/RESULTS.md`
(today's aegis llama.cpp b11870 C1 numbers, cross-context only).
