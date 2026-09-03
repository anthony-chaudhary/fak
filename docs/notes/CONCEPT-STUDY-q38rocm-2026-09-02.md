# Study Note: julianmb/q38rocm — Qwen 3.8 27B ROCmFP4 on AMD Strix Halo

**Source:** https://github.com/julianmb/q38rocm
**Pinned Revision:** `6c142530031c923ece1adf4d8c9c824b7369fef8`
**Study Date:** 2026-09-02
**Study Depth:** Deep (fan-out across all subsystems)
**Durable Study Receipt:** `study_6777fd46608e1a1293a3f111c4a4f17a9aab28c16431e689f0e011cffc965cb6`
**Parent Epics:** #6221 (quantization interoperability), #10193 (Qwen3.8 native performance), #2236 (serving superset)

---

## Repository Overview

`q38rocm` is a dedicated single-model deployment package for **Qwen 3.8 27B** quantized with **ROCmFP4_FAST** (4.26 bpw) on **AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S / gfx1151)**. It achieves **30.56–36.04 tok/s** via:

- **ROCmFP4 block quantization** (block size 32, matching Wave64 half-wave alignment)
- **MTP (Multi-Token Prediction) Speculative Decoding** (embedded MTP heads, 75–88% acceptance at temp 0)
- **Asymmetric TurboQuant KV Cache** (Keys Q8_0, Values turbo4 4-bit)
- **Backend crossover**: ROCm0 for prefill, Vulkan0 (RADV Wave64 + KHR_coopmat) for decode

The repository builds on upstream **ROCmFPX** (`ROCmFPX/ROCmFPX` org repo, pinned commit `75e67a92b2d230849aec2d6c1f7b1d1fd624e0e0`).

---

## Fan-Out Coverage (Completeness Critic)

| Subsystem | Coverage | Notes |
|-----------|----------|-------|
| Quantization pipeline (`convert_and_quant.sh`, `QUANTIZATION_RECIPES.md`) | ✅ Full | BF16→GGUF→ROCmFP4_FAST/STRIX_LEAN/ROCmFP8/Q3/Q2 |
| MTP speculative decoding (`tune_mtp.py`, `run_server.sh` profiles) | ✅ Full | Automated sweep, strict-greedy agent mode, DFlash2 alternative |
| KV cache & prompt caching (`cache_profile.sh`, `long_context_cache_benchmark.py`) | ✅ Full | TurboQuant asymmetric, RAM checkpoints, 23–800× cache speedup |
| Benchmarking & quality (`benchmark.py`, `quality_eval.py`, `context_scaling_benchmark.py`) | ✅ Full | Multi-task, deterministic validators, context scaling curves |
| NPU hybrid pipeline (`run_pipeline.py`, `npu_sidecar_drafter.py`, `NPU_INTEGRATION.md`) | ✅ Full | 1.8× TTFT burst, ~2W intent routing, negative results documented |
| Engine build & provenance (`build_engine.sh`, `BUILD_INFO.txt`) | ✅ Full | Pinned commit, patches, prebuilt download, SHA verification |
| Server launcher & profiles (`run_server.sh`, `setup_env.sh`) | ✅ Full | speed/agent/cache/safe/structured, auto-device detection |
| Docker & client integration (`docker-compose.yml`, `CLIENT_INTEGRATION.md`) | ✅ Full | Open WebUI, Continue.dev, Cursor, LiteLLM |
| Upstream tracking (`UPSTREAM_TRACKING.md`, `patches/`) | ✅ Full | Spec-boundary-mismatch fix, router timeout, DFlash2 migration |

**Completeness Critic Verdict:** Nothing material left unopened. All load-bearing subsystems read at `path:line@sha`.

---

## Candidate Borrows Table

| # | Borrow (one technique) | Source `path:line@sha` | Axis Optimized | Their Worldview Reason | Witness on fak (PRESENT/PARTIAL/ABSENT/DIVERGENT) | Inspire/Integrate | Filed Issue |
|---|------------------------|------------------------|----------------|------------------------|--------------------------------------------------|-------------------|-------------|
| 1 | **ROCmFP4 block quantization (block=32, Wave64-aligned, zero dequant stalls)** | `docs/QUANTIZATION_RECIPES.md:22-26@6c14253` | Memory bandwidth per token | RDNA 3.5 cooperative matrix ALUs execute FP4 dequant + GEMM in single Wave64 pass; 13.55 GB payload → 36 tok/s with MTP | **PARTIAL** — fak has FP4 metadata adjudication (`fp4meta.go`) supporting MXFP4 (32-element blocks, E8M0 scale, 1 level) and NVFP4 (16-element blocks, E4M3 scale, 2 levels), but no ROCmFP4-specific kernel or hardware-aligned quantization for AMD RDNA 3.5 cooperative matrix | **INSPIRE** (upstream ROCmFPX is Apache-2.0; kernel ideas transferable; fak's MXFP4 support is close but not hardware-aligned to Wave64) | #10730 |
| 2 | **Embedded MTP Speculative Decoding (K=4, strict-greedy for agents)** | `run_server.sh:335-339@6c14253`, `docs/UPSTREAM_TRACKING.md:93-94@6c14253` | Tokens per memory pass | Qwen 3.8 has MTP heads; preserving them in FP16/Q8 keeps draft acceptance 75–88%; strict mode (`--spec-mtp-strict-qwen`) for boundary-safe agent decoding | **PRESENT-on-axis** — fak has native Qwen3.5 MTP depth-N in `internal/model/qwen35_mtp_draft.go:338-479` with `Qwen35MTPMaxDraftDepth=4`, strict verification, checkpoint/restore, and `SpecDecodeGreedyQwen35MTPDepthN` entry point. The polymodel speculative loop (`polymodel.SpecDecode`) is lossless with bit-exact KV rollback. | **INTEGRATE** (fak already has this; no issue filed) | — |
| 3 | **Asymmetric TurboQuant KV Cache (K=Q8_0, V=turbo4)** | `run_server.sh:53-54,77-78@6c14253`, `docs/QUANTIZATION_RECIPES.md:66-70@6c14253` | KV cache memory at long context | Keys need precision for attention routing; Values compress to 4-bit with near-zero PPL impact; 262K context fits in 20.08 GiB | **PARTIAL** — fak has 4-bit KV cache codec `KVQuant4` in `internal/model/kvquant.go:33-119` with group size 32, affine (asymmetric) quantization, per-group scale/min, and `KVQuantLayers()` to target only full-attention layers. However, it's not yet wired into the decode path (gated at gen/next per comments), and no Q8_0 for K + turbo4 for V asymmetric split. | **INSPIRE** (fak's KVQuant4 is very close; needs asymmetric K/V split and decode-path wiring) | #10731 |
| 4 | **Backend crossover rule (ROCm0 prefill, Vulkan0 decode)** | `README.md:281-284@6c14253`, `run_server.sh:252-265@6c14253` | Prefill throughput vs decode throughput | ROCm0 HIP achieves 329 tok/s prefill at 16K; Vulkan0 RADV Wave64 achieves 36 tok/s decode via KHR_coopmat | **DIVERGENT** — fak is CPU-first with optional GPU; different hardware target. Tradeoff: fak optimizes for portability; they optimize for Strix Halo unified memory | **WATCH** (architecture insight for multi-backend routing) | — |
| 5 | **Prompt cache + MTP coexistence (23–800× cache speedup)** | `long_context_cache_benchmark.py:130-155@6c14253`, `README.md:198-210@6c14253` | Repeated long-document latency | MTP checkpoint rollback tolerates empty `data_spec`; divergent tails fall back to cold prefill without losing MTP on subsequent turns | **PARTIAL** — fak has prompt caching (`cachedemo.go`, `cachevalue.go`) and MTP (`qwen35_mtp_draft.go`), but the coexistence logic (empty `data_spec` tolerance, spec-boundary-mismatch fallback) is not yet implemented. q38rocm's upstream patch for this (`mtp-prompt-cache-fix.patch`) is a direct reference. | **INSPIRE** (if fak enables MTP + prompt cache, this coexistence is required; port the upstream patch logic) | #10732 |
| 6 | **Automated MTP tuning sweep (n ∈ [0,3,4,5,6], p ∈ [0.50,0.55])** | `scripts/tune_mtp.py:80-225@6c14253` | Speculative decoding throughput | Empirical sweep finds optimal K=4 for single-stream; K=6 for 4-slot parallel; measures acceptance rate per task type | **PARTIAL** — fak has `AdaptiveDraftDepthController` in `internal/model/adaptive_draft_depth.go:10-68` with hysteresis-based depth adjustment (min/max depth, raise/lower acceptance thresholds, hysteresis windows). However, no automated sweep CLI tool like `tune_mtp.py` that tests multiple configs against a running server. | **INSPIRE** (port `tune_mtp.py` as a fak leaf tool; the adaptive controller is the core) | #10733 |
| 7 | **Deterministic quality evaluation suite (validators per task)** | `scripts/quality_eval.py:19-41,93-111@6c14253` | Regression detection for speculative decoding | Greedy (temp 0) validators for code, JSON, logic; MTP draft accuracy tracked; catches output degeneration | **PARTIAL** — fak has extensive test infrastructure (`polymodelbench` lossless witness, `quality_eval`-style tests) but no standalone quality eval CLI with task-specific validators (code, JSON, logic) and MTP draft accuracy tracking. | **INSPIRE** (portable as fak leaf; validators are simple lambdas) | #10734 |
| 8 | **NPU hybrid burst pipeline (1.8× TTFT on long prompts)** | `scripts/run_pipeline.py:1-175@6c14253`, `docs/NPU_INTEGRATION.md:37-60@6c14253` | First-token latency on long context | NPU bursts first ~24 tokens (370 tok/s prefill) while iGPU loads weights; handoff to iGPU for sustained 33.8 tok/s decode | **ABSENT** — fak has no NPU integration; different hardware scope | **WATCH** (architectural pattern: heterogeneous burst + sustained) | — |
| 9 | **Engine provenance tracking (BUILD_INFO.txt with pinned commit, patches, binary SHA)** | `build_engine.sh:26-44,138-141@6c14253` | Debuggability & reproducibility | Every binary maps to source revision + applied patches + tarball SHA; `llama-server --version` string verified | **PRESENT-on-axis** — fak has `fak version modules` with `module@rev` and `dos verify`; more comprehensive | **INTEGRATE** (fak already leads here) | — |
| 10 | **Reasoning budget control (`--reasoning-budget`, `--reasoning off`)** | `run_server.sh:342-346@6c14253`, `README.md:501-512@6c14253` | Token budget for thinking models | Qwen 3.8 defaults to high reasoning; cap at 1024 tokens or disable for instant responses | **PRESENT-on-axis** — fak has `ThinkBudget` in `internal/agent/thinkbudget.go:38-132` with per-turn reasoning token budget, force-close at budget, and open/close marker handling. | **INTEGRATE** (fak already has this; no issue filed) | — |
| 11 | **Hardware governor & TTM memory auto-config** | `apply_hardware_tweaks.sh:1-5@6c14253`, `README.md:474-486@6c14253` | Sustained performance / OOM prevention | Lock GPU to "high" perf level; expand TTM pages_limit from 50% to ~56 GiB (64GB RAM) or 120 GiB (128GB RAM) | **ABSENT** — fak has software governors (loop governor, vCache governor, memory-stability-governor) but no hardware-level GPU governor or TTM memory auto-config for AMD GPUs. | **INSPIRE** (portable as operator leaf tool for AMD GPU targets) | #10735 |
| 12 | **Structured profile with DFlash2 block diffusion (adaptive K=3–7)** | `run_server.sh:318-334@6c14253`, `docs/DFLASH2_ALTERNATIVE.md@6c14253` | JSON/code structured output throughput | DFlash2 draft model achieves 42 tok/s structured; adaptive draft sizing; fails closed if unsupported | **ABSENT** — fak has no block diffusion speculative decoding; adaptive draft depth exists but for MTP, not DFlash2. Upstream ROCmFPX now has native DFlash2. | **WATCH** (upstream ROCmFPX has native DFlash2; track for future if fak adds block diffusion) | — |

---

## Worldview Findings (No Axis, Design-Level)

| Finding | Design Evidence | fak Implication |
|---------|----------------|-----------------|
| **Unified memory APU changes the optimization target** — 128 GB LPDDR5X shared by CPU+iGPU means weight streaming bandwidth is the *only* bottleneck; compute is free. | `README.md:44-50@6c14253` — bandwidth math: 200 GB/s / 13.55 GB = 14 tok/s unassisted | fak's kernel optimization should consider memory-bound regimes on unified-memory hardware |
| **Embedded MTP heads share target weights → zero extra memory traffic** — separate drafters (EAGLE, NPU) lose to embedded MTP because they add bus traffic. | `NPU_INTEGRATION.md:55-60@6c14253` — "any separate drafter loses to the model's own embedded MTP heads" | If fak adds speculative decoding, prefer embedded/weight-sharing designs over sidecar drafters |
| **Greedy/low-temp is where MTP wins** — at temp 0.8, acceptance collapses to ~25% and MTP reverts to unassisted speed. | `README.md:135-142@6c14253` — measured 88% at temp 0 vs 5% late at temp 0.8 | fak's speculative decoding should be temp-aware; document optimal sampling regime |
| **Prompt cache + MTP coexistence requires empty-`data_spec` tolerance** — without it, long-context cache hits fall back to cold prefill. | `UPSTREAM_TRACKING.md:94-95@6c14253` — upstream blocker #2; their patch enables 23–800× speedup | Critical integration point if fak adopts both prompt caching and MTP |
| **K=4 is the single-stream sweet spot on 256-bit bus** — K=6 saturates bus; K=8 causes rollback degradation. | `README.md:162-163@6c14253` — empirical sweeps; `tune_mtp.py:167-168@6c14253` | Hardware-aware MTP depth tuning is essential; not a static config |

---

## License & Provenance

- **Base Model:** Qwen 3.8 27B (Alibaba Cloud) — Apache-2.0 compatible
- **ROCmFPX Toolchain & Optimizations:** Apache-2.0 License (see `LICENSE`)
- **q38rocm Deployment Stack:** Apache-2.0
- **Upstream ROCmFPX:** Apache-2.0 (org repo `ROCmFPX/ROCmFPX`)
- **NPU/XRT:** Xilinx/AMD proprietary runtime (Linux); Lemonade OGA on Windows
- **All borrowed techniques:** INSPIRE-only (no expressive code copied; independently implement)

---

## Dismissals (Earned by Ablation)

| Candidate | Reason |
|-----------|--------|
| NPU co-decoder / sustained decode | Measured: degrades to ~14 tok/s under bus contention; embedded MTP wins (`NPU_INTEGRATION.md:64-65@6c14253`) |
| EAGLE-3 compressed head | 7.4% acceptance; 32k draft vocab covers only 18.5k/248k tokens (`NPU_INTEGRATION.md:66@6c14253`) |
| Backend crossover (ROCm/Vulkan) | fak targets CPU-first portable deployment; different hardware target. Tradeoff: fak optimizes for portability; they optimize for Strix Halo unified memory |

---

## Filed Issues

The following 5 issues have been filed as bounded, independently shippable leaves:

1. **#10730** — `feat(model): add ROCmFP4 hardware-aligned block quantization layout and loader (q38rocm borrow)` (Parent: #6221)
2. **#10731** — `feat(model): add asymmetric K/V bit allocation to 4-bit KV cache codec (q38rocm borrow)` (Parent: #2236)
3. **#10732** — `feat(model): support MTP speculative decoding and prompt cache coexistence with empty-data_spec tolerance (q38rocm borrow)` (Parent: #10193)
4. **#10733** — `feat(cmd): add automated MTP speculative parameter tuning sweep tool (q38rocm borrow)` (Parent: #10193)
5. **#10734** — `feat(test): add deterministic multi-task quality and smoke evaluation suite for speculative decoding (q38rocm borrow)` (Parent: #10193)
6. **#10735** — `feat(tools): add AMD GPU hardware governor and TTM memory limit configurator (q38rocm borrow)` (Parent: #10193)

## Companions

- Epics: #6221 (quantization interoperability), #10193 (Qwen3.8 native performance), #2236 (serving superset)
- Study receipt: `study_6777fd46608e1a1293a3f111c4a4f17a9aab28c16431e689f0e011cffc965cb6`
- Upstream: `https://github.com/julianmb/q38rocm` @ `6c142530031c923ece1adf4d8c9c824b7369fef8`