# Concept study: varjoranta/turboquant-vllm

**Verdict:** TurboQuant-vLLM is an applied inference-compression plugin for vLLM whose primary value to fak is its **operational and kernel negative knowledge** and **served-benchmark-witnessed designs**, not the raw algorithm (which converged onto scalar HIGGS). At pinned revision `c8a7e0a73b2b9bb93dc66c9380dceab985a0fbc5`, it provides production-measured evidence that decompressing only router-activated experts into a shared scratch pool is an 8.4× bs=1 decode win on MoE models, that shape-gain norm correction and asymmetric K4/V3 bits hold conversation quality where raw scalar quantization degrades, that QJL residual correction degrades attention in practice, and that CUDA-graph safety requires unconditional constant uploads and autotune pre-warming. Two new issues were filed (#10709, #10710).

- **Source:** `https://github.com/varjoranta/turboquant-vllm`
- **Pin:** `c8a7e0a73b2b9bb93dc66c9380dceab985a0fbc5`
- **Observed:** 2026-09-02
- **Filed issues:** #10709 (sparse MoE dequant into shared scratch pool), #10710 (3/4-bit rotated ladder rung design dossier)
- **Durable receipt:** `study_993da97ddccc68c1158488bcaddb101c9df8f9000eff30abc235171bc564c5a0`
- **Prior notes deduped:** [#1266](RESEARCH-turboquant-kv-quant-triage-1266.md) (paper/repo triage), [#9342](RESEARCH-google-turboquant-release-9342.md) (Google release study)

---

## Worldview

TurboQuant-vLLM was built by a solo maintainer (Hannu Varjoranta / Varjosoft Oy) assisting via an agentic workflow, optimizing primarily for **hardware cost reduction on metered cloud GPUs** and **local Apple Silicon validation**. Headline numbers focus on memory footprints that enable serving on smaller/fewer GPUs (e.g. GLM-5.1 754B in 309 GB on 2×H200 vs 8×H200; Gemma 4 26B in 12 GB on an L40S 48GB).

Its development doctrine is characterized by **radical honesty in performance reporting**:
- The README explicitly publishes that TQ3 decode throughput is only ~7–17% of BF16 on current vLLM (6.98 vs 91.71 tok/s on Qwen3-8B), stating clearly: *"TQ3 is a memory win, not a speed win, on current vLLM 0.19."*
- When throughput was initially overclaimed, commit `e38728c` corrected it: *"Fix overclaimed throughput numbers — scale depends on KV cache share of VRAM."*
- The stated project endgame is dissolution into upstream vLLM: GQA/MHA KV compression was surrendered to upstream vLLM PR #38479 (merged 2026-04-15), weight compression is submitted as upstream PR #39970, and the plugin maintains only what upstream defers (MoE expert quantization, MLA KV cache, and Apple Silicon MLX).

---

## Shipped at the pin versus upstreamed

- **Shipped in plugin at `c8a7e0a`:**
  - 3-bit TQ3 weight compression (4.3–4.6×) with Walsh-Hadamard rotation, Lloyd-Max codebook, and shape-gain norm correction (`weight_quant.py`).
  - bs=1 CUDA GEMV kernel (`csrc/tq_weight_gemv_bs1.cu`) and bs=1 Metal GEMV kernels (`turboquant_vllm/mlx_metal_kernels.py`).
  - Sparse MoE dequant threading `topk_ids` to decompress only active experts into a shared scratch pool (`moe_quant.py`, `csrc/tq_weight_dequant.cu`).
  - Block-diagonal WHT for partial-rotary models (`csrc/tq_weight_dequant.cu`, `torch_ops.py`).
  - Native packed checkpoint format (`.tq_packed` / `.tq_norms` / `tq_config.json`) with pre-fused MoE layouts (`checkpoint.py`, `vllm_quant.py`).
  - Legacy MLA KV-cache compression monkey-patch (`vllm_patch.py`).
  - Apple Silicon MLX port (`mlx_loader.py`, `mlx_model.py`, `mlx_ops.py`).
  - vLLM 0.25 compatibility shims (`RoutedExperts` export, `VLLM_USE_AOT_COMPILE=0` default).
- **Upstreamed / Retired:**
  - GQA/MHA KV cache compression moved upstream to vLLM via PR #38479 (`--kv-cache-dtype turboquant_3bit_nc`).
  - The plugin's own native KV backend was deleted in commit `f4986db` (−4,160 lines).
  - QJL residual correction was disabled by default after research showed it degraded generation quality (`torch_ops.py:442-447`).

---

## Deep inventory

Six subsystems were investigated across the codebase:

1. **Algorithm & Storage Layer:**
   - PolarQuant: unit-normalization, `D2 @ H @ D1` WHT rotation, Lloyd-Max Gaussian codebook, and shape-gain norm correction (`torch_ops.py:171-335`).
   - 3-bit packing: 8 indices per 3 bytes (`weight_quant.py:312-347`); packed geometry is bijective with bit width and treated as ground truth (`weight_quant.py:358-381`).
   - Native packed checkpoints stream safetensors with metadata in `tq_config.json` (`checkpoint.py:284-583`).
   - Kurtosis-aware mixed precision assigns bit widths based on tensor family tails (`weight_quant.py:1204-1232`).
2. **CUDA Kernels & Build Machinery:**
   - KV write path (`reshape_and_cache_kernel`) and read path (`dequant_paged_kernel`) in `csrc/turbo_quant.cu`.
   - Weight dequantization with block-diagonal WHT butterfly templated on `BLOCK_SIZE` in `csrc/tq_weight_dequant.cu:67-153`.
   - Warp-per-output-channel bs=1 GEMV in `csrc/tq_weight_gemv_bs1.cu:55-109`.
   - Build system detects visible GPU arches and embeds forward-compatible PTX (`compute_90` or `compute_80`) with `.arches.json` verification (`turboquant_vllm/build.py`).
3. **Triton Ops & Graph Compatibility:**
   - Four-tier dispatch: Triton FWHT-on-input (rotates input once, no weight dequant), Triton fused dequant-GEMM, CUDA dequant+cuBLAS, and PyTorch CPU fallback (`weight_quant.py:384-407`).
   - Autotune key narrowed by dropping `batch_size` (`triton_ops.py:390-428`, commit `f3f08bd`), speeding up CUDA graph capture by 16×.
   - Forward pass registered via `torch.library.custom_op` with `register_fake` meta implementations to avoid Dynamo tracing errors (`triton_ops.py:669-807`).
   - Removed the FWHT-on-input host-sync `.cpu()` fingerprint cache which broke CUDA graph capture (`README.md:358`, commit `05b7692`).
4. **vLLM Integration:**
   - Hooked via `vllm.general_plugins` entry point (`pyproject.toml:30-31`, `_vllm_plugin.py:66-97`).
   - `TurboQuantOnlineLinearMethod` and `TurboQuantOnlineMoEMethod` use meta-device initialization (`uses_meta_device = True`) to maintain a ~1-layer peak memory envelope during load (`vllm_quant.py:190-464, 1035-1362`).
   - Sparse MoE dequant threads `topk_ids` into `TurboQuantFusedMoEMethod.apply()` (`moe_quant.py:238-275`).
   - Compatibility shims for vLLM 0.25: factory-function `FusedMoE` bypassed in favor of `RoutedExperts`, AOT compile disabled by default (`_vllm_plugin.py:49-63`), and mapper-aware name resolution (`vllm_quant.py:1805-1819`).
5. **KV Cache & Apple Silicon MLX:**
   - PolarQuant K/V asymmetric bit allocation with norm correction (`torch_ops.py:413-587`).
   - Phase-2 quality features: `sink_tokens` (first 4 tokens kept at FP16) and `boundary_layers` (first/last 5 layers kept at K=8) in `vllm_patch.py:41-83`.
   - MLX loader supporting dense and MoE native TQ3 checkpoints (`mlx_loader.py:305-409`).
   - Metal SIMD-group GEMV kernels (`mlx_metal_kernels.py`) including a fused-gather kernel for MoE gate/up projections (`mlx_metal_kernels.py:288-413`).
6. **History, Worldview, & Provenance:**
   - 330 commits from 2026-03-31 to 2026-07-14.
   - Raw benchmark artifacts in `results/` validating asymmetric K4/V3 (score 4.75) against FP16 baseline (4.74).

---

## Candidate decisions

| Candidate Borrow | Axis | Their-worldview reason | FAK on-axis witness | Route | Issue |
|---|---|---|---|---|---|
| **Sparse top-k MoE dequant into shared scratch pool** | Wasted decompression work on inactive experts during decode | Serving large MoEs at bs=1 on metered GPUs; inactive-expert work was 93.75% of dequant time | ABSENT. `internal/model` has FP4 plan (#3019) and GGUF load, but no runtime active-only dequant. Host experts dequant-fuse; device compressed tier has no contract. | INSPIRE | **#10709** |
| **3/4-bit rotated Lloyd-Max rung with shape-gain norm correction & asymmetric K/V** | Magnitude fidelity & capacity of sub-q8 KV tiers | Maximizing context capacity under strict hardware budget; K4/V3 won served benchmarks | PARTIAL. `internal/compute/capacity.go:103` has `f32-Kraw + q8_0-K/V` and `internal/cachemeta/quantized_demote.go` gates demote, but #2240 ladder lacks the 3/4-bit rung design. | INSPIRE | **#10710** |
| **CUDA-graph capture safety: unconditional constant upload** | Self-containment of replayed graph nodes | Mixing per-layer quantization configs broke graph replay when constant uploads were conditionally skipped | PRESENT. `internal/compute/cuda.go:178` gates capture, pre-warms, and prevents realloc. Constants are passed via kernel arguments rather than global constant banks. | EXCLUDE | — |
| **Autotune key narrowing (drop batch_size)** | Graph capture stall duration | 51 capture batch sizes × multi-layer shapes caused 10–25 min startup stalls | PRESENT. `internal/model/hal.go:636` gates capture behind `halLogitsWarm` so JIT/warmup runs outside capture. | EXCLUDE | — |
| **Prebuilt extension arch manifest (`.arches.json`)** | Distribution reach across diverse GPU architectures | Distributing Python wheels across consumer and datacenter GPUs | DIVERGENT. FAK embeds cubin tables + PTX floors at build time and classifies coverage via `cuda_arch_coverage.go`. FAK does not runtime-JIT from source. | EXCLUDE | — |
| **MLX Metal SIMD-group GEMV kernels** | Apple Silicon decode throughput | Local development without cloud GPU cost; Qwen3.5-35B on 48 GB MacBook | DIVERGENT. FAK native inference is focused on CUDA fleet nodes; Mac path uses llama.cpp/Metal reference. | WATCH | — |
| **Sub-byte 3-bit packing format (8 indices / 3 bytes)** | Memory footprint and format self-description | Eliminating byte-alignment waste at 3-bit width | PARTIAL. Absorbed into the #10710 ladder design dossier. | INSPIRE | #10710 |

---

## Negative knowledge & engineering lessons

1. **The MoE shape recovery trap (Walls #9 and #10):** Native checkpoints for models like DeepSeek-V4 and Qwen3.6 often report un-fused or pre-mapper dimensions in their HuggingFace configs. Commits `f9bd39d` and `8c9faed` established that loaders must derive expert dimensions from the **norms tensor shape** `(n_experts * out_dim, n_groups)`, never trusting the registered model config shapes.
2. **CUDA-graph constant upload corruption:** In `csrc/tq_weight_dequant.cu:25-52`, attempting to cache constant uploads (`cudaMemcpyToSymbolAsync`) by skipping when values match previous layers caused silent graph corruption. Skipped uploads produce no graph nodes; upon graph replay with different layer configs, stale constants from earlier launches were reused. The fix: **unconditionally upload constants per launch** on the capturing stream.
3. **Graph-capture host sync trap:** The FWHT-on-input cache used a host-synchronizing `.cpu()` fingerprint to detect PyTorch memory reuse. Under vLLM 0.19 graph capture, this triggered `cudaErrorStreamCaptureUnsupported`. The cache was deleted (−186 LOC, commit `05b7692`), which was acceptable because modern models use fused `qkv_proj` where input reuse across separate calls does not occur.
4. **Autotune key explosion:** Autotuning Triton kernels with `batch_size` in the key caused vLLM to run full autotune sweeps across all ~51 graph capture batch sizes, creating a 10–25 minute startup stall (`triton_ops.py:390-413`). Removing `batch_size` from the autotune key reduced capture time by 16×.
5. **QJL residual degrades attention in practice:** While mathematically elegant for unbiased inner products, QJL's 1-bit residual projection added variance that softmax amplified, degrading end-to-end conversation quality (`torch_ops.py:442-447`). The maintainer disabled QJL by default, keeping full-bit MSE PolarQuant.
6. **Meta-device materialization leakage:** Storing custom parameter wrappers via standard Python `setattr` prevents `nn.Module._apply` from visiting them. If tensors remain on `meta` device, failures only surface during forward execution. Explicit meta-sweeps and validation guards (`Compressed3D.from_packed`) were required (`weight_quant.py:908-916`).

---

## Evidence limits & completeness critic

The entire tracked repository at commit `c8a7e0a` was examined. All core algorithm files, runtime CUDA/C++ kernels, Triton dispatch layers, vLLM integration shims, MLX modules, tests, benchmark scripts, and raw result JSONs were read. The only unopened subtree was the third-party vendored FLUTE implementation under `turboquant_vllm/flute/` and `csrc/flute/`, where provenance (Meta Apache-2.0, Guo et al. 2024) and license terms were verified via `NOTICE` and header inspection. 

---

## License and provenance

- **Root project:** MIT License, © 2026 Varjosoft Oy (`LICENSE:1-3`, `pyproject.toml:6`).
- **Vendored FLUTE:** Apache-2.0, © Meta Platforms (`NOTICE:9-16`, `csrc/flute/`).
- **Vendored Marlin utils:** Apache-2.0, © Elias Frantar (`csrc/flute/marlin_utils.hpp:2-14`).
- **CUTLASS submodule:** BSD-3-Clause, © NVIDIA (`NOTICE:34-40`).
- **Hadamard transform:** `csrc/flute/hadamard_transform.cpp` cites `pytorch-labs/applied-ai` without an explicit license header.
- **Verdict for FAK (Apache-2.0):** SAFE. Both filed issues are clean-room INSPIRE borrows (design-pattern adoptions); no foreign code is copied into FAK.

---

## Companions

- Issues filed: #10709, #10710
- Prior triage: #1266, #9342
- Related FAK seams: #3019 (FP4 expert weights), #5612 (MoE expert spill), #2240 (KV quantization ladder), #1474 (quantized demote lever)
