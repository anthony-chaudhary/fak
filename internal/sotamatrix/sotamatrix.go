// Package sotamatrix is the single, in-binary source of truth for "for each
// compute operation fak's kernel actually performs, what is the production /
// SOTA stack to learn from before writing it from scratch, and how should we
// relate to it (borrow / bind / stay-minimal)."
//
// It exists because this repo's kernel work has a documented failure mode: an
// agent reaches for "implement the Mac Q6_K fused MLP from scratch" or
// "hand-roll the amd64 kquant SIMD" without first checking that llama.cpp's GGML
// kernels, the Marlin / CUTLASS / FlashInfer references, or a named paper already
// solved the same contraction — and re-derives, badly, what is known art. Prior
// art WAS being researched (idea_scout, the docs/notes/RESEARCH-* corpus, the
// dated RESEARCH-backend-sota-matrix note) but the research was inert: nothing on
// the kernel-commit path forced an agent to consult it. This package makes the
// prior-art map LOAD-BEARING — a maintained datum the gate, the `fak sota`
// command, and the coverage scorecard all read.
//
// It is read by:
//   - `fak sota <op|file>` (cmd/fak/sota.go)        - the agent-facing lookup an
//     agent runs BEFORE writing a kernel: "what is the reference for this op?"
//   - the PRIOR_ART pre-commit gate (internal/hooks/gate_priorart.go) - which,
//     when a commit touches a kernel file matched by a row's FileGlobs, prints the
//     row's SOTA reference + route and suggests a `Prior-art:` trailer (advisory).
//   - `fak sota-coverage-scorecard`                - which cross-checks every row
//     against the tree (the fak-path file must exist; the row must carry a primary
//     SOTA link and an oracle), so the matrix cannot silently drift from reality.
//
// The matrix is deliberately a flat literal, not a directory scan: it carries the
// human-meaningful ROUTE DECISION (borrow vs bind vs stay-minimal) and the SOTA
// REFERENCE for each operation, which a directory walk cannot recover. Adding a
// kernel operation means adding one row here — the same additive-leaf discipline
// the rest of the kernel uses, and the discipline the PRIOR_ART gate exists to
// keep honest.
//
// Provenance: the rows are lifted from docs/notes/RESEARCH-backend-sota-matrix-2026-06-26.md
// (read out of the tree on 2026-06-26), promoted here from a dated note into a
// maintained datum so it stops rotting.
package sotamatrix

import "sort"

// Route is fak's chosen relationship to the SOTA stack for one operation. It is
// the load-bearing decision the matrix records: the answer to "should we write
// this from scratch?" is almost never yes.
type Route string

const (
	// RouteBorrow: study the reference kernel and adapt its technique (e.g. the
	// fused tile-dequant pattern), AFTER a witness for the current path exists.
	RouteBorrow Route = "borrow"
	// RouteBind: bind directly to the production library (e.g. cuBLAS fp16) or
	// format (e.g. GGUF) rather than re-implement it.
	RouteBind Route = "bind"
	// RouteStayMinimal: fak's value here is the bit-exact contract, not beating
	// the reference at raw throughput — implement the minimum and stay there.
	RouteStayMinimal Route = "stay-minimal"
)

// Op is one compute operation fak's kernel performs, mapped to the SOTA stack to
// learn from before touching it.
type Op struct {
	// Slug is the stable lookup key for `fak sota <slug>` (kebab-case).
	Slug string
	// Title is the human name of the operation.
	Title string
	// FileGlobs are repo-relative path globs whose touched files in a staged
	// commit should trigger the PRIOR_ART advisory for THIS row. They are matched
	// with path.Match against each glob segment (see hooks.matchKernelGlob).
	FileGlobs []string
	// FakPath names where fak performs this operation today (verified against the
	// tree). The scorecard requires the first FakPath file to exist.
	FakPath string
	// SOTA is the production / reference stack worth learning from.
	SOTA string
	// PrimaryLink is the primary repo or official-docs URL for the SOTA stack —
	// the thing to actually read. The scorecard requires a non-empty http(s) link.
	PrimaryLink string
	// Route is fak's chosen relationship to that stack.
	Route Route
	// Oracle is how a fak implementation is held honest against the reference
	// (the verification witness). The scorecard requires it to be non-empty.
	Oracle string
	// Papers are named primary references (papers/blogs) for the technique, when
	// the technique has a canonical write-up beyond a repo. May be empty.
	Papers []string
	// Note is a one-line gloss on the route decision (e.g. "borrow only after the
	// witness exists").
	Note string
}

// matrix is the flat source of truth. Each row was read out of the tree, not
// assumed. Keep it sorted by Slug for stable output and easy diffing.
var matrix = []Op{
	{
		Slug:  "amd-vulkan-quant-gemm",
		Title: "AMD Vulkan Q2_K / Q4_K / Q8_0 GEMV and GEMM",
		FileGlobs: []string{
			"internal/compute/vulkan.go",
			"internal/compute/shaders/q*k_matmul.comp",
			"internal/compute/shaders/q8_matmul*.comp",
		},
		FakPath:     "internal/compute/vulkan.go; internal/compute/shaders/q2k_matmul.comp, q4k_matmul.comp, q8_matmul*.comp",
		SOTA:        "Nathanw1014/strix-halo-llamacpp@45bec945dd4c7944fba45ebd02e6b713b342d0ba gfx1151 Vulkan rowtiling, Wave32/cooperative-matrix, and dequant-once mechanisms (MIT)",
		PrimaryLink: "https://github.com/Nathanw1014/strix-halo-llamacpp/tree/45bec945dd4c7944fba45ebd02e6b713b342d0ba",
		Route:       RouteBorrow,
		Oracle:      "independent CPU dequant/GEMM over the same packed bytes: argmax exact and cosine >= 0.995 for Q2_K/Q4_K, with finite outputs, resident packed-byte accounting, and fak-native Vulkan path identity",
		Papers:      []string{"llama.cpp k-quants design (PR #1684, Kawrakow)"},
		Note:        "Borrow one measured gfx1151 mechanism at a time after a current-path witness; attribution is required and no runtime or backend fallback may receive performance credit.",
	},
	{
		Slug:  "amd-vulkan-qwen-gdn",
		Title: "AMD Vulkan Qwen3.8 Gated-DeltaNet recurrence",
		FileGlobs: []string{
			"internal/compute/vulkan_qwen35_gdn.go",
			"internal/compute/shaders/qwen35_gdn_*.comp",
			"internal/compute/shaders/qwen35_split_qg*.comp",
			"internal/compute/shaders/qwen35_partial_rope*.comp",
			"internal/compute/shaders/sigmoid_mul.comp",
		},
		FakPath:     "internal/compute/vulkan_qwen35_gdn.go; internal/compute/shaders/qwen35_gdn_*.comp",
		SOTA:        "julianmb/q38rocm@6c142530031c923ece1adf4d8c9c824b7369fef8 Strix Halo Qwen3.8 execution study (Apache-2.0); FLA@bccaf2d3cf4d9badc8be050a2c71616220b246d7 Gated-DeltaNet recurrence (MIT)",
		PrimaryLink: "https://github.com/julianmb/q38rocm/tree/6c142530031c923ece1adf4d8c9c824b7369fef8",
		Route:       RouteStayMinimal,
		Oracle:      "fak production CPU recurrence: multi-token per-step output and logit equivalence, convolution-state and recurrent-state continuity at predeclared tolerances, exact greedy-token equality, finite outputs, fak-native Vulkan identity, and zero fallback",
		Papers: []string{
			"FLA fused recurrent Gated-DeltaNet (bccaf2d3cf4d9badc8be050a2c71616220b246d7): https://github.com/fla-org/flash-linear-attention/blob/bccaf2d3cf4d9badc8be050a2c71616220b246d7/fla/ops/gated_delta_rule/fused_recurrent.py",
		},
		Note: "Stay minimal until #12026 proves production CPU parity; only then borrow a measured recurrence mechanism, with no runtime or backend fallback.",
	},
	{
		Slug:  "amd-vulkan-sequence-attention",
		Title: "AMD Vulkan Qwen3.8 sequence prefill and attention",
		FileGlobs: []string{
			"internal/compute/vulkan_qwen35_sequence.go",
			"internal/compute/shaders/qwen35_causal_attention*.comp",
		},
		FakPath:     "internal/compute/vulkan_qwen35_sequence.go; internal/compute/shaders/qwen35_causal_attention*.comp",
		SOTA:        "Nathanw1014/strix-halo-llamacpp@45bec945dd4c7944fba45ebd02e6b713b342d0ba Strix Halo Vulkan Flash-Attention and dequant-once mechanisms (MIT)",
		PrimaryLink: "https://github.com/Nathanw1014/strix-halo-llamacpp/tree/45bec945dd4c7944fba45ebd02e6b713b342d0ba",
		Route:       RouteBorrow,
		Oracle:      "fak production CPU full sequence path: exact token identity, finite logits, logit cosine >= 0.999, per-token activation parity at the declared floor, persistent-state continuity, fak-native Vulkan identity, and zero fallback",
		Papers:      []string{"FlashAttention-2 (Dao, 2023) arXiv:2307.08691"},
		Note:        "Borrow only the mechanism selected by a current full-sequence witness; preserve model semantics and permit no runtime or backend fallback.",
	},
	{
		Slug:  "cuda-qwen-residual-norm",
		Title: "CUDA Qwen ordinary residual Add and PostNorm consumer",
		FileGlobs: []string{
			"internal/compute/cuda/qwen35_residual_norm.cuh",
			"internal/compute/cuda/transformer_ops.cuh",
			"internal/compute/cuda/qwen35_mtp.cuh",
			"internal/compute/cuda_qwen35_residual_norm*.go",
		},
		FakPath:     "internal/compute/cuda/qwen35_residual_norm.cuh; internal/compute/cuda_qwen35_sequence.go",
		SOTA:        "FlashInfer fused_add_rmsnorm technique; existing fak transformer_ops.cuh residual/RMS kernel@133fc57c9bf4f209390031ed31833a420d87faf6 (Apache-2.0)",
		PrimaryLink: "https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html",
		Route:       RouteStayMinimal,
		Oracle:      "bit-identical stored residual and normalized output versus original CUDA Add plus RMSNorm, unchanged read-only operands, then ordinary model greedy/logit parity and same-card net uncached-prefill ABCCBA",
		Note:        "Default-off checked integration of an already-present kernel, not new kernel credit; preserve F32 residual boundary, 256-thread tree and original off-path allocation order. No physical parity or throughput acceptance yet.",
	},
	{
		Slug:        "dense-f32-gemm",
		Title:       "Dense f32 GEMM",
		FileGlobs:   []string{"internal/compute/cpuref.go", "internal/model/parallel.go", "internal/model/fdot_amd64.s"},
		FakPath:     "internal/compute/cpuref.go (Reference, max|Δ|=0) + internal/model/parallel.go, fdot_amd64.s",
		SOTA:        "cuBLAS / CUTLASS GEMM",
		PrimaryLink: "https://docs.nvidia.com/cutlass/",
		Route:       RouteStayMinimal,
		Oracle:      "cpuref f32 bit-identity (max|Δ|=0)",
		Note:        "fak's value is the bit-exact contract, not beating cuBLAS at GEMM.",
	},
	{
		Slug:  "dense-quant-device-gemm",
		Title: "Dense fp16 / Q8_0 / Q4_K device GEMM",
		FileGlobs: []string{
			"internal/compute/cuda.go", "internal/compute/cuda_kernels.cu",
			"internal/compute/cuda/quant_matmul.cuh",
			"internal/compute/cuda/q2k_mmq.cuh",
			"internal/compute/cuda/q3k_matmul.cuh", "internal/compute/cuda/q3k_mmq.cuh", "internal/compute/cuda/q3k_mmvq.cuh",
			"internal/compute/cuda/q4k_mmq.cuh", "internal/compute/cuda/q4k_mmvq.cuh", "internal/compute/cuda/q4k_fp8lt.cuh",
			"internal/compute/cuda/q4k_shared_input*.cuh",
			"internal/compute/cuda/q8_mmq.cuh",
			"internal/compute/cuda/iq*.cuh",
			"internal/compute/cuda/graph_quant_workspace.cuh",
			"internal/compute/prefill.go", "internal/compute/prefill_cuda.go",
			"internal/compute/graph_cuda.go", "internal/compute/tf32_cuda.go",
			"internal/compute/quant_q4k.go",
		},
		FakPath:     "internal/compute/cuda.go + cuda/quant_matmul.cuh (k_q8_gemm, k_q4k_gemm, cublasGemmEx fp16)",
		SOTA:        "cuBLAS (fp16/tensor-core); llama.cpp Q8_0/Q4_K fused dequant-GEMM",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBind,
		Oracle:      "cpuref f32, argmax-exact + cosine ≥ floor (fp16 0.997 / Q8 0.999 / Q4_K 0.995)",
		Note:        "cuBLAS fp16 already bound; borrow the fused tile-dequant pattern for the int lanes.",
	},
	{
		Slug:        "awq-int4-gemm",
		Title:       "AWQ 4-bit GEMV/GEMM",
		FileGlobs:   []string{"internal/model/awq*.go", "internal/compute/cuda_kernels.cu", "internal/compute/cuda/weight_formats.cuh"},
		FakPath:     "internal/compute/cuda.go:1101 (AWQMatMul) + cuda/weight_formats.cuh (k_awq_gemv/gemm) + internal/model/awq.go",
		SOTA:        "Marlin mixed-precision INT4 GPTQ kernel; AutoAWQ",
		PrimaryLink: "https://github.com/IST-DASLab/marlin",
		Route:       RouteBorrow,
		Oracle:      "cpuref f32 + HF AWQ reference (greedy-continuation + cosine ≥ 0.995)",
		Papers:      []string{"AWQ: Activation-aware Weight Quantization (Lin et al., 2023) arXiv:2306.00978"},
		Note:        "Borrow Marlin's fused dequant-MMA AFTER the AWQ CUDA witness exists (the recorded gap).",
	},
	{
		Slug:        "gptq-resident",
		Title:       "GPTQ resident (CPU)",
		FileGlobs:   []string{"internal/model/gptq.go", "internal/model/gptq_*.go"},
		FakPath:     "internal/model/gptq.go (LoadGPTQ, resident 4/8-bit, g_idx) — CPU-resident, honest-fenced",
		SOTA:        "AutoGPTQ / Marlin",
		PrimaryLink: "https://github.com/IST-DASLab/marlin",
		Route:       RouteBind,
		Oracle:      "HF GPTQ dequant / cpuref",
		Papers:      []string{"GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022) arXiv:2210.17323"},
		Note:        "A Marlin-style GPU kernel is the GPU rung; CPU-resident parity is the floor.",
	},
	{
		Slug:        "exl2",
		Title:       "EXL2 loader",
		FileGlobs:   []string{"internal/model/exl2.go", "internal/model/exl2_*.go"},
		FakPath:     "internal/model/exl2.go (loader)",
		SOTA:        "ExLlamaV2",
		PrimaryLink: "https://github.com/turboderp-org/exllamav2",
		Route:       RouteStayMinimal,
		Oracle:      "ExLlamaV2 reference generation",
		Note:        "Loader parity only; no kernel claim.",
	},
	{
		Slug:        "gguf-quant-resident",
		Title:       "GGUF quant-at-load (Q4_K/Q5_K/Q6_K resident)",
		FileGlobs:   []string{"internal/ggufload/*.go", "internal/model/quant_q4k*.go", "internal/model/quant_q2.go", "internal/model/quant_q4.go"},
		FakPath:     "internal/ggufload/quant_q4k_loader.go; internal/model/quant_q4k.go, qwen35_prefill_q4k.go",
		SOTA:        "llama.cpp GGUF quant + backend",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBind,
		Oracle:      "llama.cpp oracle (2-token parity, BENCHMARK-AUTHORITY Qwen3.6-27B row)",
		Note:        "Bind to the GGUF format; fak's resident-q4k is the parity target, not a new format.",
	},
	{
		Slug:  "gguf-ternary-q2-resident",
		Title: "GGUF ternary Q2_0 resident GEMV (g128 {-1,0,+1})",
		// The g128 GGUF-resident ternary path (#4870): 34-byte blocks (one f16 scale + 128
		// 2-bit codes, y=(c-1)*d) are fed VERBATIM into the CPU GEMV — no load-time
		// re-quantize (a f32->quantize round trip re-derives scales and shifts the code
		// offset) and no dequant-to-f32 residency. Same "the resident bytes ARE the GGUF
		// bytes" discipline as quant_q4k.go, so the prior art to read is ggml's packed
		// block-quant container + its dequant->dot decode, not a new format. Honest fence:
		// no upstream ggml TERNARY type is claimed here — this container arrives with the
		// Ternary-Bonsai-27B checkpoint (docs/supported/models.md).
		FileGlobs:   []string{"internal/model/quant_q2_resident*.go"},
		FakPath:     "internal/model/quant_q2_resident.go (wrapQ2G128FromRaw, q2G128MatRowsRange, dequantQ2G128Block)",
		SOTA:        "llama.cpp / ggml packed block-quant containers + their dequant->dot resident decode",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteBind,
		Oracle:      "TestQ2ResidentMatchesDequant (internal/model): resident GEMV via residentMatRows vs an independent restatement of the container dequant — argmax-exact, |Δ|<=1e-4; block dequant element-EXACT vs that reference; packed 34 B per 128 weights",
		Note:        "Bind to the container: never re-quantize at load, and never widen the resident form to f32 (~15x the footprint).",
	},
	{
		Slug:        "cpu-fp8-tiled-quantize",
		Title:       "CPU FP8 block-scale to Q8 tiled load conversion",
		FileGlobs:   []string{"internal/model/fp8_quantize*.go", "internal/model/safetensors_quant.go"},
		FakPath:     "internal/model/fp8_quantize.go (quantizeFP8BlockScaleQ8); internal/model/safetensors_quant.go (quantizeFP8BlockScaleTensorInto)",
		SOTA:        "kvcache-ai/ktransformers@0c2912a5e25dc3459d728a59be2d8c7461ba64bd GemmKernel224FP8 lookup decode and bounded tile widening (Apache-2.0)",
		PrimaryLink: "https://github.com/kvcache-ai/ktransformers/blob/0c2912a5e25dc3459d728a59be2d8c7461ba64bd/kt-kernel/operators/amx/la/amx_raw_kernels.hpp",
		Route:       RouteBorrow,
		Oracle:      "TestFP8TiledQuantLoadMatchesExpanded: all 256 E4M3FN codes against the existing decoder; loaded Q8 codes/scales bit-exact against decodeFP8BlockScale followed by quantizeQ8, including ragged scale tiles and canonical names",
		Note:        "Source-only partial #5240; verification pending. Removes the FP8 whole-tensor decode temporary, keeps Q8 residency and existing accelerator preparation, and claims neither native FP8 GEMM nor measured throughput gains.",
	},
	{
		Slug:  "cpu-quant-simd",
		Title: "CPU K-quant / Q4_K / Q6_K SIMD dequant-GEMM",
		// The biggest "from scratch" risk surface in the tree: the hand-written amd64/arm64/noasm
		// quant lanes (quant_amd64_kquant.go, quant_arm64_q6k.go, quant_gemm.go, quant_iquant.go,
		// quant_kquant*.go, quant_quantize*.go, ...). These re-implement exactly what llama.cpp's
		// ggml-quants kernels already do — read that BEFORE hand-rolling another SIMD lane.
		FileGlobs: []string{
			"internal/model/quant_amd64*.go", "internal/model/quant_arm64*.go",
			"internal/model/quant_noasm*.go", "internal/model/quant_kquant*.go",
			"internal/model/quant_iquant.go", "internal/model/quant_gemm.go",
			"internal/model/quant_gemv*.go", "internal/model/quant_forward.go",
			"internal/model/quant_accel*.go", "internal/model/quant_quantize*.go",
		},
		FakPath:     "internal/model/quant_kquant.go, quant_amd64_kquant.go, quant_arm64_q6k.go, quant_gemm.go (SIMD dequant-GEMM lanes)",
		SOTA:        "llama.cpp ggml-quants (K-quant block dequant + vec_dot SIMD kernels)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteBorrow,
		Oracle:      "cpuref f32 dequant (max|Δ|=0 vs the reference block dequant)",
		Papers:      []string{"k-quants design (llama.cpp PR #1684, Kawrakow)"},
		Note:        "Borrow the block-dequant + vec_dot layout; do NOT re-derive a SIMD lane without reading ggml-quants first.",
	},
	{
		Slug:  "iq3-xxs-int8-decode",
		Title: "IQ3_XXS codebook decode (int8 GEMV spine)",
		// The i-quant family is NOT the k-quant family: an IQ3_XXS sub-block decodes through
		// a 256-entry codebook grid plus a 7-bit sign selector, not a packed-scale bit field,
		// so the cpu-quant-simd row's K-quant advice does not transfer. The tables this file
		// indexes (internal/ggufload/gguf_iq3_tables.go) are transcribed VERBATIM from ggml's
		// ggml-common.h and the loader's dequant is held bit-faithful to llama.cpp's
		// dequantize_row_iq3_xxs — so ggml-quants is the reference for any arch lane here.
		FileGlobs:   []string{"internal/model/quant_iq3*.go"},
		FakPath:     "internal/model/quant_iq3_int8.go (iq3xxsReduceRowScalar, iq3xxsCombineRow, iq3xxsMatRowsRangeInt8); codebook tables in internal/ggufload/gguf_iq3_tables.go",
		SOTA:        "llama.cpp ggml-quants i-quants (dequantize_row_iq3_xxs; the IQ3_XXS grid + IQ2/IQ3 sign tables in ggml-common.h)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteBorrow,
		Oracle:      "TestIQ3XXSReduceScalarMatchesDequantizedIntegerOracle (each int32 sub-block dot EXACT vs an oracle rebuilt from the dequantized block) + TestW3MLPDispatchRealReductionDimensions (dispatch seam == the int8 body exactly at in=5120/17408; max|int8-f32|/referenceRMS <= 0.05)",
		Note:        "Borrow ggml's grid+sign decode; the spine is default-off (FAK_W3_MLP) and an arch lane may replace it only under the exact-reducer witness.",
	},
	{
		Slug:        "moe-expert-dispatch",
		Title:       "MoE expert dispatch",
		FileGlobs:   []string{"internal/model/moe.go", "internal/model/moe_*.go", "internal/model/glm_dsa.go"},
		FakPath:     "internal/model/moe.go, glm_dsa.go, moe_offload.go (grouped-decode expert dispatch)",
		SOTA:        "TensorRT-LLM / DeepEP expert-parallel",
		PrimaryLink: "https://github.com/deepseek-ai/DeepEP",
		Route:       RouteBorrow,
		Oracle:      "dense reference / HF",
		Papers:      []string{"DeepSeek-V3 expert-parallel report; GShard (Lepikhin et al.) arXiv:2006.16668"},
		Note:        "Borrow grouped-decode cleanup; the cross-process EP transport is the collective-comm row.",
	},
	{
		Slug:  "moe-expert-residency-ring",
		Title: "MoE expert-weight residency ring (bounded device weight cache)",
		// The WEIGHT-side twin of the kv-cache-paging row: for a MoE model the hot working
		// set is which EXPERTS are resident, not the KV. pagedRing is a byte-budgeted LRU
		// over polymodel.Pool (pinned exemption, all-or-nothing admit) staging f32 / Q8_0 /
		// Q4_K weights — the Tier-1 GPU ring of the 3-tier GPU/pinned-CPU/SSD expert cache
		// docs/awesome-caching/README.md tracks as M11. Standalone and OFF the live serve
		// path today; the open question is the PLACEMENT POLICY, which is where the prior
		// art lives — do not re-derive an expert-placement heuristic here.
		FileGlobs:   []string{"internal/model/paging_ring*.go", "internal/model/expert_warmpins*.go"},
		FakPath:     "internal/model/paging_ring.go (pagedRing: LRU byte budget over polymodel.Pool) + internal/model/expert_warmpins.go (the resident pin-set)",
		SOTA:        "ktransformers GPU/CPU expert placement (activation-aware expert orchestration); DeepSeek EPLB expert-parallel placement",
		PrimaryLink: "https://github.com/kvcache-ai/ktransformers",
		Route:       RouteBorrow,
		Oracle:      "paging_ring_test.go: ring GEMM bit-equal to a resident cpu-ref GEMM on both a HIT and a MISS (f32, Q8_0, Q4_K), exact pageIn/hit/evict counters, used()<=budget(), and an evicted weight pages back IN",
		Papers:      []string{"FlexGen: LP-scheduled GPU+CPU+disk placement of weights/KV/activations arXiv:2303.06865"},
		Note:        "Borrow the activation-aware placement policy before extending the LRU; fak's value today is the bounded-residency lifecycle plus the bit-equality witness, not a placement heuristic.",
	},
	{
		Slug:  "collective-comm",
		Title: "Collective communication (multi-GPU all-reduce / process-group)",
		// The distributed-serve transport under EP/TP: cuda_nccl.cu is the single-process
		// ncclCommInitAll set; cuda_nccl_pg.cu is the multi-PROCESS ncclCommInitRank bootstrap
		// (one OS process per GPU, the torchrun/MPI-style path). Both bind -lnccl; the host
		// wiring is collective.go + cuda_collective*.go. Do NOT hand-roll a ring/tree all-reduce.
		FileGlobs: []string{
			"internal/compute/cuda_nccl*.cu", "internal/compute/cuda_collective*.go",
			"internal/compute/collective.go",
		},
		FakPath:     "internal/compute/cuda_nccl_pg.cu (multi-process ncclCommInitRank), cuda_nccl.cu (single-process ncclCommInitAll); internal/compute/collective.go, cuda_collective*.go host wiring",
		SOTA:        "NVIDIA NCCL (ring/tree all-reduce + process-group bootstrap); NVSHMEM; MSCCL++",
		PrimaryLink: "https://github.com/NVIDIA/nccl",
		Route:       RouteBind,
		Oracle:      "host DistComm / cpuref CollectiveBackend reduce — argmax-exact + cosine (NCCL ring/tree sums in a hardware-determined order, so this is an Approx peer, never max|Δ|=0)",
		Papers:      []string{"Ring all-reduce (Gibiansky, \"Bringing HPC Techniques to Deep Learning\", 2017)", "Horovod (Sergeev & Del Balso, 2018) arXiv:1802.05799"},
		Note:        "Bind to NCCL's collectives directly (-lnccl); fak's value is the honest Approx fence around a hardware-ordered reduce, not a new all-reduce.",
	},
	{
		Slug:  "cuda-qwen-attention-postprocess",
		Title: "CUDA Qwen sequence Q/K norm, partial RoPE and owned KV append",
		FileGlobs: []string{
			"internal/compute/cuda/qwen35_attention_postprocess.cuh",
			"internal/compute/cuda_qwen35_attention_postprocess*.go",
		},
		FakPath:     "internal/compute/cuda/qwen35_attention_postprocess.cuh; internal/compute/cuda_qwen35_sequence.go",
		SOTA:        "FlashInfer fused Q/K RMSNorm + RoPE (technique reference); exact arithmetic donor is fak transformer_ops.cuh@133fc57c9bf4f209390031ed31833a420d87faf6 (Apache-2.0)",
		PrimaryLink: "https://docs.flashinfer.ai/generated/flashinfer.norm.fused_qk_rmsnorm_rope.html",
		Route:       RouteStayMinimal,
		Oracle:      "bit-identical Q, normalized K, rotated K and V versus retained CUDA seven-launch composition, unchanged KV prefix/tail bounds, then ordinary-model greedy/logit parity and same-card uncached-prefill ABCCBA",
		Note:        "Experimental default-off F32 24-query/4-KV/hd256 path; reuse the existing 256-thread RMS tree and double RoPE expressions rather than importing different precision/reduction or paged ownership. Source-only candidate, no hardware/performance qualification.",
	},
	{
		Slug:  "cuda-qwen-gdn",
		Title: "CUDA Qwen3.8 Gated-DeltaNet recurrence",
		FileGlobs: []string{
			"internal/compute/cuda_qwen35_gdn*.go",
			"internal/compute/cuda_kernels.go",
			"internal/compute/cuda_kernels.cu",
			"internal/compute/cuda/qwen35_gdn.cuh",
			"internal/compute/cuda/gdn_decode_vllm.cuh",
			"internal/compute/cuda/gdn_mtp_decode_vllm.cuh",
			"internal/compute/cuda/gdn_shared_split.cuh",
			"internal/compute/cuda/graph_gdn_capture.cuh",
		},
		FakPath:     "internal/compute/cuda_qwen35_gdn.go + cuda_qwen35_gdn_sequence.go + cuda_kernels.go + cuda/qwen35_gdn.cuh (Qwen GDN decode and sequence recurrence)",
		SOTA:        "FLA@bccaf2d3cf4d9badc8be050a2c71616220b246d7 fused-recurrent and chunked Gated-DeltaNet; llama.cpp@ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0 Qwen3.5 state order; MLX-LM@cc8521569694a3240b52c98acffd100d59b4c755 Qwen3.5 cache semantics (all MIT)",
		PrimaryLink: "https://github.com/fla-org/flash-linear-attention/blob/bccaf2d3cf4d9badc8be050a2c71616220b246d7/fla/ops/gated_delta_rule/fused_recurrent.py",
		Route:       RouteBorrow,
		Oracle:      "existing internal/model and internal/compute CPU reference: per-step output, convolution-state, and recurrent-state parity at the declared floor; exact deterministic greedy-token equality; engine/path identity fak-native CUDA; zero fallback; reject a kernel-time win that worsens end-to-end quality, synchronization, memory, or setup cost",
		Papers: []string{
			"FLA chunked lower-triangular/KKT solve and occupancy tradeoff (bccaf2d3cf4d9badc8be050a2c71616220b246d7): https://github.com/fla-org/flash-linear-attention/blob/bccaf2d3cf4d9badc8be050a2c71616220b246d7/fla/ops/gated_delta_rule/chunk_fwd.py",
			"llama.cpp Qwen3.5 convolution, normalization, recurrence, and output order (ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0): https://github.com/ggml-org/llama.cpp/blob/ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0/src/models/qwen35.cpp",
			"MLX-LM Qwen3.5 GDN/cache semantics (cc8521569694a3240b52c98acffd100d59b4c755): https://github.com/ml-explore/mlx-lm/blob/cc8521569694a3240b52c98acffd100d59b4c755/mlx_lm/models/qwen3_5.py",
		},
		Note: "Borrow the pinned state order and GPU recurrence techniques only after measuring the current recurrence; FLA is a technique reference, never a runtime dependency. Default to the smallest parity-preserving adaptation and permit no runtime or backend fallback.",
	},
	{
		Slug:  "cuda-qwen-vllm-qk",
		Title: "CUDA ordinary Qwen fused QK RMSNorm, NeoX RoPE and owned KV append",
		FileGlobs: []string{
			"internal/compute/cuda/qwen35_qknorm_rope_cache_vllm.cuh",
			"internal/compute/*qwen35_qknorm_rope_cache_vllm*.go",
			"internal/model/qwen35_fused_qk*.go",
		},
		FakPath:     "internal/compute/cuda/qwen35_qknorm_rope_cache_vllm.cuh; internal/model/qwen35_hal.go",
		SOTA:        "vLLM fused_qknorm_rope_cache_kernel.cu@fa3b9d1377219247060d3c8b49eae199c9e7e154 (Apache-2.0), directly adapted subwarp reduction and shared-memory rotary pairs",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/csrc/fused_qknorm_rope_cache_kernel.cu",
		Route:       RouteBorrow,
		Oracle:      "Q/Kraw/rotated K within 2e-6 absolute plus 2e-6 relative of retained F32 CUDA composition; exact V/input/sentinels/frontier; one enqueue and zero activation host transfer; unchanged ordinary-model greedy/logit gates and same-device net decode time before promotion",
		Note:        "Default-off single-token device-zero F32 adapter for head dimensions 64/128/192/256 and unscaled rotary dimensions divisible by eight. Existing owned contiguous cache capacity only; capture, paging, narrowed KV and scaled RoPE retain the old native path. This changes the donor's packed half activation, cached-angle and paged-cache ABI; not bit-equivalent to the old F32 reduction. Physical and performance qualification is pending.",
	},
	{
		Slug:        "retained-q56-prompt-mmq",
		Title:       "Retained Q5_K/Q6_K ordinary CUDA prompt MMQ family",
		FileGlobs:   []string{"internal/compute/cuda/q56k_mmq.cuh", "internal/compute/cuda_q56k_mmq*.go", "internal/computebuild/cuda_q56k_mmq_test.go", "cmd/fak/serve_qwen35_q56k_mmq_cuda_test.go"},
		FakPath:     "internal/compute/cuda_q56k_mmq.go + cuda/q56k_mmq.cuh; call path cmd/fak/serve.go -> existing raw K-quant loader -> Qwen sequence projections -> explicit checked MMQ ABI",
		SOTA:        "ggml-org/llama.cpp@0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd Q5 DS4 / Q6 D4 Ampere MMQ (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd/ggml/src/ggml-cuda/mmq.cuh",
		Route:       RouteBorrow,
		Oracle:      "required selected/default native archives; independent packed CPU decoder plus exact Q8_1 fixture oracle, real retained GGUF -> SessionPrefill -> two continuing decodes; matched uncached-prefill wall time and first/subsequent decode parity with all activation packing, allocation and retirement included",
		Note:        "One default-off P>=128 format family, not two kernel credits. Stored/loaded Q5_K or Q6_K stays packed; temporary Q8_1 activation is deliberate and non-bit-exact. Require actual compiled SM80+ integer MMA image and capable device; no SM100 tcgen dependency or SM120 execution claim. Default, packed-GDN native projections and P<128 retain prior math. All compiler/device/model/performance acceptance remains UNRUN until returned source-bound evidence.",
	},
	{
		Slug:        "qwen-host-tiled-attention",
		Title:       "Qwen ordinary host prefill multi-query tiled attention",
		FileGlobs:   []string{"internal/model/qwen35_prefill_tiled_attention*.go"},
		FakPath:     "internal/model/qwen35_prefill_tiled_attention.go; internal/model/qwen35_prefill.go; internal/model/qwen35_prefill_q4k.go",
		SOTA:        "ggml-org/llama.cpp@74bca3950054eed1795f525cb9cf06bfe796492d tiled CPU flash attention (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/ggml/src/ggml-cpu/ops.cpp",
		Route:       RouteBorrow,
		Oracle:      "retained score-row F32 host attention: finite causal/window/softcap tile parity, unchanged KV and custom/observer routes; ordinary Qwen Q8/Q4_K full-prefill greedy/logit parity, then matched uncached-prefill wall time with all validation/packing and process costs included",
		Note:        "Default-off multi-query K/V packing and online normalization in an owned qf loan. Existing fdot/saxpy retained; reduction order is not bit-exact. No model, hardware or performance acceptance; reject a component win that regresses ordinary request latency or memory.",
	},
	{
		Slug:        "fused-attention",
		Title:       "Fused attention (MHA/GQA/MQA)",
		FileGlobs:   []string{"internal/compute/cuda_kernels.cu", "internal/compute/cuda/attention.cuh", "internal/compute/cuda/attention_generic.cuh", "internal/compute/cuda/fattn_mma_hd256.cuh", "internal/compute/flash*.go"},
		FakPath:     "internal/compute/cuda/attention.cuh k_flash_attention (#486, cosine floor 0.999)",
		SOTA:        "FlashInfer / FlashAttention",
		PrimaryLink: "https://docs.flashinfer.ai/",
		Route:       RouteStayMinimal,
		Oracle:      "cpuref (cosine ≥ 0.999)",
		Papers:      []string{"FlashAttention-2 (Dao, 2023) arXiv:2307.08691", "FlashInfer (Ye et al., 2024) arXiv:2501.01005"},
		Note:        "Own fused online-softmax ships; consider FlashInfer only for paged/variable-length.",
	},
	{
		Slug:        "glm-sparse-attention-dsa",
		Title:       "GLM sparse attention (DSA)",
		FileGlobs:   []string{"internal/compute/dsa.go", "internal/compute/dsa_*.go"},
		FakPath:     "internal/compute/dsa.go, dsa_index.go (cosine floor 0.999)",
		SOTA:        "TensorRT-LLM custom sparse",
		PrimaryLink: "https://nvidia.github.io/TensorRT-LLM/",
		Route:       RouteStayMinimal,
		Oracle:      "cpuref",
		Note:        "Host-index + device-sparse ships; the host-side index roundtrip is the bottleneck.",
	},
	{
		Slug:        "kv-cache-paging",
		Title:       "KV cache (paging / prefix reuse)",
		FileGlobs:   []string{"internal/model/kv*.go", "internal/model/paging.go", "internal/radixkv/*.go", "internal/compute/kvprecision.go", "internal/compute/kvresidency.go"},
		FakPath:     "internal/model/kv.go, kvcache.go, kvlayout.go, paging.go, internal/radixkv (RadixAttention, bit-exact eviction)",
		SOTA:        "vLLM PagedAttention; SGLang RadixAttention; FlashInfer",
		PrimaryLink: "https://docs.vllm.ai",
		Route:       RouteStayMinimal,
		Oracle:      "bit-identity (max|Δ|=0)",
		Papers:      []string{"PagedAttention / vLLM (Kwon et al., 2023) arXiv:2309.06180", "SGLang RadixAttention (Zheng et al., 2023) arXiv:2312.07104"},
		Note:        "Differentiator is exact eviction/reuse on an owned f32 cache, not paged throughput.",
	},
	{
		Slug:  "kv-cache-transform-compression",
		Title: "KV cache transform coding / mixed-rate compression",
		FileGlobs: []string{
			"internal/model/kvquant.go", "internal/model/coldkv.go",
			"internal/engine/kv_quantization.go", "internal/compute/kvprecision.go",
			"internal/compute/cuda/kv_precision.cuh",
		},
		FakPath:     "internal/model/kvquant.go + coldkv.go; internal/engine/kv_quantization.go; internal/compute/kvprecision.go",
		SOTA:        "SPECTRA@272032275106dc5944fbfa7091a1ceb403fa7e28 activation-weighted latent transform + reverse water-filling; KIVI; KVQuant; TurboQuant; PolarQuant; Palu",
		PrimaryLink: "https://github.com/nokia-applied-research/SPECTRA/tree/272032275106dc5944fbfa7091a1ceb403fa7e28",
		Route:       RouteBorrow,
		Oracle:      "matched fak-native Qwen3.8 dense-KV and current Q4-cold-KV controls; predeclared LongBench, RULER, and agent-code quality floors; actual payload+metadata resident bytes and peak transient bytes; calibration/encode/setup cost plus end-to-end latency and throughput; exact pre-RoPE eviction/reuse compatibility; engine=fak-native and zero fallback",
		Papers: []string{
			"SPECTRA: Pushing the KV Cache Beyond the 2-Bit Cliff via Spectral Transform Coding (Zhang et al., 2026) arXiv:2608.07915v1",
			"Palu: Compressing KV-Cache with Low-Rank Projection (Chang et al., 2024) arXiv:2407.21118",
			"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024) arXiv:2402.02750",
			"The Minimax Risk of KV Cache Compression (Haverbeck et al., 2026) arXiv:2607.01520",
		},
		Note: "Borrow the transform and allocation math only after the existing Q4+recovery baseline is measured; analytical capacity, filler allocation, or a disconnected kernel microbenchmark is not a throughput witness, and no runtime/backend fallback is permitted.",
	},
	{
		Slug:        "metal-quant-gemm",
		Title:       "Metal Q4_K / Q6_K GEMM",
		FileGlobs:   []string{"internal/metalgemm/*", "internal/model/metal_q*k*.go", "internal/model/metal_prefill.go", "internal/model/*.metal", "internal/compute/metal.go"},
		FakPath:     "internal/metalgemm/; internal/model/metal_q4k*.go, metal_prefill.go; internal/compute/metal.go",
		SOTA:        "llama.cpp Metal / MLX",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBorrow,
		Oracle:      "cpuref (GEMV cosine 1.000000)",
		Note:        "Borrow the one-command-buffer resident-decode fusion; the per-token launch is the cost (epics #59/#67).",
	},
	{
		Slug:  "metal-qwen-gdn",
		Title: "Metal Qwen3.5 Gated-DeltaNet decode",
		FileGlobs: []string{
			"internal/model/qwen35.go",
			"internal/model/qwen35_gdn.go",
			"internal/metalgemm/qwen35_decode*",
		},
		FakPath:     "internal/model/qwen35.go (CPU GDN reference: linearAttnSeq / linearAttnStep)",
		SOTA:        "llama.cpp@ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0 Qwen3.5 graph semantics; MLX@43d2f06cb87e76895bf9a152bade4fee83408643 Metal CommandEncoder ownership; MLX-LM@cc8521569694a3240b52c98acffd100d59b4c755 GDN semantics",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0/src/models/qwen35.cpp",
		Route:       RouteBorrow,
		Oracle:      "internal/model/qwen35.go CPU reference: per-step output parity, convolution-state parity, and recurrent-state parity at predeclared tolerances; exact greedy-token equality over the accepted deterministic continuation; any fallback invalidates the witness",
		Papers: []string{
			"MLX Metal CommandEncoder ownership (43d2f06cb87e76895bf9a152bade4fee83408643): https://github.com/ml-explore/mlx/blob/43d2f06cb87e76895bf9a152bade4fee83408643/mlx/backend/metal/device.cpp",
			"MLX-LM Qwen3.5 GDN semantics (cc8521569694a3240b52c98acffd100d59b4c755): https://github.com/ml-explore/mlx-lm/blob/cc8521569694a3240b52c98acffd100d59b4c755/mlx_lm/models/qwen3_5.py",
		},
		Note: "Borrow the graph/state-order and encoder-ownership invariants from pinned llama.cpp ebd048f, MLX 43d2f06, and MLX-LM cc85215 only; keep execution fak-native, with no runtime or backend fallback.",
	},
	{
		Slug:        "batched-prefill-panel",
		Title:       "Batched prompt prefill panel and hybrid roofline stage attribution",
		FileGlobs:   []string{"internal/compute/prefill_batch.go", "internal/compute/prefill_attribution.go"},
		FakPath:     "internal/compute/prefill_batch.go; internal/compute/prefill_attribution.go",
		SOTA:        "vLLM / FlashAttention batched prefill panel and analytic FLOP/byte roofline stage attribution",
		PrimaryLink: "https://docs.vllm.ai",
		Route:       RouteStayMinimal,
		Oracle:      "TestBatchedPrefillMatchesSerialLoop (prefill_batch_test.go) + TestPrefillLongPromptStageAttribution (prefill_attribution_test.go): panel prefill matches serial per-token projections/RoPE/attention and pins O(P) vs O(P^2) stage scaling",
		Note:        "Pure-Go capability contract and analytic roofline attribution; backend-specific kernels hold their own rows.",
	},
	{
		Slug:        "cuda-deepgemm-wkv",
		Title:       "CUDA SM100 DeepGEMM FP8 WKV projection",
		FileGlobs:   []string{"internal/compute/cuda_deepgemm_wkv.cu", "internal/compute/cuda_deepgemm_wkv*.go", "internal/compute/cuda/blockscaled_wkv.cuh", "internal/compute/cuda/deepgemm_wkv_*.cuh", "internal/compute/cuda/mxfp8_cast.cuh"},
		FakPath:     "internal/compute/cuda_deepgemm_wkv.cu; internal/compute/cuda_deepgemm_wkv.go, cuda_deepgemm_wkv_witness.go, v41_fp8_wkv.go",
		SOTA:        "deepseek-ai/DeepGEMM@1fc9c66975ee322d628b3288534ca0e9ce12a64d SM100 FP8/FP4 1D1D GEMM + CUTLASS@f3fde58372d33e9a5650ba7b80fc48b3b49d40c8 (MIT/BSD-3)",
		PrimaryLink: "https://github.com/deepseek-ai/DeepGEMM/tree/1fc9c66975ee322d628b3288534ca0e9ce12a64d",
		Route:       RouteBind,
		Oracle:      "V41FP8WKVDeviceFacts (cuda_deepgemm_wkv_witness.go) + V41FP8WKVQualified (cuda_deepgemm_wkv.go) + packV41WKVScales (v41_fp8_wkv.go): physical SM100/SM103 image/SM-count facts and physical numerical parity against CPU F32/FP8 WKV reference",
		Note:        "Bind pinned DeepGEMM sm100_fp8_fp4_gemm_1d1d_impl inside a dedicated SM100/SM103 TU; no Torch, runtime JIT, or fallback.",
	},
	{
		Slug:        "cuda-dsv41-k-quant",
		Title:       "CUDA DSv4.1 MXFP8 K-plane quantize-insert and dequant-gather",
		FileGlobs:   []string{"internal/compute/dsv41_k_quant.cu", "internal/compute/dsv41_k_quant*.go", "internal/compute/cuda/dsv41_k_quant_vllm.cuh"},
		FakPath:     "internal/compute/dsv41_k_quant.cu; internal/compute/dsv41_k_quant.go, cuda/dsv41_k_quant_vllm.cuh",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 MXFP8 K-plane quantize-and-insert / dequantize-and-gather (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "dsv41_k_quant.go + dsv41_k_quant_test.go (internal/compute): 528-byte MXFP8 record (512 e4m3 + 16 ue8m0), 16-token paged slot addressing, and one-pass/per-block gather parity vs f32 K plane",
		Note:        "Borrow the 32-dim tile MXFP8 paged record encoding while retaining f32 K as default oracle and admission floor.",
	},
	{
		Slug:        "cuda-eplb-map-record",
		Title:       "CUDA EPLB logical-to-physical expert remap and hit recording",
		FileGlobs:   []string{"internal/compute/eplb_map_record.cu", "internal/compute/eplb_map_record*.go"},
		FakPath:     "internal/compute/eplb_map_record.cu; internal/compute/eplb_map_record.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 base_router.py _eplb_map_and_record_i32_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/vllm/model_executor/layers/fused_moe/router/base_router.py",
		Route:       RouteBorrow,
		Oracle:      "EplbMapRecord (eplb_map_record.go + eplb_map_record_test.go): Knuth multiplicative replica hash, atomic hit recording under record_enabled and num_unpadded_tokens, and ErrEplbLogicalIDOutOfRange boundary",
		Note:        "Borrow the replica-hash remap and hit counter kernel; host Go API keeps strict typed refusal on out-of-range logical ids.",
	},
	{
		Slug:        "cuda-flashmla-sparse-decode",
		Title:       "CUDA SM100 FlashMLA FP8 sparse MLA decode",
		FileGlobs:   []string{"internal/compute/cuda_flashmla_sparse_decode.cu", "internal/compute/cuda_flashmla_decode_qualification*.go", "internal/compute/testdata/flashmla_decode_device_qualification.cu"},
		FakPath:     "internal/compute/cuda_flashmla_sparse_decode.cu; internal/compute/cuda_flashmla_decode_qualification.go, testdata/flashmla_decode_device_qualification.cu",
		SOTA:        "deepseek-ai/FlashMLA@2e5429fc5653bab6e081f09477126f731882a6a9 SM100 FP8 sparse MLA decode + CUTLASS@4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b (MIT/BSD-3)",
		PrimaryLink: "https://github.com/deepseek-ai/FlashMLA/tree/2e5429fc5653bab6e081f09477126f731882a6a9",
		Route:       RouteBind,
		Oracle:      "TestFlashMLADecodeOwnedPhysical (cuda_flashmla_decode_qualification_test.go) + testdata/flashmla_decode_device_qualification.cu: sink-normalized BF16 output, sink-excluded LSE, +Inf empty-attention LSE sentinel, and split-vs-batch-invariant parity on SM100a/SM103a",
		Note:        "Qualification-only SM100a/SM103a TU binding third_party/flashmla and third_party/cutlass; no ATen/PyTorch runtime or generic sm120 admission.",
	},
	{
		Slug:        "cuda-flashmla-sparse-prefill",
		Title:       "CUDA SM100 FlashMLA sparse prefill fused attention",
		FileGlobs:   []string{"internal/compute/cuda_flashmla_sparse_prefill.cu", "internal/compute/cuda_flashmla_sparse_prefill*.go"},
		FakPath:     "internal/compute/cuda_flashmla_sparse_prefill.cu; internal/compute/cuda_flashmla_sparse_prefill.go, v41_sparse_prefill.go",
		SOTA:        "deepseek-ai/FlashMLA@2e5429fc5653bab6e081f09477126f731882a6a9 SM100 sparse prefill fused_norm_rope_attn_rope_cast_fwd + CUTLASS@4faf1a1568cf1e4ad8ff71846a13e16f2a6a6f6b (MIT/BSD-3)",
		PrimaryLink: "https://github.com/deepseek-ai/FlashMLA/tree/2e5429fc5653bab6e081f09477126f731882a6a9",
		Route:       RouteBind,
		Oracle:      "TestCUDAFlashMLAPhysicalFixture (cuda_flashmla_sparse_prefill_physical_test.go) + fak_flashmla_prefill_kernel_info: physical SM100a/SM103a manifest parity over canonical output values, max_logits, and LSE with integer IEEE FP8 unpack",
		Note:        "Qualification-only SM100a/SM103a TU; never treated as available from build success alone and no host fallback.",
	},
	{
		Slug:        "cuda-grammar-bitmask",
		Title:       "CUDA structured-output grammar bitmask logit mask",
		FileGlobs:   []string{"internal/compute/grammar_bitmask.cu", "internal/compute/grammar_bitmask*.go"},
		FakPath:     "internal/compute/grammar_bitmask.cu; internal/compute/grammar_bitmask.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 structured_outputs.py _apply_grammar_bitmask_kernel (Apache-2.0); mlc-ai/xgrammar apply_token_bitmask_inplace_triton",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/vllm/v1/worker/gpu/structured_outputs.py",
		Route:       RouteBorrow,
		Oracle:      "PackGrammarBitmask + ApplyGrammarBitmask (grammar_bitmask.go + grammar_bitmask_test.go) + internal/agent maskLogits: packed 32-token words, tail-word guard, and exact -inf (0xff800000) placement",
		Note:        "Borrow the packed 32-token bitmask application shape while host PackGrammarBitmask owns forced-stop collapse and row-index decomposition.",
	},
	{
		Slug:        "cuda-kv-swap-batch",
		Title:       "CUDA persistent-CTA batched KV block swap",
		FileGlobs:   []string{"internal/compute/kv_swap_batch.cu", "internal/compute/kv_swap_batch*.go"},
		FakPath:     "internal/compute/kv_swap_batch.cu; internal/compute/kv_swap_batch.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 swap_blocks_triton.py _swap_blocks_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/vllm/v1/kv_offload/cpu/swap_blocks_triton.py",
		Route:       RouteBorrow,
		Oracle:      "kv_swap_batch.go + kv_swap_batch_test.go (internal/compute): byte-identical copy including size%8 tails, overlap refusal, 8-byte alignment/UVA checks, and DMA threshold routing",
		Note:        "Borrow persistent-CTA job-stride block copying with strict UVA/alignment admission and safe fallback to fcuda_kv_swap_dma.",
	},
	{
		Slug:        "cuda-logprob-topk",
		Title:       "CUDA per-row top-k log-softmax receipt",
		FileGlobs:   []string{"internal/compute/logprob_topk.cu", "internal/compute/logprob_topk*.go"},
		FakPath:     "internal/compute/logprob_topk.cu; internal/compute/logprob_topk.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 logprob.py _topk_log_softmax_kernel / _ranks_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/vllm/v1/worker/gpu/sample/logprob.py",
		Route:       RouteBorrow,
		Oracle:      "logprob_topk.go + logprob_topk_test.go + agent.nativeTopLogprobs (internal/agent/native_decode_logprobs.go): fp32 1024-block streamed max+LSE, gather-only logprobs, tie-worst chosen rank, and any_non_finite refusal",
		Note:        "Borrow block-streamed row log-softmax without materializing [rows,vocab]; float64 host oracle remains accuracy reference and fallback.",
	},
	{
		Slug:        "cuda-moe-align-sort",
		Title:       "CUDA MoE block-aligned token sort and weighted sum",
		FileGlobs:   []string{"internal/compute/moe_align_sort.cu", "internal/compute/moe_align_sort*.go"},
		FakPath:     "internal/compute/moe_align_sort.cu; internal/compute/moe_align_sort.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 MoE align_block_size and top-k weighted sum finalize (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "MoeAlignBlockSizes + MoeSumTopK (moe_align_sort.go + moe_align_sort_test.go): padded cumsum offsets, multiset equality per expert region, and inline PTX mul.rn.f32/add.rn.f32 bit-identical weighted sum",
		Note:        "Borrow block-aligned token permutation without CUB dependency; keep PTX .rn scalar finalize for bit-identical oracle parity.",
	},
	{
		Slug:        "cuda-moe-router-gemm",
		Title:       "CUDA batched FP32 MoE router gate GEMM",
		FileGlobs:   []string{"internal/compute/moe_router_gemm.cu", "internal/compute/moe_router_gemm*.go"},
		FakPath:     "internal/compute/moe_router_gemm.cu; internal/compute/moe_router_gemm.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 csrc/fp32_router_gemm.cu fp32_router_gemm_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/csrc/fp32_router_gemm.cu",
		Route:       RouteBorrow,
		Oracle:      "MoeRouterGemmFp32 (moe_router_gemm.go + moe_router_gemm_test.go): every element within MoeRouterGemmParityMaxULPs (64 ULPs) of Go oracle across M<=48, experts<=512, hidden=5120",
		Note:        "Borrow float4 K-sliced warp butterfly reduction for chunked-prefill batches; decode stays on V41RouterGateTopKBackend.",
	},
	{
		Slug:        "cuda-rmsnorm-fp8-quant",
		Title:       "CUDA fused RMSNorm and static-scale FP8 E4M3 quantization",
		FileGlobs:   []string{"internal/compute/rmsnorm_fp8_quant.cu", "internal/compute/rmsnorm_fp8_quant*.go"},
		FakPath:     "internal/compute/rmsnorm_fp8_quant.cu; internal/compute/rmsnorm_fp8_quant.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 csrc/layernorm_quant_kernels.cu rms_norm_static_fp8_quant_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/csrc/layernorm_quant_kernels.cu",
		Route:       RouteBorrow,
		Oracle:      "RmsNormFp8Quant (rmsnorm_fp8_quant.go + rmsnorm_fp8_quant_test.go): bit-identical OCP e4m3fn RNE codes via fixed sequential sumsq, 1/sqrt, and __f*_rn intrinsics",
		Note:        "Borrow one-pass RMSNorm+FP8 quantization shape with sequential index-order sumsq and OCP e4m3fn grid encoder.",
	},
	{
		Slug:        "cuda-routed-expert-gemv",
		Title:       "CUDA routed-expert FP8 block-32 and MXFP4 E8M0 GEMV",
		FileGlobs:   []string{"internal/compute/routed_expert_gemv.cu", "internal/compute/cuda_routed_expert_gemv.go", "internal/compute/fp8_block32.go"},
		FakPath:     "internal/compute/routed_expert_gemv.cu; internal/compute/cuda_routed_expert_gemv.go, fp8_block32.go",
		SOTA:        "OCP Microscaling (MXFP4 E2M1 / FP8 block-32 E8M0) + deepseek-ai/DeepGEMM block-scaled FP8",
		PrimaryLink: "https://github.com/deepseek-ai/DeepGEMM",
		Route:       RouteStayMinimal,
		Oracle:      "GemvFP8Block32E8M0 + GemvMXFP4E8M0 (fp8_block32.go + fp8_block32_test.go + cuda_routed_expert_gemv_test.go) + TestRoutedExpertGemvSourceIsFusionFree (routed_expert_gemv_source_test.go): bit-identical inline PTX mul.rn.f32/add.rn.f32 GEMV",
		Note:        "Portable SIMT packed byte-in/byte-out GEMV with zero FMA/FTZ; keeps bit-identity with host oracles across sm_80..sm_120.",
	},
	{
		Slug:        "cuda-sampling-topk",
		Title:       "CUDA wide-vocabulary top-k and nucleus sampling truncation",
		FileGlobs:   []string{"internal/compute/sampling_topk.cu", "internal/compute/sampling_topk*.go", "internal/compute/cuda/sampling_candidates.cuh"},
		FakPath:     "internal/compute/sampling_topk.cu; internal/compute/sampling_topk.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 csrc/persistent_topk.cuh / csrc/sampled_topk.cuh / csrc/sampler.cu (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/csrc/persistent_topk.cuh",
		Route:       RouteBorrow,
		Oracle:      "sampling_topk_test.go + internal/agent/inkernel_sampling.go (applyCPUSamplingTruncation): descending-prob ascending-index order, bit-equal surviving mass, and non-finite/zero-mass refusal",
		Note:        "Unwired approximate float32 fragment; served sampling stays on host float64/53-bit seeded chain until exact parity is established.",
	},
	{
		Slug:        "cuda-silu-mul-fp8-quant",
		Title:       "CUDA fused SiLU-mul and group-128 FP8 E4M3 quantization",
		FileGlobs:   []string{"internal/compute/silu_mul_fp8_quant.cu", "internal/compute/silu_mul_fp8_quant*.go"},
		FakPath:     "internal/compute/silu_mul_fp8_quant.cu; internal/compute/silu_mul_fp8_quant.go",
		SOTA:        "vllm-project/vllm@b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd csrc/libtorch_stable/quantization/activation_kernels.cu silu_mul_fp8_quant_deep_gemm_kernel (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd/csrc/libtorch_stable/quantization/activation_kernels.cu",
		Route:       RouteBorrow,
		Oracle:      "silu_mul_fp8_quant.go + silu_mul_fp8_quant_test.go (internal/compute): per-128-group amax-driven ue8m0 scale, OCP e4m3fn RNE payload, ragged tails, and fault status refusal",
		Note:        "Borrow group-128 SiLU-mul FP8 quantization shape while adding ragged-tail support and checked non-finite fault status.",
	},
	{
		Slug:        "cuda-sm10x-tmem-probe",
		Title:       "CUDA SM100/SM103 TMEM tcgen05 build-lane probe",
		FileGlobs:   []string{"internal/compute/cuda_sm10x_probe.cu"},
		FakPath:     "internal/compute/cuda_sm10x_probe.cu; internal/compute/cuda_sm10x_probe.h",
		SOTA:        "NVIDIA PTX ISA 9.0 tcgen05.alloc / tcgen05.dealloc / tcgen05.relinquish_alloc_permit (SM100a/SM103a TMEM witness)",
		PrimaryLink: "https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf",
		Route:       RouteStayMinimal,
		Oracle:      "fcuda_sm10x_probe (cuda_sm10x_probe.h) + lane_ptxas target-arch gate (testdata/lane_ptxas/sample.log): explicit SM100/SM103 enqueue witness only, never called at backend init",
		Note:        "Minimal build-lane PTX ISA 9.0 TMEM witness; refuses non-10.0/10.3 devices and active graph capture.",
	},
	{
		Slug:        "cuda-tree-verify-attention",
		Title:       "CUDA one-pass tree-masked speculative verification attention",
		FileGlobs:   []string{"internal/compute/tree_attention.cu", "internal/compute/tree_attention.go", "internal/compute/spec_verify.go", "internal/compute/cuda_spec_verify.go", "internal/compute/cuda/spec_graph.cuh"},
		FakPath:     "internal/compute/tree_attention.cu; internal/compute/tree_attention.go, spec_verify.go",
		SOTA:        "Medusa tree-masked speculative verification attention; FlashInfer masked attention",
		PrimaryLink: "https://docs.flashinfer.ai/",
		Route:       RouteStayMinimal,
		Oracle:      "specVerifyAttentionHost (spec_verify.go) + spec_verify_test.go + TestTreeVerifyAttentionCUDASourceContract: compact 32-row uint32 ancestor-mask online softmax within 1e-6",
		Papers:      []string{"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Cai et al., 2024) arXiv:2401.10774"},
		Note:        "Compact 32-row mask travels in CUDA launch-parameter space with zero global mask allocation; approximate f32 peer to host oracle.",
	},
	{
		Slug:        "cuda-v41-compressor-pool",
		Title:       "CUDA DeepSeek-V4.1 compressor gated token pool",
		FileGlobs:   []string{"internal/compute/cuda_v41_compressor_pool.cu", "internal/compute/cuda_v41_compressor_pool*.go", "internal/compute/cuda/v41_compressor*.cuh"},
		FakPath:     "internal/compute/cuda_v41_compressor_pool.cu; internal/compute/cuda_v41_compressor_pool.go, v41_compressor_pool.go",
		SOTA:        "DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277 compressor gated token pool; ported from fak shaders/v41_compressor_pool.comp (Apache-2.0)",
		PrimaryLink: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277",
		Route:       RouteStayMinimal,
		Oracle:      "TestCUDAV41CompressorPoolDeviceWitness (cuda_v41_compressor_pool_witness_test.go): cudaPoolWitnessReference binary32 serial softmax pool within 8 ulps of carried F64 magnitude",
		Note:        "Direct CUDA port of the checked Vulkan compressor pool; Cody-Waite exp and explicit __f*_rn rounding with no fast-math or FTZ.",
	},
	{
		Slug:        "cuda-v41-engram-gate",
		Title:       "CUDA DeepSeek-V4.1 Engram gate and mHC stream injection",
		FileGlobs:   []string{"internal/compute/cuda_v41_engram_gate.cu", "internal/compute/cuda_v41_engram_gate*.go"},
		FakPath:     "internal/compute/cuda_v41_engram_gate.cu; internal/compute/cuda_v41_engram_gate.go, v41_engram_gate.go",
		SOTA:        "DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277 Engram gate/add over mHC streams; ported from fak shaders/v41_engram_gate.comp (Apache-2.0)",
		PrimaryLink: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277",
		Route:       RouteStayMinimal,
		Oracle:      "cudaEngramWitnessRef (cuda_v41_engram_gate_witness_test.go) mirroring Model.v41EngramInjectPrepared: bit-exact host rsqrt/sqrt and within 1 BF16 ulp with checked status refusal",
		Note:        "Direct CUDA port preserving scalar serial reduction and checked device-to-host status boundary.",
	},
	{
		Slug:        "cuda-v41-indexer",
		Title:       "CUDA DeepSeek-V4.1 DSA ReLU-weighted indexer score",
		FileGlobs:   []string{"internal/compute/cuda_v41_indexer.cu", "internal/compute/cuda_v41_indexer*.go"},
		FakPath:     "internal/compute/cuda_v41_indexer.cu; internal/compute/cuda_v41_indexer.go, v41_indexer_score.go",
		SOTA:        "DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277 / FlashMLA DSA ReLU-weighted indexer score (MIT)",
		PrimaryLink: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277",
		Route:       RouteStayMinimal,
		Oracle:      "TestCUDAV41IndexerSIMTExact (cuda_v41_indexer_test.go) + V41IndexerScore (v41_indexer_score.go): separately rounded binary32 (--ftz=false) dimension/head order matching host oracle",
		Note:        "SIMT binary32 exact-order indexer candidate; no FMA or reassociation permitted.",
	},
	{
		Slug:        "cuda-v41-mhc",
		Title:       "CUDA DeepSeek-V4.1 mHC hyper-connection stream and Sinkhorn split",
		FileGlobs:   []string{"internal/compute/cuda_v41_mhc.cu", "internal/compute/cuda_v41_mhc*.go", "internal/compute/cuda/v41_checked_stream.cuh"},
		FakPath:     "internal/compute/cuda_v41_mhc.cu; internal/compute/cuda_v41_mhc.go, v41_mhc.go",
		SOTA:        "ggml-org/llama.cpp dsv4-hc.cu@3c809f44383bcebc318b8cddd8c6d11aaa861e9b DeepSeek-V4.1 mHC hyper-connection ops (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/tree/3c809f44383bcebc318b8cddd8c6d11aaa861e9b",
		Route:       RouteBorrow,
		Oracle:      "TestCUDAV41MHCCandidateWitness (cuda_v41_mhc_witness_test.go) + v41_mhc.go: initial/pre/post bit-identical to host reference; split/rmssplit within bounded ulps via checked readback",
		Note:        "Borrow mHC stream and Sinkhorn normalization ops under contiguous FAK ABI and checked quarantine.",
	},
	{
		Slug:        "cuda-v41-routed-mmid",
		Title:       "CUDA DeepSeek-V4.1 routed Q2_K/Q3_K expert id and fused SwiGLU/down GEMV",
		FileGlobs:   []string{"internal/compute/cuda_v41_routed_mmid.cu", "internal/compute/cuda_v41_routed_mmid*.go", "internal/compute/cuda_v41_mmid*.go"},
		FakPath:     "internal/compute/cuda_v41_routed_mmid.cu; internal/compute/cuda_v41_routed_mmid.go, v41_routed_fused.go",
		SOTA:        "ggml-org/llama.cpp@571d0d540df04f25298d0e159e520d9fc62ed121 mul_mat_vec_q Q2_K/Q3_K MMVQ ids (MIT); antirez/ds4@80ebbc396aee40eedc1d829222f3362d10fa4c6c clamped SwiGLU (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/tree/571d0d540df04f25298d0e159e520d9fc62ed121",
		Route:       RouteBorrow,
		Oracle:      "cuda_v41_routed_mmid_test.go (internal/compute): float64 host replay of q8_1 activation, Q2_K/Q3_K dequant, V4.1 clamp-SwiGLU, and ordered BF16 down sum within cudaQ2KCosineMin/cudaQ3KCosineMin",
		Note:        "Borrow llama.cpp Q2_K/Q3_K MMVQ ids with in-kernel slot/order bounds checks and checked BF16 fault records.",
	},
	{
		Slug:        "cuda-v41-routed-swiglu",
		Title:       "CUDA DeepSeek-V4.1 clamped routed SwiGLU",
		FileGlobs:   []string{"internal/compute/cuda_v41_routed_swiglu.cu", "internal/compute/cuda_v41_routed_swiglu*.go"},
		FakPath:     "internal/compute/cuda_v41_routed_swiglu.cu; internal/compute/cuda_v41_routed_swiglu.go, v41_routed_swiglu.go",
		SOTA:        "antirez/ds4@80ebbc396aee40eedc1d829222f3362d10fa4c6c clamped SwiGLU (MIT); ported from fak shaders/v41_routed_swiglu.comp",
		PrimaryLink: "https://github.com/antirez/ds4/tree/80ebbc396aee40eedc1d829222f3362d10fa4c6c",
		Route:       RouteBorrow,
		Oracle:      "cuda_v41_routed_swiglu_witness_test.go (cudaRoutedWitnessHostLane): BF16 publication within 2-ULP expf bound and checked 16-byte status fault record",
		Note:        "Shares routed_swiglu_lane with cuda_v41_routed_mmid.cu; preserves precise donor division and non-finite fault records.",
	},
	{
		Slug:        "cuda-v41-router",
		Title:       "CUDA DeepSeek-V4.1 warp top-k and token-hash MoE router",
		FileGlobs:   []string{"internal/compute/cuda_v41_router.cu", "internal/compute/cuda_v41_router*.go"},
		FakPath:     "internal/compute/cuda_v41_router.cu; internal/compute/cuda_v41_router.go, v41_router.go",
		SOTA:        "antirez/ds4@80ebbc396aee40eedc1d829222f3362d10fa4c6c warp top-k router (MIT); vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 topkGatingSoftplusSqrt/dsv4HashTopkSoftplusSqrt (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/blob/ae065918640a198a56f6f17eb45e251c3229fe41/csrc/moe/topk_softplus_sqrt_kernels.cu",
		Route:       RouteBorrow,
		Oracle:      "cuda_v41_router_test.go (requireV41RouterParity against v41_router.go within 4e-6 rel): stable log1p(exp(-|z|)) softplus-sqrt scoring, lower-expert tie-break, and four-word fault record",
		Note:        "Borrow warp-shuffle argmax and softplus-sqrt gating while replacing log(1+exp) underflow with log1p(exp(-|z|)).",
	},
	{
		Slug:        "cuda-v41-shared-attention",
		Title:       "CUDA DeepSeek-V4.1 SIMT split-KV shared-latent sink attention",
		FileGlobs:   []string{"internal/compute/cuda_v41_shared_attention.cu", "internal/compute/cuda_v41_shared_attention*.go"},
		FakPath:     "internal/compute/cuda_v41_shared_attention.cu; internal/compute/cuda_v41_shared_attention.go, v41_sparse_mla_decode.go",
		SOTA:        "vllm-project/vllm@b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd merge_attn_states LSE combine (Apache-2.0); FlashMLA split-KV base-2 online-softmax sink pattern",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd",
		Route:       RouteBorrow,
		Oracle:      "TestCUDAV41SharedAttentionMatchesFlashMLAReference (cuda_v41_shared_attention_test.go): matches v41SparseMLADecodeReference and v41SharedAttentionFloat64Oracle within 4e-6 at actual tile/split partition",
		Note:        "Portable SM80..SM120 SIMT FlashMLA-class split-KV online softmax and N-way exp2 LSE merge with sink folded last.",
	},
	{
		Slug:        "gguf-q3k-iq3s-dequant",
		Title:       "GGUF Q3_K and IQ3_S scalar row dequantizers",
		FileGlobs:   []string{"internal/model/quant_q3k_iq3s.go"},
		FakPath:     "internal/model/quant_q3k_iq3s.go",
		SOTA:        "ggml-org/llama.cpp@6fe74980162af0ed5e559870d5deccafaa034e7c Q3_K and IQ3_S row dequantizers (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/tree/6fe74980162af0ed5e559870d5deccafaa034e7c",
		Route:       RouteBind,
		Oracle:      "TestQ3KIQ3SNativeReference (quant_q3k_iq3s_test.go): SHA-256 bit-exact match against llama.cpp@6fe7498 C row dequantizers",
		Note:        "Bind load-time expansion and resident native math to the exact ggml Q3_K / IQ3_S 110-byte super-block layout.",
	},
	{
		Slug:        "gguf-quant-registry-admission",
		Title:       "GGUF quant descriptor registry, load-time capability census, and source retention plan",
		FileGlobs:   []string{"internal/model/quant_registry.go", "internal/model/quant_capability_load.go", "internal/model/quant_storage_plan.go"},
		FakPath:     "internal/model/quant_registry.go; internal/model/quant_capability_load.go, quant_storage_plan.go",
		SOTA:        "llama.cpp GGML ggml_type_traits / GGUF quant block descriptors and fail-closed backend capability gates",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteStayMinimal,
		Oracle:      "TestQuantRegistryBuiltinRegistration (quant_registry_test.go) + TestQuantCapabilityRefusalNamesEveryUnsupportedType (quant_capability_load_test.go): exhaustive kQuantKind descriptors and fail-closed backend refusal",
		Note:        "Metadata and fail-closed admission seam over retained GGUF block formats; prevents silent unbounded CPU dequant fallback.",
	},
	{
		Slug:        "gguf-row-normalized-kquant",
		Title:       "GGUF native row-normalized K-quant residency for Qwen3.5 hybrid projections",
		FileGlobs:   []string{"internal/model/quant_row_normalized.go"},
		FakPath:     "internal/model/quant_row_normalized.go",
		SOTA:        "ggml-org/llama.cpp GGUF row-wise K-quant super-blocks (Q4_K/Q5_K/Q6_K) for output-row-permuted hybrid projections",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteBind,
		Oracle:      "quant_row_normalized_test.go (internal/model): Qwen3.5 row-permuted projections served from native q4kw/kqw stores match Q8 store logits on batched prefill and per-token decode",
		Note:        "Permute whole 256-weight K-quant output rows into canonical model order to avoid Q8 re-quantization when Metal exact-Q8 is off.",
	},
	{
		Slug:        "gguf-ternary-q2-prefill",
		Title:       "GGUF ternary Q2_0 G128 and Q2 G32 batched prefill tile GEMM",
		FileGlobs:   []string{"internal/model/quant_q2_prefill.go"},
		FakPath:     "internal/model/quant_q2_prefill.go",
		SOTA:        "llama.cpp / ggml block-quant dequant reuse across prefill token tiles",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c",
		Route:       RouteBind,
		Oracle:      "TestQ2PrefillBlockReuseMatchesTokenRows (quant_q2_prefill_test.go): bit-exact float32 output vs per-token q2MatRowsRange for both g128 and g32",
		Note:        "Reuse immutable block dequantization across 16-row prefill tiles while preserving exact per-token accumulator fold.",
	},
	{
		Slug:        "host-q8-packed-kv-attention",
		Title:       "Host Q8_0 packed KV decode attention with bounded head-tile reconstruction",
		FileGlobs:   []string{"internal/model/quant_attn_q8_packed*.go"},
		FakPath:     "internal/model/quant_attn_q8_packed.go; internal/model/quant_attn_q8_packed_default.go, quant_attn_q8_packed_qualification.go",
		SOTA:        "ggml-org/llama.cpp@83078fec0db82d6b5a00d9599062c38c39145755 Q8_0 row dequantizer (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp/tree/83078fec0db82d6b5a00d9599062c38c39145755",
		Route:       RouteBorrow,
		Oracle:      "quant_attn_q8_packed_test.go (q8PackedReference): O(workers*headDim) tile reconstruction matches materialized K/V attention scores and outputs within tolerance",
		Note:        "Reconstruct one K/V head row per worker into bounded F32 tile scratch for GQA dot/saxpy reuse without whole-layer expansion.",
	},
	{
		Slug:        "computetune-kernel-selection",
		Title:       "Compatibility-bound kernel autotuning and storage-compute arbitration",
		FileGlobs:   []string{"internal/computetune/*.go"},
		FakPath:     "internal/computetune/tune.go; internal/computetune/matmul.go, arbitrate.go, variants.go",
		SOTA:        "Triton autotune / TVM MetaSchedule compatibility-bound kernel selection and storage-compute arbitration",
		PrimaryLink: "https://triton-lang.org/main/python-api/generated/triton.autotune.html",
		Route:       RouteStayMinimal,
		Oracle:      "internal/computetune/tune_test.go + matmul_test.go + arbitrate_test.go: fail-closed manifest compatibility, reference-parity admission before timing, and deterministic storage-compute arbitration",
		Note:        "Stay minimal: deterministic fail-closed manifest selection and offline arbitration; never admit unverified candidate timings.",
	},
	{
		Slug:        "cuda-inkling2-moe-decode",
		Title:       "CUDA inkling2 single-pass MoE decode, prefill, and NVFP4 lanes",
		FileGlobs:   []string{"internal/compute/cuda/inkling2_*.cuh"},
		FakPath:     "internal/compute/cuda/inkling2_loop.cuh; internal/compute/cuda/inkling2_prefill.cuh, inkling2_expert_gemv.cuh",
		SOTA:        "realugbun/llama.cpp@2eb2530ec4 ggml/src/ggml-cuda/inkling2.cu single-pass MIX + llama.cpp MMQ/NVFP4 (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBorrow,
		Oracle:      "cpuref / llama.cpp greedy oracle and per-stage parity across inkling2 decode, prefill, MoE, and NVFP4 lanes",
		Note:        "Borrow the single-pass inkling2 MIX and MMQ/NVFP4 tile schemes; keep fak-native execution and checked fault boundaries.",
	},
	{
		Slug:        "cuda-kv-fp8-storage",
		Title:       "CUDA FP8 E4M3 KV-cache storage, append, and lifecycle",
		FileGlobs:   []string{"internal/compute/cuda/kv_fp8*.cuh"},
		FakPath:     "internal/compute/cuda/kv_fp8.cuh; internal/compute/cuda/kv_fp8_append.cuh, kv_fp8_lifecycle.cuh",
		SOTA:        "vllm-project/vllm FP8 E4M3 KV-cache storage and dynamic scaling (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm",
		Route:       RouteBorrow,
		Oracle:      "F32 shadow KV parity vs FP8 E4M3 attended planes with zero unowned host round-trip",
		Note:        "Borrow vLLM FP8 KV storage and append layout while keeping F32 shadow parity oracle.",
	},
	{
		Slug:        "cuda-mega-situ-moe",
		Title:       "CUDA SM90 MegaMoE FP4, situ MoE, and warp NVFP4 GEMV",
		FileGlobs:   []string{"internal/compute/cuda/mega_moe_fp4.cuh", "internal/compute/cuda/situ_moe.cuh", "internal/compute/cuda/nvfp4_warp_gemv.cuh"},
		FakPath:     "internal/compute/cuda/mega_moe_fp4.cuh; internal/compute/cuda/situ_moe.cuh, nvfp4_warp_gemv.cuh",
		SOTA:        "flashinfer-ai/flashinfer PR #7961 SM90 pull-style FP4 MegaMoE; llama.cpp greedy oracle (Apache-2.0/MIT)",
		PrimaryLink: "https://docs.flashinfer.ai/",
		Route:       RouteBorrow,
		Oracle:      "k_nvfp4_w4a16_f32 reference and llama.cpp greedy parity with bounded K128 stage-ordering invariants",
		Note:        "Borrow SM90 pull-style MegaMoE stage ordering and warp NVFP4 decode; hold to reference GEMV and greedy parity.",
	},
	{
		Slug:        "cuda-minference-vertical-slash",
		Title:       "CUDA MInference vertical-slash sparse attention index",
		FileGlobs:   []string{"internal/compute/cuda/minference_vertical_slash_index.cuh"},
		FakPath:     "internal/compute/cuda/minference_vertical_slash_index.cuh",
		SOTA:        "microsoft/MInference csrc/vertical_slash_index.cu dynamic sparse attention index (MIT)",
		PrimaryLink: "https://github.com/microsoft/MInference",
		Route:       RouteBorrow,
		Oracle:      "exact vertical/slash index selection parity against host sparse-index reference",
		Note:        "Borrow MInference vertical-slash index kernel for long-context sparse prefill.",
	},
	{
		Slug:        "cuda-runtime-graph-staging",
		Title:       "CUDA runtime memory pool, graph capture, async upload, and peer handoff",
		FileGlobs:   []string{"internal/compute/cuda/async_upload.cuh", "internal/compute/cuda/graph_capture_once.cuh", "internal/compute/cuda/kernel_profile.cuh", "internal/compute/cuda/model_candidates.cuh", "internal/compute/cuda/peer_handoff.cuh", "internal/compute/cuda/prefill_input_owner.cuh", "internal/compute/cuda/runtime_memory.cuh", "internal/compute/cuda/runtime_services.cuh"},
		FakPath:     "internal/compute/cuda/graph_capture_once.cuh; internal/compute/cuda/async_upload.cuh, runtime_memory.cuh, runtime_services.cuh",
		SOTA:        "vllm-project/vllm offloader/prefetch.py (Apache-2.0); FlashML-org/FreeToken@de77d16a5ac36494d85964f846b50ad36f7acc21 ple_hash.py (Apache-2.0); NVIDIA CUDA Graphs",
		PrimaryLink: "https://github.com/vllm-project/vllm",
		Route:       RouteStayMinimal,
		Oracle:      "bounded pool cap, strict allocation ownership, CUDA graph capture replay parity, and peer-handoff integrity",
		Note:        "Stay minimal on runtime memory/stream/graph ownership; borrow vLLM async prefetch and FreeToken PLE hash where needed.",
	},
	{
		Slug:        "cuda-v41-embed-gather",
		Title:       "CUDA DeepSeek-V4.1 32-lane selected-row embedding gather",
		FileGlobs:   []string{"internal/compute/cuda/v41_embed_gather.cuh"},
		FakPath:     "internal/compute/cuda/v41_embed_gather.cuh",
		SOTA:        "ggml-org/llama.cpp 32-lane selected-row embedding gather (MIT)",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBorrow,
		Oracle:      "bit-exact selected-row embedding gather against host GGUF/IQ4_NL row decode",
		Note:        "Borrow llama.cpp 32-lane selected-row gather mapping for DeepSeek-V4.1 engram and embedding rows.",
	},
	{
		Slug:        "cuda-vllm-paged-mla-attention",
		Title:       "CUDA vLLM paged attention and MLA cache insert",
		FileGlobs:   []string{"internal/compute/cuda/paged_attention_vllm.cuh", "internal/compute/cuda/mla_cache_insert_vllm.cuh"},
		FakPath:     "internal/compute/cuda/paged_attention_vllm.cuh; internal/compute/cuda/mla_cache_insert_vllm.cuh",
		SOTA:        "vllm-project/vllm PagedAttention v1 and MLA cache insert kernels (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm",
		Route:       RouteBorrow,
		Oracle:      "f32 reference attention and KV cache insert parity across page boundaries in fak_vllm_pa / fak_vllm_mla",
		Note:        "Borrow vLLM paged attention and MLA cache insert under fak_vllm_pa / fak_vllm_mla namespaces.",
	},
	{
		Slug:        "grammar-constrained-decoding",
		Title:       "GBNF, JSON-schema, and vocab-trie constrained decoding",
		FileGlobs:   []string{"internal/grammar/*.go"},
		FakPath:     "internal/grammar/grammar.go; internal/grammar/gbnf_parse.go, vocab_trie.go, batch_bitmask_fsm.go, eos_batch_terminal.go",
		SOTA:        "ggml-org/llama.cpp@cd8427d018c07eefab6c110f7f115c31ec4c6157 GBNF (MIT); mlc-ai/xgrammar subtree-reject vocab trie (Apache-2.0); vllm-project/vllm@4593e73c85440e81a10565b4017a9ced421f2208 backend_guidance",
		PrimaryLink: "https://github.com/mlc-ai/xgrammar",
		Route:       RouteBorrow,
		Oracle:      "internal/grammar/*_test.go: GBNF parse/mask parity, JSON schema cursor/mask invariants, vocab trie subtree-reject exactness, and EOS batch terminal membership",
		Note:        "Borrow GBNF lowering, xgrammar vocab-trie subtree rejection, and vLLM/SGLang batch bitmask FSM contracts.",
	},
	{
		Slug:        "toolgrammar-jumpforward",
		Title:       "Discriminated-union EBNF tool grammar and PrefixVocab jump-forward",
		FileGlobs:   []string{"internal/toolgrammar/*.go"},
		FakPath:     "internal/toolgrammar/toolgrammar.go; internal/toolgrammar/jumpforward.go",
		SOTA:        "otheru-ai/ember src/model/tool_grammar.c discriminated-union EBNF; SGLang jump-forward compressed FSM decoding",
		PrimaryLink: "https://github.com/sgl-project/sglang",
		Route:       RouteBorrow,
		Oracle:      "internal/toolgrammar/toolgrammar_test.go + jumpforward_test.go: DSML/XML/JSON EBNF compilation, byte-level BPE space protection, and exact PrefixVocab longest-prefix jump-forward",
		Note:        "Clean-room EBNF tool grammar compilation and zero-allocation PrefixVocab jump-forward token forcing.",
	},
	{
		Slug:        "cuda-alpha-moe-fused",
		Title:       "CUDA W8A8 MoE fused up, activation, down, and weighted accumulate",
		FileGlobs:   []string{"internal/compute/cuda/alpha_moe_fused.cuh"},
		FakPath:     "internal/compute/cuda/alpha_moe_fused.cuh; internal/compute/alpha_moe_fused_test.go",
		SOTA:        "Aleph-Alpha/Alpha-MoE@0fbed62bdda3702fd5d4b8c497e68746f0304c3a fused_moe_w8a8_up_down_acc.cu (Apache-2.0)",
		PrimaryLink: "https://github.com/Aleph-Alpha/Alpha-MoE/tree/0fbed62bdda3702fd5d4b8c497e68746f0304c3a",
		Route:       RouteBorrow,
		Oracle:      "TestAlphaMoeFused (internal/compute/alpha_moe_fused_test.go): W8A8 up-act-down-weighted-accumulate parity in registers/SMEM with zero global intermediate writes",
		Note:        "Structure-only header contract and in-memory oracle; gated under FAK_CUDA_ALPHA_MOE_FUSED with no device claim.",
	},
	{
		Slug:        "cuda-dsa-indexer-k-quant",
		Title:       "CUDA DSA lightning indexer symmetric INT8 key cache helpers",
		FileGlobs:   []string{"internal/compute/cuda/indexer_k_quant_vllm.cuh"},
		FakPath:     "internal/compute/cuda/indexer_k_quant_vllm.cuh; internal/compute/dsa_index_test.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 symmetric INT8 indexer K quantization and paged write/gather helpers (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "hostIndexSelectF64 + dsa_index_test.go (internal/compute): symmetric INT8 indexer key write-map validation and paged gather parity",
		Note:        "Borrow vLLM symmetric INT8 indexer key write/gather layout under fak_vllm_indexer with strict page-map bounds.",
	},
	{
		Slug:        "cuda-fp8-gemm-allreduce",
		Title:       "CUDA SM90 fused FP8 rowwise GEMM and P2P all-reduce contract",
		FileGlobs:   []string{"internal/compute/cuda/fp8_gemm_allreduce.cuh"},
		FakPath:     "internal/compute/cuda/fp8_gemm_allreduce.cuh; internal/compute/fp8_gemm_allreduce_test.go",
		SOTA:        "flashinfer-ai/flashinfer@139888400ff008a4136ce6b35e2f7320bb0c0f40 cake_fp8_rowwise_gemm_allreduce (Apache-2.0)",
		PrimaryLink: "https://github.com/flashinfer-ai/flashinfer/tree/139888400ff008a4136ce6b35e2f7320bb0c0f40",
		Route:       RouteBorrow,
		Oracle:      "TestFP8GemmAllreduce (internal/compute/fp8_gemm_allreduce_test.go): two-peer SM90 m64n128k32 e4m3 5-stage SMEM pipeline and atomic-flag P2P partial publish",
		Note:        "Structure-only header contract and two-peer in-memory oracle; no NVLink hardware run or speed claim.",
	},
	{
		Slug:        "cuda-inkling2-nvfp4-w4a4-gemm",
		Title:       "CUDA Blackwell sm_120f NVFP4 W4A4 Tensor Core GEMM",
		FileGlobs:   []string{"internal/compute/cuda/nvfp4_w4a4_gemm.cuh", "internal/compute/cuda_nvfp4_gemm*.go"},
		FakPath:     "internal/compute/cuda/nvfp4_w4a4_gemm.cuh; internal/compute/cuda_nvfp4_gemm.go",
		SOTA:        "NVIDIA PTX ISA 9.0 mma.sync.aligned.m16n8k64 block-scaled mxf4nvf4 E2M1/UE4M3FN Tensor Core MMA (sm_120f)",
		PrimaryLink: "https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf",
		Route:       RouteStayMinimal,
		Oracle:      "TestCUDAInk2NVFP4W4A4GEMM (internal/compute/cuda_nvfp4_gemm_test.go): block-16 dynamic row-quantized E2M1/UE4M3FN GEMM parity for T>=16 prefill and K>=4 speculative verify",
		Note:        "Fak-native sm_120f Inkling-2 prefill/verify GEMM gated behind FAK_INKLING2_W4A4_GEMM=1; single-token decode remains separate.",
	},
	{
		Slug:        "cuda-mamba-selective-scan-prefill",
		Title:       "CUDA Mamba selective-scan chunked prefix prefill contract",
		FileGlobs:   []string{"internal/compute/cuda/selective_scan_prefill.cuh", "internal/compute/selective_scan_prefill*.go"},
		FakPath:     "internal/compute/cuda/selective_scan_prefill.cuh; internal/compute/selective_scan_prefill_test.go",
		SOTA:        "state-spaces/mamba@e9594ce1c732 csrc/selective_scan/selective_scan_fwd_kernel.cuh chunked associative scan (Apache-2.0)",
		PrimaryLink: "https://github.com/state-spaces/mamba",
		Route:       RouteBorrow,
		Oracle:      "TestSelectiveScanPrefill (internal/compute/selective_scan_prefill_test.go): chunked (a,b) associative scan with running-prefix carry vs sequential recurrence",
		Note:        "Structure-only header contract and sequential-recurrence host oracle; no hardware run or speed claim.",
	},
	{
		Slug:        "cuda-megakernel-nvfp4-decode",
		Title:       "CUDA Blackwell NVFP4 persistent decode megakernel contract",
		FileGlobs:   []string{"internal/compute/cuda/megakernel_nvfp4_decode.cuh"},
		FakPath:     "internal/compute/cuda/megakernel_nvfp4_decode.cuh; internal/compute/megakernel_nvfp4_decode_test.go",
		SOTA:        "Luce-Org/lucebox-hub@f3cc3ec9c56810c724981e8ae5114539e9500b8a kernel_gb10_nvfp4.cu persistent decode (Apache-2.0)",
		PrimaryLink: "https://github.com/Luce-Org/lucebox-hub/tree/f3cc3ec9c56810c724981e8ae5114539e9500b8a",
		Route:       RouteBorrow,
		Oracle:      "TestMegakernelNVFP4Decode (internal/compute/megakernel_nvfp4_decode_test.go): E2M1 4-bit lookup dequant and persistent sm_120/sm_121a header constants",
		Note:        "Structure-only contract under FAK_CUDA_MEGAKERNEL_NVFP4_DECODE; no device run or loader change.",
	},
	{
		Slug:        "cuda-megakernel-sm120-decode",
		Title:       "CUDA SM120 persistent decode flag-in-data handoff contract",
		FileGlobs:   []string{"internal/compute/cuda/megakernel_sm120_decode.cuh"},
		FakPath:     "internal/compute/cuda/megakernel_sm120_decode.cuh; internal/compute/megakernel_sm120_decode_test.go",
		SOTA:        "SamuelReeder/megakernels@10bd202c2cdceaf0703ea3bbb88ad03eaa308766 qwen3.5-0.8b persistent megakernel (MIT)",
		PrimaryLink: "https://github.com/SamuelReeder/megakernels/tree/10bd202c2cdceaf0703ea3bbb88ad03eaa308766",
		Route:       RouteBorrow,
		Oracle:      "TestMegakernelSM120Decode (internal/compute/megakernel_sm120_decode_test.go): flag-in-data producer ring handoff and fused int8 weight dot-dequant oracle",
		Note:        "Structure-only contract under FAK_CUDA_MEGAKERNEL_SM120_DECODE; no SM120 tuning or speed claim.",
	},
	{
		Slug:        "cuda-megakernel-taskgraph",
		Title:       "CUDA host-built decode megakernel task-graph ABI contract",
		FileGlobs:   []string{"internal/compute/cuda/megakernel_taskgraph.cuh"},
		FakPath:     "internal/compute/cuda/megakernel_taskgraph.cuh; internal/compute/megakernel_taskgraph_test.go",
		SOTA:        "cohere-ai/cohere-megakernel@67d0b9ca22ea3652796b715d1d1863459e0e2c3c src/decode/abi.h & megakernel.cuh (Apache-2.0)",
		PrimaryLink: "https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c",
		Route:       RouteBorrow,
		Oracle:      "TestMegakernelTaskgraph (internal/compute/megakernel_taskgraph_test.go): fixed-width int32 descriptor layout, dependency counters, and 8/1/1/1 warp-role split",
		Note:        "Structure-only ABI contract under FAK_CUDA_MEGAKERNEL_TASKGRAPH; no kernel declaration or device claim.",
	},
	{
		Slug:        "cuda-minference-online-sparse-index",
		Title:       "CUDA MInference online diagonal-sum sparse index approximation",
		FileGlobs:   []string{"internal/compute/cuda/online_sparse_index.cuh"},
		FakPath:     "internal/compute/cuda/online_sparse_index.cuh; internal/compute/online_sparse_index_test.go",
		SOTA:        "microsoft/MInference@29ef1974fcb5a1440ccf92d7d7996900230fddcf vertical_slash_utils.py online calc_index (MIT)",
		PrimaryLink: "https://github.com/microsoft/MInference/tree/29ef1974fcb5a1440ccf92d7d7996900230fddcf",
		Route:       RouteBorrow,
		Oracle:      "TestOnlineSparseIndexApprox (internal/compute/online_sparse_index_test.go): last-q block vertical and slash top-k approximation without full recompute",
		Note:        "Structure-only header contract and host sparse-index oracle; no hardware run or speed claim.",
	},
	{
		Slug:        "cuda-mkernel-collective",
		Title:       "CUDA persistent fused collective-compute phase contract",
		FileGlobs:   []string{"internal/compute/cuda/mkernel_collective.cuh"},
		FakPath:     "internal/compute/cuda/mkernel_collective.cuh; internal/compute/mkernel_collective_test.go",
		SOTA:        "uccl-project/mKernel@ef6225fdde493f4119f519c99bfa4ad9234e323b gemm_ar & ag_gemm fused collectives (MIT)",
		PrimaryLink: "https://github.com/uccl-project/mKernel/tree/ef6225fdde493f4119f519c99bfa4ad9234e323b",
		Route:       RouteBorrow,
		Oracle:      "TestMKernelCollective (internal/compute/mkernel_collective_test.go): two-rank session phase progression for fused GEMM+all-reduce, all-gather, and reduce-scatter",
		Note:        "Structure-only phase contract under FAK_CUDA_MKERNEL_COLLECTIVE; no multi-GPU or NVLink run claim.",
	},
	{
		Slug:        "cuda-moe-grouped-topk",
		Title:       "CUDA group-constrained noaux_tc MoE top-k routing",
		FileGlobs:   []string{"internal/compute/moe_grouped_topk.cu", "internal/compute/moe_grouped_topk*.go"},
		FakPath:     "internal/compute/moe_grouped_topk.cu; internal/compute/moe_grouped_topk.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 csrc/moe/grouped_topk_kernels.cu (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "TestGroupedTopKMatchesOracle + TestGroupedTopKCUDASourceContract (internal/compute/moe_grouped_topk_test.go): bit-exact grouped routing vs glm5next_forward and host oracle",
		Note:        "Unwired standalone fragment re-expressed for fak flat C ABI; gated on qualification before libfakcuda integration.",
	},
	{
		Slug:        "cuda-moe-topk-gating",
		Title:       "CUDA general softmax/sigmoid MoE top-k gating",
		FileGlobs:   []string{"internal/compute/moe_topk_gating.cu", "internal/compute/moe_topk_gating*.go"},
		FakPath:     "internal/compute/moe_topk_gating.cu; internal/compute/moe_topk_gating.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 csrc/moe/topk_softmax_kernels.cu (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "TestMoeTopKGatingMatchesOracle + TestMoeTopKGatingCUDASourceContract (internal/compute/moe_topk_gating_test.go): scoring, renormalization, and deterministic tie-breaking vs MoeTopKGating",
		Note:        "Unwired standalone fragment for non-V4.1 MoE routers; gated behind qualification before libfakcuda wiring.",
	},
	{
		Slug:        "cuda-pdl-chain",
		Title:       "CUDA programmatic dependent launch (PDL) decode chain contract",
		FileGlobs:   []string{"internal/compute/cuda/pdl_chain.cuh"},
		FakPath:     "internal/compute/cuda/pdl_chain.cuh; internal/compute/pdl_chain_test.go",
		SOTA:        "tie-pilot-qxw/pdl-megakernel-reconstruction@8664720072fefb62cc3a097c9c5bba8c936294c5 PDL decode graph (MIT)",
		PrimaryLink: "https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction/tree/8664720072fefb62cc3a097c9c5bba8c936294c5",
		Route:       RouteBorrow,
		Oracle:      "TestPDLChain (internal/compute/pdl_chain_test.go): PDL graph edge topology and griddepcontrol wait dependency order invariants",
		Note:        "Structure-only header contract under FAK_CUDA_PDL_CHAIN; no hardware run or speed claim.",
	},
	{
		Slug:        "cuda-pit-sparse-prefill",
		Title:       "CUDA PIT mixed-sparse diagonal and vertical block prefill contract",
		FileGlobs:   []string{"internal/compute/cuda/pit_sparse_prefill.cuh"},
		FakPath:     "internal/compute/cuda/pit_sparse_prefill.cuh; internal/compute/pit_sparse_prefill_test.go",
		SOTA:        "microsoft/MInference@29ef1974fcb5a1440ccf92d7d7996900230fddcf pit_sparse_flash_attention_v2.py (MIT)",
		PrimaryLink: "https://github.com/microsoft/MInference/tree/29ef1974fcb5a1440ccf92d7d7996900230fddcf",
		Route:       RouteBorrow,
		Oracle:      "TestPITSparsePrefillKernel (internal/compute/pit_sparse_prefill_test.go): 64x64 PIT slash/vertical mixed-sparse block prefill with {16,32,64,128} zero-padding",
		Note:        "Structure-only header contract and host PIT sparse oracle; no hardware run or speed claim.",
	},
	{
		Slug:        "cuda-vllm-paged-chunked-prefill",
		Title:       "CUDA vLLM paged chunked-prefill attention",
		FileGlobs:   []string{"internal/compute/prefill_paged.cu"},
		FakPath:     "internal/compute/prefill_paged.cu; internal/compute/prefill_paged_test.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 paged chunked-prefill attention over block-table KV arenas (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "TestPagedChunkedPrefillMatchesContiguousPanel (internal/compute/prefill_paged_test.go): bitwise contiguous-panel parity across cold, single-block, page-crossing, and all-paged paths",
		Note:        "Borrow vLLM block-table paged prefill over x=4 K/V layouts under fak_vllm_prefill namespace.",
	},
	{
		Slug:        "cuda-vllm-rope-position-array",
		Title:       "CUDA vLLM persistent cos/sin table and position-id array RoPE",
		FileGlobs:   []string{"internal/compute/rope_position_array.cu", "internal/compute/rope_position_array*.go"},
		FakPath:     "internal/compute/rope_position_array.cu; internal/compute/rope_position_array.go",
		SOTA:        "vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 persistent cos/sin table and position-id array RoPE rotation (Apache-2.0)",
		PrimaryLink: "https://github.com/vllm-project/vllm/tree/ae065918640a198a56f6f17eb45e251c3229fe41",
		Route:       RouteBorrow,
		Oracle:      "TestRopePositionArrayMatchesRowwiseOracle (internal/compute/rope_position_array_test.go): bitwise NeoX, interleaved, partial-offset, and inverse rotation parity vs cpuref.RoPE",
		Note:        "Borrow persistent [max_position, rot_dim] cos/sin table and gapped position-array RoPE indexing.",
	},
}

// Operations returns every matrix row, sorted by Slug. The returned slice is a
// copy; callers may not mutate the matrix.
func Operations() []Op {
	out := make([]Op, len(matrix))
	copy(out, matrix)
	sort.Slice(out, func(i, j int) bool { return out[i].Slug < out[j].Slug })
	return out
}

// BySlug returns the row with the given slug, or ok=false if none.
func BySlug(slug string) (Op, bool) {
	for _, o := range matrix {
		if o.Slug == slug {
			return o, true
		}
	}
	return Op{}, false
}
