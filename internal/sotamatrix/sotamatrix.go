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
//   - tools/sota_coverage_scorecard.py             - which cross-checks every row
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
			"internal/compute/prefill.go", "internal/compute/prefill_cuda.go",
			"internal/compute/graph_cuda.go", "internal/compute/tf32_cuda.go",
			"internal/compute/quant_q4k.go",
		},
		FakPath:     "internal/compute/cuda.go + cuda_kernels.cu (k_q8_gemm, k_q4k_gemm, cublasGemmEx fp16)",
		SOTA:        "cuBLAS (fp16/tensor-core); llama.cpp Q8_0/Q4_K fused dequant-GEMM",
		PrimaryLink: "https://github.com/ggml-org/llama.cpp",
		Route:       RouteBind,
		Oracle:      "cpuref f32, argmax-exact + cosine ≥ floor (fp16 0.997 / Q8 0.999 / Q4_K 0.995)",
		Note:        "cuBLAS fp16 already bound; borrow the fused tile-dequant pattern for the int lanes.",
	},
	{
		Slug:        "awq-int4-gemm",
		Title:       "AWQ 4-bit GEMV/GEMM",
		FileGlobs:   []string{"internal/model/awq*.go", "internal/compute/cuda_kernels.cu"},
		FakPath:     "internal/compute/cuda.go:1101 (AWQMatMul) + cuda_kernels.cu (k_awq_gemv/gemm) + internal/model/awq.go",
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
		Slug:        "fused-attention",
		Title:       "Fused attention (MHA/GQA/MQA)",
		FileGlobs:   []string{"internal/compute/cuda_kernels.cu", "internal/compute/flash*.go"},
		FakPath:     "internal/compute/cuda_kernels.cu k_flash_attention (#486, cosine floor 0.999)",
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
