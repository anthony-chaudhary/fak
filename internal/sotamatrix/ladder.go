package sotamatrix

import "sort"

// ladder.go — the companion to the kernel prior-art matrix. Where `matrix`
// answers "before writing THIS one kernel, what reference already solved the
// contraction?", the milestone LADDERS answer the coarser, strategic question
// the matrix cannot: "for a whole capability — attention, batching, quantization
// — what are the recognized SOTA baseline milestones, in order, and which rung is
// fak actually on?"
//
// It exists because "should we build the next thing, and what IS the next thing?"
// kept being answered from memory. An agent would reach for "add FlashAttention"
// without knowing fak already ships a fused online-softmax (the FlashAttention-1
// rung) and that the open rung is FA-2 work-partitioning / FA-3 FP8 / FlashInfer
// paging — or reach for "batch requests" without knowing fak's batcher is
// dynamic, padding-aware composition (per internal/gateway/batchsched.go) and that
// the named-but-deferred rung is continuous / in-flight batching. The ladder makes
// that landscape a maintained datum instead of folklore: each rung is an external,
// dated, citable milestone (not a self-claim), and FakRung records which rung fak
// has reached, grounded in the tree-verified matrix Op it maps to.
//
// It is read by `fak sota milestones [axis]` (cmd/fak/sota.go). Each Ladder either
// maps to a kernel Op (OpSlug set — the ladder's FakRung is grounded in that row's
// tree-verified FakPath/Note) or is a serving-level capability with no single
// kernel (OpSlug empty — batching, speculative decoding), where FakRung is the
// serving position or -1 when fak does not implement the axis at all.
//
// Discipline: a rung is EXTERNAL PRIOR ART — a real technique with a real
// reference — never a fak self-report. The only fak-facing claim is FakRung, and
// it is deliberately conservative and pinned to what the matrix already witnesses
// against the tree. Adding a rung means adding a real, citable milestone; do NOT
// invent a rung to flatter fak's position.

// Milestone is one recognized rung on a capability's SOTA ladder: a named, dated
// technique that moved the baseline. Ordered by Level within a Ladder.
type Milestone struct {
	// Level is the rung index, 0 = the naive baseline, ascending. Levels within a
	// ladder are contiguous (0,1,2,…) so "the next rung" is FakRung+1.
	Level int
	// Name is the technique's canonical name ("FlashAttention-2").
	Name string
	// Year is when the milestone landed ("2023"), for ordering intuition.
	Year string
	// Ref is the canonical paper/repo to actually read for this rung.
	Ref string
	// Adds is one line: what THIS rung added over the rung below it (for Level 0,
	// what the baseline IS and its limitation).
	Adds string
}

// Ladder is the ordered set of SOTA baseline milestones for one capability axis.
type Ladder struct {
	// Axis is the stable lookup key for `fak sota milestones <axis>` (kebab-case).
	Axis string
	// Title is the human name of the capability.
	Title string
	// OpSlug is the matrix Op this axis maps to, or "" for a serving-level axis
	// with no single kernel row (batching, speculative-decoding). When set it MUST
	// resolve via BySlug — the cross-reference is the ladder's honesty anchor.
	OpSlug string
	// Summary is a one-line gloss on what the axis is and why the ladder matters.
	Summary string
	// FakRung is the highest Level fak's implementation has reached, grounded in the
	// mapped Op's tree-verified FakPath/Note (or the serving position). It is -1 when
	// fak does not implement this axis at all. Advisory: the witness is the Op row
	// and the cited code, not this integer.
	FakRung int
	// Rungs are the milestones, ordered by ascending Level (0,1,2,…).
	Rungs []Milestone
}

// NextRung returns the milestone one rung above FakRung — the next baseline
// milestone to target — and ok=false when fak is already on the top rung or the
// axis is not implemented (FakRung < 0).
func (l Ladder) NextRung() (Milestone, bool) {
	if l.FakRung < 0 {
		return Milestone{}, false
	}
	for _, r := range l.Rungs {
		if r.Level == l.FakRung+1 {
			return r, true
		}
	}
	return Milestone{}, false
}

// ladders is the flat source of truth for the milestone ladders. Keep it sorted by
// Axis for stable output. Every Rung is external prior art with a real reference;
// FakRung is pinned conservatively to what the mapped matrix Op already witnesses.
var ladders = []Ladder{
	{
		Axis:    "attention",
		Title:   "Fused attention",
		OpSlug:  "fused-attention",
		Summary: "How the softmax-attention core is computed: from an O(N²)-memory score matrix to IO-aware fused kernels. fak ships its own fused online-softmax kernel (the FlashAttention-1 rung), cosine floor 0.999.",
		FakRung: 1,
		Rungs: []Milestone{
			{Level: 0, Name: "Naive softmax attention", Year: "2017", Ref: "Vaswani et al., \"Attention Is All You Need\" arXiv:1706.03762", Adds: "Baseline: materializes the full N×N score matrix in HBM — O(N²) memory, memory-bandwidth bound."},
			{Level: 1, Name: "FlashAttention-1", Year: "2022", Ref: "Dao et al. arXiv:2205.14135", Adds: "IO-aware tiling + online softmax: computes exact attention without ever materializing the N×N scores."},
			{Level: 2, Name: "FlashAttention-2", Year: "2023", Ref: "Dao arXiv:2307.08691", Adds: "Better work partitioning across warps and fewer non-matmul FLOPs — roughly 2× FlashAttention-1."},
			{Level: 3, Name: "FlashAttention-3", Year: "2024", Ref: "Shah et al. arXiv:2407.08608", Adds: "Hopper warp-specialization + async pipelining and FP8, exploiting the newer tensor-core path."},
			{Level: 4, Name: "FlashInfer", Year: "2024", Ref: "Ye et al. arXiv:2501.01005", Adds: "Paged / variable-length + block-sparse attention with JIT'd kernels — the serving-oriented rung."},
		},
	},
	{
		Axis:    "batching",
		Title:   "Request batching",
		OpSlug:  "", // serving-level: fak batches at internal/gateway/batchsched.go, no single kernel row.
		Summary: "How many requests share a forward pass and how prefill/decode interleave — the serving-throughput axis. fak's batcher is dynamic, padding-aware composition (internal/gateway/batchsched.go); its own honest fence names continuous/in-flight batching as the deferred rung.",
		FakRung: 2,
		Rungs: []Milestone{
			{Level: 0, Name: "Static / single-request", Year: "2020", Ref: "single-slot decode baseline (e.g. llama.cpp one-slot)", Adds: "Baseline: one sequence per forward pass — the weight stream is not amortized, GPU idles between requests."},
			{Level: 1, Name: "Static padded batching", Year: "2021", Ref: "NVIDIA FasterTransformer github.com/NVIDIA/FasterTransformer", Adds: "Group N requests padded to a common length; head-of-line blocking — the batch waits for the longest sequence."},
			{Level: 2, Name: "Dynamic batching", Year: "2021", Ref: "NVIDIA Triton dynamic batcher (Triton Inference Server user guide)", Adds: "Form a batch from a server-side queue within a latency window, bounding padding — no client-side batching required."},
			{Level: 3, Name: "Continuous / in-flight batching", Year: "2022", Ref: "Orca (Yu et al., OSDI 2022)", Adds: "Iteration-level scheduling: finished sequences leave and new ones join every decode step — no waiting for the slowest."},
			{Level: 4, Name: "Chunked prefill", Year: "2023", Ref: "SARATHI arXiv:2308.16369; Sarathi-Serve (OSDI 2024) arXiv:2403.02310", Adds: "Split long prefills into chunks piggybacked onto decode steps so a big prefill never stalls in-flight decodes."},
			{Level: 5, Name: "Disaggregated prefill/decode", Year: "2024", Ref: "DistServe arXiv:2401.09670; Splitwise arXiv:2311.18677", Adds: "Run prefill and decode on separate GPU pools so their different compute profiles stop interfering; KV is shipped between them."},
		},
	},
	{
		Axis:    "quantization",
		Title:   "Weight quantization",
		OpSlug:  "awq-int4-gemm", // the 4-bit rung fak sits on; ladder spans several matrix quant rows.
		Summary: "The weight (and activation) precision axis: from fp16 to fused 4-bit and microscaled fp8. fak sits at INT4 PTQ (AWQ/GPTQ, gguf k-quants); the fused-4-bit Marlin kernel is the matrix's recorded gap.",
		FakRung: 2,
		Rungs: []Milestone{
			{Level: 0, Name: "FP16 / BF16 weights", Year: "2017", Ref: "Micikevicius et al., \"Mixed Precision Training\" arXiv:1710.03740", Adds: "Baseline: 16-bit weights — the full-precision-ish reference every quant rung is measured against."},
			{Level: 1, Name: "INT8 weight quant", Year: "2022", Ref: "LLM.int8() (Dettmers et al.) arXiv:2208.07339", Adds: "8-bit weights with outlier handling — ~2× smaller, near-lossless, no special kernel needed."},
			{Level: 2, Name: "INT4 PTQ: GPTQ / AWQ", Year: "2023", Ref: "GPTQ arXiv:2210.17323; AWQ arXiv:2306.00978", Adds: "4-bit weight-only via error-compensated (GPTQ) or activation-aware (AWQ) rounding — the accuracy method."},
			{Level: 3, Name: "Fused INT4 kernel: Marlin", Year: "2024", Ref: "Marlin github.com/IST-DASLab/marlin", Adds: "A fused dequant-MMA kernel so 4-bit weights run at near-fp16 tensor-core throughput — the kernel, not just the format."},
			{Level: 4, Name: "FP8 / MXFP4 microscaling", Year: "2024", Ref: "OCP Microscaling (MX) Formats spec; NVIDIA Transformer Engine", Adds: "Hardware fp8 (Hopper/Blackwell) and microscaled MXFP4/MXFP8 quantizing activations too, not only weights."},
		},
	},
	{
		Axis:    "kv-cache",
		Title:   "KV cache management",
		OpSlug:  "kv-cache-paging",
		Summary: "How the attention KV cache is laid out and reused across requests. fak owns an exact f32 cache with block paging and RadixAttention prefix reuse (internal/radixkv); its differentiator is bit-exact eviction, not paged throughput.",
		FakRung: 2,
		Rungs: []Milestone{
			{Level: 0, Name: "Contiguous per-sequence cache", Year: "2020", Ref: "HuggingFace Transformers past_key_values (contiguous baseline)", Adds: "Baseline: a pre-allocated max-length contiguous buffer per sequence — heavy fragmentation and wasted capacity."},
			{Level: 1, Name: "PagedAttention", Year: "2023", Ref: "vLLM (Kwon et al.) arXiv:2309.06180", Adds: "Block-paged KV like OS virtual memory — near-zero fragmentation, far higher batch occupancy."},
			{Level: 2, Name: "RadixAttention prefix reuse", Year: "2023", Ref: "SGLang (Zheng et al.) arXiv:2312.07104", Adds: "A radix tree over KV blocks shares common prefixes across requests, skipping recomputation of shared context."},
			{Level: 3, Name: "KV quantization / offload", Year: "2024", Ref: "KIVI (Liu et al.) arXiv:2402.02750", Adds: "KV stored in int8/fp8 and tiered to host memory to extend context length past device HBM."},
		},
	},
	{
		Axis:    "speculative-decoding",
		Title:   "Speculative decoding",
		OpSlug:  "", // serving-level; fak's exact-decode model runner does not implement draft-token speculation.
		Summary: "Generate multiple tokens per verification pass to break the memory-bound one-token-per-step decode floor, while preserving the model's exact output distribution. fak's model runner is exact-decode and does not (yet) implement draft-token speculation — this ladder is the recognized external landscape.",
		FakRung: -1,
		Rungs: []Milestone{
			{Level: 0, Name: "Autoregressive decode", Year: "2017", Ref: "one-token-per-step baseline (Vaswani et al. arXiv:1706.03762)", Adds: "Baseline: one token per forward pass — memory-bandwidth bound, the whole weight set streamed per token."},
			{Level: 1, Name: "Draft-model speculative decoding", Year: "2023", Ref: "Leviathan et al. arXiv:2211.17192; Chen et al. arXiv:2302.01318", Adds: "A small draft model proposes k tokens; the large model verifies them in one pass — provably the same distribution."},
			{Level: 2, Name: "Medusa (self-speculation)", Year: "2024", Ref: "Cai et al. arXiv:2401.10774", Adds: "Extra decoding heads on the base model propose continuations — no separate draft model to serve."},
			{Level: 3, Name: "EAGLE", Year: "2024", Ref: "Li et al. arXiv:2401.15077", Adds: "Feature-level autoregression for the draft, raising the accept rate well above Medusa's token-level heads."},
			{Level: 4, Name: "Lookahead / n-gram drafts", Year: "2024", Ref: "Lookahead decoding (Fu et al.) arXiv:2402.02057", Adds: "Drafts pulled from n-grams / the prompt with no trained draft model — zero extra parameters."},
		},
	},
}

// Ladders returns every milestone ladder, sorted by Axis. The returned slice (and
// each Rungs slice) is a copy; callers may not mutate the source.
func Ladders() []Ladder {
	out := make([]Ladder, len(ladders))
	for i, l := range ladders {
		l.Rungs = append([]Milestone(nil), l.Rungs...)
		out[i] = l
	}
	sortLadders(out)
	return out
}

// LadderByAxis returns the ladder with the given axis, or ok=false if none. The
// returned Rungs slice is a copy.
func LadderByAxis(axis string) (Ladder, bool) {
	for _, l := range ladders {
		if l.Axis == axis {
			l.Rungs = append([]Milestone(nil), l.Rungs...)
			return l, true
		}
	}
	return Ladder{}, false
}

// LaddersForOp returns the milestone ladders whose OpSlug maps to the given matrix
// Op slug (sorted by Axis) — the bridge from `fak sota <slug>` to its ladder.
func LaddersForOp(slug string) []Ladder {
	var out []Ladder
	for _, l := range ladders {
		if l.OpSlug == slug {
			l.Rungs = append([]Milestone(nil), l.Rungs...)
			out = append(out, l)
		}
	}
	sortLadders(out)
	return out
}

func sortLadders(ls []Ladder) {
	sort.Slice(ls, func(i, j int) bool { return ls[i].Axis < ls[j].Axis })
}
