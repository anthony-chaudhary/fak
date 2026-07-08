package model

import "os"

// kvgroups.go — the hybrid-model KV MEMORY PLANE: a layer-group vocabulary plus the
// per-group budget arithmetic the whole plane sizes against (#2241, R1 of M9, epic #2236).
//
// fak already DECODES hybrid architectures — the kernels exist and are CPU-proven: SWA
// (swa.go), Qwen3.5/3.6 Gated-DeltaNet linear-attention layers (qwen35.go), GLM MLA/DSA.
// But the memory plane still treats every layer as uniform full-attention KV: KVCache
// (kvcache.go) keeps ONE per-layer K/V run per position for every layer, and PagedKVPool
// (pagedkv.go) pages every layer into the same fixed block. That over-allocates two ways a
// hybrid model never uses:
//
//   - a SLIDING-WINDOW layer retains full history it can provably never attend to again (a
//     position older than the window is un-attended — the correctness fact swa.go's
//     TrimToWindow already exploits), yet the uniform plane keeps every position resident;
//   - a RECURRENT state-space layer (Gated-DeltaNet / Mamba) holds an ACCUMULATED state, not
//     a token-indexed span (see RecurrentEvictUnsupportedError in kvcache.go), so its
//     footprint is O(1) per sequence — but the uniform plane budgets it as if it held ctx
//     token-slots of K/V it never had.
//
// The engines solved exactly this: vLLM's Hybrid Memory Allocator groups layers by KV shape
// (full / sliding-window / Mamba state) with per-group budgets, and SGLang's UnifiedTree
// offloads hybrids by group. This file lands the fak parity SPINE for that: a group
// vocabulary (KVLayerGroup) fak can classify any decoded layer into, and the per-group
// budget arithmetic (window layers cap at the window; state layers are O(1) per sequence).
//
// Scope (honest, R1): the vocabulary and the budget arithmetic, gated OFF by default behind
// FAK_HYBRID_KV (gen/next: near-term foundation, dogfood-gated until promotion evidence
// lands). When the gate is off, ResidentKVFloats reports the uniform allocation and nothing
// in the decode path changes. A genuinely per-group PagedKVPool (separate block pools per
// group), the per-group radix payload for prefix reuse, and swap/restore round-trips are the
// R2 follow-ons; the resident-bytes-vs-uniform witness on Qwen3.6-GDN is R3. What lands here
// is the classification + arithmetic those steps size against, proven on a hybrid config.

// KVLayerGroup classifies a decoder layer by the SHAPE of the attention state its KV memory
// must hold — the vocabulary the hybrid memory plane budgets against. It is orthogonal to
// the decode kernel (two layers in the same group may run different kernels); what it
// captures is how the layer's resident footprint scales with context length.
type KVLayerGroup uint8

const (
	// KVGroupFull is a full-causal-attention layer: token-indexed K/V, one row per position,
	// resident footprint GROWS linearly with context (every past position stays attendable).
	KVGroupFull KVLayerGroup = iota
	// KVGroupSlidingWindow is a sliding-window-attention layer (SWA, e.g. Gemma3 local
	// layers): token-indexed K/V, but a position older than the window is provably
	// un-attended (swa.go), so the resident footprint CAPS at the window size.
	KVGroupSlidingWindow
	// KVGroupRecurrent is a recurrent state-space layer (Gated-DeltaNet / Mamba linear
	// attention): its "KV" is an ACCUMULATED recurrent state plus a short causal-conv window,
	// not a token-indexed span, so the resident footprint is O(1) per sequence — constant in
	// context length.
	KVGroupRecurrent
)

// String renders the group for witnesses and operator readouts.
func (g KVLayerGroup) String() string {
	switch g {
	case KVGroupFull:
		return "full-attention"
	case KVGroupSlidingWindow:
		return "sliding-window"
	case KVGroupRecurrent:
		return "recurrent-state"
	default:
		return "unknown"
	}
}

// kvGroupPlanes is the per-position plane count the token-indexed groups budget against —
// K, Kraw (the pre-RoPE key kept so an evicted survivor re-rotates in one step), and V,
// matching model.KVCache's own three-plane layout (kvcache.go). Recurrent layers hold no
// token-indexed plane, so this factor does not apply to them.
const kvGroupPlanes = 3

// KVLayerGroupOf classifies decoder layer l into its KV memory group. The order matters:
// a recurrent linear-attention layer is checked first (it never has a token window), then a
// configured sliding window, else full causal. A model with no windows and no linear layers
// classifies every layer KVGroupFull — the uniform case, so a non-hybrid model's budget is
// byte-for-byte the pre-grouping allocation.
func (c Config) KVLayerGroupOf(l int) KVLayerGroup {
	if c.isLinearAttnLayer(l) {
		return KVGroupRecurrent
	}
	if c.windowForLayer(l) > 0 {
		return KVGroupSlidingWindow
	}
	return KVGroupFull
}

// recurrentStateFloats is the O(1) per-sequence float32 footprint of ONE Gated-DeltaNet
// linear-attention layer: the accumulated recurrent state (nV value heads, each a
// keyHeadDim*valueHeadDim matrix) plus the short causal-conv window (K-1 retained rows of
// convDim), matching newLinearAttnLayerState / pushConvRow in qwen35.go. Both terms are
// derived from the model geometry alone — they do NOT depend on context length, which is the
// whole point of the recurrent group. A degenerate (non-hybrid) config yields 0.
func (c Config) recurrentStateFloats() int {
	_, nV, kHd, vHd, _, _, convDim := c.linearAttnDims()
	recurrent := nV * kHd * vHd
	conv := 0
	if c.LinearConvKernelDim > 1 {
		conv = (c.LinearConvKernelDim - 1) * convDim
	}
	total := recurrent + conv
	if total < 0 {
		return 0
	}
	return total
}

// KVLayerResidentPositions is the number of token-slots layer l must keep RESIDENT at
// context length ctx under grouped allocation:
//   - full-attention: ctx (every position stays attendable);
//   - sliding-window: min(ctx, window) (older positions are provably un-attended, so
//     they may be dropped — the swa.go TrimToWindow correctness fact);
//   - recurrent: 0 (a state-space layer holds no token-indexed span at all).
//
// A negative ctx is clamped to 0.
func (c Config) KVLayerResidentPositions(l, ctx int) int {
	if ctx < 0 {
		ctx = 0
	}
	switch c.KVLayerGroupOf(l) {
	case KVGroupRecurrent:
		return 0
	case KVGroupSlidingWindow:
		if w := c.windowForLayer(l); w > 0 && ctx > w {
			return w
		}
		return ctx
	default:
		return ctx
	}
}

// KVLayerResidentFloats is layer l's resident float32 footprint at context length ctx:
// its O(1) recurrent state for a recurrent layer, else its resident token-slots times the
// three-plane (K/Kraw/V) per-position stride. This is the per-layer term the aggregate
// budget sums.
func (c Config) KVLayerResidentFloats(l, ctx int) int {
	if c.KVLayerGroupOf(l) == KVGroupRecurrent {
		return c.recurrentStateFloats()
	}
	stride := c.NumKVHeads * c.HeadDim
	if stride < 0 {
		stride = 0
	}
	return c.KVLayerResidentPositions(l, ctx) * stride * kvGroupPlanes
}

// KVGroupBudget is a model's resident KV footprint at a fixed context length, split by
// layer group — the hybrid memory plane's core accounting. Layer counts and float32 totals
// are tracked per group so a caller (pool sizing, preemption arithmetic, an operator
// readout) can see WHERE the memory goes and how each group scales with ctx.
type KVGroupBudget struct {
	Ctx             int // context length this budget was computed at
	FullLayers      int // number of full-attention layers
	WindowLayers    int // number of sliding-window layers
	RecurrentLayers int // number of recurrent state-space layers
	FullFloats      int // resident float32 across all full-attention layers (grows with Ctx)
	WindowFloats    int // resident float32 across all sliding-window layers (caps at window)
	RecurrentFloats int // resident float32 across all recurrent layers (O(1) in Ctx)
}

// TotalFloats is the resident float32 across every group.
func (b KVGroupBudget) TotalFloats() int {
	return b.FullFloats + b.WindowFloats + b.RecurrentFloats
}

// KVGroupBudgetAt computes the per-group resident budget at context length ctx by
// classifying every decoder layer and summing its per-layer resident footprint. This is the
// grouped allocation the hybrid plane sizes against; compare TotalFloats() to
// UniformKVFloats(ctx) to see the over-allocation the plane removes.
func (c Config) KVGroupBudgetAt(ctx int) KVGroupBudget {
	b := KVGroupBudget{Ctx: ctx}
	for l := 0; l < c.NumLayers; l++ {
		switch c.KVLayerGroupOf(l) {
		case KVGroupRecurrent:
			b.RecurrentLayers++
			b.RecurrentFloats += c.KVLayerResidentFloats(l, ctx)
		case KVGroupSlidingWindow:
			b.WindowLayers++
			b.WindowFloats += c.KVLayerResidentFloats(l, ctx)
		default:
			b.FullLayers++
			b.FullFloats += c.KVLayerResidentFloats(l, ctx)
		}
	}
	return b
}

// UniformKVFloats is the resident KV footprint the PRE-grouping plane allocates: EVERY layer
// keeps ctx token-slots of K/Kraw/V regardless of its group. This is the over-allocation the
// hybrid plane removes — a sliding-window layer keeps full history it can never attend to,
// and a recurrent layer keeps token spans it never had. It is the baseline the grouped
// budget is measured against (and, with the gate off, the footprint ResidentKVFloats
// reports — preserving today's behavior exactly).
func (c Config) UniformKVFloats(ctx int) int {
	if ctx < 0 {
		ctx = 0
	}
	stride := c.NumKVHeads * c.HeadDim
	if stride < 0 {
		stride = 0
	}
	nLayers := c.NumLayers
	if nLayers < 0 {
		nLayers = 0
	}
	return nLayers * ctx * stride * kvGroupPlanes
}

// HybridKVGroupsEnabled reports whether the layer-group hybrid KV memory plane is active.
// It is gated OFF by default — gen/next generation intent: a near-term foundation kept
// gated until promotion evidence (the R2 hybrid-radix witnesses and the R3 resident-bytes
// benchmark) lands — and turned on by FAK_HYBRID_KV=1. When off, the plane budgets every
// layer uniformly and nothing in the decode path changes; when on, budgets are per-group.
func HybridKVGroupsEnabled() bool {
	return os.Getenv("FAK_HYBRID_KV") == "1"
}

// ResidentKVFloats is the single seam the rest of the memory plane consults for a model's
// resident KV budget at context length ctx: the per-group total when the hybrid plane is
// enabled, else the uniform allocation. Routing every caller (pool sizing, preemption
// arithmetic) through here means the FAK_HYBRID_KV flag flips the whole plane in one place.
func (c Config) ResidentKVFloats(ctx int) int {
	if HybridKVGroupsEnabled() {
		return c.KVGroupBudgetAt(ctx).TotalFloats()
	}
	return c.UniformKVFloats(ctx)
}

// GroupBudget exposes this cache's per-group resident budget at context length ctx, so a
// caller holding a live KVCache can see the hybrid split without reaching for the Config.
// The arithmetic lives on Config; this is the KVCache-facing accessor named in the #2241
// scope ("layer-group vocabulary in model.KVCache/PagedKVPool").
func (c *KVCache) GroupBudget(ctx int) KVGroupBudget {
	return c.cfg.KVGroupBudgetAt(ctx)
}

// GroupBudgetBlocks expresses a per-group budget in PagedKVPool blocks for a single sequence
// of ctx tokens: each token-indexed group costs ceil(residentPositions/blockTokens) blocks
// PER LAYER in that group, while the recurrent group's O(1) state is not block-paged (it
// rides a per-sequence state slab, not the token-block pool) and contributes 0 blocks. This
// is the pool-sizing arithmetic a future per-group PagedKVPool draws on (R2); it reports the
// block demand today so preemption/offload accounting can already budget hybrids per group.
func (p *PagedKVPool) GroupBudgetBlocks(cfg Config, ctx int) (full, window, recurrent int) {
	bt := p.blockTokens
	if bt <= 0 {
		bt = 16
	}
	for l := 0; l < cfg.NumLayers; l++ {
		switch cfg.KVLayerGroupOf(l) {
		case KVGroupRecurrent:
			// O(1) recurrent state: not paged into the token-block pool.
		case KVGroupSlidingWindow:
			window += ceilDivPositions(cfg.KVLayerResidentPositions(l, ctx), bt)
		default:
			full += ceilDivPositions(cfg.KVLayerResidentPositions(l, ctx), bt)
		}
	}
	return full, window, recurrent
}

// ceilDivPositions is ceil(pos/blockTokens) with pos>=0 and blockTokens>0 guaranteed by the
// caller — the block count a run of pos token-slots occupies in a paged pool.
func ceilDivPositions(pos, blockTokens int) int {
	if pos <= 0 {
		return 0
	}
	return (pos + blockTokens - 1) / blockTokens
}
