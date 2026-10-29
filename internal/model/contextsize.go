package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// ContextSizeConfig projects this model Config into the compute-level geometry the
// context auto-sizer consumes (compute.AutoSizeContextPlan, #1049). It is the one
// projection both the serve boot path (cmd/fak/serve.go) and the in-kernel per-request
// planner (internal/agent/inkernel_planner.go) share, so they size a context plan from
// a single source of geometry and cannot drift apart on the same model.
//
// # Per-layer window (#5520)
//
// It is also the FIRST consumer of the corrected KV geometry KVCacheShape computes: an
// interleaved local/global checkpoint (gemma-4's five sliding layers per global one)
// hands compute.KVConfig.WindowPerLayer the same per-layer bound the Shape's
// kvbudget.LayerProfile carries, so EstimateKVStoreBytes charges a windowed layer
// min(window, tokens) positions instead of the whole context. Before this the Shape was
// computed and discarded — every number a user or the auto-sizer saw came from the flat
// NumLayers x tokens product, which over-charges the pinned gemma-4-26B cadence (30
// layers, 25 windowed at 1024, 5 global) by 32/7 ~= 4.57x at a 16384-token context.
//
// GATED, and deliberately so. The discount is only TRUE if the cache actually drops
// aged-out positions on a windowed layer, and today it does not: Session.TrimToWindow
// (swa.go:48) has no production caller — swa.go:14 says so and a repo-wide grep agrees —
// and Config.MaxWindow (swa.go:25) returns 0 unless EVERY layer declares a window, which
// an interleaved model never does. Charging the discount unconditionally would turn a
// conservative over-reservation into an UNDER-reservation and trade a refusal for an OOM.
// So it rides FAK_HYBRID_KV, the flag kvgroups.go:220 already gates this exact
// grouped-vs-uniform saving behind ("dogfood-gated until promotion evidence lands"). Off
// (the default) the projection is byte-for-byte what it was; on, the planner reads the
// corrected shape. Flipping it on by default is the follow-on, and it is blocked on the
// cache realizing the window, not on this projection.
func (c Config) ContextSizeConfig() compute.ContextSizeConfig {
	return c.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
}

// ContextSizeConfigWithPrecision is ContextSizeConfig with the KV store's realized
// storage tier set. A caller serving a q8 KV cache passes compute.KVPrecisionQ8 so the
// context auto-sizer and the fit checks charge the denser mixed layout (f32 pre-RoPE K
// + q8_0 K/V) the engine actually allocates — otherwise the admission gate would plan
// for f32 while the engine resides q8, refusing a context that fits. The F32 zero
// value reproduces ContextSizeConfig byte-for-byte.
func (c Config) ContextSizeConfigWithPrecision(prec compute.KVPrecision) compute.ContextSizeConfig {
	if c.IsDeepSeekV41() {
		// V41Prefill/V41Step bypass the generic q8 KV cache and retain F32
		// host/device rows. BF16 numerical boundaries do not narrow that
		// storage. Keep the conservative F32 tier until V4.1 realizes q8.
		prec = compute.KVPrecisionF32
	}
	kv := compute.KVConfig{
		NumLayers:  c.NumLayers,
		NumKVHeads: c.NumKVHeads,
		HeadDim:    c.HeadDim,
		RopeTheta:  c.RopeTheta,
		Precision:  prec,
	}
	var sessionState compute.MemoryPlan
	var compressedKV []compute.CompressedKVLayer
	if c.IsQwen35Hybrid() {
		// The live Qwen35 paths never append token-indexed K/Kraw/V for a
		// linear_attention layer. The direct path returns through linearAttnStep
		// before the cache append; the HAL path remaps each full-attention layer
		// onto a compact KV index. Charge exactly those compact layers plus the
		// fixed recurrent matrix and short-convolution window.
		//
		// If fixed-state arithmetic cannot be proved, retain the uniform layer
		// count. A malformed/overflowing config must stay conservatively
		// over-reserved instead of receiving an unbacked hybrid discount.
		if recurrentBytes, ok := recurrentGeometryBytes(c); ok {
			fullLayers := 0
			for l := 0; l < c.NumLayers; l++ {
				if !c.isLinearAttnLayer(l) {
					fullLayers++
				}
			}
			kv.NumLayers = fullLayers
			if recurrentBytes > 0 {
				sessionState = compute.MemoryPlan{{
					Class:  compute.MemoryKVCache,
					Bytes:  recurrentBytes,
					Detail: "qwen35-gdn-recurrent-conv-state",
					DType:  compute.F32.String(),
				}}
			}
		}
	}
	if HybridKVGroupsEnabled() {
		if shape, ok := c.v4FlashCompressedKVShape(); ok {
			compressedKV = make([]compute.CompressedKVLayer, shape.Layers)
			var inflightBytes int64
			for l, ratio := range shape.PerLayer.CompressRatio {
				compressedKV[l] = compute.CompressedKVLayer{
					WindowRows:    shape.PerLayer.Window[l],
					Ratio:         ratio,
					RowBytes:      int64(shape.KVLoraRank) * 4,
					IndexRowBytes: int64(shape.PerLayer.IndexHeadDim[l]) * 4,
				}
				rows := ratio - 1
				if ratio == 4 {
					rows = 2*ratio - 1 // overlapping compressor retains both groups
				}
				if ratio > 0 {
					inflightBytes += int64(rows) * int64(c.HiddenSize+c.HeadDim) * 4
				}
			}
			sessionState = append(sessionState, compute.MemoryDemand{
				Class: compute.MemoryKVCache, Bytes: inflightBytes,
				Detail: "deepseek-v4-compressor-inflight-rows", DType: compute.F32.String(),
			})
		}
		if c.IsDeepSeekV41() {
			// DeepSeek-V4.1 keeps a per-layer sliding-window row mirror plus the
			// shared compressed rows and indexer keys its source layers publish
			// (#13914). The uniform product charges three f32 rows per layer per
			// token while the kept KV is a bounded mirror plus ~8.4 KB/token of
			// published rows; an owners-only charge would under-reserve because
			// the mirror is per layer, so the charge folds each layer's mirror
			// into its own entry and the compressor in-flight rows into
			// SessionState. Anything the published-schedule check cannot prove
			// keeps the uniform charge (fail closed, conservative over-
			// reservation).
			if layers, inflight, ok := c.v41CompressedKVCharge(); ok {
				compressedKV = layers
				sessionState = append(sessionState, compute.MemoryDemand{
					Class: compute.MemoryKVCache, Bytes: inflight,
					Detail: "deepseek-v41-compressor-inflight-rows", DType: compute.F32.String(),
				})
			}
		}
		// KVCacheShape already normalizes the loader's -1 full-attention sentinel to
		// kvbudget's non-positive "no window" spelling, which is the spelling
		// compute.KVConfig.WindowPerLayer reads — so the slice crosses verbatim. A
		// uniformly-global model leaves PerLayer (or PerLayer.Window) nil and the
		// projection stays exactly the uniform one.
		if p := c.KVCacheShape().PerLayer; p != nil && !c.hasKVCompressionSchedule() {
			kv.WindowPerLayer = p.Window
		}
	}
	return compute.ContextSizeConfig{
		KV:           kv,
		SessionState: sessionState,
		CompressedKV: compressedKV,
		Scratch: compute.TransformerScratchConfig{
			HiddenSize:       c.HiddenSize,
			IntermediateSize: c.IntermediateSize,
			VocabSize:        c.VocabSize,
			NumLayers:        c.NumLayers,
			NumHeads:         c.NumHeads,
			NumKVHeads:       c.NumKVHeads,
			HeadDim:          c.HeadDim,
			IncludeLogits:    true,
		},
		MaxContext: c.MaxPositionEmbeddings,
	}
}

// v41CompressedKVCharge projects the published DeepSeek-V4.1-Flash retention
// schedule onto compute.CompressedKVLayer entries plus the compressor's
// in-flight row demand (#13914).
//
// Every decoder layer keeps a v41WindowSize-row device row mirror of one f32
// HeadDim row per kept position (ring-wrapped at the window edge); the KV
// source layers publish one compressed HeadDim row per completed ratio group,
// and the index source layers one IndexHeadDim key per published compressed
// row. Each layer's entry folds its mirror into the window term
// (WindowRows = v41WindowSize, RowBytes = HeadDim*4) so the kept rows are
// charged per layer instead of dropped by an owners-only accounting, and the
// owner's compressed rows ride the same entry's Ratio term. An index-only
// source — a layer that publishes keys but no latents — gets a second
// zero-width-base entry (WindowRows = 1 pins the base term at one 0-byte row)
// whose ceil(tokens/Ratio) index keys are the whole charge. The compressor's
// in-flight rows — at most ratio-1 pre-attention input carriers plus ratio-1
// projected rows per compressing layer, the bound V41AttentionState's append
// guards enforce — are returned separately for SessionState, mirroring the
// V4 Flash in-flight demand.
//
// ok=false (uniform fallback) unless every axis the arithmetic trusts is the
// published one: the 40-layer decoder schedule (which also pins the ratios),
// the published source lists, and positive pinned widths.
func (c Config) v41CompressedKVCharge() ([]compute.CompressedKVLayer, int64, bool) {
	if !c.IsDeepSeekV41() || !v41PublishedDecoderSchedule(c) {
		return nil, 0, false
	}
	m := c.DeepSeekV41
	if m == nil || !v41EqualInts(m.KVSourceLayerIDs, v41PublishedKVSourceLayerIDs) ||
		!v41EqualInts(m.IndexSourceLayerIDs, v41PublishedIndexSourceLayerIDs) {
		return nil, 0, false
	}
	if c.HeadDim <= 0 || c.HiddenSize <= 0 || c.IndexHeadDim <= 0 {
		return nil, 0, false
	}
	rowBytes, indexBytes := int64(c.HeadDim)*4, int64(c.IndexHeadDim)*4
	kvSource := make(map[int]bool, len(v41PublishedKVSourceLayerIDs))
	for _, l := range v41PublishedKVSourceLayerIDs {
		kvSource[l] = true
	}
	indexOnly := make(map[int]bool, len(v41PublishedIndexSourceLayerIDs))
	for _, l := range v41PublishedIndexSourceLayerIDs {
		if !kvSource[l] {
			indexOnly[l] = true
		}
	}
	n := v41PublishedNumLayers
	layers := make([]compute.CompressedKVLayer, 0, n+len(indexOnly))
	var inflight int64
	for l := 0; l < n; l++ {
		ratio := c.CompressRatios[l]
		entry := compute.CompressedKVLayer{WindowRows: v41WindowSize, RowBytes: rowBytes}
		switch {
		case kvSource[l]:
			// All four published KV sources are also index sources: their
			// compressed rows and index keys fold into the layer's own entry.
			entry.Ratio = ratio
			entry.IndexRowBytes = indexBytes
		case indexOnly[l]:
			// Keys without latents: a zero-width-base entry charges exactly
			// its ceil(tokens/ratio) index keys.
			layers = append(layers, entry, compute.CompressedKVLayer{
				WindowRows:    1,
				Ratio:         ratio,
				IndexRowBytes: indexBytes,
			})
			if ratio > 1 {
				inflight += int64(ratio-1) * int64(c.HiddenSize+c.HeadDim) * 4
			}
			continue
		}
		if ratio > 1 {
			inflight += int64(ratio-1) * int64(c.HiddenSize+c.HeadDim) * 4
		}
		layers = append(layers, entry)
	}
	return layers, inflight, true
}
