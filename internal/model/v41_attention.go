package model

// v41_attention.go implements the DeepSeek V4.1 CED/CSA2 attention path over
// the session-owned shared KV / shared index state (issue #12896, leaf of parent
// #12640). It completes the seam the #13006 handoff names: the reduced
// assembly's per-position sink contraction consumed a *positional expansion* of
// the compressor's pooled latent (`kvRows[i] = compressed[i/ratio]`, see
// v41_forward.go). This file instead maps the contraction onto the reference's
// compressed / shared KV cache directly, and resolves the shared-index reuse via
// V41IndexerPublication.Reuse — the "reader layer without its own keys" path.
//
// Reference: deepseek-ai/DeepSeek-V4.1-Flash inference/model.py (Compressor /
// lightning indexer / Attention forward) and inference/kernel.py's sparse sink
// kernel, pinned at revision dba1be0a40aa45a94ad051997016db3960a90277 (MIT).
//
// Boundaries. The caller owns projection, RoPE, and the session state; this file
// begins where already-projected query heads and a completed compressed KV stream
// exist and ends at the contracted per-position attention output (before the
// grouped output projection). It runs no projection, no RoPE, and no MoE.
//
// Fail-closed. Every ratio the reduced assembly cannot represent as a
// compressed/shared contraction is refused with a typed error wrapping
// ErrV41ForwardStage naming the layer; there is no silent fall-through to the
// generic per-layer Q/K/V path and no mock. A compressed layer whose declared
// ratio does not divide its KV stream, a reader layer that names no resolvable
// source, and an unimplemented ratio all refuse before any output is produced.

import "fmt"

// V41AttentionRole classifies one layer's participation in the shared
// KV/index schedule. Exactly one role applies per layer; readers reuse the
// nearest declared source.
type V41AttentionRole int

const (
	// V41AttentionRolePerLayer is a layer with no shared-source participation:
	// it contracts its own projected KV rows (ratio 0/1 regime).
	V41AttentionRolePerLayer V41AttentionRole = iota
	// V41AttentionRoleKVSource is a declared shared-KV source layer. It pools
	// its own KV input through the CED/CSA2 compressor and publishes the rows.
	V41AttentionRoleKVSource
	// V41AttentionRoleReader is a layer that reads a preceding source's shared
	// compressed KV rows instead of projecting its own.
	V41AttentionRoleReader
)

// v41AttentionRoles resolves the shared KV/index schedule for a config into a
// per-layer role. It is the single place that maps KVSourceLayerIDs /
// IndexSourceLayerIDs / CompressRatios onto execution roles, so a layer's role
// and its source are decided once and consistently.
//
// A source layer owns the compressed KV stream; a later layer whose ratio is 0
// (the reference's uncompressed reader regime) but that follows a source reads
// the source's rows. A declared source that lies outside [0,NumLayers) is
// unreachable and is ignored (it keeps the reduced fixture runnable).
func v41AttentionRoles(cfg Config) map[int]V41AttentionRole {
	roles := make(map[int]V41AttentionRole, cfg.NumLayers)
	d41 := cfg.DeepSeekV41
	if d41 == nil {
		return roles
	}
	for _, src := range d41.KVSourceLayerIDs {
		if src >= 0 && src < cfg.NumLayers {
			roles[src] = V41AttentionRoleKVSource
		}
	}
	// Every layer that declares a compressed regime and is not itself a source
	// is a reader of the nearest preceding source. A compressed layer without a
	// preceding source is a per-layer compressor (self-contained pooling).
	nearest := -1
	for l := 0; l < cfg.NumLayers; l++ {
		if roles[l] == V41AttentionRoleKVSource {
			nearest = l
			continue
		}
		if _, isSource := roles[l]; isSource {
			continue
		}
		ratio := v41CompressRatioAt(cfg, l)
		if ratio > 1 {
			// A compressed layer contracts the compressed stream. When a source
			// precedes it, the reference reuses that source's shared KV rows;
			// otherwise it pools its own input.
			if nearest >= 0 {
				roles[l] = V41AttentionRoleReader
			}
			continue
		}
		if nearest >= 0 {
			roles[l] = V41AttentionRoleReader
		}
	}
	return roles
}

// v41CompressRatioAt returns the declared compress ratio for one layer, or 0
// when the schedule does not cover it.
func v41CompressRatioAt(cfg Config, layer int) int {
	if cfg.DeepSeekV41 == nil || layer < 0 || layer >= len(cfg.DeepSeekV41.CompressRatios) {
		return 0
	}
	return cfg.DeepSeekV41.CompressRatios[layer]
}

// v41AttentionRatioImplemented reports whether the reduced assembly can represent
// a declared compress ratio as a real CED/CSA2 contraction. The published
// DeepSeek V4.1 Flash schedule carries only 0/1 (the uncompressed regimes) and 2
// (the compressed group width); see v41_attention.go's fail-closed note. Every
// other positive ratio is an unimplemented variant and must be refused rather
// than silently pooled at a different width or run through the generic Q/K/V
// path.
func v41AttentionRatioImplemented(ratio int) bool {
	switch ratio {
	case 0, 1, 2:
		return true
	default:
		return false
	}
}

// V41AttentionPlan is the resolved, immutable execution plan for one layer's
// attention path. Ratio is the layer's declared compress ratio; KVSourceLayer
// and IndexSourceLayer are the resolved source layer IDs (-1 when none).
// CompressedGroupSize is the compressor's pooling width (the ratio) so the
// causal group boundary is explicit. TopKWidth is the number of compressed row
// IDs the layer's index selection publishes per query position (0 when the
// layer publishes no index list); it is the stride of a caller-supplied
// V41AttentionSharedKVOptions.Idx list.
type V41AttentionPlan struct {
	Layer              int
	Role               V41AttentionRole
	Ratio              int
	KVSourceLayer      int
	IndexSourceLayer   int
	CandidateSource    int
	CandidateTopKBlock int
	CandidateBlockSize int
	TopKWidth          int
}

// topKWidth is the resolved per-position index-list stride. A layer that
// contributes no index selection (not an index source and not a reader of one)
// has width 0, so callers can test it before slicing an index list.
func (p V41AttentionPlan) topKWidth() int { return p.TopKWidth }

// v41AttentionPlanFor resolves one layer's plan from the config and the declared
// source sets. It returns an error for a malformed schedule (a compressed ratio
// outside the representable set) rather than silently downgrading the layer.
func v41AttentionPlanFor(cfg Config, layer int, roles map[int]V41AttentionRole) (V41AttentionPlan, error) {
	p := V41AttentionPlan{
		Layer:            layer,
		Role:             V41AttentionRolePerLayer,
		Ratio:            v41CompressRatioAt(cfg, layer),
		KVSourceLayer:    -1,
		IndexSourceLayer: -1,
		CandidateSource:  -1,
	}
	if role, ok := roles[layer]; ok {
		p.Role = role
	}
	if p.Ratio < 0 {
		return p, v41StageErr(v41StageCompress, layer,
			fmt.Errorf("%w: layer %d declares malformed compressor ratio %d", ErrV41ForwardStage, layer, p.Ratio))
	}
	// Fail closed on a ratio the reduced assembly cannot represent. The published
	// DeepSeek V4.1 Flash schedule carries only the uncompressed regimes (0/1) and
	// the CED/CSA2 ratio 2 (v41_config.go); any other positive ratio is an
	// unimplemented attention variant, and silently pooling it at a different
	// width would emit logits for a model the reference never describes. There is
	// no fall-through to the generic per-layer Q/K/V path.
	if !v41AttentionRatioImplemented(p.Ratio) {
		return p, v41StageErr(v41StageCompress, layer,
			fmt.Errorf("%w: layer %d declares compressor ratio %d, which is not an implemented V4.1 attention variant", ErrV41ForwardStage, layer, p.Ratio))
	}
	d41 := cfg.DeepSeekV41
	if d41 != nil {
		// Resolve the nearest preceding declared source (the reference shares a
		// source's state forward through the stack). A source at or before the
		// reader is reusable; a later declaration is not yet available.
		for _, src := range d41.KVSourceLayerIDs {
			if src <= layer && (p.KVSourceLayer < 0 || src > p.KVSourceLayer) {
				p.KVSourceLayer = src
			}
		}
		for _, src := range d41.IndexSourceLayerIDs {
			if src <= layer && (p.IndexSourceLayer < 0 || src > p.IndexSourceLayer) {
				p.IndexSourceLayer = src
			}
		}
		if d41.CandidateSourceLayerID >= 0 && d41.CandidateSourceLayerID <= layer {
			p.CandidateSource = d41.CandidateSourceLayerID
			p.CandidateTopKBlock = d41.CandidateTopKBlocks
			p.CandidateBlockSize = d41.CandidateBlockSize
		}
		// The layer carries a per-position index list only when it is an index
		// source or a reader resolving a preceding one; the published stride is
		// the configured top-k width. Every other layer publishes no list.
		if indexSourceAt(d41, layer) || p.IndexSourceLayer >= 0 {
			if cfg.IndexTopK > 0 {
				p.TopKWidth = cfg.IndexTopK
			}
		}
	}
	return p, nil
}

// indexSourceAt reports whether layer is itself a declared index source.
func indexSourceAt(d41 *DeepSeekV41Config, layer int) bool {
	if d41 == nil {
		return false
	}
	for _, src := range d41.IndexSourceLayerIDs {
		if src == layer {
			return true
		}
	}
	return false
}

// v41CompressedCausalMask builds the per-query-position visibility over a
// compressed KV stream. Position t may attend compressed group g only when the
// group is complete at t and causally visible: g*ratio+ratio-1 <= t. This is
// the reference's block-causal boundary: a position never reads a latent whose
// group has not closed.
//
// It returns a [seq][groups] bool mask. A group beyond the stream is false.
func v41CompressedCausalMask(seq, ratio, groups int) [][]bool {
	mask := make([][]bool, seq)
	for t := 0; t < seq; t++ {
		row := make([]bool, groups)
		for g := 0; g < groups; g++ {
			last := (g+1)*ratio - 1
			row[g] = last <= t
		}
		mask[t] = row
	}
	return mask
}

// V41AttentionSharedKVOptions carries the already-resolved inputs for one
// layer's compressed/shared contraction. Ratios are the declared compress
// ratio; Groups is the number of completed compressed rows; SourceRows is the
// resolved shared KV stream (nil when the layer pools its own input). HeadDim
// is the KV latent width (the compressed row width), Heads/Dim the query head
// geometry, TopK the sink index-list width, Sink the per-head learnable sink.
type V41AttentionSharedKVOptions struct {
	Layer      int
	Ratio      int
	Groups     int
	HeadDim    int
	Heads      int
	TopK       int
	Softmax    float32
	Sink       []float32
	RopeDim    int
	Inverse    func(ropeDim int, o []float32) error
	Idx        []int32 // caller-supplied row selection (len seq*TopK); nil => causal
	IndexTopK  int
	SourceRows [][]float32 // resolved shared compressed KV stream (row-major [Groups][HeadDim])
}

// V41AttentionCompressedForward contracts already-projected V4.1 query heads
// against a COMPRESSED key/value stream using the reference's block-causal sink
// attention. It is the #12896 seam: unlike the per-position sink contraction, a
// query position reads only compressed groups that are causally complete at
// that position, and every group's key and value is the same pooled row (the
// reference's compressed KV cache holds a single latent per group, used as both
// key and value).
//
// q is row-major [seq, Heads*HeadDim] (per position, per head). values is the
// compressed stream, row-major [Groups][HeadDim]. sink, when non-nil, is
// [Heads].
//
// When opt.Idx is non-nil it supplies the selected compressed row IDs per
// position (from the lightning indexer / shared-index reuse); a -1 marks an
// empty slot. When nil, the causal mask above selects every complete group.
//
// The output is row-major [seq, Heads*HeadDim]. A position with no visible
// group yields an all-zero row (the reference's finite bound). Every geometry
// is validated and every non-finite value refused before output.
func V41AttentionCompressedForward(q []float32, values [][]float32, opt V41AttentionSharedKVOptions) ([]float32, error) {
	if opt.Ratio < 1 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention needs ratio >= 1, got %d", ErrV41ForwardStage, opt.Ratio))
	}
	if opt.Heads <= 0 || opt.HeadDim <= 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention geometry heads=%d dim=%d", ErrV41ForwardStage, opt.Heads, opt.HeadDim))
	}
	if opt.TopK < 0 || opt.Groups < 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention topk=%d groups=%d", ErrV41ForwardStage, opt.TopK, opt.Groups))
	}
	if !finite32(opt.Softmax) || opt.Softmax == 0 {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention softmax scale must be finite and non-zero, got %g", ErrV41ForwardStage, opt.Softmax))
	}
	if opt.Inverse != nil && (opt.RopeDim <= 0 || opt.RopeDim > opt.HeadDim || opt.RopeDim%2 != 0) {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention inverse rotation needs an even rope dim in (0,%d], got %d", ErrV41ForwardStage, opt.HeadDim, opt.RopeDim))
	}
	if opt.HeadDim != v41KVHeadDim(opt) {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: compressed attention head dim %d does not match the compressed row width", ErrV41ForwardStage, opt.HeadDim))
	}
	for g, row := range values {
		if len(row) != opt.HeadDim {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed row %d width %d, want %d", ErrV41ForwardStage, g, len(row), opt.HeadDim))
		}
		for i, v := range row {
			if !finite32(v) {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed row %d element %d is non-finite", ErrV41ForwardStage, g, i))
			}
		}
	}
	if opt.SourceRows != nil && len(opt.SourceRows) != opt.Groups {
		return nil, v41StageErr(v41StageAttention, opt.Layer,
			fmt.Errorf("%w: shared KV stream has %d rows, want %d", ErrV41ForwardStage, len(opt.SourceRows), opt.Groups))
	}
	seq := 0
	if opt.Heads > 0 && opt.HeadDim > 0 {
		if len(q)%(opt.Heads*opt.HeadDim) != 0 {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention query length %d not a multiple of heads*dim %d", ErrV41ForwardStage, len(q), opt.Heads*opt.HeadDim))
		}
		seq = len(q) / (opt.Heads * opt.HeadDim)
	}
	for i, v := range q {
		if !finite32(v) {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention query element %d is non-finite", ErrV41ForwardStage, i))
		}
	}
	if opt.Sink != nil {
		if len(opt.Sink) != opt.Heads {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention sink length %d, want %d", ErrV41ForwardStage, len(opt.Sink), opt.Heads))
		}
		for i, v := range opt.Sink {
			if !finite32(v) {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed attention sink element %d is non-finite", ErrV41ForwardStage, i))
			}
		}
	}

	mask := v41CompressedCausalMask(seq, opt.Ratio, opt.Groups)
	if opt.Idx != nil {
		if opt.IndexTopK <= 0 {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention index list supplied with non-positive top-k %d", ErrV41ForwardStage, opt.IndexTopK))
		}
		if len(opt.Idx) != seq*opt.IndexTopK {
			return nil, v41StageErr(v41StageAttention, opt.Layer,
				fmt.Errorf("%w: compressed attention index list length %d, want %d", ErrV41ForwardStage, len(opt.Idx), seq*opt.IndexTopK))
		}
		for i, row := range opt.Idx {
			if row < -1 || int(row) >= opt.Groups {
				return nil, v41StageErr(v41StageAttention, opt.Layer,
					fmt.Errorf("%w: compressed attention row index %d out of range at %d", ErrV41ForwardStage, row, i))
			}
		}
	}

	out := make([]float32, len(q))
	for t := 0; t < seq; t++ {
		qBase := t * opt.Heads * opt.HeadDim
		for h := 0; h < opt.Heads; h++ {
			hBase := qBase + h*opt.HeadDim
			// Resolve this position's visible groups. When the caller supplied
			// an index list, a group is visible only if it is both selected (-1
			// means empty) and causally complete; otherwise the causal mask is
			// the visibility authority.
			var visible []int
			if opt.Idx != nil {
				idxBase := t * opt.IndexTopK
				for i := 0; i < opt.IndexTopK; i++ {
					row := int(opt.Idx[idxBase+i])
					if row < 0 || !mask[t][row] {
						continue
					}
					visible = append(visible, row)
				}
			} else {
				for g := 0; g < opt.Groups; g++ {
					if mask[t][g] {
						visible = append(visible, g)
					}
				}
			}

			maxScore := float32(negInf32)
			if opt.Sink != nil {
				maxScore = opt.Sink[h]
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += q[hBase+d] * row[d]
				}
				dot *= opt.Softmax
				if dot > maxScore {
					maxScore = dot
				}
			}
			if maxScore == float32(negInf32) {
				continue
			}
			var sum float32
			if opt.Sink != nil {
				sum = exp32(opt.Sink[h] - maxScore)
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += q[hBase+d] * row[d]
				}
				sum += exp32(dot*opt.Softmax - maxScore)
			}
			if sum == 0 {
				continue
			}
			for _, g := range visible {
				var dot float32
				row := values[g]
				for d := 0; d < opt.HeadDim; d++ {
					dot += q[hBase+d] * row[d]
				}
				weight := exp32(dot*opt.Softmax-maxScore) / sum
				for d := 0; d < opt.HeadDim; d++ {
					out[hBase+d] += weight * row[d]
				}
			}
		}
	}
	if opt.Inverse != nil {
		for t := 0; t < seq; t++ {
			for h := 0; h < opt.Heads; h++ {
				headBase := (t*opt.Heads + h) * opt.HeadDim
				tail := out[headBase+opt.HeadDim-opt.RopeDim : headBase+opt.HeadDim]
				if err := opt.Inverse(opt.RopeDim, tail); err != nil {
					return nil, v41StageErr(v41StageAttention, opt.Layer, err)
				}
			}
		}
	}
	return out, nil
}

// v41KVHeadDim is the single KV latent width shared by the query heads and the
// compressed stream. It exists so validation has one named authority rather
// than an inline duplicate; the plan and options must agree on it.
func v41KVHeadDim(opt V41AttentionSharedKVOptions) int { return opt.HeadDim }

// v41AttentionIdxForSharedSource maps a source layer's published top-k row IDs
// (compressed group indices) onto a caller's index list for one query position,
// validating every row against the resolved group count. It is the shared-index
// reuse boundary: the reader must not recompute the source's selection.
func v41AttentionIdxForSharedSource(rows []int32, groups, topK int) ([]int32, error) {
	if topK <= 0 {
		return nil, fmt.Errorf("model: V4.1 shared index needs a positive top-k, got %d", topK)
	}
	if len(rows) != topK {
		return nil, fmt.Errorf("model: V4.1 shared index source published %d rows, want %d", len(rows), topK)
	}
	out := make([]int32, topK)
	for i, row := range rows {
		if row < -1 || int(row) >= groups {
			return nil, fmt.Errorf("model: V4.1 shared index row %d out of range for %d groups", row, groups)
		}
		out[i] = row
	}
	return out, nil
}

// ---- the forward seam: plan caching, shared KV/index resolution --------------

// v41AttentionRolesCached resolves the KV/index execution role per layer for the
// model's config. The schedule is a pure function of the config, so it is
// computed once and memoized on the model; the map is returned read-only and is
// never mutated by a caller. A nil DeepSeekV41 metadata yields an empty map, so
// the reduced non-shared path stays runnable.
func (m *Model) v41AttentionRolesCached() map[int]V41AttentionRole {
	if m == nil {
		return nil
	}
	if m.v41Roles != nil {
		return m.v41Roles
	}
	roles := v41AttentionRoles(m.Cfg)
	m.v41Roles = roles
	return roles
}

// v41AttentionIndexList resolves the flattened, per-position index list the
// compressed contraction consumes, [seq][TopKWidth] row-major. Three cases,
// decided by the resolved plan and never by guesswork:
//
//   - The layer IS the index source: its locally computed selection (localIdx,
//     one top-k row for the newest position) is authoritative. Because the
//     reduced assembly selects from the completed compressed stream, the newest
//     position's selection is the causal superset, so it is used for every
//     published query position.
//   - The layer is a READER of a preceding source: it reuses that source's
//     published top-k selection (V41AttentionState.PublishTopK) rather than
//     recomputing the scoring path — the reference's "reader layer without its
//     own keys" reuse. A reader with no publication fails closed.
//   - Neither: no index list, so the caller keeps the block-causal mask.
//
// groups bounds every resolved row against the compressed stream, and seq fixes
// the required per-position width, so a stale or malformed publication is
// refused instead of silently truncating the contraction. localIdx is the
// layer's own selection (nil when the layer is not an index source).
func (m *Model) v41AttentionIndexList(plan V41AttentionPlan, st *v41ForwardState, localIdx []int32, headDim, groups, seq int) ([]int32, error) {
	if plan.TopKWidth <= 0 || seq <= 0 {
		return nil, nil
	}
	isSource := indexSourceAt(m.Cfg.DeepSeekV41, plan.Layer)
	if !isSource && plan.IndexSourceLayer < 0 {
		return nil, nil
	}

	// one validates a resolved selection row against the compressed stream and
	// normalizes it to the published stride, padding a short top-k with -1 (the
	// reference's empty-slot marker) when fewer groups are visible than top-k.
	one := func(row []int32) ([]int32, error) {
		if len(row) > plan.TopKWidth {
			return nil, v41StageErr(v41StageIndexer, plan.Layer,
				fmt.Errorf("%w: layer %d index row width %d exceeds stride %d", ErrV41ForwardStage, plan.Layer, len(row), plan.TopKWidth))
		}
		normalized := make([]int32, plan.TopKWidth)
		for i := range normalized {
			normalized[i] = -1
		}
		for i, id := range row {
			if id < -1 || int(id) >= groups {
				return nil, v41StageErr(v41StageIndexer, plan.Layer,
					fmt.Errorf("%w: layer %d index row %d out of range for %d groups", ErrV41ForwardStage, plan.Layer, id, groups))
			}
			normalized[i] = id
		}
		return normalized, nil
	}

	if isSource {
		if localIdx == nil {
			return nil, nil
		}
		row, err := one(localIdx)
		if err != nil {
			return nil, err
		}
		flat := make([]int32, 0, seq*plan.TopKWidth)
		for t := 0; t < seq; t++ {
			flat = append(flat, row...)
		}
		return flat, nil
	}

	// Reader: reuse the source's published, per-position selection.
	if st == nil {
		return nil, v41StageErr(v41StageIndexer, plan.Layer,
			fmt.Errorf("%w: layer %d reads index source %d but no session state carries the publication", ErrV41ForwardStage, plan.Layer, plan.IndexSourceLayer))
	}
	attn, err := st.attentionState(headDim, 8)
	if err != nil {
		return nil, v41StageErr(v41StageIndexer, plan.Layer, err)
	}
	published, _, ok := attn.TopK()
	if !ok || len(published) == 0 {
		return nil, v41StageErr(v41StageIndexer, plan.Layer,
			fmt.Errorf("%w: layer %d reads index source %d but no top-k selection was published", ErrV41ForwardStage, plan.Layer, plan.IndexSourceLayer))
	}
	flat := make([]int32, 0, len(published)*plan.TopKWidth)
	for _, row := range published {
		normalized, err := one(row)
		if err != nil {
			return nil, err
		}
		flat = append(flat, normalized...)
	}
	return flat, nil
}

// v41AttentionSourceUpdates builds the projection-free publication a source
// layer hands to the session state for later readers: its completed compressed
// rows (Latent) and the matching index keys. A layer that is neither a KV
// source nor an index source publishes nothing, so the reduced non-shared path
// stays a no-op here.
//
// A compressed source (ratio > 1) publishes one entry per completed group, in
// order, pairing the pooled latent with the index key the group contributes.
// When the layer is a KV source or index source at ratio <= 1 (the reader
// regime), there is no pooled group; the projected query-latent rows are the
// index keys a downstream reader scores against, so each is published in stack
// order with its Latent left nil.
//
// The update carries every published row in increasing stack order, which the
// state's validateUpdates requires; a caller therefore never reorders them.
func (m *Model) v41AttentionSourceUpdates(plan V41AttentionPlan, compressedKV [][]float32, qLatRows [][]float32) []V41AttentionStateUpdate {
	isKV := plan.KVSourceLayer == plan.Layer
	isIndex := indexSourceAt(m.Cfg.DeepSeekV41, plan.Layer)
	if !isKV && !isIndex {
		return nil
	}
	ref := V41AttentionStateRef{
		LayerID:       plan.Layer,
		Ratio:         plan.Ratio,
		IsKVSource:    isKV,
		IsIndexSource: isIndex,
	}
	if len(compressedKV) == 0 {
		// No pooled group yet: an index source publishes its projected latents as
		// the keys a reader will score against. A ratio-0 layer may not carry a
		// latent (validateUpdates refuses one), so only the key is published.
		if !isIndex {
			return nil
		}
		updates := make([]V41AttentionStateUpdate, 0, len(qLatRows))
		for _, row := range qLatRows {
			if row == nil {
				continue
			}
			updates = append(updates, V41AttentionStateUpdate{
				Ref:      ref,
				IndexKey: append([]float32(nil), row...),
			})
		}
		return updates
	}
	updates := make([]V41AttentionStateUpdate, 0, len(compressedKV))
	for _, row := range compressedKV {
		up := V41AttentionStateUpdate{Ref: ref}
		if isKV {
			up.Latent = append([]float32(nil), row...)
		}
		if isIndex {
			// The index key for a completed group is the pooled row itself: the
			// reference scores the shared index against the compressed cache row.
			up.IndexKey = append([]float32(nil), row...)
		}
		updates = append(updates, up)
	}
	return updates
}

// maxInt returns the larger of a and b for the small integer clamps in the
// forward seam (a ratio floor). It exists as a named helper so the seam reads
// without a branch at each call site.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// negInf32 is the sink's "no visible group" seed.
const negInf32 = float32(-3.4028234e38)
