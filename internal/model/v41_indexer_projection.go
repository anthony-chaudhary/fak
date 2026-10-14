package model

import (
	"fmt"
	"math"
)

func v41IndexerProjectionLeaf(leaf string) bool {
	return leaf == "indexer.wq_b.weight" || leaf == "indexer.wk.weight" || leaf == "indexer.weights_proj.weight"
}

func (m *Model) v41IndexKeys(l int, rows [][]float32, project v41DenseProjectionFunc) ([][]float32, error) {
	return m.v41IndexKeysWithOperations(l, rows, project, nil)
}

func (m *Model) v41IndexKeysWithOperations(l int, rows [][]float32, project v41DenseProjectionFunc, normalize v41IndexKeyNormFunc) ([][]float32, error) {
	return m.v41IndexKeysPublished(l, rows, project, normalize, nil)
}

func (m *Model) v41IndexKeysPublished(l int, rows [][]float32, project v41DenseProjectionFunc, normalize v41IndexKeyNormFunc, publish v41BF16BoundaryFunc) ([][]float32, error) {
	dim := m.Cfg.IndexHeadDim
	if dim <= 0 {
		return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index head width", ErrV41ForwardStage))
	}
	keys := make([][]float32, len(rows))
	if len(rows) == 0 {
		return keys, nil
	}
	norm := m.tensor(layerName(l, "indexer.k_norm.weight"))
	for i, row := range rows {
		key, err := m.v41ProjMatRowsWithProjection(l, "indexer.wk.weight", row, dim, len(row), project)
		if err != nil {
			return nil, err
		}
		// The reference's BF16 wk output is widened for RMSNorm arithmetic.
		key, err = v41IndexBF16Publish(l, "indexer.wk.weight", "projection", key, publish)
		if err != nil {
			return nil, err
		}
		key, err = m.v41IndexKeyNorm(l, key, norm, float32(m.Cfg.RMSNormEps), normalize)
		if err != nil {
			return nil, err
		}
		// RMSNorm multiplies its learned gain in F32, then casts once.
		key, err = v41IndexBF16Publish(l, "indexer.k_norm.weight", "normalization", key, publish)
		if err != nil {
			return nil, err
		}
		keys[i] = key
	}
	return keys, nil
}

func (m *Model) v41IndexRowsProjected(l, pos int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc) ([]int32, error) {
	return m.v41IndexRowsWithOperations(l, pos, qLat, hidden, keys, project, nil, nil)
}

func (m *Model) v41IndexRowsWithOperations(l, pos int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc) ([]int32, error) {
	return m.v41IndexRowsWithDeviceOps(l, pos, qLat, hidden, keys, project, score, health, nil)
}

// v41IndexRowsWithDeviceOps also offers the row selection to the session's
// device index-select seam when ops binds one (Ref #13668).
func (m *Model) v41IndexRowsWithDeviceOps(l, pos int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc, ops *v41DeviceGatherOps, views ...*v41IndexerKeyView) ([]int32, error) {
	return m.v41IndexRowsWithPreparedKeys(l, pos, qLat, hidden, keys, project, score, health, ops, nil, views...)
}

// v41ColdIndexerKeyPanel snapshots a completed layer's stable keys. A malformed
// panel declines reuse so the ordinary per-query validator still reports a bad
// row only when that row becomes visible. Finiteness is checked by the scorer.
func v41ColdIndexerKeyPanel(keys [][]float32, dim int) []float32 {
	elements, valid := checkedMulInt(len(keys), dim)
	if dim <= 0 || !valid || elements == 0 {
		return nil
	}
	for _, key := range keys {
		if len(key) != dim {
			return nil
		}
	}
	panel := make([]float32, elements)
	for i, key := range keys {
		copy(panel[i*dim:(i+1)*dim], key)
	}
	return panel
}

// Only the scalar scorer borrows a prepared prefix. Arbitrary callbacks retain
// their independent mutable flat input, including when a panel is supplied.
// No prepared storage is retained in continuation or publication state.
func (m *Model) v41IndexRowsWithPreparedKeys(l, pos int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc, ops *v41DeviceGatherOps, prepared []float32, views ...*v41IndexerKeyView) ([]int32, error) {
	ordinary := v41IndexerScoreBinding{ordinary: score}
	return m.v41IndexRowsWithPreparedBinding(l, pos, qLat, hidden, keys, project, &ordinary, health, ops, prepared, nil, views...)
}

// Native borrowing requires the same immutable binding that supplies the
// invoked scorer. Ordinary callers cannot inherit this synchronous authority.
func (m *Model) v41IndexRowsWithPreparedBinding(l, pos int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc, binding *v41IndexerScoreBinding, health v41IndexerScoreHealthFunc, ops *v41DeviceGatherOps, prepared []float32, borrow *v41IndexerScoreBorrow, views ...*v41IndexerKeyView) ([]int32, error) {
	score := binding.scoreFunc()
	if borrow == nil || borrow.binding != binding || !borrow.valid() {
		borrow = nil
	}
	d41 := m.Cfg.DeepSeekV41
	if !indexSourceAt(d41, l) {
		return nil, nil
	}
	cfg := m.Cfg
	nHeads, dim := cfg.IndexNHeads, cfg.IndexHeadDim
	width, valid := checkedMulInt(nHeads, dim)
	if nHeads <= 0 || dim <= 0 || !valid {
		return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index geometry", ErrV41ForwardStage))
	}
	elements, valid := checkedMulInt(len(keys), dim)
	if !valid {
		return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: index key geometry overflow", ErrV41ForwardStage))
	}
	for _, key := range keys {
		if len(key) != dim {
			return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index key width", ErrV41ForwardStage))
		}
	}
	resident := false
	if score != nil && len(views) == 1 {
		if selected := ops.residentIndexerScore(m, l, pos, keys, views[0], score); selected != nil {
			score, resident = selected, true
			borrow = nil
		}
	}
	var flat []float32
	if !resident && (score == nil || borrow != nil) && prepared != nil && len(prepared) == elements {
		flat = prepared[:elements:elements]
	} else if !resident {
		borrow = nil
		flat = make([]float32, 0, elements)
		for _, key := range keys {
			flat = append(flat, key...)
		}
	}
	q, err := m.v41ProjMatRowsWithProjection(l, "indexer.wq_b.weight", qLat, width, cfg.QLoraRank, project)
	if err != nil {
		return nil, err
	}
	// Both the BF16 linear and the reference's FP8 GEMM return BF16 in the
	// pinned BF16 execution profile. This does not emulate FP8 input quantization.
	publish := ops.bf16PublishFunc()
	q, err = v41IndexBF16Publish(l, "indexer.wq_b.weight", "projection", q, publish)
	if err != nil {
		return nil, err
	}
	if err := m.v41IndexRoPEPublished(l, pos, q, nHeads, ops.indexRoPEFunc(), publish); err != nil {
		return nil, err
	}
	weights, err := m.v41ProjMatRowsWithProjection(l, "indexer.weights_proj.weight", hidden, nHeads, cfg.HiddenSize, project)
	if err != nil {
		return nil, err
	}
	weights, err = v41IndexWeightsBF16Device(l, weights, dim, nHeads, publish, ops.indexWeightScaleFunc())
	if err != nil {
		return nil, err
	}
	topKBlocks, blockSize := 0, 0
	if d41.CandidateSourceLayerID == l {
		topKBlocks, blockSize = d41.CandidateTopKBlocks, d41.CandidateBlockSize
	}
	opened := m.v41NowNanos()
	defer m.v41NoteIndexerScoring(opened)
	return v41IndexRowsSelectedWithBorrow(l, q, flat, weights, nHeads, dim, len(keys), topKBlocks, blockSize, cfg.IndexTopK, score, health, ops.indexSelectFunc(), borrow)
}

// v41CompressedPublicationRoPE runs only on a producer's fresh rows, after the
// index key has been projected and normalized from the unrotated latent. The
// position is the first absolute token of the completed group. Readers and
// restored publications already contain this rotation and must not call it.
// Normalized latent and index key are widened BF16: the reference's in-place
// rotary copyback rounds their tails back to BF16. Reference FP4 quantization
// remains separate work.
func (m *Model) v41CompressedPublicationRoPE(l, pos int, latent, indexKey []float32) error {
	return m.v41CompressedPublicationRoPEWithDevice(l, pos, latent, indexKey, nil)
}

func (m *Model) v41CompressedPublicationRoPEWithDevice(l, pos int, latent, indexKey []float32, rotate v41CompressedPublicationRoPEFunc) error {
	cfg := m.Cfg
	rd := cfg.QKRopeHeadDim
	if pos < 0 || rd <= 0 || rd%2 != 0 || rd > cfg.HeadDim || len(latent) != cfg.HeadDim {
		return v41StageErr(v41StageCompress, l, fmt.Errorf("%w: invalid compressed rotary geometry or position", ErrV41ForwardStage))
	}
	if indexKey != nil && (rd > cfg.IndexHeadDim || len(indexKey) != cfg.IndexHeadDim) {
		return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: index rotary width %d exceeds or disagrees with key width %d", ErrV41ForwardStage, rd, cfg.IndexHeadDim))
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	if rotate != nil {
		rows := 1
		if indexKey != nil {
			rows = 2
		}
		// Only consumed tails enter the operation. Prefix bits (including
		// non-finite payloads) retain their historical untouched semantics.
		tails := make([]float32, 0, rows*rd)
		if indexKey != nil {
			tails = append(tails, indexKey[len(indexKey)-rd:]...)
		}
		tails = append(tails, latent[len(latent)-rd:]...)
		finished, err := rotate(l, tails, append([]float32(nil), cos...), append([]float32(nil), sin...), rows, rd)
		if err == nil && len(finished) != len(tails) {
			err = errV41TailRoPEResult
		}
		if err != nil {
			return &V41TailRoPEOperationError{Layer: l, Cause: v41StageErr(v41StageCompress, l, err)}
		}
		// The captured session has completed both checked BF16 conversions
		// on owned storage. Publish neither tail until that whole call succeeds.
		if indexKey != nil {
			copy(indexKey[len(indexKey)-rd:], finished[:rd])
		}
		copy(latent[len(latent)-rd:], finished[(rows-1)*rd:])
		return nil
	}
	if indexKey != nil {
		applyRopeTailInterleaved(indexKey, cos, sin, rd)
		tail, err := v41IndexBF16Copyback(l, "indexer.k_norm.weight", "rotary", indexKey[len(indexKey)-rd:])
		if err != nil {
			return err
		}
		copy(indexKey[len(indexKey)-rd:], tail)
	}
	applyRopeTailInterleaved(latent, cos, sin, rd)
	for i := len(latent) - rd; i < len(latent); i++ {
		latent[i] = v41RoundBF16(latent[i])
	}
	return nil
}

func (m *Model) v41IndexRoPE(l, pos int, values []float32, heads int) error {
	return m.v41IndexRoPEWithDevice(l, pos, values, heads, nil)
}

func (m *Model) v41IndexRoPEWithDevice(l, pos int, values []float32, heads int, rotate v41IndexRoPEFunc) error {
	return m.v41IndexRoPEPublished(l, pos, values, heads, rotate, nil)
}

func (m *Model) v41IndexRoPEPublished(l, pos int, values []float32, heads int, rotate v41IndexRoPEFunc, publish v41BF16BoundaryFunc) error {
	cfg := m.Cfg
	dim, rd := cfg.IndexHeadDim, cfg.QKRopeHeadDim
	width, valid := checkedMulInt(heads, dim)
	if pos < 0 || heads <= 0 || dim <= 0 || !valid || len(values) != width || rd <= 0 || rd%2 != 0 || rd > dim {
		return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index rotary geometry or position", ErrV41ForwardStage))
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	if rotate != nil {
		fail := func(cause error) error {
			return &V41ProjectionOperationError{Layer: l, Leaf: "indexer.wq_b.weight", Stage: string(v41StageIndexer), Cause: v41StageErr(v41StageIndexer, l, cause)}
		}
		rotated, err := rotate(l, append([]float32(nil), values...), append([]float32(nil), cos...), append([]float32(nil), sin...), heads, dim, rd)
		if err != nil {
			return fail(err)
		}
		if len(rotated) != width {
			return fail(errV41TailRoPEResult)
		}
		owned := append([]float32(nil), rotated...)
		for i, value := range owned {
			if !finite32(value) || (i%dim < dim-rd && math.Float32bits(value) != math.Float32bits(values[i])) {
				return fail(errV41TailRoPEResult)
			}
		}
		if err := v41IndexTailsBF16(l, owned, heads, dim, rd, publish); err != nil {
			return fail(err)
		}
		copy(values, owned)
		return nil
	}

	if publish == nil {
		for h := 0; h < heads; h++ {
			head := values[h*dim : (h+1)*dim]
			applyRopeTailInterleaved(head, cos, sin, rd)
			tail, err := v41IndexBF16Copyback(l, "indexer.wq_b.weight", "rotary", head[dim-rd:])
			if err != nil {
				return err
			}
			copy(head[dim-rd:], tail)
		}
		return nil
	}
	for h := 0; h < heads; h++ {
		applyRopeTailInterleaved(values[h*dim:(h+1)*dim], cos, sin, rd)
	}
	return v41IndexTailsBF16(l, values, heads, dim, rd, publish)
}

// v41IndexTailsBF16 publishes every head's rotary tail as one row and writes the
// results back only after the whole publication succeeds. Without a bound
// publication each tail keeps its own host copy, as before.
func v41IndexTailsBF16(l int, values []float32, heads, dim, rd int, publish v41BF16BoundaryFunc) error {
	if publish == nil {
		for h := 0; h < heads; h++ {
			tail := values[(h+1)*dim-rd : (h+1)*dim]
			rounded, err := v41IndexBF16Copyback(l, "indexer.wq_b.weight", "rotary", tail)
			if err != nil {
				return err
			}
			copy(tail, rounded)
		}
		return nil
	}
	tails := make([]float32, 0, heads*rd)
	for h := 0; h < heads; h++ {
		tails = append(tails, values[(h+1)*dim-rd:(h+1)*dim]...)
	}
	rounded, err := v41IndexBF16Publish(l, "indexer.wq_b.weight", "rotary", tails, publish)
	if err != nil {
		return err
	}
	for h := 0; h < heads; h++ {
		copy(values[(h+1)*dim-rd:(h+1)*dim], rounded[h*rd:(h+1)*rd])
	}
	return nil
}

// v41IndexBF16Publish is v41IndexBF16Copyback routed through the session's
// ordinary device publication when one is bound.
func v41IndexBF16Publish(l int, leaf, phase string, values []float32, publish v41BF16BoundaryFunc) ([]float32, error) {
	return v41PublishBF16(l, leaf+" "+phase, values, publish, func(values []float32) ([]float32, error) {
		return v41IndexBF16Copyback(l, leaf, phase, values)
	})
}

// v41IndexBF16Copyback owns its result so a projection or norm callback's
// storage is never changed by the dtype boundary. Validate before bit rounding:
// a NaN payload can otherwise wrap into a finite value, and finite F32 can
// overflow BF16. Neither failure can be repaired by replaying token history.
func v41IndexBF16Copyback(l int, leaf, phase string, values []float32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, value := range values {
		if finite32(value) {
			out[i] = v41RoundBF16(value)
			if finite32(out[i]) {
				continue
			}
		}
		return nil, &V41ProjectionOperationError{
			Layer: l, Leaf: leaf, Stage: string(v41StageIndexer),
			Cause: v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: %s BF16 %s value[%d] is non-finite or overflows", ErrV41ForwardStage, leaf, phase, i)),
		}
	}
	return out, nil
}
