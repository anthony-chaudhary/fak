package model

import "fmt"

func v41IndexerProjectionLeaf(leaf string) bool {
	return leaf == "indexer.wq_b.weight" || leaf == "indexer.wk.weight" || leaf == "indexer.weights_proj.weight"
}

func (m *Model) v41IndexKeys(l int, rows [][]float32, project v41DenseProjectionFunc) ([][]float32, error) {
	return m.v41IndexKeysWithOperations(l, rows, project, nil)
}

func (m *Model) v41IndexKeysWithOperations(l int, rows [][]float32, project v41DenseProjectionFunc, normalize v41IndexKeyNormFunc) ([][]float32, error) {
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
		key, err = v41IndexBF16Copyback(l, "indexer.wk.weight", "projection", key)
		if err != nil {
			return nil, err
		}
		key, err = m.v41IndexKeyNorm(l, key, norm, float32(m.Cfg.RMSNormEps), normalize)
		if err != nil {
			return nil, err
		}
		// RMSNorm multiplies its learned gain in F32, then casts once.
		key, err = v41IndexBF16Copyback(l, "indexer.k_norm.weight", "normalization", key)
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
	flat := make([]float32, 0, elements)
	for _, key := range keys {
		if len(key) != dim {
			return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index key width", ErrV41ForwardStage))
		}
		flat = append(flat, key...)
	}
	q, err := m.v41ProjMatRowsWithProjection(l, "indexer.wq_b.weight", qLat, width, cfg.QLoraRank, project)
	if err != nil {
		return nil, err
	}
	// Both the BF16 linear and the reference's FP8 GEMM return BF16 in the
	// pinned BF16 execution profile. This does not emulate FP8 input quantization.
	q, err = v41IndexBF16Copyback(l, "indexer.wq_b.weight", "projection", q)
	if err != nil {
		return nil, err
	}
	if err := m.v41IndexRoPE(l, pos, q, nHeads); err != nil {
		return nil, err
	}
	weights, err := m.v41ProjMatRowsWithProjection(l, "indexer.weights_proj.weight", hidden, nHeads, cfg.HiddenSize, project)
	if err != nil {
		return nil, err
	}
	weights, err = v41IndexWeightsBF16(l, weights, dim, nHeads)
	if err != nil {
		return nil, err
	}
	topKBlocks, blockSize := 0, 0
	if d41.CandidateSourceLayerID == l {
		topKBlocks, blockSize = d41.CandidateTopKBlocks, d41.CandidateBlockSize
	}
	opened := m.v41NowNanos()
	defer m.v41NoteIndexerScoring(opened)
	return v41IndexRowsWithScore(l, q, flat, weights, nHeads, dim, len(keys), topKBlocks, blockSize, cfg.IndexTopK, score, health)
}

// v41CompressedPublicationRoPE runs only on a producer's fresh rows, after the
// index key has been projected and normalized from the unrotated latent. The
// position is the first absolute token of the completed group. Readers and
// restored publications already contain this rotation and must not call it.
// Normalized latent and index key are widened BF16: the reference's in-place
// rotary copyback rounds their tails back to BF16. Reference FP4 quantization
// remains separate work.
func (m *Model) v41CompressedPublicationRoPE(l, pos int, latent, indexKey []float32) error {
	cfg := m.Cfg
	rd := cfg.QKRopeHeadDim
	if pos < 0 || rd <= 0 || rd%2 != 0 || rd > cfg.HeadDim || len(latent) != cfg.HeadDim {
		return v41StageErr(v41StageCompress, l, fmt.Errorf("%w: invalid compressed rotary geometry or position", ErrV41ForwardStage))
	}
	if indexKey != nil && (rd > cfg.IndexHeadDim || len(indexKey) != cfg.IndexHeadDim) {
		return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: index rotary width %d exceeds or disagrees with key width %d", ErrV41ForwardStage, rd, cfg.IndexHeadDim))
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
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
	cfg := m.Cfg
	dim, rd := cfg.IndexHeadDim, cfg.QKRopeHeadDim
	width, valid := checkedMulInt(heads, dim)
	if pos < 0 || heads <= 0 || dim <= 0 || !valid || len(values) != width || rd <= 0 || rd%2 != 0 || rd > dim {
		return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: invalid index rotary geometry or position", ErrV41ForwardStage))
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
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
