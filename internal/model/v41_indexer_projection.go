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
		key, err = m.v41IndexKeyNorm(l, key, norm, float32(m.Cfg.RMSNormEps), normalize)
		if err != nil {
			return nil, err
		}
		keys[i] = key
	}
	return keys, nil
}

func (m *Model) v41IndexRowsProjected(l int, qLat, hidden []float32, keys [][]float32, project v41DenseProjectionFunc) ([]int32, error) {
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
	weights, err := m.v41ProjMatRowsWithProjection(l, "indexer.weights_proj.weight", hidden, nHeads, cfg.HiddenSize, project)
	if err != nil {
		return nil, err
	}
	for h := range weights {
		weights[h] *= cfg.attnScale() * float32(1/math.Sqrt(float64(nHeads)))
	}
	topKBlocks, blockSize := 0, 0
	if d41.CandidateSourceLayerID == l {
		topKBlocks, blockSize = d41.CandidateTopKBlocks, d41.CandidateBlockSize
	}
	opened := m.v41NowNanos()
	defer m.v41NoteIndexerScoring(opened)
	pub, err := NewV41IndexerPublication(l, q, flat, weights, nHeads, dim, len(keys), topKBlocks, blockSize, cfg.IndexTopK, 0)
	if err != nil {
		return nil, v41StageErr(v41StageIndexer, l, err)
	}
	return pub.Rows(), nil
}
