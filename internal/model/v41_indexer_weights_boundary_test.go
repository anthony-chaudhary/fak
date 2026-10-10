package model

import (
	"errors"
	"math"
	"testing"
)

// The real projected-weight call site must deliver the fixed BF16 operand to
// scoring and refuse a BF16 overflow before any scoring or callback mutation.
// This is a source-contract regression, not a runtime or hardware receipt.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexerWeightsBF16ScoreBoundary(t *testing.T) {
	t.Parallel()
	for _, selected := range []bool{false, true} {
		name := "host-projection"
		if selected {
			name = "selected-projection"
		}
		t.Run(name, func(t *testing.T) {
			cfg := Config{
				HiddenSize: 1, QLoraRank: 1, HeadDim: 8, IndexHeadDim: 3,
				IndexNHeads: 1, IndexTopK: 1, QKRopeHeadDim: 2,
				DeepSeekV41: &DeepSeekV41Config{
					CompressRatios: []int{2}, IndexSourceLayerIDs: []int{0},
					CandidateSourceLayerID: -1, CompressRopeTheta: 10000,
				},
			}
			manifest, raw := synthBuildRaw([]synthTensor{
				{layerName(0, "indexer.wq_b.weight"), []int{3, 1}},
				{layerName(0, "indexer.weights_proj.weight"), []int{1, 1}},
			}, func(string, func() float32) float32 { return 1 })
			m := &Model{Cfg: cfg, manifest: manifest, raw: raw}
			projected := []float32{math.Float32frombits(0x3f848000)}
			v41WriteTensorF32(t, m, layerName(0, "indexer.weights_proj.weight"), projected)
			var project v41DenseProjectionFunc
			weightCalls := 0
			if selected {
				project = func(l int, leaf string, input []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
					if leaf == "indexer.weights_proj.weight" {
						weightCalls++
						return projected, v41ProjectionHandled, nil
					}
					values, err := m.v41ProjectionRows(l, leaf, input, out, in, rows, nil)
					return values, v41ProjectionHandled, err
				}
			}
			scoreCalls := 0
			score := func(_ int, _, _, weights []float32, heads, dim, rows int) ([]float32, error) {
				scoreCalls++
				if heads != 1 || dim != 3 || rows != 1 || len(weights) != 1 || math.Float32bits(weights[0]) != 0x3f180000 {
					t.Fatalf("wrong BF16 score operand: heads=%d dim=%d rows=%d weights=%v", heads, dim, rows, weights)
				}
				return []float32{1}, nil
			}
			keys := [][]float32{{1, 0, 0}}
			ids, err := m.v41IndexRowsWithOperations(0, 0, []float32{1}, []float32{1}, keys, project, score, nil)
			if err != nil || len(ids) != 1 || ids[0] != 0 || scoreCalls != 1 || (selected && weightCalls != 1) {
				t.Fatalf("weight boundary not exercised once: ids=%v scores=%d projections=%d err=%v", ids, scoreCalls, weightCalls, err)
			}
			if math.Float32bits(projected[0]) != 0x3f848000 {
				t.Fatal("success mutated the projection callback's buffer")
			}
			projected[0] = math.MaxFloat32
			v41WriteTensorF32(t, m, layerName(0, "indexer.weights_proj.weight"), projected)
			ids, err = m.v41IndexRowsWithOperations(0, 0, []float32{1}, []float32{1}, keys, project, score, nil)
			var operation *V41ProjectionOperationError
			if ids != nil || !errors.Is(err, ErrV41ForwardStage) || !errors.As(err, &operation) || operation.Layer != 0 || operation.Leaf != "indexer.weights_proj.weight" || operation.Stage != string(v41StageIndexer) {
				t.Fatalf("weight overflow lost typed failure: ids=%v err=%v", ids, err)
			}
			if scoreCalls != 1 || (selected && weightCalls != 2) || math.Float32bits(projected[0]) != 0x7f7fffff {
				t.Fatalf("failure scored, replayed, or mutated projection storage: scores=%d projections=%d value=%08x", scoreCalls, weightCalls, math.Float32bits(projected[0]))
			}
		})
	}
}
