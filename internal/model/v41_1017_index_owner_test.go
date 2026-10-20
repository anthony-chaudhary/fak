package model

import (
	"reflect"
	"testing"
)

// The 1017 reference rebinds shared_attn.index_k even without a new latent.
// FAK's registry instead resolves persistent publications by source layer. Two
// owners distinguish retained source history from a stale last-writer pointer.
// This witnesses registry semantics only, not checkpoint or FP8 parity (#13891).
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41IndexOwnersRetainHistoryDuringIncompleteGroups(t *testing.T) {
	t.Parallel()
	newState := func() *V41AttentionState {
		t.Helper()
		state, err := NewV41AttentionState(4, 2)
		if err != nil {
			t.Fatal(err)
		}
		return state
	}
	registry := newState()
	owners := []*V41AttentionState{newState(), newState()}
	m := &Model{Cfg: Config{NumLayers: 4, DeepSeekV41: &DeepSeekV41Config{
		CompressRatios:      []int{2, 2, 2, 2},
		KVSourceLayerIDs:    []int{0, 2},
		IndexSourceLayerIDs: []int{0, 1, 2, 3},
	}}}
	keys := [][]float32{{1, 2, 3, 4}, {9, 8, 7, 6}}
	projections := [2]int{}
	appendRow := func(owner, pos int) {
		t.Helper()
		layer := 2 * owner
		pool, err := NewV41CompressorPool(2, 4)
		if err != nil {
			t.Fatal(err)
		}
		project := func(input []float32) ([]float32, error) {
			return append([]float32(nil), input...), nil
		}
		score := func([]float32) ([]float32, error) { return make([]float32, 4), nil }
		_, _, emitted, _, _, err := owners[owner].appendCompressorSourcePublication(
			layer, 2, pos, []float32{1, 2, 3, 4}, pool, project, score, []float32{1, 1, 1, 1}, 1e-5,
			v41CompressorPublication{
				registry: registry,
				ref:      V41AttentionStateRef{LayerID: layer, Ratio: 2, IsKVSource: true, IsIndexSource: true},
				projectIndex: func([]float32) ([]float32, error) {
					projections[owner]++
					return append([]float32(nil), keys[owner]...), nil
				},
			})
		if err != nil || emitted != (pos == 1) {
			t.Fatalf("owner %d position %d: emitted=%v error=%v", layer, pos, emitted, err)
		}
	}
	checkReader := func(owner, positions int) {
		t.Helper()
		layer := 2*owner + 1
		plan, err := m.v41AttentionPlan(layer)
		if err != nil {
			t.Fatal(err)
		}
		rows, got, err := m.v41CompressedReaderStepRows(plan, registry, positions)
		if err != nil {
			t.Fatalf("reader %d: %v", layer, err)
		}
		if positions == 1 {
			if len(rows) != 0 || len(got) != 0 {
				t.Fatal("first incomplete group fabricated a publication")
			}
			return
		}
		if len(rows) != 1 || !reflect.DeepEqual(got, [][]float32{keys[owner]}) {
			t.Fatalf("reader %d resolved wrong owner history: %v", layer, got)
		}
	}
	for pos := 0; pos < 3; pos++ {
		// Publish both owners before reading either: a last-writer registry would
		// incorrectly feed owner 2's keys to reader 1, including at position 2.
		for owner := range owners {
			appendRow(owner, pos)
		}
		for owner := range owners {
			checkReader(owner, pos+1)
		}
	}
	if projections != [2]int{1, 1} {
		t.Fatalf("incomplete group reprojected committed keys: %v", projections)
	}
	// Public snapshots stay owned even when readers use the borrowed step view.
	copyKeys, _ := registry.IndexKeys(0)
	copyKeys[0][0] = -999
	checkReader(0, 3)
}
