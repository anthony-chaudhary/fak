package model

import (
	"errors"
	"reflect"
	"testing"
)

// The estimates below are unmeasured. These authored witnesses establish the
// rollback staging contract, not measured allocation counts or decode speed.
// fak-test:runtime fast est=20ms lane=default
func TestV41StepRegistryJournalBoundAndOwnership(t *testing.T) {
	for _, history := range []int{1, 128} {
		state, err := NewV41AttentionState(2, 8)
		if err != nil {
			t.Fatal(err)
		}
		for layer := 0; layer < 3; layer++ {
			for row := 0; row < history; row++ {
				state.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: layer, Ratio: 2, IsKVSource: true, IsIndexSource: true}, Latent: []float32{float32(row), 1}, IndexKey: []float32{2, float32(row)}}})
			}
		}
		topk := make([][]int32, history)
		for i := range topk {
			topk[i] = []int32{int32(i)}
		}
		if err := state.PublishTopK(2, topk); err != nil {
			t.Fatal(err)
		}
		if err := state.PublishCandidates(2, []bool{true, false}); err != nil {
			t.Fatal(err)
		}
		before := state.clone()
		oldTopK, oldCandidates := state.topk, state.candidates
		kvAlias, indexAlias := state.kvPublications, state.indexPublications
		first := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
		kvFirst, indexFirst := kvAlias[first], indexAlias[first]
		undo := stageV41StepRegistry(state, 3)
		if len(undo.kv) != 3 || cap(undo.kv) != 3 || len(undo.index) != 3 || cap(undo.index) != 3 {
			t.Fatal("journal extent depends on retained prefix")
		}
		if &undo.topk[0][0] != &oldTopK[0][0] || &undo.candidates[0] != &oldCandidates[0] {
			t.Fatal("journal cloned immutable selection payload")
		}
		for _, entries := range [][]v41StepPublicationUndo{undo.kv, undo.index} {
			for _, entry := range entries {
				if entry.rowSet || entry.row != nil || entry.key.start != history {
					t.Fatal("journal captured historical payload instead of next key")
				}
			}
		}
		for layer := 0; layer < 3; layer++ {
			state.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: layer, Ratio: 2, IsKVSource: true, IsIndexSource: true}, Latent: []float32{9, 9}, IndexKey: []float32{8, 8}}})
		}
		if err := state.PublishTopK(4, [][]int32{{9}}); err != nil {
			t.Fatal(err)
		}
		if err := state.PublishCandidates(4, []bool{false}); err != nil {
			t.Fatal(err)
		}
		undo.restore()
		undo.restore() // outer panic rollback may follow an explicit error rollback
		if !reflect.DeepEqual(state, before) || len(kvAlias) != 3*history || len(indexAlias) != 3*history {
			t.Fatal("journal did not restore in-place registry content")
		}
		if &state.kvPublications[first][0] != &kvFirst[0] || &state.indexPublications[first][0] != &indexFirst[0] || &state.topk[0][0] != &oldTopK[0][0] || &state.candidates[0] != &oldCandidates[0] {
			t.Fatal("rollback replaced retained payload ownership")
		}
		// Public reads and external snapshots must remain deep copies even though
		// the internal transactional journal now retains immutable references.
		kv, _ := state.KVSourceRows(0)
		keys, _ := state.IndexKeys(0)
		selections, _, _ := state.TopK()
		candidates, _, _ := state.Candidates()
		kv[0][0], keys[0][0], selections[0][0], candidates[0] = 99, 99, 99, false
		cloned := state.clone()
		cloned.kvPublications[first][0] = 77
		cloned.indexPublications[first][0] = 77
		cloned.topk[0][0] = 77
		cloned.candidates[0] = false
		if !reflect.DeepEqual(state, before) {
			t.Fatal("public reader or external snapshot aliases retained registry")
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41StepRegistryJournalPresenceSemantics(t *testing.T) {
	for _, presentEnd := range []bool{false, true} {
		for _, presentRow := range []bool{false, true} {
			state, _ := NewV41AttentionState(1, 8)
			key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
			original := []float32{3}
			if presentEnd {
				state.kvPublishedEnd[0] = 0
				state.indexPublishedEnd[0] = 0
			}
			if presentRow {
				state.kvPublications[key] = original
				state.indexPublications[key] = original
			}
			before := state.clone()
			undo := stageV41StepRegistry(state, 1)
			state.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0, Ratio: 2, IsKVSource: true, IsIndexSource: true}, Latent: []float32{4}, IndexKey: []float32{5}}})
			undo.restore()
			_, kvEnd := state.kvPublishedEnd[0]
			_, indexEnd := state.indexPublishedEnd[0]
			_, kvRow := state.kvPublications[key]
			_, indexRow := state.indexPublications[key]
			if kvEnd != presentEnd || indexEnd != presentEnd || kvRow != presentRow || indexRow != presentRow || !reflect.DeepEqual(state, before) {
				t.Fatal("absent versus present-zero key/cursor semantics lost")
			}
			if presentRow && (&state.kvPublications[key][0] != &original[0] || &state.indexPublications[key][0] != &original[0]) {
				t.Fatal("preexisting touched row was cloned")
			}
		}
	}
	// No publication is legal on nil maps, but staging/restoring untouched lazy
	// state must preserve nil rather than eagerly allocating maps or selections.
	state := &V41AttentionState{}
	undo := stageV41StepRegistry(state, 2)
	undo.restore()
	if !reflect.DeepEqual(state, &V41AttentionState{}) {
		t.Fatal("untouched lazy state changed")
	}
}

// fak-test:runtime medium est=10s lane=default
func TestV41StepRegistryLiveFailureRestoration(t *testing.T) {
	for _, scenario := range []struct {
		name   string
		prefix int
	}{
		{"returned late layer", 3}, {"panic late layer", 3}, {"head failure", 3}, {"incomplete group", 4},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			m := v41CompressorTestFixture(t)
			prefix := []int{1, 2, 3, 4}[:scenario.prefix]
			st := v41IncrementalState(t, m, prefix)
			registry := st.attn
			if registry == nil {
				t.Fatal("role fixture has no registry")
			}
			if err := registry.PublishCandidates(2, []bool{true, false}); err != nil {
				t.Fatal(err)
			}
			oldTopK, oldCandidates := registry.topk, registry.candidates
			kvAlias, indexAlias := registry.kvPublications, registry.indexPublications
			oldKVEnd, oldIndexEnd := registry.kvPublishedEnd[0], registry.indexPublishedEnd[0]
			if oldKVEnd == 0 || oldKVEnd != oldIndexEnd || len(registry.topk) == 0 || len(registry.topk[0]) == 0 {
				t.Fatal("fixture did not seed paired publications and top-k")
			}
			key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
			oldKV, oldIndex := kvAlias[key], indexAlias[key]
			sentinel := errors.New("selected journal rollback fault")
			sawPublication := false
			// Source expert execution follows compression publication and top-k
			// publication, so this observes real live mutation before the later fault.
			st.expertGateUp = func(layer int, _ string, _ []float32) ([]float32, v41ExpertGateUpOutcome, error) {
				if layer == 0 {
					expected := oldKVEnd
					if scenario.prefix%2 == 1 {
						expected++
					}
					if registry.kvPublishedEnd[0] != expected || registry.indexPublishedEnd[0] != expected {
						t.Fatal("fault witness did not observe expected source publication")
					}
					if len(registry.topk) != 1 || &registry.topk[0][0] == &oldTopK[0][0] {
						t.Fatal("source did not replace top-k before fault")
					}
					sawPublication = true
				}
				return nil, v41GateUpDeclined, nil
			}
			switch scenario.name {
			case "returned late layer", "incomplete group":
				st.layers[1].nextWindowPos += 7
			case "panic late layer":
				st.denseProjection = func(layer int, _ string, _ []float32, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
					if layer == 1 {
						panic(sentinel)
					}
					return nil, v41ProjectionDeclined, nil
				}
			case "head failure":
				st.finalNorm = func([]float32) ([]float32, error) { return nil, sentinel }
			}
			before := captureV41ForwardSnapshot(st)
			var returned error
			var stats v41StepStats
			panicked := recoverError(func() { _, stats, returned = m.forwardV41Step(5, st, &v41ProjScratch{}) })
			if scenario.name == "returned late layer" || scenario.name == "incomplete group" {
				if returned == nil || panicked != nil || stats.Committed {
					t.Fatalf("returned=%v panic=%v stats=%+v", returned, panicked, stats)
				}
			} else if !errors.Is(panicked, sentinel) {
				t.Fatalf("panic lost cause: %v", panicked)
			}
			if !sawPublication || st.attn != registry || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
				t.Fatal("failure did not restore complete continuation state")
			}
			if registry.kvPublishedEnd[0] != oldKVEnd || registry.indexPublishedEnd[0] != oldIndexEnd || len(kvAlias) != len(registry.kvPublications) || len(indexAlias) != len(registry.indexPublications) {
				t.Fatal("registry maps/cursors were replaced or partially published")
			}
			if &registry.kvPublications[key][0] != &oldKV[0] || &registry.indexPublications[key][0] != &oldIndex[0] || &registry.topk[0][0] != &oldTopK[0][0] || &registry.candidates[0] != &oldCandidates[0] {
				t.Fatal("rollback replaced retained publication/selection payload")
			}
		})
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41StepRegistryLiveSuccess(t *testing.T) {
	m := v41CompressorTestFixture(t)
	st := v41IncrementalState(t, m, []int{1, 2, 3})
	registry := st.attn
	key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
	oldKV, oldIndex := registry.kvPublications[key], registry.indexPublications[key]
	beforeKV, beforeIndex := append([]float32(nil), oldKV...), append([]float32(nil), oldIndex...)
	beforeEnd := registry.kvPublishedEnd[0]
	if _, stats, err := m.forwardV41Step(4, st, &v41ProjScratch{}); err != nil || !stats.Committed {
		t.Fatalf("step=%+v err=%v", stats, err)
	}
	if st.attn != registry || len(st.history) != 4 || registry.kvPublishedEnd[0] != beforeEnd+1 || registry.indexPublishedEnd[0] != beforeEnd+1 {
		t.Fatal("successful source append not committed once")
	}
	if &registry.kvPublications[key][0] != &oldKV[0] || &registry.indexPublications[key][0] != &oldIndex[0] || !reflect.DeepEqual(oldKV, beforeKV) || !reflect.DeepEqual(oldIndex, beforeIndex) {
		t.Fatal("successful append changed retained publication")
	}
}
