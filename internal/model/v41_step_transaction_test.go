package model

import (
	"errors"
	"reflect"
	"testing"
)

// Estimate is unmeasured; this is a small synthetic correctness witness, not a
// throughput benchmark or a receipt for cross-session shared projection work.
// fak-test:runtime fast est=500ms lane=default
func TestV41StepTransactionInterleavedSessions(t *testing.T) {
	m := v41CompressorTestFixtureIndex(t, 128)
	var sessions, serial [2]*Session
	var states, refs [2]*v41ForwardState
	var scratch [2]v41ProjScratch
	var calls [2]int
	active := -1
	for lane, prefix := range [][]int{{1, 2, 3}, {1, 2, 3, 4}} {
		source := m.NewSession()
		t.Cleanup(source.Close)
		if len(source.Prefill(prefix)) == 0 {
			t.Fatal("prefill returned no logits")
		}
		// A restored session must rebind its own callbacks, never keep those
		// of the prefix source. Make accidental source invocation fatal.
		source.v41State().setDenseProjection(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			t.Fatal("restored transaction called prefix source callback")
			return nil, v41ProjectionDeclined, nil
		})
		snapshot := captureV41ForwardSnapshot(source.v41State())
		sessions[lane], serial[lane] = m.NewSession(), m.NewSession()
		t.Cleanup(sessions[lane].Close)
		t.Cleanup(serial[lane].Close)
		sessions[lane].v41Forward = snapshot.clone().restore()
		serial[lane].v41Forward = snapshot.clone().restore()
		states[lane], refs[lane] = sessions[lane].v41State(), serial[lane].v41State()
		if states[lane].callbackOwner != sessions[lane] || refs[lane].callbackOwner != serial[lane] {
			t.Fatal("restored callbacks do not belong to destination sessions")
		}
		states[lane].setDenseProjection(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			if active != lane {
				t.Fatalf("lane %d callback ran while lane %d owned the step", lane, active)
			}
			calls[lane]++
			return nil, v41ProjectionDeclined, nil
		})
		// Match the callback's arithmetic path in the serial reference.
		refs[lane].setDenseProjection(nil)
	}
	runPair := func(fail bool) {
		t.Helper()
		var tx [2]*v41StepTransaction
		var want [2][]float32
		var before [2]*v41ForwardSnapshot
		var registry [2]*V41AttentionState
		var pointers [2][]*V41AttentionState
		sentinel := errors.New("lane one selected head failure")
		for lane := range states {
			// Populate the next overwritten slot so rollback must restore actual
			// bytes even with this bounded, short-prefix fixture.
			for _, st := range []*v41ForwardState{states[lane], refs[lane]} {
				for _, layer := range st.layers {
					slot := layer.nextWindowPos % layer.windowSize
					layer.window[slot] = append([]float32(nil), layer.window[0]...)
				}
			}
			if fail && lane == 1 {
				states[lane].finalNorm = func([]float32) ([]float32, error) { return nil, sentinel }
			}
			before[lane] = captureV41ForwardSnapshot(states[lane])
			registry[lane] = states[lane].attn
			pointers[lane] = append([]*V41AttentionState(nil), states[lane].layers...)
			if !fail || lane == 0 {
				var err error
				want[lane], _, err = m.forwardV41Step(6+lane, refs[lane], nil)
				if err != nil {
					t.Fatal(err)
				}
			}
			active = lane
			var err error
			tx[lane], err = m.beginV41Step(6+lane, states[lane], &scratch[lane])
			if err != nil {
				t.Fatal(err)
			}
			defer tx[lane].abort()
		}
		for layer := 0; layer < m.Cfg.NumLayers; layer++ {
			for lane := range tx {
				active = lane
				if err := tx[lane].advanceLayer(); err != nil {
					t.Fatal(err)
				}
			}
		}
		if fail && (registry[1].kvPublishedEnd[0] != before[1].attn.kvPublishedEnd[0]+1 || registry[1].indexPublishedEnd[0] != before[1].attn.indexPublishedEnd[0]+1) {
			t.Fatal("failure witness did not mutate both source publication ends")
		}
		// Neither history may commit before its own head succeeds.
		for lane := range states {
			if !reflect.DeepEqual(states[lane].history, before[lane].history) {
				t.Fatal("premature commit")
			}
		}
		active = 0
		got, err := tx[0].finalize()
		if err != nil || !reflect.DeepEqual(got, want[0]) || !tx[0].stats.Committed {
			t.Fatalf("lane zero logits/commit: %v", err)
		}
		committed := captureV41ForwardSnapshot(states[0])
		active = 1
		if fail {
			panicked := recoverError(func() { _, err = tx[1].finalize() })
			if !errors.Is(panicked, sentinel) {
				t.Fatalf("head failure cause lost: returned=%v panic=%v", err, panicked)
			}
			if tx[1].stats.Committed || tx[1].stats.LayersRolledBack != m.Cfg.NumLayers || !tx[1].closed {
				t.Fatalf("failed lane stats: %+v", tx[1].stats)
			}
			if states[1].attn != registry[1] || !reflect.DeepEqual(captureV41ForwardSnapshot(states[1]), before[1]) {
				t.Fatal("failed lane ring/publication/history rollback differs")
			}
			for layer, pointer := range pointers[1] {
				if states[1].layers[layer] != pointer {
					t.Fatal("rollback replaced retained layer identity")
				}
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(states[0]), committed) {
				t.Fatal("lane one rollback changed committed lane zero")
			}
		} else {
			got, err = tx[1].finalize()
			if err != nil || !reflect.DeepEqual(got, want[1]) || !tx[1].stats.Committed {
				t.Fatalf("lane one logits/commit: %v", err)
			}
		}
		for lane := range states {
			if !fail || lane == 0 {
				if !reflect.DeepEqual(captureV41ForwardSnapshot(states[lane]), captureV41ForwardSnapshot(refs[lane])) {
					t.Fatalf("lane %d differs from serial continuation", lane)
				}
			}
			if scratch[lane].denseProjection() != nil || scratch[lane].mhcCarry != nil {
				t.Fatal("transaction retained session scratch callbacks")
			}
			tx[lane].abort() // no duplicate rollback and no committed-token undo
		}
	}
	runPair(false)
	runPair(true)
	if calls[0] == 0 || calls[1] == 0 {
		t.Fatal("session callback ownership witness was vacuous")
	}
}
