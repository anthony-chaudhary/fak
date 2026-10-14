package model

import (
	"math"
	"testing"
)

func v41IndexKeyMemoCountingSession(t *testing.T, m *Model, calls *int) *Session {
	t.Helper()
	s := m.NewSession()
	t.Cleanup(s.Close)
	st := s.v41State()
	st.setDenseProjection(func(layer int, leaf string, _ []float32, _, _, rows int) ([]float32, v41DenseProjectionOutcome, error) {
		if leaf == "indexer.wk.weight" {
			*calls += rows
		}
		return nil, v41ProjectionDeclined, nil
	})
	return s
}

func v41IndexKeyMemoSameBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: width %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %v, want bit-identical %v", what, i, got[i], want[i])
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41StepFallbackReprojectsOnlyNewIndexKeyGroups(t *testing.T) {
	t.Parallel()
	var calls int
	s := v41IndexKeyMemoCountingSession(t, v41CompressorTestFixtureIndex(t, 128), &calls)
	history := []int{1, 2, 3, 4, 5, 6}
	s.Prefill(history)
	if calls != 3 {
		t.Fatalf("cold prefill projected %d index keys, want 3 completed groups", calls)
	}
	for _, step := range []struct {
		id, wantCalls int
	}{{7, 0}, {3, 1}, {5, 0}, {2, 1}} {
		s.v41Forward.layers = nil
		if s.v41IncrementalEligible() {
			t.Fatal("forced fallback still reports incremental eligibility")
		}
		calls = 0
		got := s.Step(step.id)
		history = append(history, step.id)
		if calls != step.wantCalls {
			t.Fatalf("fallback step %d projected %d index keys, want %d new groups only", step.id, calls, step.wantCalls)
		}
		var coldCalls int
		cold := v41IndexKeyMemoCountingSession(t, v41CompressorTestFixtureIndex(t, 128), &coldCalls)
		want := cold.Prefill(history)
		if coldCalls != len(history)/2 {
			t.Fatalf("cold oracle projected %d index keys, want %d", coldCalls, len(history)/2)
		}
		v41IndexKeyMemoSameBits(t, "fallback logits", got, want)
		keys, ok := s.v41Forward.attn.IndexKeys(0)
		wantKeys, wantOK := cold.v41Forward.attn.IndexKeys(0)
		if !ok || !wantOK || len(keys) != len(wantKeys) || len(keys) != len(history)/2 {
			t.Fatalf("published index keys %d/%t, cold %d/%t", len(keys), ok, len(wantKeys), wantOK)
		}
		for g := range keys {
			v41IndexKeyMemoSameBits(t, "published index key", keys[g], wantKeys[g])
		}
	}
}
