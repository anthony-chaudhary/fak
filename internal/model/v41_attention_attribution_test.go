package model

import (
	"errors"
	"math"
	"testing"
)

// v41AttributionClock advances by a fixed amount on every read. The attention
// entrypoint reads it immediately before and after one contraction, making the
// expected elapsed time independent of wall-clock scheduling.
func v41AttributionClock(step int64) func() int64 {
	var now int64
	return func() int64 {
		now += step
		return now
	}
}

// TestV41AttentionAttributionActualPaths is a [SW-VERIFIED] accounting witness.
// It drives the real reduced V4.1 Prefill/Step paths for both plain and ratio-2
// attention; it does not claim physical-device timing.
// fak-test:runtime fast est=2s lane=default
func TestV41AttentionAttributionActualPaths(t *testing.T) {
	for _, tc := range []struct {
		name            string
		model           func(*testing.T) *Model
		wantDecodeCalls int
	}{
		{name: "plain", model: v41ReducedModel, wantDecodeCalls: 1},
		// The compressed Step currently replays the three-position prefix through
		// the full ratio-2 path, so all three attempted contractions belong to the
		// decode phase of that one Step.
		{name: "compressed_ratio_2", model: v41AttentionModel, wantDecodeCalls: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model(t)
			m.v41SetExpertTimeClock(v41AttributionClock(11))
			s := &Session{M: m}

			if got := s.Prefill([]int{2, 4}); len(got) == 0 {
				t.Fatal("Prefill returned no logits")
			}
			prefill := m.V41ExpertFaultAttribution()
			if prefill.Prefill.AttentionContractionCalls != 2 {
				t.Fatalf("prefill attention calls = %d, want 2 (one attempted contraction per token)", prefill.Prefill.AttentionContractionCalls)
			}
			if prefill.Prefill.AttentionContractionNanos != 22 {
				t.Fatalf("prefill attention nanos = %d, want 22", prefill.Prefill.AttentionContractionNanos)
			}
			if prefill.Decode.AttentionContractionCalls != 0 || prefill.Decode.AttentionContractionNanos != 0 {
				t.Fatalf("decode attention after prefill = %d calls/%dns, want 0/0", prefill.Decode.AttentionContractionCalls, prefill.Decode.AttentionContractionNanos)
			}

			if got := s.Step(6); len(got) == 0 {
				t.Fatal("Step returned no logits")
			}
			after := m.V41ExpertFaultAttribution()
			if after.Prefill.AttentionContractionCalls != 2 || after.Prefill.AttentionContractionNanos != 22 {
				t.Fatalf("decode mutated prefill attention = %d calls/%dns, want 2/22", after.Prefill.AttentionContractionCalls, after.Prefill.AttentionContractionNanos)
			}
			wantDecodeNanos := int64(tc.wantDecodeCalls * 11)
			if after.Decode.AttentionContractionCalls != tc.wantDecodeCalls || after.Decode.AttentionContractionNanos != wantDecodeNanos {
				t.Fatalf("decode attention = %d calls/%dns, want %d/%d", after.Decode.AttentionContractionCalls, after.Decode.AttentionContractionNanos, tc.wantDecodeCalls, wantDecodeNanos)
			}
		})
	}
}

// TestV41AttentionAttributionAttemptAndGuards pins the accounting boundary:
// an attempted contraction is counted even when it returns a typed stage error,
// an unknown phase is inert, and a backwards clock cannot subtract lifetime time.
// fak-test:runtime fast est=1s lane=default
func TestV41AttentionAttributionAttemptAndGuards(t *testing.T) {
	t.Run("typed_error_counts_attempt", func(t *testing.T) {
		m := v41ReducedModel(t)
		m.tensor(layerName(0, "attn.sink"))[0] = float32(math.NaN())
		m.v41SetExpertTimeClock(v41AttributionClock(13))
		m.v41SetExpertFaultPhase(V41PhasePrefill)
		_, err := m.forwardV41([]int{1}, nil)
		m.v41SetExpertFaultPhase(V41PhaseUnknown)
		var stageErr *V41ForwardError
		if err == nil || !errors.As(err, &stageErr) {
			t.Fatalf("plain attention error = %v, want *V41ForwardError", err)
		}
		if stageErr.Stage != v41StageAttention || stageErr.Layer != 0 {
			t.Fatalf("plain attention error stage=%s layer=%d, want attention/0", stageErr.Stage, stageErr.Layer)
		}
		got := m.V41ExpertFaultAttribution()
		if got.Prefill.AttentionContractionCalls != 1 || got.Prefill.AttentionContractionNanos != 13 {
			t.Fatalf("failed attention attempt = %d calls/%dns, want 1/13", got.Prefill.AttentionContractionCalls, got.Prefill.AttentionContractionNanos)
		}
	})

	t.Run("unknown_phase_inert", func(t *testing.T) {
		m := v41ReducedModel(t)
		m.v41SetExpertTimeClock(v41AttributionClock(7))
		if _, err := m.forwardV41([]int{1}, nil); err != nil {
			t.Fatalf("unscoped forward: %v", err)
		}
		got := m.V41ExpertFaultAttribution()
		if got.Prefill.AttentionContractionCalls != 0 || got.Decode.AttentionContractionCalls != 0 {
			t.Fatalf("unscoped attention recorded prefill=%d decode=%d, want 0/0", got.Prefill.AttentionContractionCalls, got.Decode.AttentionContractionCalls)
		}
	})

	t.Run("negative_elapsed_clamped", func(t *testing.T) {
		m := v41ReducedModel(t)
		read := 0
		m.v41SetExpertTimeClock(func() int64 {
			read++
			if read == 1 {
				return 100
			}
			return 90
		})
		m.v41SetExpertFaultPhase(V41PhaseDecode)
		m.v41NoteAttentionContraction(m.v41NowNanos())
		m.v41SetExpertFaultPhase(V41PhaseUnknown)
		got := m.V41ExpertFaultAttribution().Decode
		if got.AttentionContractionCalls != 1 || got.AttentionContractionNanos != 0 {
			t.Fatalf("backwards-clock attention = %d calls/%dns, want 1/0", got.AttentionContractionCalls, got.AttentionContractionNanos)
		}
	})
}
