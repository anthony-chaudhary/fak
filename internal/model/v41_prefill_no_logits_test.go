package model

import (
	"math"
	"reflect"
	"testing"
)

// PrefillNoLogits must use the same V4.1 continuation state as Prefill. The
// fixture has V4.1 weight names, so entering a generic forward is also a failure.
// fak-test:runtime fast est=500ms lane=default
func TestV41PrefillNoLogitsContinuation(t *testing.T) {
	for _, tc := range []struct {
		name string
		seed bool
	}{
		{"cold", false},
		{"after-prefill", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			s := m.NewSession()
			control := m.NewSession()
			t.Cleanup(s.Close)
			t.Cleanup(control.Close)
			prefix, _ := v41IncrementalPrefix(m.Cfg, 8)
			var history []int
			checkState := func() {
				t.Helper()
				if s.v41Forward == nil || !reflect.DeepEqual(s.v41Forward.history, history) {
					t.Fatalf("V4.1 history missing or differs from %v", history)
				}
				for l := 0; l < m.Cfg.NumLayers; l++ {
					if got := s.v41Forward.layerState(l).nextWindowPos; got != len(history) {
						t.Fatalf("layer %d cursor = %d, want %d", l, got, len(history))
					}
				}
			}
			if tc.seed {
				s.Prefill(prefix)
				control.Prefill(prefix)
				history = append(history, prefix...)
			} else {
				s.PrefillNoLogits(nil)
				if s.v41Forward != nil {
					t.Fatal("empty cold prefill created V4.1 state")
				}
				s.PrefillNoLogits(prefix)
				control.Prefill(prefix)
				history = append(history, prefix...)
			}
			checkState()
			if !s.v41IncrementalEligible() {
				t.Fatal("prefill did not seed incremental continuation")
			}

			// Multiple suffixes catch replacement, duplication and lost cursors.
			for _, n := range []int{1, 4} {
				suffix := v41SuffixTail(m.Cfg, len(history), n)
				s.PrefillNoLogits(suffix)
				control.Prefill(suffix)
				history = append(history, suffix...)
				checkState()
				s.PrefillNoLogits([]int{})
				checkState()
			}

			next := v41SuffixTail(m.Cfg, len(history), 1)[0]
			got := append([]float32(nil), s.Step(next)...)
			wantControl := append([]float32(nil), control.Step(next)...)
			history = append(history, next)
			checkState()
			ref, err := m.forwardV41(history, nil)
			if err != nil {
				t.Fatalf("full-history reference: %v", err)
			}
			if len(ref.Logits) == 0 {
				t.Fatal("full-history reference returned no logits")
			}
			for _, want := range [][]float32{wantControl, ref.Logits[len(ref.Logits)-1]} {
				if len(got) != m.Cfg.VocabSize || len(want) != len(got) {
					t.Fatalf("logit widths: got %d, reference %d, vocab %d", len(got), len(want), m.Cfg.VocabSize)
				}
				for i := range got {
					g, w := float64(got[i]), float64(want[i])
					if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) || math.Abs(g-w) > 1e-6 {
						t.Fatalf("next-step logits[%d] = %g, want finite %g within 1e-6", i, g, w)
					}
				}
			}
		})
	}
}
