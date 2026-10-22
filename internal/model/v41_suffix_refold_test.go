package model

import (
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41SuffixRefoldEligibleBounds(t *testing.T) {
	for _, tc := range []struct {
		name       string
		maxHistory string
		history    int
		suffix     int
		want       bool
	}{
		{"unset", "", 2, v41SuffixRefoldMinRows, false},
		{"zero", "0", 2, v41SuffixRefoldMinRows, false},
		{"negative", "-4", 2, v41SuffixRefoldMinRows, false},
		{"invalid", "two", 2, v41SuffixRefoldMinRows, false},
		{"cold-session", "8", 0, v41SuffixRefoldMinRows, false},
		{"history-over-cap", "1", 2, v41SuffixRefoldMinRows, false},
		{"suffix-under-panel", "8", 2, v41SuffixRefoldMinRows - 1, false},
		{"at-cap", "2", 2, v41SuffixRefoldMinRows, true},
		{"under-cap-long-suffix", "8", 3, 4 * v41SuffixRefoldMinRows, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(v41SuffixRefoldMaxHistoryEnv, tc.maxHistory)
			if got := v41SuffixRefoldEligible(tc.history, tc.suffix); got != tc.want {
				t.Fatalf("v41SuffixRefoldEligible(%d, %d) with cap %q = %t, want %t", tc.history, tc.suffix, tc.maxHistory, got, tc.want)
			}
		})
	}
}

// A tiny restored history followed by a long suffix re-folds through the cold
// grouped prefill only when opted in; every other setting keeps the per-token
// suffix and its counters byte-for-byte.
//
// fak-test:runtime medium est=10s lane=default
func TestV41SuffixRefoldShortRestore(t *testing.T) {
	const historyLen = 2
	m := v41IncrementalPlainModel(t, 2)
	defer m.CloseWeights()
	ids := v41ParityPrompt(m.Cfg.VocabSize, historyLen+v41SuffixRefoldMinRows)
	next := v41SuffixTail(m.Cfg, len(ids), 1)[0]

	t.Setenv(v41SuffixUbatchEnv, "")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	cold := m.NewSession()
	defer cold.Close()
	wantLogits := append([]float32(nil), cold.Prefill(ids)...)
	wantNext := append([]float32(nil), cold.Step(next)...)

	var serialLogits, serialNext []float32
	for _, tc := range []struct {
		name       string
		maxHistory string
		suffixLen  int
		refold     bool
	}{
		{"off", "", v41SuffixRefoldMinRows, false},
		{"zero", "0", v41SuffixRefoldMinRows, false},
		{"invalid", "two", v41SuffixRefoldMinRows, false},
		{"history-over-cap", "1", v41SuffixRefoldMinRows, false},
		{"suffix-under-panel", "2", v41SuffixRefoldMinRows - 1, false},
		{"on", "2", v41SuffixRefoldMinRows, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(v41SuffixRefoldMaxHistoryEnv, tc.maxHistory)
			history := ids[:historyLen+tc.suffixLen]
			s := m.NewSession()
			defer s.Close()
			s.Prefill(ids[:historyLen])
			if !s.v41IncrementalEligible() {
				t.Fatal("restored history is not incrementally eligible")
			}
			read, restore := v41SuffixPrefillCalls(t)
			panelsBefore := v41PanelInvocationCount.Load()
			got := append([]float32(nil), s.Prefill(history[historyLen:])...)
			panels := v41PanelInvocationCount.Load() - panelsBefore
			steps := read()
			restore()
			v41RequireCursors(t, s, history)
			if !tc.refold {
				if steps != tc.suffixLen || panels != 0 {
					t.Fatalf("serial suffix steps=%d panels=%d, want steps=%d panels=0", steps, panels, tc.suffixLen)
				}
				if tc.suffixLen != v41SuffixRefoldMinRows {
					return
				}
				gotNext := append([]float32(nil), s.Step(next)...)
				if serialLogits == nil {
					serialLogits, serialNext = got, gotNext
					return
				}
				v41RequireSameLogits(t, "inert setting vs unset", got, serialLogits)
				v41RequireSameLogits(t, "step after inert setting vs unset", gotNext, serialNext)
				return
			}
			if steps != 0 || panels == 0 {
				t.Fatalf("refold steps=%d panels=%d, want steps=0 and a grouped panel", steps, panels)
			}
			v41RequireSameLogits(t, "refold vs cold prefill", got, wantLogits)
			v41RequireSameLogits(t, "step after refold vs cold", s.Step(next), wantNext)
			v41RequireCursors(t, s, append(append([]int(nil), history...), next))
		})
	}
}
