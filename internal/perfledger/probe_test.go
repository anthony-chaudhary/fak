package perfledger

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestIsProbeBoundary(t *testing.T) {
	probe := Record{Schema: Schema, PromptTokens: 4, CachedTokens: 21, TTFTMS: 150}
	if s := Summarize([]Record{probe, probe}); s.Probes != 2 || s.Count != 0 {
		t.Fatalf("probes/count = %d/%d, want 2/0", s.Probes, s.Count)
	}
	for _, tc := range []struct {
		prompt, cached int
		want           bool
	}{
		{4, 21, true},
		{ProbeMaxPromptTokens - 1, 1, true},
		{ProbeMaxPromptTokens, 1, false},
		{25, 0, false},
		{0, 0, false},
		{2000, 30000, false},
	} {
		if got := (Record{PromptTokens: tc.prompt, CachedTokens: tc.cached}).IsProbe(); got != tc.want {
			t.Errorf("IsProbe(%d+%d) = %v, want %v", tc.prompt, tc.cached, got, tc.want)
		}
	}
}
