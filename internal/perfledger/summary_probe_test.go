package perfledger

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestSummarizeExcludesProbeTurnsFromServedStats(t *testing.T) {
	// The witnessed Halo shape (fak-private#3151): a 4+21-token probe every ~70s
	// beside real partial-regime agent turns. Two probes would otherwise pull the
	// partial p50 down to the probe's ~150 ms.
	probe := Record{Schema: Schema, PromptTokens: 4, CachedTokens: 21, CompletionTokens: 21, TTFTMS: 150, E2EMS: 740, PrefillTPS: 27}
	served := []Record{
		{Schema: Schema, PromptTokens: 5000, CachedTokens: 15000, CompletionTokens: 60, TTFTMS: 7000, E2EMS: 9000},
		{Schema: Schema, PromptTokens: 6000, CachedTokens: 14000, CompletionTokens: 60, TTFTMS: 8000, E2EMS: 10000},
	}
	s := Summarize([]Record{probe, served[0], probe, served[1], probe})

	if s.Count != 2 {
		t.Fatalf("count = %d, want 2 served turns (3 probes excluded)", s.Count)
	}
	if s.TTFTP50MS < 7000 {
		t.Fatalf("ttft p50 = %v ms, want a served-turn value (>= 7000), not the probe's 150", s.TTFTP50MS)
	}
	if got := s.ByRegime["partial"].Count; got != 2 {
		t.Fatalf("partial regime count = %d, want 2 (probes excluded)", got)
	}
	if want := 29000.0 / 40000.0; s.CacheHitShare < want-0.001 || s.CacheHitShare > want+0.001 {
		t.Fatalf("cache_hit_share = %v, want %.4f over served turns only", s.CacheHitShare, want)
	}
}
