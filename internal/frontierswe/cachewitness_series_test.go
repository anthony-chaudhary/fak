package frontierswe

import "testing"

// A synthetic long-horizon trajectory: cumulative counters that grow every scrape,
// with reuse compounding (turn 1 is all-cold prefill, turns 2..N reuse most of the
// resident prefix). This is the exact shape a real FrontierSWE /metrics series has.
func trajectory() []CacheSample {
	return []CacheSample{
		{Turn: 1, PromptTokens: 1000, ReusedTokens: 0, ProviderCacheReadTokens: 0},    // cold first prefill
		{Turn: 2, PromptTokens: 2100, ReusedTokens: 1000, ProviderCacheReadTokens: 0}, // reused turn-1 prefix
		{Turn: 3, PromptTokens: 3300, ReusedTokens: 2100, ProviderCacheReadTokens: 0}, // reused turns 1..2
		{Turn: 4, PromptTokens: 4600, ReusedTokens: 3300, ProviderCacheReadTokens: 0}, // reused turns 1..3
	}
}

func TestFoldCacheWitness_RealizedReuseRateAndCumulative(t *testing.T) {
	s := FoldCacheWitness(trajectory())

	if s.Schema != CacheWitnessSchema {
		t.Errorf("schema = %q, want %q", s.Schema, CacheWitnessSchema)
	}
	if got, want := len(s.Points), 4; got != want {
		t.Fatalf("points = %d, want %d", got, want)
	}
	// Realized reuse rate = final cumulative reused/prompt = 3300/4600.
	if want := 3300.0 / 4600.0; !approxEq(s.RealizedReuseRate, want) {
		t.Errorf("realized reuse rate = %v, want %v", s.RealizedReuseRate, want)
	}
	if s.FinalPromptTokens != 4600 || s.FinalReusedTokens != 3300 {
		t.Errorf("finals = %d/%d, want 4600/3300", s.FinalPromptTokens, s.FinalReusedTokens)
	}
	if !s.CacheBit {
		t.Errorf("CacheBit = false, want true (reuse compounded turns 2..4)")
	}

	// Per-turn DELTA is the turn-2..N cache bite: e.g. turn 3 ingested 1200 new
	// prefill tokens (3300-2100) of which 1100 (2100-1000) were served from the KV.
	p3 := s.Points[2]
	if p3.DeltaPromptTokens != 1200 || p3.DeltaReusedTokens != 1100 {
		t.Errorf("turn-3 deltas = %d/%d, want 1200/1100", p3.DeltaPromptTokens, p3.DeltaReusedTokens)
	}
	if want := 1100.0 / 1200.0; !approxEq(p3.DeltaReuseRatio, want) {
		t.Errorf("turn-3 delta reuse ratio = %v, want %v", p3.DeltaReuseRatio, want)
	}
	// Cumulative reuse ratio rises turn over turn — the whole point of the curve.
	for i := 1; i < len(s.Points); i++ {
		if s.Points[i].CumReuseRatio < s.Points[i-1].CumReuseRatio {
			t.Errorf("cum reuse ratio dropped at turn %d: %v < %v",
				s.Points[i].Turn, s.Points[i].CumReuseRatio, s.Points[i-1].CumReuseRatio)
		}
	}
}

func TestFoldCacheWitness_ProvenanceNeverConflated(t *testing.T) {
	// Provider cache_read (OBSERVED) must be echoed separately and NEVER folded into
	// the WITNESSED reused-token totals — the conflation-scorecard line.
	samples := []CacheSample{
		{Turn: 1, PromptTokens: 1000, ReusedTokens: 0, ProviderCacheReadTokens: 500},
		{Turn: 2, PromptTokens: 2000, ReusedTokens: 900, ProviderCacheReadTokens: 1400},
	}
	s := FoldCacheWitness(samples)

	if s.Provenance["cum_reused_tokens"] != witnessed {
		t.Errorf("cum_reused_tokens provenance = %q, want WITNESSED", s.Provenance["cum_reused_tokens"])
	}
	if s.Provenance["provider_cache_read_tokens"] != observed {
		t.Errorf("provider provenance = %q, want OBSERVED", s.Provenance["provider_cache_read_tokens"])
	}
	// The realized reuse rate is over fak's OWN reuse only (900/2000), untouched by
	// the 1400 provider cache_read — proof the two caches are not summed.
	if want := 900.0 / 2000.0; !approxEq(s.RealizedReuseRate, want) {
		t.Errorf("realized reuse rate = %v, want %v (provider cache_read must not leak in)", s.RealizedReuseRate, want)
	}
	if s.ProviderCacheReadTokens != 1400 {
		t.Errorf("final provider cache_read = %d, want 1400 (echoed separately)", s.ProviderCacheReadTokens)
	}
}

func TestFoldCacheWitness_GatewayRestartClampsDelta(t *testing.T) {
	// A mid-trial gateway restart resets the cumulative counters; the fold must clamp
	// the backwards delta to 0 and flag it, not wrap a uint64 negative.
	samples := []CacheSample{
		{Turn: 1, PromptTokens: 3000, ReusedTokens: 2000},
		{Turn: 2, PromptTokens: 500, ReusedTokens: 100}, // restart: counters dropped
	}
	s := FoldCacheWitness(samples)
	p := s.Points[1]
	if !p.Regressed {
		t.Errorf("restarted scrape not flagged Regressed")
	}
	if p.DeltaPromptTokens != 0 || p.DeltaReusedTokens != 0 {
		t.Errorf("clamped deltas = %d/%d, want 0/0", p.DeltaPromptTokens, p.DeltaReusedTokens)
	}
}

func TestFoldCacheWitness_AllColdDoesNotBite(t *testing.T) {
	// A trajectory whose cache never engaged reports "did not bite", not a small win.
	s := FoldCacheWitness([]CacheSample{
		{Turn: 1, PromptTokens: 1000, ReusedTokens: 0},
		{Turn: 2, PromptTokens: 2000, ReusedTokens: 0},
	})
	if s.CacheBit {
		t.Errorf("CacheBit = true on an all-cold trajectory")
	}
	if s.RealizedReuseRate != 0 {
		t.Errorf("realized reuse rate = %v, want 0", s.RealizedReuseRate)
	}
}

func TestFoldCacheWitness_Empty(t *testing.T) {
	s := FoldCacheWitness(nil)
	if len(s.Points) != 0 || s.CacheBit || s.RealizedReuseRate != 0 {
		t.Errorf("empty fold not zero-valued: %+v", s)
	}
	if s.Schema != CacheWitnessSchema {
		t.Errorf("schema not stamped on empty fold")
	}
}

func approxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}
