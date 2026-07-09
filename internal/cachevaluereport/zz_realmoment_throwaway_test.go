package cachevaluereport

import (
	"math"
	"testing"
)

// Throwaway cross-check: feed Fable's FableMoment the REAL fleet aggregate
// (docs/nightrun/gateway-usage.jsonl, 684 sessions) under both model labels and
// confirm it reproduces the independently-computed savings moment
// ($6,371.68 Opus / $12,743.37 Fable). Deleted after running.
func TestRealFleetMomentThrowaway(t *testing.T) {
	agg := func(model string) ModelSession {
		return ModelSession{
			Model:                    model,
			InputTokens:              16311574,
			CacheReadInputTokens:     1457109810,
			CacheCreationInputTokens: 148247520,
			OutputTokens:             15439222,
		}
	}
	prices := map[string]ModelPrice{
		"claude-opus-4-8": {InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
		"claude-fable-5":  {InputPerMTokUSD: 10, OutputPerMTokUSD: 50},
	}
	groups, unpriced := FableMoment([]ModelSession{agg("claude-opus-4-8"), agg("claude-fable-5")}, prices)
	if len(unpriced) != 0 {
		t.Fatalf("unexpected unpriced: %v", unpriced)
	}
	want := map[string]float64{"claude-opus-4-8": 6371.68, "claude-fable-5": 12743.37}
	for _, g := range groups {
		t.Logf("%-16s billed=$%.2f  net-cache-saving=$%.2f", g.Model, g.BilledUSD, g.NetCacheSavingUSD)
		if math.Abs(g.NetCacheSavingUSD-want[g.Model]) > 0.5 {
			t.Fatalf("%s net saving %.2f, want ~%.2f", g.Model, g.NetCacheSavingUSD, want[g.Model])
		}
	}
	if len(groups) == 2 && math.Abs(groups[0].NetCacheSavingUSD*2-groups[1].NetCacheSavingUSD) > 0.5 {
		t.Fatalf("fable not exactly 2x opus")
	}
}
