package main

// Tests for the cache-read-share/billed_cache_read_share top-line cell
// (#4498/#4484): the pure corpus-wide fold and the catalog lockstep witness,
// mirroring the provider-cache (#4504) witnesses. Where provider-cache splits
// the billed cache_read tokens per provider into a leaderboard, this cell folds
// the SAME tokens across ALL providers into one headline fraction.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestDojoCacheReadShareEpisodeFromSessions(t *testing.T) {
	// Two billed providers, each contributing input 100 + cache_read 800 +
	// cache_creation 100 = 1000 input-side tokens: summed corpus-wide that is
	// cache_read 1600 / (input 200 + cache_read 1600 + cache_creation 200 = 2000)
	// = 0.8 over 10 billed turns. A synthetic (non-billed) row and an unreadable
	// session never fold in.
	sessions := []sessionaudit.Session{
		{PerModel: map[string]sessionaudit.ModelCounts{
			"claude-fable-5": {Turns: 6, Input: 100, CacheRead: 800, CacheCreate: 100},
			// A synthetic (non-billed) row must never fold into the top line.
			"<synthetic>": {Turns: 5, Input: 999, CacheRead: 999},
		}},
		{PerModel: map[string]sessionaudit.ModelCounts{
			"gpt-5-codex": {Turns: 4, Input: 100, CacheRead: 800, CacheCreate: 100},
		}},
		// An unreadable session never folds.
		{Error: "unreadable", PerModel: map[string]sessionaudit.ModelCounts{
			"claude-fable-5": {Turns: 9, Input: 9999, CacheRead: 1},
		}},
	}
	ins := cacheReadShareEpisodeFromSessions(sessions)
	if len(ins) != 1 {
		t.Fatalf("the top-line cell folds ONE overall episode, got %d: %+v", len(ins), ins)
	}
	e := ins[0]
	if e.Prediction.Lever != "cache-read-share" || e.Prediction.Metric != "billed_cache_read_share" {
		t.Fatalf("episode cell wrong: %s/%s", e.Prediction.Lever, e.Prediction.Metric)
	}
	// The pinned claim literal (#4498/#4484): a seeded genuine estimate, not a floor.
	if e.Prediction.Claimed != 0.8 || e.Prediction.IntentionalFloor || e.Prediction.LowerIsBetter {
		t.Fatalf("pinned cache-read-share claim drifted: %+v", e.Prediction)
	}
	if !e.Outcome.Measured || e.Outcome.Provenance != dojo.Observed {
		t.Fatalf("a corpus with cache billing must measure (OBSERVED): %+v", e.Outcome)
	}
	if e.Outcome.Realized != 0.8 || e.Outcome.Sample != 10 {
		t.Fatalf("top-line fold wrong: %+v, want share 0.8 over 10 billed turns", e.Outcome)
	}
	if !strings.Contains(e.Outcome.Source, "top-line") {
		t.Fatalf("measured source should name the top-line fold: %q", e.Outcome.Source)
	}

	// A corpus that bills input but relays NO cache fields anywhere must be
	// UNMEASURED, never a fabricated 0.0 (the provider-cache honesty rule applied
	// to the aggregate): the transcript shape cannot tell "billed cold everywhere"
	// from "no cache-read field relayed".
	allCold := cacheReadShareEpisodeFromSessions([]sessionaudit.Session{
		{PerModel: map[string]sessionaudit.ModelCounts{"claude-fable-5": {Turns: 3, Input: 500}}},
		{PerModel: map[string]sessionaudit.ModelCounts{"gpt-5-codex": {Turns: 2, Input: 300}}},
	})
	if len(allCold) != 1 || allCold[0].Outcome.Measured || allCold[0].Outcome.Sample != 5 {
		t.Fatalf("an all-cold corpus must yield one UNMEASURED episode over its billed turns, got %+v", allCold)
	}
	if !strings.Contains(allCold[0].Outcome.Source, "no cache_read/cache_creation") {
		t.Fatalf("the UNMEASURED reason must name the missing cache fields: %q", allCold[0].Outcome.Source)
	}

	// An empty (or missing) corpus still surfaces the cell — honestly UNMEASURED.
	empty := cacheReadShareEpisodeFromSessions(nil)
	if len(empty) != 1 || empty[0].Outcome.Measured {
		t.Fatalf("an empty corpus must yield one UNMEASURED episode, got %+v", empty)
	}
}

func TestDojoCatalogMatchesCacheReadShareEmittedMetrics(t *testing.T) {
	// the static catalog must match the metric the cache-read-share lever emits.
	emitted := map[string]bool{}
	for _, in := range cacheReadShareEpisodeFromSessions([]sessionaudit.Session{
		{PerModel: map[string]sessionaudit.ModelCounts{"claude-fable-5": {Turns: 1, Input: 1, CacheRead: 1}}},
	}) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "cache-read-share" {
			continue
		}
		for _, m := range lv.Metrics {
			if !emitted[m.Name] {
				t.Fatalf("catalog advertises metric %q the lever never emits", m.Name)
			}
		}
		if len(lv.Metrics) != len(emitted) {
			t.Fatalf("catalog lists %d metrics but the lever emits %d", len(lv.Metrics), len(emitted))
		}
	}
}
