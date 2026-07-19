package main

// Tests for the token-economy/tokens_saved_ratio cell's shell adapter (#4487): the
// session-corpus reduction sums the ON side but reports no paired OFF baseline, so
// the fold scores UNMEASURED naming the missing baseline — the shape `fak dojo run`
// scores today — and the catalog stays in lockstep with the emitted metric.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestTokenEconomyBilledOnCorpusSumsOnSideButHasNoBaseline(t *testing.T) {
	sessions := []sessionaudit.Session{
		{AssistantTurns: 2, PerModel: map[string]sessionaudit.ModelCounts{
			// Input-side billed tokens fold: input + cache_read + cache_creation
			// (output is the model's generation, not something the levers shed).
			"claude-opus-4-8": {Turns: 2, Input: 100, Output: 999, CacheRead: 300, CacheCreate: 100},
			// A harness-synthetic row never totals into the billed ON side.
			"<synthetic>": {Turns: 5, Input: 777},
		}},
		{AssistantTurns: 1, PerModel: map[string]sessionaudit.ModelCounts{
			"glm-5.2": {Turns: 1, Input: 50, CacheRead: 50},
		}},
		// An unreadable session carries no billed tokens to fold.
		{Error: "unreadable", PerModel: map[string]sessionaudit.ModelCounts{"claude-opus-4-8": {Input: 9999}}},
	}
	corpus := tokenEconomyBilledOnCorpus(sessions)
	// 100+300+100 (claude) + 50+50 (glm) = 600 input-side ON tokens; output and the
	// synthetic + unreadable rows are excluded.
	if corpus.OnTokens != 600 {
		t.Fatalf("ON-side input-side token sum = %d, want 600", corpus.OnTokens)
	}
	if corpus.BaselinePaired {
		t.Fatal("an ordinary session corpus has no no-lever OFF baseline; BaselinePaired must be false")
	}

	got := dojo.TokenEconomyEpisodes(corpus)
	if len(got) != 1 || got[0].Outcome.Measured {
		t.Fatalf("want one UNMEASURED episode (no paired baseline), got %+v", got)
	}
	if got[0].Outcome.Sample != 600 {
		t.Fatalf("the UNMEASURED episode should carry the ON-side sample (600), got %+v", got[0].Outcome)
	}
	if !strings.Contains(got[0].Outcome.Source, "no paired") || !strings.Contains(got[0].Outcome.Source, "OFF baseline") {
		t.Fatalf("the UNMEASURED source must name the missing paired OFF baseline, got %q", got[0].Outcome.Source)
	}
	if got[0].Prediction.Claimed != 0.30 {
		t.Fatalf("the UNMEASURED episode must carry the registered claim (0.30), got %v", got[0].Prediction.Claimed)
	}
}

// TestTokenEconomyLeverEpisodesEmpty pins the fail-open: an empty (or missing)
// corpus still surfaces the cell — honestly UNMEASURED with a zero ON-side sample,
// never a crash or a fabricated ratio.
func TestTokenEconomyLeverEpisodesEmpty(t *testing.T) {
	corpus := tokenEconomyBilledOnCorpus(nil)
	if corpus.OnTokens != 0 || corpus.BaselinePaired {
		t.Fatalf("empty corpus should yield zero ON tokens and no baseline, got %+v", corpus)
	}
	got := dojo.TokenEconomyEpisodes(corpus)
	if len(got) != 1 || got[0].Outcome.Measured {
		t.Fatalf("want one UNMEASURED episode from an empty corpus, got %+v", got)
	}
}

// TestDojoCatalogMatchesTokenEconomyEmittedMetrics keeps the static `dojo list`
// catalog in lockstep with the metric the token-economy lever actually emits.
func TestDojoCatalogMatchesTokenEconomyEmittedMetrics(t *testing.T) {
	emitted := map[string]bool{}
	for _, in := range dojo.TokenEconomyEpisodes(tokenEconomyBilledOnCorpus(nil)) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "token-economy" {
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
