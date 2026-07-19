package main

// Tests for the reconcile-ledger fold of provider-completion/verified_completion_rate
// (#4492). They pin the facts reduction (SPAWNED rows keyed per provider, everything
// else skipped), the honest per-provider UNMEASURED while closed_now stays an
// unattributed aggregate, the measured seam the fold already supports for the day a
// provider-attributed close record lands, the empty-ledger episode, and the
// catalog-vs-emitted lockstep for the registered lever.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestProviderCompletionLedgerFactsKeysSpawnedRowsPerProvider(t *testing.T) {
	events := []loopmgr.Event{
		// Two claude dispatches and one codex dispatch (codex keys to "gpt" through
		// the shared leaderboard keying #4505).
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"},
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"},
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "codex"},
		// A harness-synthetic principal carries no billed provider to key by.
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "<synthetic>"},
		// A non-SPAWNED start row is not a dispatched worker.
		{Kind: loopmgr.EventStart, Reason: "RESUMED", Principal: "claude"},
		// The closure auditor's end row: an aggregate count, no provider attribution.
		{Kind: loopmgr.EventEnd, Metrics: map[string]int64{"closed_now": 2}},
	}
	got := providerCompletionLedgerFacts(events)
	if len(got) != 2 {
		t.Fatalf("want two provider rows (claude, gpt), got %+v", got)
	}
	if got[0].Provider != "claude" || got[0].Dispatched != 2 {
		t.Fatalf("claude row should carry 2 dispatched, got %+v", got[0])
	}
	if got[1].Provider != "gpt" || got[1].Dispatched != 1 {
		t.Fatalf("gpt (codex) row should carry 1 dispatched, got %+v", got[1])
	}
	for _, r := range got {
		if r.ClosesAttributed || r.VerifiedClosed != 0 {
			t.Fatalf("closed_now is an unattributed aggregate; no row may claim attributed closes, got %+v", r)
		}
	}
}

// TestProviderCompletionLedgerUnmeasuredPerProvider pins today's honest shape: each
// dispatched provider folds one UNMEASURED episode against the ONE registered
// provider-completion claim, carrying its witnessed dispatched count as the sample
// and naming the missing provider-attributed close record — never a fabricated
// rate. The episode survives the real scorer as UNMEASURED end-to-end.
func TestProviderCompletionLedgerUnmeasuredPerProvider(t *testing.T) {
	events := []loopmgr.Event{
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"},
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "claude"},
		{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "gemini"},
		{Kind: loopmgr.EventEnd, Metrics: map[string]int64{"closed_now": 3}},
	}
	got := providerCompletionLedgerEpisodes(providerCompletionLedgerFacts(events))
	if len(got) != 2 {
		t.Fatalf("want one episode per dispatched provider (claude, gemini), got %+v", got)
	}
	claude := got[0]
	if claude.Prediction.Lever != "provider-completion" || claude.Prediction.Metric != "verified_completion_rate" {
		t.Fatalf("episode must address the ONE registered provider-completion cell, got %+v", claude.Prediction)
	}
	if claude.Prediction.Claimed != 0.5 {
		t.Fatalf("episode must carry the one registered claim (0.5), got %v", claude.Prediction.Claimed)
	}
	if claude.Prediction.Unit != "fraction" {
		t.Fatalf("episode unit = %q, want fraction", claude.Prediction.Unit)
	}
	if claude.Outcome.Measured {
		t.Fatalf("an unattributed-close ledger must score UNMEASURED, got %+v", claude.Outcome)
	}
	if claude.Outcome.Sample != 2 {
		t.Fatalf("the UNMEASURED episode should carry claude's dispatched count (2), got %+v", claude.Outcome)
	}
	if !strings.Contains(claude.Outcome.Source, "provider claude") ||
		!strings.Contains(claude.Outcome.Source, "no per-provider close attribution") {
		t.Fatalf("the UNMEASURED source must name the provider and the missing attribution seam, got %q", claude.Outcome.Source)
	}
	if got[1].Outcome.Sample != 1 || !strings.Contains(got[1].Outcome.Source, "provider gemini") {
		t.Fatalf("gemini row should carry its own dispatched count, got %+v", got[1].Outcome)
	}
	e := dojo.Score("reconcile-ledger", claude.Prediction, claude.Outcome, dojo.DefaultCalibBand())
	if e.Verdict != dojo.VerdictUnmeasured {
		t.Fatalf("an unattributed-close episode must score %s end-to-end, got %s (%+v)", dojo.VerdictUnmeasured, e.Verdict, e)
	}
}

// TestProviderCompletionLedgerMeasuresWhenClosesAttributed pins the extension seam:
// the moment a provider-attributed VERIFIED-close record lands on the ledger, the
// same pure fold measures VerifiedClosed/Dispatched WITNESSED — and a provider
// whose every dispatch went unclosed measures a REAL 0.0, distinct from
// unattributed.
func TestProviderCompletionLedgerMeasuresWhenClosesAttributed(t *testing.T) {
	got := providerCompletionLedgerEpisodes([]providerDispatchCompletion{
		{Provider: "claude", Dispatched: 4, VerifiedClosed: 2, ClosesAttributed: true},
		{Provider: "glm", Dispatched: 3, VerifiedClosed: 0, ClosesAttributed: true},
	})
	if len(got) != 2 {
		t.Fatalf("want two measured episodes, got %+v", got)
	}
	claude := got[0]
	if !claude.Outcome.Measured || claude.Outcome.Realized != 0.5 {
		t.Fatalf("claude should measure 2/4 = 0.5, got %+v", claude.Outcome)
	}
	if claude.Outcome.Provenance != dojo.Witnessed {
		t.Fatalf("an attributed reconcile-ledger close is WITNESSED, got %+v", claude.Outcome)
	}
	if claude.Outcome.Sample != 4 {
		t.Fatalf("sample should be the dispatched denominator (4), got %+v", claude.Outcome)
	}
	glm := got[1]
	if !glm.Outcome.Measured || glm.Outcome.Realized != 0.0 {
		t.Fatalf("an all-unclosed provider measures a real 0.0, not UNMEASURED, got %+v", glm.Outcome)
	}
}

// TestProviderCompletionLedgerEmpty pins the fail-open: a missing or empty ledger
// (no SPAWNED rows at all) yields ONE honest UNMEASURED episode, never a crash or
// a fabricated rate.
func TestProviderCompletionLedgerEmpty(t *testing.T) {
	for _, events := range [][]loopmgr.Event{
		nil,
		{{Kind: loopmgr.EventEnd, Metrics: map[string]int64{"closed_now": 5}}},
		{{Kind: loopmgr.EventStart, Reason: "SPAWNED", Principal: "<synthetic>"}},
	} {
		got := providerCompletionLedgerEpisodes(providerCompletionLedgerFacts(events))
		if len(got) != 1 || got[0].Outcome.Measured {
			t.Fatalf("want one UNMEASURED episode from a dispatch-free ledger, got %+v", got)
		}
		if got[0].Prediction.Claimed != 0.5 {
			t.Fatalf("the UNMEASURED episode must still carry the registered claim, got %+v", got[0].Prediction)
		}
		if !strings.Contains(got[0].Outcome.Source, "no SPAWNED start rows") {
			t.Fatalf("the UNMEASURED source must name the empty dispatched population, got %q", got[0].Outcome.Source)
		}
	}
}

// TestDojoCatalogMatchesProviderCompletionLedgerEmittedMetrics keeps the static
// `dojo list` catalog row of the registered lever in lockstep with the metric the
// lever actually emits.
func TestDojoCatalogMatchesProviderCompletionLedgerEmittedMetrics(t *testing.T) {
	emitted := map[string]bool{}
	for _, in := range providerCompletionLedgerEpisodes(providerCompletionLedgerFacts(nil)) {
		emitted[in.Prediction.Metric] = true
	}
	found := false
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "provider-completion-ledger" {
			continue
		}
		found = true
		for _, m := range lv.Metrics {
			if !emitted[m.Name] {
				t.Fatalf("catalog advertises metric %q the lever never emits", m.Name)
			}
		}
		if len(lv.Metrics) != len(emitted) {
			t.Fatalf("catalog lists %d metrics but the lever emits %d", len(lv.Metrics), len(emitted))
		}
	}
	if !found {
		t.Fatal("provider-completion-ledger is not in the dojo lever catalog; RegisterLever did not fold its row")
	}
}
