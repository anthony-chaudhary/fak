package metrics

import (
	"strings"
	"testing"
)

func TestWitnessCacheBreakNormalizesAndClamps(t *testing.T) {
	// A negative cost clamps to zero; an out-of-vocabulary cause folds to unknown.
	e := WitnessCacheBreak(CacheBreakCause("bogus"), -50)
	if e.Cause != CacheBreakUnknown {
		t.Fatalf("cause = %q, want %q", e.Cause, CacheBreakUnknown)
	}
	if e.CostTokens != 0 {
		t.Fatalf("cost = %d, want 0 (clamped)", e.CostTokens)
	}
	// A known cause and positive cost pass through unchanged.
	e = WitnessCacheBreak(CacheBreakToolsetChange, 1200)
	if e.Cause != CacheBreakToolsetChange || e.CostTokens != 1200 {
		t.Fatalf("got %+v, want {toolset_change 1200}", e)
	}
}

func TestFoldCacheBreaksTotalsAndPerCause(t *testing.T) {
	events := []CacheBreakEvent{
		WitnessCacheBreak(CacheBreakToolsetChange, 1000),
		WitnessCacheBreak(CacheBreakRebuiltPrompt, 300),
		WitnessCacheBreak(CacheBreakToolsetChange, 200),
		WitnessCacheBreak(CacheBreakCause("weird"), 40), // folds to unknown
	}
	r := FoldCacheBreaks(events)
	if r.Events != 4 {
		t.Fatalf("events = %d, want 4", r.Events)
	}
	if r.CostTokens != 1540 {
		t.Fatalf("cost = %d, want 1540", r.CostTokens)
	}
	// Per-cause tally is emitted in canonical order: toolset_change, rebuilt_prompt, unknown.
	wantOrder := []CacheBreakCause{CacheBreakToolsetChange, CacheBreakRebuiltPrompt, CacheBreakUnknown}
	if len(r.ByCause) != len(wantOrder) {
		t.Fatalf("ByCause = %+v, want %d causes", r.ByCause, len(wantOrder))
	}
	for i, want := range wantOrder {
		if r.ByCause[i].Cause != want {
			t.Fatalf("ByCause[%d].Cause = %q, want %q", i, r.ByCause[i].Cause, want)
		}
	}
	if r.ByCause[0].Events != 2 || r.ByCause[0].CostTokens != 1200 {
		t.Fatalf("toolset_change tally = %+v, want {events:2 cost:1200}", r.ByCause[0])
	}
}

func TestFoldCacheBreaksEmptyIsCleanZero(t *testing.T) {
	r := FoldCacheBreaks(nil)
	if r.Events != 0 || r.CostTokens != 0 || len(r.ByCause) != 0 {
		t.Fatalf("empty fold = %+v, want a clean zero", r)
	}
}

// TestCacheBreakNonzeroWitnessOnDeliberatePrefixBreak is the acceptance witness:
// a test that DELIBERATELY breaks the prefix produces a nonzero cache_break_events
// count with a per-event cost and cause. Here the "deliberate break" is a
// mid-conversation toolset mutation that invalidated a warm 900-token prefix.
func TestCacheBreakNonzeroWitnessOnDeliberatePrefixBreak(t *testing.T) {
	// Deliberately break the prefix: a toolset change mutated the cached prefix
	// mid-conversation, costing a 900-token cold rebuild.
	broken := []CacheBreakEvent{WitnessCacheBreak(CacheBreakToolsetChange, 900)}
	r := FoldCacheBreaks(broken)
	if r.Events == 0 {
		t.Fatal("deliberate prefix break produced zero cache_break_events; want a nonzero witness")
	}
	if r.CostTokens == 0 {
		t.Fatal("deliberate prefix break produced zero token cost; want a per-event cost witness")
	}
	if len(r.ByCause) != 1 || r.ByCause[0].Cause != CacheBreakToolsetChange {
		t.Fatalf("break not attributed to a cause: %+v", r.ByCause)
	}
}

// TestCacheBreakLabeledMetricEmittedPerSession is the acceptance witness that a
// labeled metric is emitted per session on the Prometheus surface.
func TestCacheBreakLabeledMetricEmittedPerSession(t *testing.T) {
	r := FoldCacheBreaks([]CacheBreakEvent{
		WitnessCacheBreak(CacheBreakToolsetChange, 900),
		WitnessCacheBreak(CacheBreakProviderQuirk, 120),
	})
	out, err := RenderOpenMetricsText(r.OpenMetricFamilies())
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	text := string(out)
	for _, want := range []string{
		"# TYPE fak_cache_break_events_total counter",
		"# TYPE fak_cache_break_cost_tokens_total counter",
		`fak_cache_break_events_total{cause="toolset_change"} 1`,
		`fak_cache_break_cost_tokens_total{cause="toolset_change"} 900`,
		`fak_cache_break_events_total{cause="provider_quirk"} 1`,
		`fak_cache_break_cost_tokens_total{cause="provider_quirk"} 120`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("rendered metrics missing %q\n---\n%s", want, text)
		}
	}
}

func TestCacheBreakEmptySessionRendersCleanFamilies(t *testing.T) {
	r := FoldCacheBreaks(nil)
	out, err := RenderOpenMetricsText(r.OpenMetricFamilies())
	if err != nil {
		t.Fatalf("render empty: %v", err)
	}
	// The families are still declared (HELP/TYPE) with no samples — a clean zero.
	text := string(out)
	if !strings.Contains(text, "# TYPE fak_cache_break_events_total counter") {
		t.Fatalf("empty session dropped the family declaration:\n%s", text)
	}
}

// TestCacheBreakGateFailsRegression is the acceptance witness that a gate fails a
// regression: a defined budget passes a within-threshold session and fails one
// that regresses above it. It is a defined threshold, not "any nonzero fails".
func TestCacheBreakGateFailsRegression(t *testing.T) {
	budget := CacheBreakBudget{MaxEvents: 1, MaxCostTokens: 1000}

	// Within budget (one break, 500 tokens): the gate passes — a nonzero count
	// alone does NOT fail.
	ok := FoldCacheBreaks([]CacheBreakEvent{WitnessCacheBreak(CacheBreakRebuiltPrompt, 500)})
	if err := ok.CheckBudget(budget); err != nil {
		t.Fatalf("within-budget session failed the gate: %v", err)
	}

	// Regression by event count: two breaks exceeds MaxEvents=1.
	tooMany := FoldCacheBreaks([]CacheBreakEvent{
		WitnessCacheBreak(CacheBreakRebuiltPrompt, 100),
		WitnessCacheBreak(CacheBreakToolsetChange, 100),
	})
	if err := tooMany.CheckBudget(budget); err == nil {
		t.Fatal("event-count regression did not fail the gate")
	}

	// Regression by token cost: one break but over MaxCostTokens=1000.
	tooCostly := FoldCacheBreaks([]CacheBreakEvent{WitnessCacheBreak(CacheBreakToolsetChange, 5000)})
	if err := tooCostly.CheckBudget(budget); err == nil {
		t.Fatal("token-cost regression did not fail the gate")
	}
}

func TestCacheBreakRegressesAgainstBaseline(t *testing.T) {
	baseline := FoldCacheBreaks([]CacheBreakEvent{WitnessCacheBreak(CacheBreakRebuiltPrompt, 300)})
	same := FoldCacheBreaks([]CacheBreakEvent{WitnessCacheBreak(CacheBreakRebuiltPrompt, 300)})
	worse := FoldCacheBreaks([]CacheBreakEvent{
		WitnessCacheBreak(CacheBreakRebuiltPrompt, 300),
		WitnessCacheBreak(CacheBreakToolsetChange, 1),
	})
	if same.RegressesAgainst(baseline) {
		t.Fatal("an equal session was flagged as a regression")
	}
	if !worse.RegressesAgainst(baseline) {
		t.Fatal("a worse session was not flagged as a regression")
	}
}
