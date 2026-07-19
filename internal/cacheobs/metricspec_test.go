package cacheobs

import (
	"strings"
	"testing"
)

// wellFormedTurns is a fixture where every turn obeys reused <= cacheable <= eligible <=
// prompt, so no collection path has to clamp — the two paths must agree exactly.
func wellFormedTurns() []Event {
	return []Event{
		{Kind: EventTurn, PromptTokens: 100, CacheableTokens: 80, ReusedTokens: 60, EligibleTokens: 90},
		{Kind: EventTurn, PromptTokens: 200, CacheableTokens: 150, ReusedTokens: 120, EligibleTokens: 180},
		{Kind: EventTurn, PromptTokens: 50, CacheableTokens: 40, ReusedTokens: 40, EligibleTokens: 50},
	}
}

// TestDefaultSpecsCoverSnapshotSchemaExactlyOnce is the LMCache mp_continuous.py:99-112 check:
// the registry must cover every additive snapshot field exactly once, with no gaps and no
// rival definitions. This is the structural guard that makes reporter drift impossible — if
// it fails, a metric is either undefined or doubly-defined before any fold runs.
func TestDefaultSpecsCoverSnapshotSchemaExactlyOnce(t *testing.T) {
	if err := CheckCoverage(DefaultSpecs(), SnapshotFields()); err != nil {
		t.Fatalf("default specs must cover the snapshot schema exactly once: %v", err)
	}
}

// TestCheckCoverageFlagsGapsDuplicatesAndUnknowns proves the coverage guard actually bites:
// a missing field, a doubly-covered field, and a spec targeting a field outside the schema
// each produce an error naming the offending field.
func TestCheckCoverageFlagsGapsDuplicatesAndUnknowns(t *testing.T) {
	// Gap: drop the reused_tokens spec.
	var withGap []MetricSpec
	for _, s := range DefaultSpecs() {
		if s.Field != FieldReusedTokens {
			withGap = append(withGap, s)
		}
	}
	if err := CheckCoverage(withGap, SnapshotFields()); err == nil || !strings.Contains(err.Error(), FieldReusedTokens) {
		t.Fatalf("a missing field must fail coverage naming %q, got %v", FieldReusedTokens, err)
	}

	// Duplicate: cover prompt_tokens twice.
	dup := append(DefaultSpecs(), MetricSpec{Event: EventTurn, Field: FieldPromptTokens, Extract: func(Event) float64 { return 0 }, Reduce: sumSamples})
	if err := CheckCoverage(dup, SnapshotFields()); err == nil || !strings.Contains(err.Error(), "want exactly 1") {
		t.Fatalf("a doubly-covered field must fail coverage, got %v", err)
	}

	// Unknown: a spec targeting a field the schema does not declare.
	extra := append(DefaultSpecs(), MetricSpec{Event: EventTurn, Field: "made_up_field", Extract: func(Event) float64 { return 0 }, Reduce: sumSamples})
	if err := CheckCoverage(extra, SnapshotFields()); err == nil || !strings.Contains(err.Error(), "made_up_field") {
		t.Fatalf("an unknown-field spec must fail coverage, got %v", err)
	}
}

// TestReconcileAgreesAcrossCollectionPaths is the parity guarantee on well-formed events: the
// declarative SpecFold and the imperative ObserverFold reduce the SAME registry to the SAME
// value for every covered field, so Reconcile finds nothing to report.
func TestReconcileAgreesAcrossCollectionPaths(t *testing.T) {
	specs := DefaultSpecs()
	events := wellFormedTurns()

	diffs := Reconcile(specs, events,
		NamedReporter{Name: "spec", Fold: SpecFold},
		NamedReporter{Name: "observer", Fold: ObserverFold},
	)
	if len(diffs) != 0 {
		t.Fatalf("well-formed events must reconcile with no divergence, got %+v", diffs)
	}

	// Spot-check the reference reduction so "agreement" is not agreement on wrong numbers.
	r := SpecFold(specs, events)
	if r[FieldTurns] != 3 {
		t.Fatalf("turns = %v, want 3", r[FieldTurns])
	}
	if r[FieldPromptTokens] != 350 {
		t.Fatalf("prompt_tokens = %v, want 350", r[FieldPromptTokens])
	}
	if r[FieldReusedTokens] != 220 {
		t.Fatalf("reused_tokens = %v, want 220", r[FieldReusedTokens])
	}
	// The hit ratio is a pure function of the covered fields: 220/350.
	if got, want := ReuseRatioFrom(r), 220.0/350.0; got < want-1e-9 || got > want+1e-9 {
		t.Fatalf("derived reuse ratio = %v, want %v", got, want)
	}
}

// TestReconcileCatchesClampDrift is the loud-failure case with two GENUINE collection paths.
// A turn reporting reused > prompt is malformed; the imperative Observer clamps reused into
// [0, prompt] while the naive declarative sum does not. The two paths therefore disagree on
// reused_tokens, and Reconcile must name exactly that field and reporter — a real drift
// between real reporters, caught structurally rather than shipped as a silent mismatch.
func TestReconcileCatchesClampDrift(t *testing.T) {
	specs := DefaultSpecs()
	events := []Event{
		// reused (500) exceeds prompt (100): ObserverFold clamps to 100, SpecFold sums 500.
		{Kind: EventTurn, PromptTokens: 100, CacheableTokens: 100, ReusedTokens: 500, EligibleTokens: 100},
	}

	diffs := Reconcile(specs, events,
		NamedReporter{Name: "spec", Fold: SpecFold},
		NamedReporter{Name: "observer", Fold: ObserverFold},
	)
	if len(diffs) == 0 {
		t.Fatal("clamp drift between the two paths must be flagged, got none")
	}
	var found *Disagreement
	for i := range diffs {
		if diffs[i].Field == FieldReusedTokens {
			found = &diffs[i]
			break
		}
	}
	if found == nil {
		t.Fatalf("expected a reused_tokens divergence, got %+v", diffs)
	}
	if found.Reporter != "observer" || found.Reference != 500 || found.Got != 100 {
		t.Fatalf("divergence = %+v, want observer reused_tokens reference=500 got=100", *found)
	}
}

// TestReconcileCatchesWrongReducer is the other loud-failure case: a reporter that reduces one
// field with the WRONG arithmetic (mean instead of sum) must be caught. This is the drift a
// hand-maintained second emitter is most likely to introduce, and the shared registry plus
// Reconcile turns it into a named, evidenced disagreement.
func TestReconcileCatchesWrongReducer(t *testing.T) {
	specs := DefaultSpecs()
	events := wellFormedTurns()

	// A buggy path that averages reused_tokens instead of summing them.
	buggy := func(specs []MetricSpec, events []Event) Report {
		r := SpecFold(specs, events)
		var n float64
		for _, e := range events {
			if e.Kind == EventTurn {
				n++
			}
		}
		if n > 0 {
			r[FieldReusedTokens] = r[FieldReusedTokens] / n // wrong: mean, not sum
		}
		return r
	}

	diffs := Reconcile(specs, events,
		NamedReporter{Name: "spec", Fold: SpecFold},
		NamedReporter{Name: "buggy", Fold: buggy},
	)
	if len(diffs) != 1 {
		t.Fatalf("wrong reducer must produce exactly one divergence, got %+v", diffs)
	}
	d := diffs[0]
	if d.Reporter != "buggy" || d.Field != FieldReusedTokens {
		t.Fatalf("divergence = %+v, want buggy reused_tokens", d)
	}
	if d.Reference != 220 || d.Got < 220.0/3-1e-9 || d.Got > 220.0/3+1e-9 {
		t.Fatalf("divergence values = (ref=%v got=%v), want (220, 220/3)", d.Reference, d.Got)
	}
}

// TestReconcileSingleReporterIsNoOp pins the boundary: with fewer than two paths there is
// nothing to cross-check, so Reconcile reports no divergence rather than comparing a path to
// itself.
func TestReconcileSingleReporterIsNoOp(t *testing.T) {
	if diffs := Reconcile(DefaultSpecs(), wellFormedTurns(), NamedReporter{Name: "spec", Fold: SpecFold}); diffs != nil {
		t.Fatalf("a single reporter has nothing to reconcile against, got %+v", diffs)
	}
}

// TestObserverFoldMatchesSnapshotAndRatio ties the declarative reduction back to the existing
// imperative surface: for well-formed events the spec-derived ratio equals the Observer's own
// Snapshot().ReuseRatio, so the registry does not invent a second, disagreeing headline.
func TestObserverFoldMatchesSnapshotAndRatio(t *testing.T) {
	events := wellFormedTurns()
	o := New()
	for _, e := range events {
		o.ObserveLabeled(Labels{}, int(e.PromptTokens), int(e.CacheableTokens), int(e.ReusedTokens), int(e.EligibleTokens))
	}
	snap := o.Snapshot()

	derived := ReuseRatioFrom(SpecFold(DefaultSpecs(), events))
	if derived < snap.ReuseRatio-1e-9 || derived > snap.ReuseRatio+1e-9 {
		t.Fatalf("spec-derived ratio %v must match imperative Snapshot ReuseRatio %v", derived, snap.ReuseRatio)
	}
}
