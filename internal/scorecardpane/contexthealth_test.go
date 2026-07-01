package scorecardpane

import "testing"

// contexthealth_test.go — issue #1579 witness: `go test ./internal/scorecardpane
// ./cmd/fak -run ContextHealth`.

// TestContextHealthSeverityClosedVocabulary mirrors memview's
// TestProvenanceEventKindClosedVocabulary shape: every declared severity is a member
// of the closed set, and an unrecognized value fails closed to "unknown(...)" rather
// than silently rendering as if it were a real severity.
func TestContextHealthSeverityClosedVocabulary(t *testing.T) {
	all := []ContextHealthSeverity{
		ContextHealthFresh, ContextHealthConstrained, ContextHealthQueryNeeded,
		ContextHealthStaleRisk, ContextHealthBudgetDraining, ContextHealthResetImminent,
	}
	if len(all) != len(validContextHealthSeverities) {
		t.Fatalf("declared %d severities but membership set has %d — keep them in sync",
			len(all), len(validContextHealthSeverities))
	}
	for _, s := range all {
		if !ValidContextHealthSeverity(s) {
			t.Errorf("%q should be a valid ContextHealthSeverity", s)
		}
		if s.String() != string(s) {
			t.Errorf("String() of a valid severity must be itself, got %q", s.String())
		}
	}
	bogus := ContextHealthSeverity("on_fire")
	if ValidContextHealthSeverity(bogus) {
		t.Error("bogus severity must not be valid")
	}
	if bogus.String() != "unknown(on_fire)" {
		t.Errorf("String() of an invalid severity = %q, want unknown(on_fire)", bogus.String())
	}
	if ContextHealthSeverity("").String() != "(unset)" {
		t.Errorf("String() of empty severity = %q, want (unset)", ContextHealthSeverity("").String())
	}
}

// TestClassifyContextHealthAllClearIsFresh proves the zero-signal input classifies
// as fresh — the all-clear default, not an omitted/zero-value accident.
func TestClassifyContextHealthAllClearIsFresh(t *testing.T) {
	got := ClassifyContextHealth(ContextHealthSignals{})
	if got != ContextHealthFresh {
		t.Fatalf("ClassifyContextHealth(zero signals) = %q, want fresh", got)
	}
}

// TestClassifyContextHealthSingleSignals proves each individual failing area maps to
// its named severity from the issue's In-scope line.
func TestClassifyContextHealthSingleSignals(t *testing.T) {
	cases := []struct {
		name   string
		in     ContextHealthSignals
		wanted ContextHealthSeverity
	}{
		{"visibility", ContextHealthSignals{VisibilityFailing: true}, ContextHealthConstrained},
		{"query", ContextHealthSignals{QueryFailing: true}, ContextHealthQueryNeeded},
		{"cache", ContextHealthSignals{CacheFailing: true}, ContextHealthStaleRisk},
		{"memory", ContextHealthSignals{MemoryFailing: true}, ContextHealthStaleRisk},
		{"budget", ContextHealthSignals{BudgetFailing: true}, ContextHealthBudgetDraining},
		{"reset", ContextHealthSignals{ResetFailing: true}, ContextHealthResetImminent},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyContextHealth(tc.in)
			if got != tc.wanted {
				t.Errorf("ClassifyContextHealth(%+v) = %q, want %q", tc.in, got, tc.wanted)
			}
		})
	}
}

// TestClassifyContextHealthWorstFirstIsDeterministic proves that when multiple areas
// fail at once, the classifier always returns the single most-severe verdict
// (reset-imminent outranks everything), regardless of which other signals are also
// set — the product surface renders one closed enum, never a free-text list.
func TestClassifyContextHealthWorstFirstIsDeterministic(t *testing.T) {
	got := ClassifyContextHealth(ContextHealthSignals{
		VisibilityFailing: true, QueryFailing: true, CacheFailing: true,
		BudgetFailing: true, ResetFailing: true, MemoryFailing: true,
	})
	if got != ContextHealthResetImminent {
		t.Fatalf("ClassifyContextHealth(all failing) = %q, want reset_imminent (worst-first)", got)
	}

	got = ClassifyContextHealth(ContextHealthSignals{BudgetFailing: true, QueryFailing: true, VisibilityFailing: true})
	if got != ContextHealthBudgetDraining {
		t.Fatalf("ClassifyContextHealth(budget+query+visibility) = %q, want budget_draining", got)
	}
}

// TestContextHealthSignalsFromSLORowsMapsAreas proves the JSON-row adapter reads the
// same {area,status} shape internal/productscorecard.ManagedContextSLOReport emits,
// without scorecardpane importing that (tier-1-only) package.
func TestContextHealthSignalsFromSLORowsMapsAreas(t *testing.T) {
	rows := []map[string]any{
		{"area": "visibility", "status": "pass"},
		{"area": "reset", "status": "FAIL"},
		{"area": "budget", "status": "pass"},
		{"area": "query", "status": "missing"},
		{"area": "cache", "status": "pass"},
		{"area": "memory", "status": "pass"},
	}
	sig := ContextHealthSignalsFromSLORows(rows)
	if sig.VisibilityFailing || sig.BudgetFailing || sig.CacheFailing || sig.MemoryFailing {
		t.Fatalf("passing rows must not be marked failing: %+v", sig)
	}
	if !sig.ResetFailing {
		t.Error("reset row status=FAIL must mark ResetFailing (case-insensitive)")
	}
	if !sig.QueryFailing {
		t.Error("query row status=missing must mark QueryFailing (anything but pass is failing)")
	}
	got := ClassifyContextHealth(sig)
	if got != ContextHealthResetImminent {
		t.Fatalf("ClassifyContextHealth(from rows) = %q, want reset_imminent", got)
	}
}

// TestContextHealthSignalsFromSLORowsEmptyIsAllClear proves an empty/nil row set (no
// managed-context SLO data available) classifies as fresh, not an error — mirroring
// ManagedContextSLOReport's own "no data" -> nil posture.
func TestContextHealthSignalsFromSLORowsEmptyIsAllClear(t *testing.T) {
	sig := ContextHealthSignalsFromSLORows(nil)
	if sig.AnyFailing() {
		t.Fatalf("empty rows must produce no failing signals, got %+v", sig)
	}
	if got := ClassifyContextHealth(sig); got != ContextHealthFresh {
		t.Fatalf("ClassifyContextHealth(from empty rows) = %q, want fresh", got)
	}
}
