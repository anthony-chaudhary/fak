package metrics

import (
	"strings"
	"testing"
)

func labeledFigure(account, metric string, p SpendProvenance) SpendFigure {
	return SpendFigure{
		Account:        account,
		Metric:         metric,
		Value:          1,
		Unit:           "count",
		ValuationBasis: "fak watchdog session registry (sessions.json)",
		Provenance:     p,
	}
}

// TestGateSpendLabeledPassesFullyLabeledRollup locks the pass condition: every
// figure carries a valuation basis and a closed-set provenance label.
func TestGateSpendLabeledPassesFullyLabeledRollup(t *testing.T) {
	r := SpendRollup{Schema: SpendRollupSchema, Figures: []SpendFigure{
		labeledFigure("alpha", "live_sessions", SpendWitnessed),
		labeledFigure("alpha", "usage_throttled", SpendObserved),
	}}
	if defects := GateSpendLabeled(r); len(defects) != 0 {
		t.Fatalf("fully labeled rollup failed the gate: %v", defects)
	}
}

// TestGateSpendLabeledFailsUnlabeledFigure is #2903's witness: a spend figure
// with no provenance label fails the rollup.
func TestGateSpendLabeledFailsUnlabeledFigure(t *testing.T) {
	unlabeled := labeledFigure("alpha", "live_sessions", "")
	r := SpendRollup{Schema: SpendRollupSchema, Figures: []SpendFigure{unlabeled}}
	defects := GateSpendLabeled(r)
	if len(defects) != 1 {
		t.Fatalf("unlabeled figure: got %d defects, want 1: %v", len(defects), defects)
	}
	if !strings.Contains(defects[0], "unlabeled") {
		t.Fatalf("defect does not name the unlabeled figure: %q", defects[0])
	}
}

// TestGateSpendLabeledFailsOpenSetProvenance locks the set closed: a plausible
// but unrecognized token ("provider-reported") is unlabeled, not a third label.
func TestGateSpendLabeledFailsOpenSetProvenance(t *testing.T) {
	r := SpendRollup{Schema: SpendRollupSchema, Figures: []SpendFigure{
		labeledFigure("alpha", "usage_throttled", "provider-reported"),
	}}
	if defects := GateSpendLabeled(r); len(defects) != 1 {
		t.Fatalf("open-set provenance: got %d defects, want 1: %v", len(defects), defects)
	}
}

// TestGateSpendLabeledFailsMissingValuationBasis: a labeled provenance does not
// excuse a figure from naming what the number is denominated in.
func TestGateSpendLabeledFailsMissingValuationBasis(t *testing.T) {
	f := labeledFigure("alpha", "live_sessions", SpendWitnessed)
	f.ValuationBasis = ""
	r := SpendRollup{Schema: SpendRollupSchema, Figures: []SpendFigure{f}}
	defects := GateSpendLabeled(r)
	if len(defects) != 1 || !strings.Contains(defects[0], "valuation basis") {
		t.Fatalf("missing basis: got %v, want one valuation-basis defect", defects)
	}
}

// TestWeakestSpendProvenance locks the no-upgrade rule: one OBSERVED input makes
// the total OBSERVED, and any input outside the closed set poisons the total to
// unlabeled (which the gate then refuses).
func TestWeakestSpendProvenance(t *testing.T) {
	cases := []struct {
		name string
		in   []SpendProvenance
		want SpendProvenance
	}{
		{"all witnessed stays witnessed", []SpendProvenance{SpendWitnessed, SpendWitnessed}, SpendWitnessed},
		{"one observed drags the total", []SpendProvenance{SpendWitnessed, SpendObserved}, SpendObserved},
		{"unknown token poisons to unlabeled", []SpendProvenance{SpendWitnessed, "guessed"}, ""},
		{"empty input is unlabeled", nil, ""},
	}
	for _, tc := range cases {
		if got := WeakestSpendProvenance(tc.in...); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}
