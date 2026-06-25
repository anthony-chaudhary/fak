package callavoid

import (
	"strings"
	"testing"
)

// gameability_test pins the #816 anti-gaming invariant for the graded amplification:
// the SPECULATIVE productive-deny axis is a counterfactual no kernel counter measures,
// so it must never move the grade, and its reported upper bound must be bounded in
// aggregate. These tests are the regression floor so the gameable
// `naive += float64(pruned)` form cannot silently return.

// TestGradeIsImmuneToRedirectFlood: a 100000×large redirect flood CANNOT move the graded
// amplification/grade/status — they are identical with vs. without the flood — and the
// speculative upper bound saturates at the aggregate cap instead of inflating to ~1e11.
func TestGradeIsImmuneToRedirectFlood(t *testing.T) {
	base := Account(Tally{Execute: 1, MemoHit: 9})
	flood := make([]int, 100000)
	for i := range flood {
		flood[i] = 1 << 20 // each far over the per-deny cap
	}
	withFlood := Account(Tally{Execute: 1, MemoHit: 9, Redirects: flood})

	// The grade-bearing fields are IDENTICAL — the speculative axis cannot touch them.
	if !approx(base.Amplification, withFlood.Amplification) ||
		base.Grade != withFlood.Grade || base.Status != withFlood.Status {
		t.Fatalf("flood moved the grade: base amp/grade/status = %v/%s/%s, with flood = %v/%s/%s",
			base.Amplification, base.Grade, base.Status,
			withFlood.Amplification, withFlood.Grade, withFlood.Status)
	}
	// The grade-relevant turn accounting is realized-only and unchanged by the flood.
	if !approx(base.EffectiveTurns, withFlood.EffectiveTurns) || !approx(base.ExecutedTurns, withFlood.ExecutedTurns) {
		t.Fatalf("flood changed the realized turn accounting: base eff/exec = %v/%v, with flood = %v/%v",
			base.EffectiveTurns, base.ExecutedTurns, withFlood.EffectiveTurns, withFlood.ExecutedTurns)
	}
	// The speculative upper bound is BOUNDED in aggregate, not ~1e11.
	if withFlood.SpeculativePrunedTurns != DefaultMaxSpeculativePrunedTurns || !withFlood.SpeculativeAggregateCapped {
		t.Fatalf("speculative pruned = %d capped=%v, want saturated at %d and flagged",
			withFlood.SpeculativePrunedTurns, withFlood.SpeculativeAggregateCapped, DefaultMaxSpeculativePrunedTurns)
	}
	if !hasRisk(withFlood.Risks, "SPECULATIVE") {
		t.Errorf("the flood window must surface a mandatory SPECULATIVE risk:\n%v", withFlood.Risks)
	}
}

// TestZeroRealWorkNeverGradesA: the original adversarial case — one execute, a huge
// redirect fan-out, zero realized avoidance — must NOT grade A. The fan-out is
// speculative; with no realized amplification the window is break-even (grade F).
func TestZeroRealWorkNeverGradesA(t *testing.T) {
	r := Account(Tally{Execute: 1, Redirects: []int{100000, 100000, 100000}})
	if r.Grade == "A" || r.Status == "amplifying" {
		t.Fatalf("zero-real-work window graded %s/%s — the speculative fan-out must not earn a grade", r.Grade, r.Status)
	}
	if r.SpeculativePrunedTurns <= 0 {
		t.Errorf("the pruned fan-out should still be reported on the speculative axis, got %d", r.SpeculativePrunedTurns)
	}
}

func hasRisk(risks []string, needle string) bool {
	for _, r := range risks {
		if strings.Contains(r, needle) {
			return true
		}
	}
	return false
}
