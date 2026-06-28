package callavoid

import (
	"reflect"
	"testing"
)

// coverage_test.go — regression floor for #820 acceptance (1): the witnessed
// productive-deny fan-out is produced FROM a deny rule's structural coverage, not a guess.
// The crediting side (acceptance 2 & 3) is covered in witness_test.go; these tests bind
// the PRODUCER: a witness may only credit variants the rule structurally covers, the
// fan-out is bounded by the rule's own declaration, and a deny that covers nothing earns
// no realized credit.

// TestWitnessFromCoverageCreditsOnlyCovered: a productive deny credits exactly the pruned
// variants that lie in the rule's declared structural coverage. A "pruned" variant the
// rule does NOT cover is a guess and is dropped — it can never reach the realized headline.
func TestWitnessFromCoverageCreditsOnlyCovered(t *testing.T) {
	cov := DenyRuleCoverage{
		Rule:    "policy.read-scope",
		Covered: []string{"read:/etc", "read:/var", "read:/root"},
	}
	// The deny pruned two covered variants plus one OUTSIDE the rule's coverage ("read:/home"
	// — a guess) and a blank.
	w := WitnessFromCoverage(cov, []string{"read:/etc", "read:/home", "", "read:/root"})
	if w.Rule != "policy.read-scope" {
		t.Errorf("rule = %q, want policy.read-scope", w.Rule)
	}
	// Only the two in-coverage variants survive; the out-of-coverage guess and blank are gone.
	if !reflect.DeepEqual(w.Variants, []string{"read:/etc", "read:/root"}) {
		t.Fatalf("variants = %v, want [read:/etc read:/root] (out-of-coverage guess dropped)", w.Variants)
	}
	// And the witness credits realized fan-out 2 through Account — grounded, not asserted.
	r := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{w}})
	if r.WitnessedDenies != 1 || r.WitnessedPruned != 2 {
		t.Errorf("witnessed denies/pruned = %d/%d, want 1/2", r.WitnessedDenies, r.WitnessedPruned)
	}
	if !approx(r.Amplification, 3) { // 1 execute + 2 grounded variants = 3 effective / 1 paid.
		t.Errorf("amplification = %v, want 3 (structural fan-out is realized credit)", r.Amplification)
	}
}

// TestWitnessFromCoverageRefusesPureGuess: a deny that pruned variants NONE of which the
// rule covers yields a variant-less witness — it credits nothing and is surfaced as an
// empty witness. A "productive" deny whose claimed pruning is entirely outside its
// structural coverage can never inflate the realized headline.
func TestWitnessFromCoverageRefusesPureGuess(t *testing.T) {
	cov := DenyRuleCoverage{Rule: "policy.x", Covered: []string{"a", "b"}}
	w := WitnessFromCoverage(cov, []string{"c", "d"}) // both outside coverage — pure guess.
	if len(w.Variants) != 0 {
		t.Fatalf("variants = %v, want none (all pruned variants are outside the rule's coverage)", w.Variants)
	}
	r := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{w}})
	if r.WitnessedDenies != 0 || r.WitnessedPruned != 0 {
		t.Errorf("witnessed denies/pruned = %d/%d, want 0/0 (a pure guess credits nothing)", r.WitnessedDenies, r.WitnessedPruned)
	}
	if r.WitnessedEmpty != 1 {
		t.Errorf("witnessed empty = %d, want 1 (the guess is surfaced as an empty witness)", r.WitnessedEmpty)
	}
	if !approx(r.Amplification, 1) || r.Status != "break_even" {
		t.Errorf("amp/status = %v/%s, want 1/break_even", r.Amplification, r.Status)
	}
}

// TestWitnessFromCoverageDedupsAndSorts: the producer is non-gameable on enumeration —
// repeating a covered variant credits it once — and emits a deterministic (sorted) order so
// the witness is auditable.
func TestWitnessFromCoverageDedupsAndSorts(t *testing.T) {
	cov := DenyRuleCoverage{Covered: []string{"x", "y", "z"}}
	w := WitnessFromCoverage(cov, []string{"z", "x", "x", "y", "z"})
	if !reflect.DeepEqual(w.Variants, []string{"x", "y", "z"}) {
		t.Fatalf("variants = %v, want [x y z] (deduped + sorted)", w.Variants)
	}
}

// TestWitnessFromCoverageHonorsRuleMaxFanout: a rule's OWN declared MaxFanout caps the
// witness below the package per-deny cap — a rule that covers a large domain but declares
// it only ever prunes a small bounded slice can never credit more than it claims.
func TestWitnessFromCoverageHonorsRuleMaxFanout(t *testing.T) {
	covered := []string{"a", "b", "c", "d", "e"}
	cov := DenyRuleCoverage{Rule: "cap.fs", Covered: covered, MaxFanout: 2}
	w := WitnessFromCoverage(cov, covered) // pruned all five, but the rule declares a ceiling of 2.
	if len(w.Variants) != 2 {
		t.Fatalf("variants = %v, want 2 (clamped to the rule's declared MaxFanout)", w.Variants)
	}
	r := Account(Tally{Execute: 1, WitnessedRedirects: []WitnessedRedirect{w}})
	if r.WitnessedPruned != 2 {
		t.Errorf("witnessed pruned = %d, want 2 (the rule's own ceiling binds)", r.WitnessedPruned)
	}
}

// TestWitnessesFromCoverageBatch: a batch of fired deny rules maps to one witness each, in
// order, ready to hand to TallyFromCountersWitnessed. A productive deny grounded in
// coverage credits; one whose pruning is entirely uncovered is surfaced as empty.
func TestWitnessesFromCoverageBatch(t *testing.T) {
	fired := []FiredDeny{
		{Coverage: DenyRuleCoverage{Rule: "r1", Covered: []string{"a", "b"}}, Pruned: []string{"a", "b"}},
		{Coverage: DenyRuleCoverage{Rule: "r2", Covered: []string{"x"}}, Pruned: []string{"q"}}, // uncovered guess.
	}
	wits := WitnessesFromCoverage(fired)
	if len(wits) != 2 || wits[0].Rule != "r1" || wits[1].Rule != "r2" {
		t.Fatalf("batch shape wrong: %+v", wits)
	}
	if len(wits[0].Variants) != 2 || len(wits[1].Variants) != 0 {
		t.Errorf("variants = %v / %v, want 2 covered for r1, 0 for the r2 guess", wits[0].Variants, wits[1].Variants)
	}

	// End-to-end: fold against the session counters. Two denies recorded; one witnessed
	// productive (r1), one effectively hard (r2's pruning was uncovered).
	tally := TallyFromCountersWitnessed(Counters{EngineCalls: 1, Denies: 2}, wits)
	r := Account(tally)
	if r.WitnessedDenies != 1 || r.WitnessedPruned != 2 {
		t.Errorf("witnessed denies/pruned = %d/%d, want 1/2", r.WitnessedDenies, r.WitnessedPruned)
	}
	if r.WitnessedEmpty != 1 {
		t.Errorf("witnessed empty = %d, want 1 (r2's uncovered pruning)", r.WitnessedEmpty)
	}
	// A grounded productive deny adds realized credit; the uncovered one does NOT.
	if !approx(r.EffectiveTurns, 3) { // 1 execute + 2 grounded variants.
		t.Errorf("effective = %v, want 3 (only the covered deny amplifies)", r.EffectiveTurns)
	}
}

// TestWitnessFromCoverageDeterministic: pure — identical coverage + pruning yields an
// identical witness.
func TestWitnessFromCoverageDeterministic(t *testing.T) {
	cov := DenyRuleCoverage{Rule: "r", Covered: []string{"a", "b", "c"}, MaxFanout: 2}
	a := WitnessFromCoverage(cov, []string{"c", "a", "b"})
	b := WitnessFromCoverage(cov, []string{"c", "a", "b"})
	if !reflect.DeepEqual(a, b) {
		t.Errorf("WitnessFromCoverage is not deterministic: %v vs %v", a, b)
	}
}
