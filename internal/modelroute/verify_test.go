package modelroute

import (
	"reflect"
	"testing"
)

// verifyApproxEq compares two coverage fractions with a small tolerance, so a
// ratio that is not exactly representable in binary float (e.g. 2/5) still asserts
// cleanly.
func verifyApproxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

// TestVerificationPredicatesFailClosed witnesses the load-bearing safety rule:
// only the two KNOWN independent tiers (judge, witness) are Trusted; only witness
// is Witnessed; and an ABSENT or UNRECOGNISED provenance is treated as unverified
// by every predicate — never silently promoted.
func TestVerificationPredicatesFailClosed(t *testing.T) {
	cases := []struct {
		v         Verification
		trusted   bool
		witnessed bool
		rank      int
		label     string
	}{
		{VerifyNone, false, false, 0, "self-reported"},
		{VerifyJudge, true, false, 1, "judge"},
		{VerifyWitness, true, true, 2, "witness"},
		// An unrecognised token: counted below, but trusted/witnessed by NOTHING.
		{Verification("bogus"), false, false, 0, "bogus"},
	}
	for _, c := range cases {
		if c.v.Trusted() != c.trusted {
			t.Errorf("%q.Trusted() = %v, want %v", c.v, c.v.Trusted(), c.trusted)
		}
		if c.v.Witnessed() != c.witnessed {
			t.Errorf("%q.Witnessed() = %v, want %v", c.v, c.v.Witnessed(), c.witnessed)
		}
		if c.v.Rank() != c.rank {
			t.Errorf("%q.Rank() = %d, want %d", c.v, c.v.Rank(), c.rank)
		}
		if c.v.Label() != c.label {
			t.Errorf("%q.Label() = %q, want %q", c.v, c.v.Label(), c.label)
		}
	}
	// The zero value of Outcome.Verify must be the fail-closed VerifyNone.
	var o Outcome
	if o.Verify != VerifyNone {
		t.Errorf("zero Outcome.Verify = %q, want VerifyNone (self-reported)", o.Verify)
	}
}

// TestVerificationCoverageMath witnesses the coverage headline: over a mix of
// self-reported / judge / witness / unknown outcomes, Trusted counts judge+witness
// only, Witnessed counts witness only, an unknown token is counted in Total but
// trusted by nothing, and the derived fractions are correct.
func TestVerificationCoverageMath(t *testing.T) {
	m := manifestForOutcomes()
	tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})

	var j OutcomeJournal
	j.Record(m.Version, tool, Outcome{Quality: 0.9, Verify: VerifyWitness})
	j.Record(m.Version, tool, Outcome{Quality: 0.8, Verify: VerifyJudge})
	j.Record(m.Version, tool, Outcome{Quality: 1.0, Verify: VerifyNone})
	j.Record(m.Version, tool, Outcome{Quality: 0.7}) // Verify zero == VerifyNone
	j.Record(m.Version, tool, Outcome{Quality: 0.5, Verify: Verification("bogus")})

	c := j.VerificationCoverage()

	if c.Total != 5 {
		t.Fatalf("Total: got %d, want 5", c.Total)
	}
	if c.Trusted != 2 { // witness + judge
		t.Errorf("Trusted: got %d, want 2 (witness+judge)", c.Trusted)
	}
	if c.Witnessed != 1 { // witness only
		t.Errorf("Witnessed: got %d, want 1 (witness only)", c.Witnessed)
	}
	if c.SelfReported() != 3 { // 2 none + 1 unknown, all fail-closed untrusted
		t.Errorf("SelfReported: got %d, want 3 (2 none + 1 unknown)", c.SelfReported())
	}
	if !verifyApproxEq(c.Coverage(), 2.0/5.0) {
		t.Errorf("Coverage: got %v, want 0.4", c.Coverage())
	}
	if !verifyApproxEq(c.WitnessCoverage(), 1.0/5.0) {
		t.Errorf("WitnessCoverage: got %v, want 0.2", c.WitnessCoverage())
	}
	// The unknown token is VISIBLE in the per-provenance breakdown (not hidden),
	// even though it never counts as trusted.
	if c.ByProvenance[Verification("bogus")] != 1 {
		t.Errorf("unknown provenance not surfaced in ByProvenance: %+v", c.ByProvenance)
	}
	if c.ByProvenance[VerifyNone] != 2 {
		t.Errorf("VerifyNone count: got %d, want 2", c.ByProvenance[VerifyNone])
	}
	// WitnessCoverage is always <= Coverage (witness is a subset of trusted).
	if c.WitnessCoverage() > c.Coverage() {
		t.Errorf("WitnessCoverage %v exceeded Coverage %v", c.WitnessCoverage(), c.Coverage())
	}
	// SortedProvenance is most-trusted-first, so witness leads.
	sorted := c.SortedProvenance()
	if len(sorted) == 0 || sorted[0] != VerifyWitness {
		t.Errorf("SortedProvenance should lead with witness: got %v", sorted)
	}
}

// TestVerificationEmptyJournal witnesses that an empty journal reports zero
// coverage without dividing by zero, and no risky buckets.
func TestVerificationEmptyJournal(t *testing.T) {
	var j OutcomeJournal
	c := j.VerificationCoverage()
	if c.Total != 0 || c.Coverage() != 0 || c.WitnessCoverage() != 0 {
		t.Errorf("empty coverage: got %+v / cov=%v / wcov=%v, want all zero", c, c.Coverage(), c.WitnessCoverage())
	}
	if len(j.RiskyBuckets(0.9)) != 0 {
		t.Errorf("empty journal has no routes, so no risky buckets: got %v", j.RiskyBuckets(0.9))
	}
}

// TestRiskyBucketsLocaliseUnsafeRoutes witnesses the actionable safety output: the
// per-(aspect,rule) fold localises WHICH buckets a lower tier serves with weak
// verification coverage, and RiskyBuckets returns exactly the buckets below the
// threshold, sorted worst-coverage-first and deterministically.
func TestRiskyBucketsLocaliseUnsafeRoutes(t *testing.T) {
	m := manifestForOutcomes()
	tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"}) // rule tool-writes
	hard := m.Route(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})
	def := m.Route(Subject{Aspect: AspectRequest}) // no rule matches => default ("")

	if def.RuleName != "" {
		t.Fatalf("setup: expected the request subject to hit the default, got rule %q", def.RuleName)
	}

	var j OutcomeJournal
	// tool bucket: 1 witness + 1 self-reported => coverage 0.5
	j.Record(m.Version, tool, Outcome{Verify: VerifyWitness})
	j.Record(m.Version, tool, Outcome{Verify: VerifyNone})
	// hard bucket: 2 self-reported => coverage 0.0 (the most unsafe)
	j.Record(m.Version, hard, Outcome{Verify: VerifyNone})
	j.Record(m.Version, hard, Outcome{Verify: VerifyNone})
	// default bucket: 1 witness => coverage 1.0 (fully covered, never risky)
	j.Record(m.Version, def, Outcome{Verify: VerifyWitness})

	byKey := j.VerificationByKey()
	if len(byKey) != 3 {
		t.Fatalf("VerificationByKey: got %d buckets, want 3", len(byKey))
	}
	toolKey := AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"}
	if got := byKey[toolKey].Coverage(); !verifyApproxEq(got, 0.5) {
		t.Errorf("tool bucket coverage: got %v, want 0.5", got)
	}

	// Threshold 0.75: tool (0.5) and hard (0.0) are unsafe; default (1.0) is safe.
	risky := j.RiskyBuckets(0.75)
	if len(risky) != 2 {
		t.Fatalf("RiskyBuckets(0.75): got %d, want 2 (tool + hard, not default)", len(risky))
	}
	// Worst-coverage-first: hard (0.0) must precede tool (0.5).
	if risky[0].Key.Rule != "hard" {
		t.Errorf("RiskyBuckets order: got %q first, want the worst (hard, coverage 0.0)", risky[0].Key.Rule)
	}
	if risky[1].Key.Rule != "tool-writes" {
		t.Errorf("RiskyBuckets order: got %q second, want tool-writes", risky[1].Key.Rule)
	}
	// The safe default bucket must NOT appear.
	for _, b := range risky {
		if b.Key.Rule == "" {
			t.Errorf("fully-covered default bucket leaked into RiskyBuckets: %+v", b)
		}
	}

	// A threshold of 0 flags nothing (no coverage can be strictly < 0); a threshold
	// clamped from >1 down to 1 flags every bucket that is not fully covered.
	if got := len(j.RiskyBuckets(0)); got != 0 {
		t.Errorf("RiskyBuckets(0): got %d, want 0 (nothing is below zero coverage)", got)
	}
	if got := len(j.RiskyBuckets(2.0)); got != 2 { // clamped to 1.0; only the 1.0 default is safe
		t.Errorf("RiskyBuckets(2.0->clamped 1.0): got %d, want 2 (all but the fully-covered default)", got)
	}
}

// TestVerificationFoldDeterministic witnesses that the coverage fold is pure and
// order-independent: the same outcomes journaled in any order fold to an equal
// VerificationCounts, and folding one journal twice is stable.
func TestVerificationFoldDeterministic(t *testing.T) {
	m := manifestForOutcomes()
	tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})

	verifs := []Verification{VerifyWitness, VerifyNone, VerifyJudge, VerifyNone, VerifyWitness}

	var jA OutcomeJournal
	for _, v := range verifs {
		jA.Record(m.Version, tool, Outcome{Verify: v})
	}
	var jB OutcomeJournal
	for i := len(verifs) - 1; i >= 0; i-- {
		jB.Record(m.Version, tool, Outcome{Verify: verifs[i]})
	}

	if !reflect.DeepEqual(jA.VerificationCoverage(), jB.VerificationCoverage()) {
		t.Fatalf("coverage fold is order-dependent:\n A=%+v\n B=%+v", jA.VerificationCoverage(), jB.VerificationCoverage())
	}
	if !reflect.DeepEqual(jA.VerificationCoverage(), jA.VerificationCoverage()) {
		t.Fatal("VerificationCoverage is not pure: two folds of one journal differ")
	}
}
