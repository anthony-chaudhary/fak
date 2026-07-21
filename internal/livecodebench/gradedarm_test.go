package livecodebench

import (
	"strings"
	"testing"
)

// gradedSuite builds a minimal suite carrying the given question ids. The
// injected oracle grader ignores test cases, so the problems need no starter
// code or tests — the seam only forwards the Problem to the grader.
func gradedSuite(ids ...string) Suite {
	ps := make([]Problem, len(ids))
	for i, id := range ids {
		ps[i] = Problem{QuestionID: id, Scenario: ScenarioCodeGeneration}
	}
	return Suite{Model: "m", ReleaseVersion: "release_v6", Problems: ps}
}

// rawArm / fakArm build arm reports over (id -> completions) rows in order.
func rawArm(n int, rows ...RawArmProblem) RawArmReport {
	return RawArmReport{Arm: "raw", Model: "m", N: n, Temperature: 0.2, Release: "release_v6", Problems: rows}
}
func fakArm(n int, rows ...RawArmProblem) FakArmReport {
	return FakArmReport{Arm: "fak", Model: "m", N: n, Temperature: 0.2, Release: "release_v6", Problems: rows}
}

// row5 is one problem's five completions (pass@5 needs at least five samples).
func row5(id, code string) RawArmProblem {
	f := fence(code)
	return RawArmProblem{QuestionID: id, Completions: []string{f, f, f, f, f}}
}

// oracleGrader passes exactly when the extracted code equals want.
func oracleGrader(want string) CodegenGrader {
	return func(_ Problem, code string) (bool, error) { return code == want, nil }
}

// TestGradedArmDelta_FakHigher grades two witnessed arms where fak solves a
// problem raw misses, and asserts a positive, local-only pass@1 delta.
func TestGradedArmDelta_FakHigher(t *testing.T) {
	suite := gradedSuite("q1", "q2")
	raw := rawArm(5, row5("q1", "return wrong"), row5("q2", "return wrong"))
	fak := fakArm(5, row5("q1", "return ok"), row5("q2", "return wrong"))

	c, err := GradedArmDelta(suite, raw, fak, oracleGrader("return ok"))
	if err != nil {
		t.Fatalf("GradedArmDelta: %v", err)
	}
	if !c.Delta || c.Verdict != GradedVerdictLocalFakHigher {
		t.Fatalf("delta=%t verdict=%q, want delta + %q", c.Delta, c.Verdict, GradedVerdictLocalFakHigher)
	}
	if !c.FairnessWitnessed || !c.GraderAvailable {
		t.Fatalf("fences: fairness=%t grader=%t, want both true", c.FairnessWitnessed, c.GraderAvailable)
	}
	// raw solves 0/2 problems, fak solves 1/2 -> pass@1 delta = 0.5.
	if !closeEnough(c.Raw.Pass1, 0) || !closeEnough(c.Fak.Pass1, 0.5) {
		t.Fatalf("pass@1 raw/fak = %v/%v, want 0/0.5", c.Raw.Pass1, c.Fak.Pass1)
	}
	if c.Pass1Delta <= 0 {
		t.Fatalf("pass_1_delta = %v, want > 0", c.Pass1Delta)
	}
	// Honesty invariants: always local-ungraded, never claimable.
	if c.EvidenceClass != EvidenceLocalUngraded || c.ResultClaimAllowed {
		t.Fatalf("evidence=%q claim=%t, want %q + false", c.EvidenceClass, c.ResultClaimAllowed, EvidenceLocalUngraded)
	}
	if c.Schema != GradedABSchema {
		t.Fatalf("schema = %q, want %q", c.Schema, GradedABSchema)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate on a well-formed delta: %v", err)
	}
	// The markdown leads with the claim boundary and the verdict.
	md := RenderGradedABMarkdown(c)
	if !strings.Contains(md, "LOCAL sandbox signal only") || !strings.Contains(md, GradedVerdictLocalFakHigher) {
		t.Fatalf("markdown missing claim boundary or verdict:\n%s", md)
	}
}

// TestGradedArmDelta_RawHigher is the mirror: raw solves what fak misses.
func TestGradedArmDelta_RawHigher(t *testing.T) {
	suite := gradedSuite("q1", "q2")
	raw := rawArm(5, row5("q1", "return ok"), row5("q2", "return wrong"))
	fak := fakArm(5, row5("q1", "return wrong"), row5("q2", "return wrong"))

	c, err := GradedArmDelta(suite, raw, fak, oracleGrader("return ok"))
	if err != nil {
		t.Fatalf("GradedArmDelta: %v", err)
	}
	if c.Verdict != GradedVerdictLocalRawHigher || c.Pass1Delta >= 0 {
		t.Fatalf("verdict=%q delta=%v, want %q + negative", c.Verdict, c.Pass1Delta, GradedVerdictLocalRawHigher)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

// TestGradedArmDelta_Tie: identical outcomes -> zero delta, LOCAL_TIE.
func TestGradedArmDelta_Tie(t *testing.T) {
	suite := gradedSuite("q1", "q2")
	raw := rawArm(5, row5("q1", "return ok"), row5("q2", "return wrong"))
	fak := fakArm(5, row5("q1", "return ok"), row5("q2", "return wrong"))

	c, err := GradedArmDelta(suite, raw, fak, oracleGrader("return ok"))
	if err != nil {
		t.Fatalf("GradedArmDelta: %v", err)
	}
	if c.Verdict != GradedVerdictLocalTie || !closeEnough(c.Pass1Delta, 0) {
		t.Fatalf("verdict=%q delta=%v, want %q + 0", c.Verdict, c.Pass1Delta, GradedVerdictLocalTie)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

// TestGradedArmDelta_Unwitnessed: the arms ran different problems, so the
// fairness fence fails FIRST — no delta, no error, and the grader is never
// called (proving the fence gates grading).
func TestGradedArmDelta_Unwitnessed(t *testing.T) {
	suite := gradedSuite("q1", "q2")
	raw := rawArm(5, row5("q1", "return ok"), row5("q2", "return ok"))
	fak := fakArm(5, row5("q1", "return ok"), row5("q3", "return ok")) // q3 != q2

	called := false
	grade := func(_ Problem, _ string) (bool, error) { called = true; return true, nil }

	c, err := GradedArmDelta(suite, raw, fak, grade)
	if err != nil {
		t.Fatalf("GradedArmDelta: unexpected error %v (a fairness mismatch is an outcome, not an error)", err)
	}
	if called {
		t.Fatal("grader was called despite a failed fairness fence")
	}
	if c.Delta || c.Verdict != GradedVerdictUnwitnessed {
		t.Fatalf("delta=%t verdict=%q, want no delta + %q", c.Delta, c.Verdict, GradedVerdictUnwitnessed)
	}
	if c.FairnessWitnessed || c.FairnessReason == "" {
		t.Fatalf("fairness witnessed=%t reason=%q, want false + a reason", c.FairnessWitnessed, c.FairnessReason)
	}
	if c.Pass1Delta != 0 || c.Pass5Delta != 0 {
		t.Fatalf("abstain carried a delta: %v/%v", c.Pass1Delta, c.Pass5Delta)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate on an unwitnessed abstain: %v", err)
	}
}

// TestGradedArmDelta_Ungraded: witnessed arms but no grader -> honest abstain.
func TestGradedArmDelta_Ungraded(t *testing.T) {
	suite := gradedSuite("q1", "q2")
	raw := rawArm(5, row5("q1", "return ok"), row5("q2", "return ok"))
	fak := fakArm(5, row5("q1", "return ok"), row5("q2", "return ok"))

	c, err := GradedArmDelta(suite, raw, fak, nil)
	if err != nil {
		t.Fatalf("GradedArmDelta: %v", err)
	}
	if c.Delta || c.Verdict != GradedVerdictUngraded {
		t.Fatalf("delta=%t verdict=%q, want no delta + %q", c.Delta, c.Verdict, GradedVerdictUngraded)
	}
	if !c.FairnessWitnessed {
		t.Fatal("fairness should be witnessed (the arms ran the same problems)")
	}
	if c.GraderAvailable {
		t.Fatal("grader_available should be false with a nil grader")
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate on an ungraded abstain: %v", err)
	}
}

// TestGradedArmDelta_ArmIDMissingFromSuite: a witnessed, gradeable arm names a
// problem the suite does not define -> a real error (no test cases to grade).
func TestGradedArmDelta_ArmIDMissingFromSuite(t *testing.T) {
	suite := gradedSuite("q1") // q2 absent
	raw := rawArm(5, row5("q1", "return ok"), row5("q2", "return ok"))
	fak := fakArm(5, row5("q1", "return ok"), row5("q2", "return ok"))

	if _, err := GradedArmDelta(suite, raw, fak, oracleGrader("return ok")); err == nil {
		t.Fatal("expected an error when an arm question_id is absent from the suite")
	}
}

// TestGradedArmComparison_ValidateRejects checks the structural honesty
// invariants: no artifact may carry a claimable delta, promote its evidence
// class, or smuggle a number through an abstain.
func TestGradedArmComparison_ValidateRejects(t *testing.T) {
	base := func() GradedArmComparison {
		return GradedArmComparison{
			Schema:             GradedABSchema,
			Scenario:           ScenarioCodeGeneration,
			EvidenceClass:      EvidenceLocalUngraded,
			ResultClaimAllowed: false,
			FairnessWitnessed:  true,
			GraderAvailable:    true,
			Delta:              true,
			Raw:                CodegenSummary{Graded: 5},
			Fak:                CodegenSummary{Graded: 5},
		}
	}
	cases := []struct {
		name   string
		mutate func(c *GradedArmComparison)
	}{
		{"wrong schema", func(c *GradedArmComparison) { c.Schema = "other" }},
		{"official evidence class", func(c *GradedArmComparison) { c.EvidenceClass = EvidenceOfficialLCBRunner }},
		{"claim allowed", func(c *GradedArmComparison) { c.ResultClaimAllowed = true }},
		{"delta without grader", func(c *GradedArmComparison) { c.GraderAvailable = false }},
		{"delta without fairness", func(c *GradedArmComparison) { c.FairnessWitnessed = false }},
		{"delta with no graded generations", func(c *GradedArmComparison) { c.Raw.Graded = 0 }},
		{"abstain smuggling a delta", func(c *GradedArmComparison) {
			c.Delta = false
			c.Pass1Delta = 0.1
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			tc.mutate(&c)
			if err := c.Validate(); err == nil {
				t.Fatalf("Validate accepted an invalid artifact (%s)", tc.name)
			}
		})
	}

	// The base itself is valid.
	if err := base().Validate(); err != nil {
		t.Fatalf("base artifact should validate: %v", err)
	}
}
