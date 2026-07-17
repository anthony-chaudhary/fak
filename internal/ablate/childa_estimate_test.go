package ablate

import "testing"

// #2831 opened KnownFeatures into a concept registry and made every arm row carry
// {token delta, provider-vs-fak split, correctness verdict, CHILD-A estimate when
// present}. registry_test.go's TestConceptRowsCarryRequiredDerivedFields pins the
// first three fields but leaves the "CHILD-A estimate when present" clause of the
// acceptance unwitnessed: annotateConceptRows copies concept.ChildAOwnServeEstimate()
// onto the arm row, yet nothing proved a supplied estimate survives the fold — nor,
// symmetrically, that a concept without one leaves the pointer nil instead of a
// fabricated zero. This closes both halves so the estimator child (#2829) has a
// witnessed seam to attach its dollar figure to.
func TestConceptRowCarriesChildAEstimateWhenPresent(t *testing.T) {
	const token = "zz_childa_estimate_concept"
	const want = 4.75
	est := want
	Register(Concept{Token: token, Runtime: func(bool) {}, Owner: "fak", Reversible: true, PrefixStable: true,
		ChildAOwnServeEstimate: func() *float64 { return &est }})

	rep := &Report{Baseline: "all-off", Runs: []AblationRun{
		{ArmID: "all-off", Arm: structArm(100, 20)},
		{ArmID: token, Features: map[string]string{token: "on"}, Arm: structArm(80, 10)},
	}}
	rep.annotateConceptRows()

	row := rep.Runs[1]
	if row.ChildAOwnServeEstimate == nil {
		t.Fatal("arm row dropped the CHILD-A own-serve estimate the concept supplied")
	}
	if got := *row.ChildAOwnServeEstimate; got != want {
		t.Fatalf("CHILD-A estimate = %v, want %v", got, want)
	}

	// "when present": the baseline arm turns no concept on, so its row must keep a nil
	// estimate pointer — the fold never fabricates a zero for an arm that carries none.
	if base := rep.Runs[0]; base.ChildAOwnServeEstimate != nil {
		t.Fatalf("baseline arm fabricated a CHILD-A estimate: %v", *base.ChildAOwnServeEstimate)
	}
}
