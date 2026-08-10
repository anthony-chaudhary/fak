package issuepolicy

import "testing"

func TestWitnessGradeAdvisoryAndStrictHold(t *testing.T) {
	base := Candidate{Schema: Schema, Key: "witness-grade", Title: "Add witness grade", CurrentState: "missing grade", WhyNow: "dispatch now", WorkingSpine: "fak issue contract", WorkUnit: "leaf", ExpectedSteps: 3, InScope: "grade witness", OutOfScope: "runtime", DoneCondition: "worker says complete", Witness: "agent reports that it finished", AcceptanceGate: "review", Lane: "issuecontract", Paths: []string{"internal/issuecontract"}}
	advisory := ReviewCandidate(base, Options{})
	if advisory.WitnessGrade.Grade != WitnessGradeForgeable {
		t.Fatalf("advisory grade = %+v", advisory.WitnessGrade)
	}
	if containsString(advisory.Reasons, ReasonWitnessForgeable) {
		t.Fatalf("advisory unexpectedly held: %+v", advisory.Reasons)
	}
	strict := ReviewCandidate(base, Options{StrictWitness: true})
	if !containsString(strict.Reasons, ReasonWitnessForgeable) || strict.Dispatchability != TriageOnly {
		t.Fatalf("strict review = %+v", strict)
	}
}

func TestWitnessGradeRecognizesIndependentOracles(t *testing.T) {
	for _, witness := range []string{
		"go test ./internal/issuecontract -run TestWitnessGrade",
		"captured JSON fixture asserts CLAIM_OUT_OF_LANE with count=1",
		"dos commit-audit returns OK and git show names the source file",
	} {
		got := witnessGrade(Candidate{DoneCondition: "effect is present", Witness: witness}, true)
		if got.Grade != WitnessGradeStrong || len(got.Flags) != 0 {
			t.Fatalf("witness %q => %+v", witness, got)
		}
	}
}
