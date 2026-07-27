package modelroute

import (
	"reflect"
	"testing"
)

// Witnesses for capability grading (epic #5416, track F).
//
// The placer already refuses to move work onto cheap hardware without a MEASURED
// capability. These are the tests of what "measured" costs — and every one of them is a
// test that the fold declines a grade it has not earned, because the failure that matters
// here is not "we graded too harshly", it is "we let a forgeable claim move real traffic".

// judged / witnessed build evidence rows at the two independent provenance tiers.
func judged(c WorkClass, attempts, successes int64) ClassEvidence {
	return ClassEvidence{Class: c, Attempts: attempts, Successes: successes, Verify: VerifyJudge}
}

func witnessed(c WorkClass, attempts, successes int64) ClassEvidence {
	return ClassEvidence{Class: c, Attempts: attempts, Successes: successes, Verify: VerifyWitness}
}

func TestASelfReportedSuccessNeverBuysAGrade(t *testing.T) {
	// A perfect record — 500 attempts, 500 successes — that the model reported about
	// itself. This is the one input that must never move traffic.
	g := GradeCapability("tiny", []ClassEvidence{
		{Class: ClassRoutine, Attempts: 500, Successes: 500, Verify: VerifyNone},
	}, DefaultGradeFloor())

	if g.Measured {
		t.Fatalf("self-report bought a grade: %+v", g)
	}
	if g.Reason != ReasonNoTrustedEvidence {
		t.Errorf("reason = %q, want %q", g.Reason, ReasonNoTrustedEvidence)
	}
	if g.Dropped != 500 {
		t.Errorf("dropped = %d, want 500 — the evidence must be reported as REFUSED, not as absent", g.Dropped)
	}
	if g.Capability != TierT0 {
		t.Errorf("an ungraded model's capability = %s, want the zero value so a caller that "+
			"ignores Measured fails toward the most demanding rung", g.Capability)
	}
}

func TestAnIndependentlyCheckedRecordBuysTheGrade(t *testing.T) {
	g := GradeCapability("tiny", []ClassEvidence{judged(ClassRoutine, 20, 18)}, DefaultGradeFloor())
	if !g.Measured {
		t.Fatalf("20 judge-scored attempts at 90%% did not clear a 20/80%% floor: %+v", g)
	}
	if g.Capability != TierT2 {
		t.Errorf("capability = %s, want T2 (the routine floor)", g.Capability)
	}
	if g.Class != ClassRoutine || g.Attempts != 20 || g.Successes != 18 {
		t.Errorf("the grade does not carry its own evidence trail: %+v", g)
	}
	if g.Reason != ReasonGradedFromEvidence {
		t.Errorf("reason = %q, want %q", g.Reason, ReasonGradedFromEvidence)
	}
}

func TestRequireWitnessRefusesAJudgeScore(t *testing.T) {
	ev := []ClassEvidence{judged(ClassRoutine, 40, 40)}
	floor := DefaultGradeFloor()
	if g := GradeCapability("tiny", ev, floor); !g.Measured {
		t.Fatalf("precondition: a judge score should grade under the default floor: %+v", g)
	}
	floor.RequireWitness = true
	g := GradeCapability("tiny", ev, floor)
	if g.Measured {
		t.Fatalf("RequireWitness accepted a relayed model opinion: %+v", g)
	}
	if g.Dropped != 40 || g.Reason != ReasonNoTrustedEvidence {
		t.Errorf("the refusal is not reported as one: %+v", g)
	}
	// The same fleet, the same numbers, established from git evidence instead.
	if g := GradeCapability("tiny", []ClassEvidence{witnessed(ClassRoutine, 40, 40)}, floor); !g.Measured {
		t.Errorf("RequireWitness refused a DOS-witnessed record: %+v", g)
	}
}

func TestAnUnrecognisedClassLabelCannotMintAGrade(t *testing.T) {
	// The hole this closes: PolicyFor maps an unknown class to the T0 floor — the right
	// conservatism when picking a floor for WORK, and a capability-minting bug when read
	// backwards. A typo must not grade a 4B local model at the frontier tier.
	g := GradeCapability("tiny", []ClassEvidence{
		{Class: "rutine", Attempts: 400, Successes: 400, Verify: VerifyWitness},
	}, DefaultGradeFloor())
	if g.Measured {
		t.Fatalf("a misspelled class label minted a %s grade: %+v", g.Capability, g)
	}
	if g.Dropped != 400 {
		t.Errorf("dropped = %d, want 400", g.Dropped)
	}
}

func TestTheGradeIsTheFloorOfTheWorkNotItsOptimalTier(t *testing.T) {
	// Security/release work has floor T1 and optimal T0. Doing it successfully evidences
	// that the model met the FLOOR; it says nothing about what an operator would prefer
	// to spend on the class, which is what optimal means.
	g := GradeCapability("corp-mid", []ClassEvidence{witnessed(ClassSecurityRelease, 50, 50)}, DefaultGradeFloor())
	if !g.Measured {
		t.Fatalf("50/50 witnessed attempts did not grade: %+v", g)
	}
	if want := PolicyFor(ClassSecurityRelease).RequiredTier; g.Capability != want {
		t.Errorf("capability = %s, want %s (the floor, not the optimal tier)", g.Capability, want)
	}
	if g.Capability == PolicyFor(ClassSecurityRelease).OptimalTier {
		t.Errorf("a perfect record on security work was read as frontier capability: %+v", g)
	}
}

func TestTheMostDemandingClassThatClearedTheBarWins(t *testing.T) {
	g := GradeCapability("frontier", []ClassEvidence{
		judged(ClassRoutine, 100, 100),
		witnessed(ClassUltraHard, 25, 22),
		judged(ClassNormalImpl, 40, 39),
	}, DefaultGradeFloor())
	if !g.Measured || g.Capability != TierT0 {
		t.Fatalf("capability = %s (measured=%v), want T0 from the ultra-hard evidence: %+v",
			g.Capability, g.Measured, g)
	}
	if g.Class != ClassUltraHard {
		t.Errorf("the grade credits class %q, want %q", g.Class, ClassUltraHard)
	}
	// Order must not decide it.
	rev := GradeCapability("frontier", []ClassEvidence{
		judged(ClassNormalImpl, 40, 39),
		witnessed(ClassUltraHard, 25, 22),
		judged(ClassRoutine, 100, 100),
	}, DefaultGradeFloor())
	if !reflect.DeepEqual(g, rev) {
		t.Errorf("the fold is order-dependent:\n %+v\n %+v", g, rev)
	}

	// The discriminating case: the most demanding qualifying class is NOT the one that
	// happens to sort last. A model that clears both ordinary implementation and routine
	// work is a T1 model — grading it T2 would keep it off the fleet rung for work it is
	// demonstrably doing, which is the same waste as the vendor escalation, one rung down.
	mid := GradeCapability("corp-mid", []ClassEvidence{
		witnessed(ClassNormalImpl, 40, 36),
		witnessed(ClassRoutine, 90, 90),
	}, DefaultGradeFloor())
	if !mid.Measured || mid.Capability != TierT1 {
		t.Fatalf("capability = %s (measured=%v), want T1 — the most demanding class that "+
			"cleared the bar, not the last one examined: %+v", mid.Capability, mid.Measured, mid)
	}
	if mid.Class != ClassNormalImpl || mid.Attempts != 40 {
		t.Errorf("the grade credits the wrong class's evidence: %+v", mid)
	}
}

func TestFailingEveryBarIsNotAGradeOfTheWorstTier(t *testing.T) {
	// 50% on routine work. The tempting answer is "well, T2 then" — but T2 is a positive
	// claim that this model CAN serve routine work, and the evidence is that it could not.
	g := GradeCapability("tiny", []ClassEvidence{judged(ClassRoutine, 40, 20)}, DefaultGradeFloor())
	if g.Measured {
		t.Fatalf("a model that failed its only bar was graded %s: %+v", g.Capability, g)
	}
	if g.Reason != ReasonBelowSuccessFloor {
		t.Errorf("reason = %q, want %q — the operator needs to know the samples were there",
			g.Reason, ReasonBelowSuccessFloor)
	}
	if g.Dropped != 0 {
		t.Errorf("dropped = %d, want 0: this evidence was counted and lost, not refused", g.Dropped)
	}
}

func TestTooLittleEvidenceIsInsufficientNotFailure(t *testing.T) {
	// Three for three is a perfect record and not a measurement. The two reasons are kept
	// distinct because they call for opposite responses: run more work, or stop trying.
	g := GradeCapability("tiny", []ClassEvidence{witnessed(ClassRoutine, 3, 3)}, DefaultGradeFloor())
	if g.Measured {
		t.Fatalf("3 attempts cleared a 20-attempt floor: %+v", g)
	}
	if g.Reason != ReasonInsufficientSamples {
		t.Errorf("reason = %q, want %q", g.Reason, ReasonInsufficientSamples)
	}
}

func TestSameClassRowsMergeBeforeTheyAreJudged(t *testing.T) {
	// Two half-sized runs of the same class are one body of evidence, not two failures to
	// reach the floor.
	g := GradeCapability("tiny", []ClassEvidence{
		witnessed(ClassRoutine, 10, 9),
		judged(ClassRoutine, 10, 9),
	}, DefaultGradeFloor())
	if !g.Measured || g.Attempts != 20 || g.Successes != 18 {
		t.Fatalf("rows did not merge per class: %+v", g)
	}
	// A grade is only as good as its weakest link: merging a witnessed run with a judged
	// one yields a judged grade, never a witnessed one.
	if g.Verify != VerifyJudge {
		t.Errorf("verify = %q, want %q (the WEAKEST provenance behind the grade)", g.Verify, VerifyJudge)
	}
}

func TestMalformedEvidenceIsClampedDownwards(t *testing.T) {
	g := GradeCapability("tiny", []ClassEvidence{
		{Class: ClassRoutine, Attempts: 20, Successes: 999, Verify: VerifyWitness},
	}, DefaultGradeFloor())
	if g.Successes != 20 {
		t.Errorf("successes = %d, want 20 — a producer bug must not inflate the record", g.Successes)
	}
	neg := GradeCapability("tiny", []ClassEvidence{
		{Class: ClassRoutine, Attempts: 40, Successes: -40, Verify: VerifyWitness},
	}, DefaultGradeFloor())
	if neg.Measured || neg.Reason != ReasonBelowSuccessFloor {
		t.Errorf("negative successes did not clamp to zero: %+v", neg)
	}
	// Zero-attempt rows are not evidence in either direction: not counted, not refused.
	empty := GradeCapability("tiny", []ClassEvidence{witnessed(ClassRoutine, 0, 0)}, DefaultGradeFloor())
	if empty.Measured || empty.Dropped != 0 || empty.Reason != ReasonNoTrustedEvidence {
		t.Errorf("an empty row was treated as evidence: %+v", empty)
	}
}

func TestAGradedModelIsWhatFinallyLetsWorkDescendTheLadder(t *testing.T) {
	// The payoff, end to end: evidence in, cheap rung out. This is the loop epic #5416
	// exists to close — nothing else in the system can move a token off a vendor.
	r := threeZoneRoster()
	evidence := map[string][]ClassEvidence{
		"tiny":     {witnessed(ClassRoutine, 60, 57)},
		"corp-mid": {witnessed(ClassNormalImpl, 60, 54)},
	}
	grades := GradeCandidates([]string{"frontier", "corp-mid", "tiny", "corp-agentic"}, evidence, DefaultGradeFloor())
	if len(grades) != 4 {
		t.Fatalf("grades = %d, want 4 (an ungraded model stays in the pool)", len(grades))
	}
	if grades[0].Model != "corp-agentic" || grades[3].Model != "tiny" {
		t.Fatalf("grades are not in a deterministic order: %+v", grades)
	}
	var candidates []Candidate
	for _, g := range grades {
		candidates = append(candidates, g.Candidate())
	}

	routine, err := r.Place(ClassRoutine, candidates)
	if err != nil {
		t.Fatalf("Place(routine): %v", err)
	}
	if routine.Zone != ZoneDevice || !routine.SelfHosted() || !routine.Measured {
		t.Errorf("routine work did not reach the engineer's own machine: zone=%s measured=%v",
			routine.Zone, routine.Measured)
	}
	impl, err := r.Place(ClassNormalImpl, candidates)
	if err != nil {
		t.Fatalf("Place(normal-impl): %v", err)
	}
	if impl.Zone != ZoneFleet || impl.Model != "corp-mid" {
		t.Errorf("ordinary implementation did not reach company hardware: zone=%s model=%s",
			impl.Zone, impl.Model)
	}
	// And the model nobody graded still cannot descend: ultra-hard work goes to the vendor
	// because that is the only rung an ungraded candidate may serve.
	hard, err := r.Place(ClassUltraHard, candidates)
	if err != nil {
		t.Fatalf("Place(ultra-hard): %v", err)
	}
	if hard.Zone != ZoneVendor || hard.Measured {
		t.Errorf("ultra-hard work: zone=%s measured=%v, want the vendor on an ungraded fallback",
			hard.Zone, hard.Measured)
	}
}

func TestGradeCandidatesKeepsEveryBoundModelAndDedupes(t *testing.T) {
	got := GradeCandidates([]string{"b", "a", "b", ""}, nil, DefaultGradeFloor())
	if len(got) != 2 || got[0].Model != "a" || got[1].Model != "b" {
		t.Fatalf("GradeCandidates = %+v, want one sorted entry each for a and b", got)
	}
	for _, g := range got {
		if g.Measured || g.Reason != ReasonNoTrustedEvidence {
			t.Errorf("a model with no evidence at all claims a grade: %+v", g)
		}
		if c := g.Candidate(); c.Measured || c.Model != g.Model {
			t.Errorf("Candidate() lost the honesty bit: %+v", c)
		}
	}
}
