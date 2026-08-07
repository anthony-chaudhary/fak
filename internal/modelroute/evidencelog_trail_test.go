package modelroute

import (
	"strings"
	"testing"
)

// Witnesses for the AUDIT PATH from a grade back to the turns that earned it (#5428,
// epic #5416 track F).
//
// The producer and the grader both landed before this: a fleet can now turn finished
// slots into a durable journal and the journal into a capability. What neither half gave
// an operator was the last question — "which turns?" — and a capability nobody can
// re-walk is a number they have to take on trust from the same pipeline that produced it.
//
// So every test here attacks the citation in the direction a broken producer would break
// it: citing turns the grader refused, citing turns for a grade it never awarded, citing
// turns the fold dropped, or citing a set that does not add up to the count it explains.

// journalGrade folds a journal and grades one model from it, returning everything a
// caller needs to check the citation against the number it is supposed to explain.
func journalGrade(t *testing.T, model string, floor GradeFloor, outcomes []TurnOutcome) (Grade, []EvidenceTrail) {
	t.Helper()
	evidence, _, trails := FoldTurnOutcomesTrail(outcomes, FoldOptions{})
	return GradeCapability(model, evidence[model], floor), trails
}

// TestTheCitationAccountsForEveryAttemptTheGradeCounted is the central invariant. A grade
// says "24 of 25 attempts at this class"; the citation must name 25 turns, or say exactly
// how many of them were unnameable. Anything else is a citation that silently explains a
// different number than the one it is printed beside.
func TestTheCitationAccountsForEveryAttemptTheGradeCounted(t *testing.T) {
	var outcomes []TurnOutcome
	for i := 0; i < 25; i++ {
		outcomes = append(outcomes, TurnOutcome{
			ID:      "slot:resolve-" + strings.Repeat("x", i%3) + itoa(i),
			Model:   "tiny",
			Class:   ClassRoutine,
			Success: i != 7, // 24/25 — over the 0.8 floor
			Verify:  VerifyWitness,
		})
	}
	floor := DefaultGradeFloor()
	g, trails := journalGrade(t, "tiny", floor, outcomes)
	if !g.Measured || g.Class != ClassRoutine {
		t.Fatalf("grade did not come from the routine evidence: %+v", g)
	}
	turns, anonymous := TurnsBehind(g, trails, floor)
	if int64(len(turns))+int64(anonymous) != g.Attempts {
		t.Errorf("citation names %d turn(s) + %d anonymous, but the grade counted %d attempt(s) — "+
			"a citation that does not add up to the number beside it is a producer bug",
			len(turns), anonymous, g.Attempts)
	}
	if anonymous != 0 {
		t.Errorf("every row carried an id; anonymous = %d", anonymous)
	}
	for _, id := range turns {
		if !strings.HasPrefix(id, "slot:") {
			t.Errorf("citation named %q, which is not a turn id from this journal", id)
		}
	}
}

// TestTheCitationNeverNamesTurnsTheGraderRefused is the attack the trail makes possible
// and would otherwise hide: a model with a mountain of SELF-REPORTED turns and a bare
// quorum of witnessed ones. GradeCapability already drops the self-reported block. If the
// citation still listed it, the audit would show a hundred turns behind a grade that
// twenty bought — over-claiming in exactly the direction the grader refused to.
func TestTheCitationNeverNamesTurnsTheGraderRefused(t *testing.T) {
	var outcomes []TurnOutcome
	for i := 0; i < 20; i++ {
		outcomes = append(outcomes, TurnOutcome{
			ID: "witnessed:" + itoa(i), Model: "tiny", Class: ClassRoutine,
			Success: true, Verify: VerifyWitness,
		})
	}
	for i := 0; i < 100; i++ {
		outcomes = append(outcomes, TurnOutcome{
			ID: "selfreport:" + itoa(i), Model: "tiny", Class: ClassRoutine,
			Success: true, Verify: VerifyNone,
		})
	}
	floor := DefaultGradeFloor()
	g, trails := journalGrade(t, "tiny", floor, outcomes)
	if !g.Measured {
		t.Fatalf("20 witnessed successes did not reach the default floor: %+v", g)
	}
	if g.Dropped != 100 {
		t.Fatalf("grader kept the self-reported block: dropped = %d, want 100", g.Dropped)
	}
	turns, _ := TurnsBehind(g, trails, floor)
	if len(turns) != 20 {
		t.Errorf("citation named %d turn(s) for a grade built on 20 attempts", len(turns))
	}
	for _, id := range turns {
		if strings.HasPrefix(id, "selfreport:") {
			t.Fatalf("citation named %q — a turn the grade explicitly refused to count", id)
		}
	}
}

// TestARequireWitnessFloorNarrowsTheCitationWithTheGrade pins the citation to the SAME
// floor the grade used rather than to a fixed notion of "trusted". An operator who
// tightened the bar to witnessed-only must not be shown judge-scored turns as the
// evidence behind a grade those turns were not allowed to move.
func TestARequireWitnessFloorNarrowsTheCitationWithTheGrade(t *testing.T) {
	var outcomes []TurnOutcome
	for i := 0; i < 25; i++ {
		outcomes = append(outcomes, TurnOutcome{
			ID: "witness:" + itoa(i), Model: "tiny", Class: ClassRoutine,
			Success: true, Verify: VerifyWitness,
		})
		outcomes = append(outcomes, TurnOutcome{
			ID: "judge:" + itoa(i), Model: "tiny", Class: ClassRoutine,
			Success: true, Verify: VerifyJudge,
		})
	}
	lenient := DefaultGradeFloor()
	strict := DefaultGradeFloor()
	strict.RequireWitness = true

	gLenient, trails := journalGrade(t, "tiny", lenient, outcomes)
	loose, _ := TurnsBehind(gLenient, trails, lenient)
	if len(loose) != 50 {
		t.Errorf("a judge-accepting floor cited %d turn(s), want both blocks (50)", len(loose))
	}

	gStrict, trails := journalGrade(t, "tiny", strict, outcomes)
	tight, _ := TurnsBehind(gStrict, trails, strict)
	if len(tight) != 25 {
		t.Errorf("--grade-floor witness cited %d turn(s), want only the witnessed 25", len(tight))
	}
	for _, id := range tight {
		if strings.HasPrefix(id, "judge:") {
			t.Fatalf("witness-only floor cited judge-scored turn %q", id)
		}
	}
}

// TestAnUnmeasuredGradeCitesNothing. An UNMEASURED grade names no class, so there is no
// set of turns that earned it. Returning the model's turns anyway would read as "here is
// the evidence behind this capability" for a capability that was never awarded — the one
// place a citation could manufacture the very claim the grader declined to make.
func TestAnUnmeasuredGradeCitesNothing(t *testing.T) {
	var outcomes []TurnOutcome
	for i := 0; i < 5; i++ { // under the 20-attempt floor
		outcomes = append(outcomes, TurnOutcome{
			ID: "slot:" + itoa(i), Model: "tiny", Class: ClassRoutine,
			Success: true, Verify: VerifyWitness,
		})
	}
	floor := DefaultGradeFloor()
	g, trails := journalGrade(t, "tiny", floor, outcomes)
	if g.Measured || g.Reason != ReasonInsufficientSamples {
		t.Fatalf("5 attempts bought a grade: %+v", g)
	}
	if turns, anonymous := TurnsBehind(g, trails, floor); len(turns) != 0 || anonymous != 0 {
		t.Errorf("an unmeasured grade cited %d turn(s) + %d anonymous, want nothing", len(turns), anonymous)
	}
	// The turns are still in the trail — explaining a REFUSAL is a different question,
	// and it is answered by reading the fold rather than by a citation for a grade that
	// does not exist.
	if len(trails) != 1 || len(trails[0].Turns) != 5 {
		t.Errorf("the fold lost the refused turns: %+v", trails)
	}
}

// TestTheTrailNeverNamesATurnTheFoldDropped. Replay is the cheapest way to manufacture a
// grade, and the fold already refuses it. A citation assembled from the raw journal
// instead of from the counting pass would re-introduce the duplicate at the exact moment
// an operator went looking for proof it had been excluded.
func TestTheTrailNeverNamesATurnTheFoldDropped(t *testing.T) {
	outcomes := []TurnOutcome{
		{ID: "slot:a", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "slot:a", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "slot:b", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness}, // no id
	}
	evidence, stats, trails := FoldTurnOutcomesTrail(outcomes, FoldOptions{})
	if stats.Duplicates != 1 || stats.Counted != 3 {
		t.Fatalf("fold stats changed shape: %+v", stats)
	}
	if len(trails) != 1 {
		t.Fatalf("want one (model,class,verify) row, got %+v", trails)
	}
	tr := trails[0]
	if len(tr.Turns) != 2 || tr.Turns[0] != "slot:a" || tr.Turns[1] != "slot:b" {
		t.Errorf("trail named %v — a replayed id must appear once, in journal order", tr.Turns)
	}
	if tr.Anonymous != 1 {
		t.Errorf("anonymous = %d, want the one counted turn that carried no id", tr.Anonymous)
	}
	if got := evidence["tiny"][0].Attempts; int(got) != len(tr.Turns)+tr.Anonymous {
		t.Errorf("%d attempt(s) but %d named + %d anonymous — the citation and the count disagree",
			got, len(tr.Turns), tr.Anonymous)
	}
}

// TestTheTrailIsRecordedByTheSameCountingPass. FoldTurnOutcomes and its trail-returning
// form must not be two different folds. The delegation is what guarantees a citation can
// never drift from the counts it explains, and this pins it rather than trusting it.
func TestTheTrailIsRecordedByTheSameCountingPass(t *testing.T) {
	outcomes := []TurnOutcome{
		{ID: "x", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness, At: at(20)},
		{ID: "y", Model: "corp", Class: ClassUltraHard, Success: false, Verify: VerifyJudge, At: at(21)},
	}
	plainEv, plainStats := FoldTurnOutcomes(outcomes, FoldOptions{})
	trailEv, trailStats, trails := FoldTurnOutcomesTrail(outcomes, FoldOptions{})
	if plainStats != trailStats {
		t.Errorf("stats diverged: %+v vs %+v", plainStats, trailStats)
	}
	if len(plainEv) != len(trailEv) {
		t.Errorf("evidence diverged: %d vs %d model(s)", len(plainEv), len(trailEv))
	}
	// Deterministic order, so a citation of a given journal is as reproducible as the
	// grade built from it.
	if len(trails) != 2 || trails[0].Model != "corp" || trails[1].Model != "tiny" {
		t.Errorf("trails are not sorted by model: %+v", trails)
	}
}
