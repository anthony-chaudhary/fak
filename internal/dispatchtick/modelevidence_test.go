package dispatchtick

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for the capability-evidence producer (#5428, epic #5416 track F).
//
// The thing being defended is subtle enough to be worth naming twice: every row this
// producer emits can be individually true while the CORPUS lies. Because evidence is
// keyed by provenance and never merged across it, filing successes and failures under
// different provenances hands the grader a bucket that is 100% successes without a single
// dishonest row in it. Most of what follows is about that.

func slot(issue int, model, claim, test string) WitnessRecord {
	r := WitnessRecord{Issue: issue, Log: fmt.Sprintf("resolve-%d.log", issue), Model: model, Claim: claim, TestClaim: test}
	if claim != ClaimNoCommit {
		r.SHA = fmt.Sprintf("%040x", issue)
		r.Verdict = "SHIPPED"
		if claim == ClaimWitnessed {
			r.Witness = WitnessOK
		}
	}
	return r
}

func routineClass(WitnessRecord) modelroute.WorkClass { return modelroute.ClassRoutine }

func stampedAt(t time.Time) func(WitnessRecord) time.Time {
	return func(WitnessRecord) time.Time { return t }
}

func TestASeatDefaultSlotIsNotEvidenceAboutAnyModel(t *testing.T) {
	// A slot with no --model pin ran on whatever the seat happened to default to. Filing
	// it under the empty model, or under the fleet default we GUESS it used, would put a
	// stranger's result on some model's record.
	records := []WitnessRecord{
		slot(1, "", ClaimWitnessed, ClaimTestGreen),
		slot(2, "qwen3.6-4b", ClaimWitnessed, ClaimTestGreen),
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: routineClass})
	if len(out) != 1 || out[0].Model != "qwen3.6-4b" {
		t.Fatalf("outcomes = %+v, want only the pinned slot", out)
	}
	if stats.Unattributed != 1 || stats.Produced != 1 {
		t.Errorf("stats = %+v, want the unpinned slot counted as unattributable", stats)
	}
}

func TestAWorkClassNobodyDeclaredIsNotGuessed(t *testing.T) {
	// PolicyFor maps an unknown class to the T0 floor. That is right when picking a floor
	// for work and a capability-MINTING hole when read backwards to grade a model, so a
	// resolver that will not say drops the record.
	records := []WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen)}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{})
	if len(out) != 0 || stats.Unclassified != 1 {
		t.Fatalf("out=%+v stats=%+v — a record with no declared class became evidence", out, stats)
	}
	// And a resolver that answers only for some slots produces evidence only for those.
	out, stats = TurnOutcomesFromWitness(
		[]WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen), slot(2, "tiny", ClaimWitnessed, ClaimTestGreen)},
		WitnessEvidenceOptions{Class: func(r WitnessRecord) modelroute.WorkClass {
			if r.Issue == 1 {
				return modelroute.ClassRoutine
			}
			return ""
		}})
	if len(out) != 1 || stats.Unclassified != 1 {
		t.Errorf("out=%+v stats=%+v", out, stats)
	}
}

func TestATestRunGradesTheProvenanceAndTheClaimGradesTheOutcome(t *testing.T) {
	cases := []struct {
		name    string
		rec     WitnessRecord
		success bool
		verify  modelroute.Verification
	}{
		{"tests ran and passed on a witnessed diff", slot(1, "m", ClaimWitnessed, ClaimTestGreen), true, modelroute.VerifyWitness},
		{"tests ran and failed", slot(2, "m", ClaimWitnessed, ClaimTestRed), false, modelroute.VerifyWitness},
		{"witnessed diff, nothing ran it", slot(3, "m", ClaimWitnessed, ClaimTestUnrun), true, modelroute.VerifyJudge},
		{"tests passed but the diff did something else", slot(4, "m", ClaimUnwitnessed, ClaimTestGreen), false, modelroute.VerifyWitness},
		{"claimed a commit that is not there", slot(5, "m", ClaimUnwitnessed, ClaimTestUnrun), false, modelroute.VerifyJudge},
		{"no commit at all", slot(6, "m", ClaimNoCommit, ""), false, modelroute.VerifyJudge},
	}
	for _, c := range cases {
		out, _ := TurnOutcomesFromWitness([]WitnessRecord{c.rec}, WitnessEvidenceOptions{Class: routineClass})
		if len(out) != 1 {
			t.Fatalf("%s: no outcome produced", c.name)
		}
		if out[0].Success != c.success || out[0].Verify != c.verify {
			t.Errorf("%s: success=%v verify=%q, want success=%v verify=%q",
				c.name, out[0].Success, out[0].Verify, c.success, c.verify)
		}
	}
}

func TestProvenanceNeverCorrelatesWithOutcome(t *testing.T) {
	// The attack this file exists to stop, run end to end: a model with a MEDIOCRE record
	// whose failures are all filed one provenance lower than its successes grades as
	// flawless, because the grader never merges across provenance and so never sees them
	// together. Here 20 green and 20 red test runs are the SAME check, so they share a
	// bucket, and the 30 no-commit slots — a different check — cannot dilute or flatter it.
	var records []WitnessRecord
	for i := 0; i < 20; i++ {
		records = append(records, slot(100+i, "tiny", ClaimWitnessed, ClaimTestGreen))
	}
	for i := 0; i < 20; i++ {
		records = append(records, slot(200+i, "tiny", ClaimWitnessed, ClaimTestRed))
	}
	for i := 0; i < 30; i++ {
		records = append(records, slot(300+i, "tiny", ClaimNoCommit, ""))
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{
		Class: routineClass, At: stampedAt(time.Date(2026, 7, 20, 12, 0, 0, 0, time.UTC))})
	if stats.Produced != 70 || stats.Undated != 0 {
		t.Fatalf("stats = %+v", stats)
	}
	ev, _ := modelroute.FoldTurnOutcomes(out, modelroute.FoldOptions{})
	for _, row := range ev["tiny"] {
		if row.Successes == row.Attempts && row.Attempts > 0 {
			t.Errorf("a provenance bucket is a perfect record: %+v — successes and failures "+
				"of the same check must share a bucket, or the split manufactures a grade", row)
		}
		if row.Verify == modelroute.VerifyWitness && (row.Attempts != 40 || row.Successes != 20) {
			t.Errorf("witness row = %+v, want 20/40 — the reds belong here with the greens", row)
		}
	}
	// 50% is nowhere near the bar, so this model earns nothing. That is the honest answer
	// for a fleet whose slots half-fail, and it is the answer the split would have hidden.
	g := modelroute.GradeCapability("tiny", ev["tiny"], modelroute.DefaultGradeFloor())
	if g.Measured {
		t.Errorf("a half-failing model was graded: %+v", g)
	}
	if g.Reason != modelroute.ReasonBelowSuccessFloor {
		t.Errorf("reason = %q, want the shortfall named as a success-rate shortfall", g.Reason)
	}
}

func TestTheSameSweepTwiceIsCountedOnce(t *testing.T) {
	// Witness sweeps are re-run over the same runs directory. An id that is not stable
	// across sweeps would let a fleet double its evidence by sweeping twice.
	records := []WitnessRecord{
		slot(1, "tiny", ClaimWitnessed, ClaimTestGreen),
		slot(2, "tiny", ClaimWitnessed, ClaimTestGreen),
	}
	first, _ := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: routineClass})
	second, _ := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: routineClass})
	ev, fold := modelroute.FoldTurnOutcomes(append(first, second...), modelroute.FoldOptions{})
	if ev["tiny"][0].Attempts != 2 || fold.Duplicates != 2 {
		t.Errorf("evidence=%+v fold=%+v — a re-sweep doubled the corpus", ev["tiny"], fold)
	}
}

func TestTheSlotsOwnLogIsTheIDEvenWhenACommitCouldHaveSuppliedOne(t *testing.T) {
	// Two things ride on the log path being tried BEFORE the commit sha.
	//
	// A slot that never committed has no sha at all, so the log is the only stable id it
	// will ever have. Losing it does not merely cost the replay check — it re-opens replay
	// for exactly the failing half of the corpus, and a re-sweep then doubles a model's
	// no-commit rows while its witnessed rows dedup normally. That is the same manufactured
	// record this file exists to stop, just pointed the other way.
	nocommit := []WitnessRecord{
		{Issue: 11, Log: "resolve-11.log", Model: "tiny", Claim: ClaimNoCommit},
		{Issue: 12, Log: "resolve-12.log", Model: "tiny", Claim: ClaimNoCommit},
	}
	first, stats := TurnOutcomesFromWitness(nocommit, WitnessEvidenceOptions{Class: routineClass})
	second, _ := TurnOutcomesFromWitness(nocommit, WitnessEvidenceOptions{Class: routineClass})
	if stats.Unidentified != 0 {
		t.Errorf("stats = %+v — a no-commit slot still has its own log to be keyed by", stats)
	}
	ev, fold := modelroute.FoldTurnOutcomes(append(first, second...), modelroute.FoldOptions{})
	if ev["tiny"][0].Attempts != 2 || fold.Duplicates != 2 || fold.Undeduplicable != 0 {
		t.Errorf("evidence=%+v fold=%+v — a re-sweep doubled a model's failures", ev["tiny"], fold)
	}
	// And where both exist the log still wins, because the sha is per-COMMIT while the log
	// is per-SLOT: two workers that land on the same commit are two attempts, and keying on
	// the sha would fold them into one.
	both, _ := TurnOutcomesFromWitness(
		[]WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen)}, WitnessEvidenceOptions{Class: routineClass})
	if both[0].ID != "slot:resolve-1.log" {
		t.Errorf("id = %q, want the per-slot log rather than the per-commit sha", both[0].ID)
	}
}

func TestASlotWithNoStableIDIsReportedNotGivenASyntheticOne(t *testing.T) {
	// The issue number is deliberately NOT a fallback id: one issue is dispatched many
	// times, so keying on it would collapse a model's whole history with that issue into
	// one attempt — a silent LOSS, where an empty id merely costs the replay check.
	records := []WitnessRecord{
		{Issue: 7, Model: "tiny", Claim: ClaimNoCommit},
		{Issue: 7, Model: "tiny", Claim: ClaimNoCommit},
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: routineClass})
	if stats.Unidentified != 2 || stats.Produced != 2 {
		t.Fatalf("stats = %+v, want both rows produced and both reported as un-identifiable", stats)
	}
	for _, o := range out {
		if o.ID != "" {
			t.Errorf("a synthetic id was invented: %q", o.ID)
		}
	}
	ev, fold := modelroute.FoldTurnOutcomes(out, modelroute.FoldOptions{})
	if ev["tiny"][0].Attempts != 2 || fold.Undeduplicable != 2 {
		t.Errorf("evidence=%+v fold=%+v — un-id'd rows must be kept AND flagged", ev["tiny"], fold)
	}
}

func TestAnUnstampedSweepIsCountedSoTheWindowGapIsVisible(t *testing.T) {
	out, stats := TurnOutcomesFromWitness(
		[]WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen)},
		WitnessEvidenceOptions{Class: routineClass})
	if stats.Undated != 1 || !out[0].At.IsZero() {
		t.Fatalf("stats=%+v at=%v — an unstamped row must be counted, not silently emitted", stats, out[0].At)
	}
	// And an unstamped corpus cannot answer a freshness window, which is the cost being
	// reported: the same rows vanish the moment anyone asks for one.
	_, fold := modelroute.FoldTurnOutcomes(out, modelroute.FoldOptions{Since: time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)})
	if fold.Counted != 0 || fold.Undated != 1 {
		t.Errorf("fold = %+v, want the unstamped row excluded from a windowed grading", fold)
	}
}

func TestAZoneIsRecordedOnlyWhenTheCallerKnowsIt(t *testing.T) {
	out, _ := TurnOutcomesFromWitness(
		[]WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen)},
		WitnessEvidenceOptions{Class: routineClass})
	if out[0].Zone != "" {
		t.Errorf("zone = %q, want empty — an unrecorded rung must not read as the device rung", out[0].Zone)
	}
	out, _ = TurnOutcomesFromWitness(
		[]WitnessRecord{slot(1, "tiny", ClaimWitnessed, ClaimTestGreen)},
		WitnessEvidenceOptions{Class: routineClass, Zone: func(WitnessRecord) modelroute.PlacementZone { return modelroute.ZoneFleet }})
	if out[0].Zone != modelroute.ZoneFleet {
		t.Errorf("zone = %q, want the caller's answer", out[0].Zone)
	}
}

func TestAWitnessSweepGradesAModelEndToEnd(t *testing.T) {
	// The whole producer chain: finished slots -> outcomes -> evidence -> a grade a
	// Candidate can carry down the ladder. Nobody asserts a capability anywhere.
	var records []WitnessRecord
	for i := 0; i < 30; i++ {
		claim, test := ClaimWitnessed, ClaimTestGreen
		if i%10 == 0 { // one in ten fails its tests
			test = ClaimTestRed
		}
		records = append(records, slot(500+i, "qwen3.6-4b", claim, test))
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{
		Class: routineClass, At: stampedAt(time.Date(2026, 7, 20, 12, 0, 0, 0, time.UTC)),
		Zone: func(WitnessRecord) modelroute.PlacementZone { return modelroute.ZoneDevice },
	})
	if stats.Produced != 30 || stats.Unattributed+stats.Unclassified+stats.Undated+stats.Unidentified != 0 {
		t.Fatalf("stats = %+v, want a fully attributed corpus", stats)
	}
	ev, _ := modelroute.FoldTurnOutcomes(out, modelroute.FoldOptions{Since: time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)})
	g := modelroute.GradeCapability("qwen3.6-4b", ev["qwen3.6-4b"], modelroute.DefaultGradeFloor())
	if !g.Measured || g.Verify != modelroute.VerifyWitness {
		t.Fatalf("grade = %+v, want a witnessed measurement", g)
	}
	if c := g.Candidate(); !c.Measured || c.Capability != modelroute.TierT2 {
		t.Errorf("candidate = %+v, want a MEASURED T2 that may descend the ladder", c)
	}
}
