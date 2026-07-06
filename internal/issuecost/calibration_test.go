package issuecost

import (
	"encoding/json"
	"strings"
	"testing"
)

// calibrationFixture is a deterministic, hand-verified join fixture. Every row's
// bucket and over-tier flag is worked out in the comment so the test PROVES the
// fold, not just self-agreement. Layout (join is by issue number):
//
//	CHOSEN T0 — over-tier successes, no rework  -> expand-cheaper
//	  10 chosen T0 optimal T2  success   OVER-TIER (T0 more demanding than T2)
//	  11 chosen T0 optimal T2  success   OVER-TIER
//	  12 chosen T0 optimal T2  success   OVER-TIER
//	  13 chosen T0 optimal T0  success   not over-tier (chosen == optimal)
//	CHOSEN T1 — has rework/escalation           -> raise-floor
//	  20 chosen T1 optimal T1  escalation
//	  21 chosen T1 optimal T1  success
//	  22 chosen T1 optimal T2  revert (also closed+witnessed -> revert WINS)
//	CHOSEN T2 — a refusal, a stall, a success   -> hold
//	  30 chosen T2 optimal T2  correct-refuse
//	  31 chosen T2 optimal T2  success
//	  32 chosen T2 optimal T2  closed+green but NOT commit-witnessed -> STALL
//	UNJOINED
//	  40 chosen T0 optimal T2  (no witnessed outcome) -> unjoined, never bucketed
//
// Totals: decisions=11 joined=10 unjoined=1
//
//	buckets: success=6 stall=1 escalation=1 revert=1 refuse=1  (sum 10 = joined)
//	over_tier_waste=3 (issues 10,11,12)
func calibrationFixture() ([]TierDecision, []WitnessedOutcome) {
	decisions := []TierDecision{
		{Issue: 10, Chosen: TierT0, Required: TierT2, Optimal: TierT2},
		{Issue: 11, Chosen: TierT0, Required: TierT2, Optimal: TierT2},
		{Issue: 12, Chosen: TierT0, Required: TierT2, Optimal: TierT2},
		{Issue: 13, Chosen: TierT0, Required: TierT0, Optimal: TierT0},
		{Issue: 20, Chosen: TierT1, Required: TierT1, Optimal: TierT1},
		{Issue: 21, Chosen: TierT1, Required: TierT1, Optimal: TierT1},
		{Issue: 22, Chosen: TierT1, Required: TierT1, Optimal: TierT2},
		{Issue: 30, Chosen: TierT2, Required: TierT0, Optimal: TierT2},
		{Issue: 31, Chosen: TierT2, Required: TierT2, Optimal: TierT2},
		{Issue: 32, Chosen: TierT2, Required: TierT2, Optimal: TierT2},
		{Issue: 40, Chosen: TierT0, Required: TierT2, Optimal: TierT2},
	}
	green := func(issue int) WitnessedOutcome {
		return WitnessedOutcome{Issue: issue, CommitWitnessed: true, TestsGreen: true, Closed: true, Turns: 3}
	}
	outcomes := []WitnessedOutcome{
		green(10), green(11), green(12), green(13),
		{Issue: 20, Escalated: true, Turns: 9},
		green(21),
		// 22: reverted even though it also closed + witnessed — revert must win.
		{Issue: 22, CommitWitnessed: true, TestsGreen: true, Closed: true, Reverted: true, Turns: 7},
		{Issue: 30, Refused: true, Turns: 1},
		green(31),
		// 32: closed and green but NO commit witness — must NOT count as success.
		{Issue: 32, Closed: true, TestsGreen: true, CommitWitnessed: false, Turns: 4},
		// issue 40 has NO outcome row on purpose.
	}
	return decisions, outcomes
}

// TestCalibrationFoldBuckets pins the overall join counts, the bucket tally, and
// the over-tier-waste total against the hand-verified fixture.
func TestCalibrationFoldBuckets(t *testing.T) {
	rep := Calibrate(calibrationFixture())
	if rep.Decisions != 11 || rep.Joined != 10 || rep.Unjoined != 1 {
		t.Errorf("join counts: got decisions=%d joined=%d unjoined=%d, want 11/10/1", rep.Decisions, rep.Joined, rep.Unjoined)
	}
	want := map[Bucket]int{
		BucketSuccess:       6,
		BucketStall:         1,
		BucketEscalation:    1,
		BucketReverted:      1,
		BucketCorrectRefuse: 1,
	}
	for b, w := range want {
		if rep.Buckets[b] != w {
			t.Errorf("bucket %s: got %d, want %d", b, rep.Buckets[b], w)
		}
	}
	// buckets must sum back to joined.
	var sum int
	for _, b := range bucketOrder {
		sum += rep.Buckets[b]
	}
	if sum != rep.Joined {
		t.Errorf("bucket sum %d != joined %d", sum, rep.Joined)
	}
	if rep.OverTierWaste != 3 {
		t.Errorf("over_tier_waste: got %d, want 3", rep.OverTierWaste)
	}
}

// TestCalibrationPerTier pins the per-chosen-tier roll-up.
func TestCalibrationPerTier(t *testing.T) {
	rep := Calibrate(calibrationFixture())

	t0 := rep.PerTier[TierT0]
	if t0.N != 4 || t0.Buckets[BucketSuccess] != 4 || t0.OverTierWaste != 3 {
		t.Errorf("T0: got n=%d success=%d waste=%d, want 4/4/3", t0.N, t0.Buckets[BucketSuccess], t0.OverTierWaste)
	}
	t1 := rep.PerTier[TierT1]
	if t1.N != 3 || t1.Buckets[BucketSuccess] != 1 || t1.Buckets[BucketEscalation] != 1 || t1.Buckets[BucketReverted] != 1 {
		t.Errorf("T1: got %+v, want n=3 success=1 escalation=1 revert=1", t1)
	}
	if t1.OverTierWaste != 0 {
		t.Errorf("T1 over_tier_waste: got %d, want 0 (a revert is not a success)", t1.OverTierWaste)
	}
	t2 := rep.PerTier[TierT2]
	if t2.N != 3 || t2.Buckets[BucketSuccess] != 1 || t2.Buckets[BucketCorrectRefuse] != 1 || t2.Buckets[BucketStall] != 1 {
		t.Errorf("T2: got %+v, want n=3 success=1 refuse=1 stall=1", t2)
	}
}

// TestCalibrationRecommendations proves the advisory proposals: T0 (over-tier
// successes, no rework) -> expand-cheaper; T1 (rework) -> raise-floor; T2 (thin
// signal) -> hold. Every recommendation MUST be advisory (auto_apply=false).
func TestCalibrationRecommendations(t *testing.T) {
	rep := Calibrate(calibrationFixture())
	got := map[Tier]string{}
	for _, rec := range rep.Recommendations {
		if rec.AutoApply {
			t.Errorf("recommendation for %s has auto_apply=true; calibration must stay advisory/shadow", rec.Tier)
		}
		got[rec.Tier] = rec.Action
	}
	want := map[Tier]string{
		TierT0: ActionExpandCheaper,
		TierT1: ActionRaiseFloor,
		TierT2: ActionHold,
	}
	for tier, action := range want {
		if got[tier] != action {
			t.Errorf("recommendation %s: got %q, want %q", tier, got[tier], action)
		}
	}
	// exactly one recommendation per active tier (3), in most-demanding-first order.
	if len(rep.Recommendations) != 3 {
		t.Fatalf("got %d recommendations, want 3", len(rep.Recommendations))
	}
	order := []Tier{rep.Recommendations[0].Tier, rep.Recommendations[1].Tier, rep.Recommendations[2].Tier}
	if order[0] != TierT0 || order[1] != TierT1 || order[2] != TierT2 {
		t.Errorf("recommendation order: got %v, want [T0 T1 T2]", order)
	}
}

// TestCalibrationWitnessSources is the done-condition witness: the report names a
// witness source for EVERY outcome bucket, so no metric is a bare, unbacked count.
func TestCalibrationWitnessSources(t *testing.T) {
	rep := Calibrate(calibrationFixture())
	for _, b := range bucketOrder {
		src := rep.WitnessSources[b]
		if strings.TrimSpace(src) == "" {
			t.Errorf("bucket %s has no named witness source", b)
		}
		if WitnessSource(b) != src {
			t.Errorf("WitnessSource(%s)=%q disagrees with report=%q", b, WitnessSource(b), src)
		}
	}
	// success must rest on a NON-forgeable commit witness, not just a close.
	if !strings.Contains(rep.WitnessSources[BucketSuccess], "commit-audit") {
		t.Errorf("success witness source should cite commit-audit: %q", rep.WitnessSources[BucketSuccess])
	}
}

// TestCalibrationClassifyPrecedence proves the load-bearing classification order:
// a refusal is correct behavior first; a revert is rework even when it also
// closed+witnessed; a bare close with no commit witness is a stall, not success.
func TestCalibrationClassifyPrecedence(t *testing.T) {
	cases := []struct {
		name string
		o    WitnessedOutcome
		want Bucket
	}{
		{"refuse-beats-all", WitnessedOutcome{Refused: true, Reverted: true, Escalated: true, Closed: true, CommitWitnessed: true, TestsGreen: true}, BucketCorrectRefuse},
		{"revert-beats-close", WitnessedOutcome{Reverted: true, Closed: true, CommitWitnessed: true, TestsGreen: true}, BucketReverted},
		{"escalation-beats-close", WitnessedOutcome{Escalated: true, Closed: true, CommitWitnessed: true, TestsGreen: true}, BucketEscalation},
		{"full-witness-is-success", WitnessedOutcome{Closed: true, CommitWitnessed: true, TestsGreen: true}, BucketSuccess},
		{"closed-no-commit-is-stall", WitnessedOutcome{Closed: true, CommitWitnessed: false, TestsGreen: true}, BucketStall},
		{"closed-no-tests-is-stall", WitnessedOutcome{Closed: true, CommitWitnessed: true, TestsGreen: false}, BucketStall},
		{"empty-is-stall", WitnessedOutcome{}, BucketStall},
	}
	for _, c := range cases {
		if got := classify(c.o); got != c.want {
			t.Errorf("%s: classify = %s, want %s", c.name, got, c.want)
		}
	}
}

// TestCalibrationEmpty: no decisions -> a zero report with non-nil maps and an
// explicit empty Render, plus a full witness-source table (self-documenting even
// when there is nothing to fold).
func TestCalibrationEmpty(t *testing.T) {
	rep := Calibrate(nil, nil)
	if rep.Decisions != 0 || rep.Joined != 0 || rep.Unjoined != 0 || rep.OverTierWaste != 0 {
		t.Errorf("empty report: got %+v, want all-zero counts", rep)
	}
	if rep.Buckets == nil || rep.PerTier == nil || rep.WitnessSources == nil {
		t.Error("empty report has a nil map; callers must be able to index without a nil check")
	}
	if len(rep.WitnessSources) != len(bucketOrder) {
		t.Errorf("witness sources: got %d, want %d (all buckets named even when empty)", len(rep.WitnessSources), len(bucketOrder))
	}
	if len(rep.Recommendations) != 0 {
		t.Errorf("empty report should make no recommendations, got %d", len(rep.Recommendations))
	}
	if got := rep.Render(); !strings.Contains(got, "decisions=0") || !strings.Contains(got, "no rows") {
		t.Errorf("empty Render: %q", got)
	}
}

// TestCalibrationUnjoinedNotBucketed: a decision whose issue has no witnessed
// outcome is counted as unjoined and never lands in a bucket (nothing to
// calibrate against without a witness).
func TestCalibrationUnjoinedNotBucketed(t *testing.T) {
	decisions := []TierDecision{{Issue: 1, Chosen: TierT1, Required: TierT1, Optimal: TierT1}}
	rep := Calibrate(decisions, nil)
	if rep.Unjoined != 1 || rep.Joined != 0 {
		t.Errorf("got joined=%d unjoined=%d, want 0/1", rep.Joined, rep.Unjoined)
	}
	var sum int
	for _, b := range bucketOrder {
		sum += rep.Buckets[b]
	}
	if sum != 0 {
		t.Errorf("unjoined decision leaked into buckets: sum=%d, want 0", sum)
	}
	if len(rep.Recommendations) != 0 {
		t.Errorf("no witnessed outcome -> no recommendation, got %d", len(rep.Recommendations))
	}
}

// TestCalibrationLastOutcomeWins: when two witnessed outcomes share an issue, the
// LAST one supersedes, matching a durable ledger where a later witness replaces an
// earlier one.
func TestCalibrationLastOutcomeWins(t *testing.T) {
	decisions := []TierDecision{{Issue: 5, Chosen: TierT1, Required: TierT1, Optimal: TierT1}}
	outcomes := []WitnessedOutcome{
		{Issue: 5, Closed: true, CommitWitnessed: true, TestsGreen: true}, // success first
		{Issue: 5, Reverted: true},                                        // later revert supersedes
	}
	rep := Calibrate(decisions, outcomes)
	if rep.Buckets[BucketReverted] != 1 || rep.Buckets[BucketSuccess] != 0 {
		t.Errorf("last outcome should win: got revert=%d success=%d, want 1/0", rep.Buckets[BucketReverted], rep.Buckets[BucketSuccess])
	}
}

// TestCalibrationReportJSON is the acceptance-gate artifact: the report marshals
// to JSON, round-trips, and carries the recommendations + witness sources an
// operator captures. The marshalled report is logged so the gate has a captured
// JSON report to inspect.
func TestCalibrationReportJSON(t *testing.T) {
	rep := Calibrate(calibrationFixture())
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	t.Logf("captured calibration report:\n%s", raw)

	var back CalibrationReport
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if back.Decisions != rep.Decisions || back.Joined != rep.Joined || back.OverTierWaste != rep.OverTierWaste {
		t.Errorf("round-trip scalars: got %+v, want %+v", back, rep)
	}
	if len(back.Recommendations) != len(rep.Recommendations) {
		t.Errorf("round-trip recommendations: got %d, want %d", len(back.Recommendations), len(rep.Recommendations))
	}
	// the JSON must carry the human-visible advisory bit and a named witness.
	s := string(raw)
	for _, want := range []string{"\"auto_apply\": false", "\"witness_sources\"", "commit-audit", ActionExpandCheaper, ActionRaiseFloor} {
		if !strings.Contains(s, want) {
			t.Errorf("JSON report missing %q", want)
		}
	}
}

// TestCalibrationRender surfaces the join counts, the bucket tally, and the
// advisory recommendations in the operator readout.
func TestCalibrationRender(t *testing.T) {
	out := Calibrate(calibrationFixture()).Render()
	for _, want := range []string{
		"decisions=11", "joined=10", "unjoined=1", "over_tier_waste=3",
		"success=6", "revert-reopen=1",
		"rec T0 -> expand-cheaper", "rec T1 -> raise-floor", "rec T2 -> hold",
		"auto_apply=false",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("Render missing %q in:\n%s", want, out)
		}
	}
}
