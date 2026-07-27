package ablate

// tier_scorer_replay_test.go — the witness for the deterministic-replay scorer (#5413).
//
// The one claim that matters: production can now produce a MEASURED verdict. Every test
// below drives the real AnnotateModelStrength classifier rather than asserting on the
// scorer in isolation, because "the scorer returned a number" is not the property — "the
// axis graded LOAD_BEARING / REDUNDANT / HOBBLING instead of UNMEASURED, off recorded
// data, at $0" is.
//
// The second claim, equally load-bearing: every refusal path is still an honest
// Measured=false and never a fabricated 0.0. A missing rung and a measured zero must stay
// distinguishable, because #4414 files code-deletion issues off these grades.

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// The headline: a recorded table turns the production UNMEASURED posture into real
// grades. Same three trajectories the classifier fixture uses, served from records
// instead of a fake — so this is the end-to-end "the axis now earns something" witness.
func TestReplayTierScorerProducesMeasuredVerdicts(t *testing.T) {
	scorer, err := NewReplayTierScorer(TierScoreTable{
		Source: "recorded-sweep-2026-07",
		Records: []TierScoreRecord{
			{Tier: TierWeak, ArmID: "all-off", Score: 0.40},
			{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.60},
			{Tier: TierWeak, ArmID: FeatureRadix, Score: 0.60},
			{Tier: TierStrong, ArmID: "all-off", Score: 0.90},
			{Tier: TierStrong, ArmID: FeatureVDSO, Score: 0.98},  // +0.08 => still pulls weight
			{Tier: TierStrong, ArmID: FeatureRadix, Score: 0.90}, // +0.00 => the strong model erased it
		},
	})
	if err != nil {
		t.Fatalf("NewReplayTierScorer: %v", err)
	}

	// The table's own ladder, so the caller never guesses a rung it cannot serve.
	if got := scorer.Tiers(); !reflect.DeepEqual(got, []string{TierWeak, TierStrong}) {
		t.Fatalf("Tiers() = %v, want the canonical weak,strong the table records", got)
	}

	rep := strengthFixtureReport()
	if err := AnnotateModelStrength(context.Background(), rep, scorer.Tiers(), scorer, StrengthParams{}); err != nil {
		t.Fatalf("AnnotateModelStrength: %v", err)
	}

	got := map[string]string{}
	for _, v := range rep.Verdicts {
		got[v.ID] = v.Grade
	}
	want := map[string]string{
		FeatureVDSO:  VerdictLoadBearing,
		FeatureRadix: VerdictRedundant,
		// normgate is in the fixture report but NOT in the table: an arm nobody recorded
		// stays UNMEASURED rather than collapsing to a zero delta and grading REDUNDANT,
		// which would auto-file a deletion issue for a feature never measured at all.
		FeatureNormgate: VerdictUnmeasured,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("verdicts = %v, want %v", got, want)
	}

	// The measured grades must carry provenance back to the table that produced them.
	for _, run := range rep.Runs {
		if run.ArmID != FeatureVDSO {
			continue
		}
		if run.ModelStrength.StrongestTier != TierStrong {
			t.Fatalf("classified on %q, want the strongest recorded rung %q", run.ModelStrength.StrongestTier, TierStrong)
		}
		for _, td := range run.ModelStrength.Tiers {
			if !strings.Contains(td.Detail, "recorded-sweep-2026-07") {
				t.Fatalf("replayed rung %q does not name its source: %q", td.Tier, td.Detail)
			}
		}
	}
}

// A rung with no record is UNMEASURED with a Detail naming the miss — never Score 0 with
// Measured true. This is the fabrication fence, asserted at the scorer boundary.
func TestReplayTierScorerMissIsUnmeasuredNotZero(t *testing.T) {
	scorer, err := NewReplayTierScorer(TierScoreTable{
		Source:  "partial-table",
		Records: []TierScoreRecord{{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.5}},
	})
	if err != nil {
		t.Fatalf("NewReplayTierScorer: %v", err)
	}
	out, err := scorer.ScoreArm(context.Background(), TierStrong, FeatureVDSO, nil)
	if err != nil {
		t.Fatalf("ScoreArm: %v", err)
	}
	if out.Measured {
		t.Fatalf("an unrecorded rung reported Measured=true: %+v", out)
	}
	if out.Score != 0 {
		t.Fatalf("unmeasured outcome carries a score %v, which reads as data", out.Score)
	}
	for _, want := range []string{FeatureVDSO, TierStrong, "partial-table"} {
		if !strings.Contains(out.Detail, want) {
			t.Fatalf("miss detail %q does not name %q", out.Detail, want)
		}
	}
}

// Construction refuses the artifacts that would make a replay non-deterministic or
// silently empty. Each of these would otherwise surface as a wall of UNMEASURED with no
// stated cause.
func TestReplayTierScorerRefusesMalformedTable(t *testing.T) {
	cases := []struct {
		name    string
		records []TierScoreRecord
		want    string
	}{
		{
			name:    "unknown tier",
			records: []TierScoreRecord{{Tier: "galaxy", ArmID: FeatureVDSO, Score: 1}},
			want:    "unknown model tier",
		},
		{
			name:    "empty arm id",
			records: []TierScoreRecord{{Tier: TierWeak, ArmID: "  ", Score: 1}},
			want:    "empty arm_id",
		},
		{
			name: "same rung recorded twice with different scores",
			records: []TierScoreRecord{
				{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.1},
				{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.2},
			},
			want: "would not be deterministic",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewReplayTierScorer(TierScoreTable{Records: tc.records}); err == nil {
				t.Fatalf("accepted a malformed table")
			} else if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not explain the refusal (want %q)", err, tc.want)
			}
		})
	}

	// Re-recording the IDENTICAL measurement is idempotent, not ambiguous: refusing it
	// would make an append-only recorder unusable.
	scorer, err := NewReplayTierScorer(TierScoreTable{Records: []TierScoreRecord{
		{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.25},
		{Tier: TierWeak, ArmID: FeatureVDSO, Score: 0.25},
	}})
	if err != nil {
		t.Fatalf("an identical duplicate was refused: %v", err)
	}
	if scorer.Rungs() != 1 {
		t.Fatalf("Rungs() = %d, want the duplicate collapsed to 1", scorer.Rungs())
	}
}

// The determinism claim, end to end: grade a report, freeze the measurements to a table
// on disk, replay that table through a fresh scorer, and get BYTE-IDENTICAL verdicts.
// This is what makes an expensive measurement a $0 artifact forever after.
func TestRecordThenReplayReproducesVerdicts(t *testing.T) {
	tiers := []string{TierWeak, TierMid, TierStrong}

	live := strengthFixtureReport()
	if err := AnnotateModelStrength(context.Background(), live, tiers, strengthFixtureScorer(), StrengthParams{}); err != nil {
		t.Fatalf("AnnotateModelStrength (measured): %v", err)
	}

	path := filepath.Join(t.TempDir(), "tier-scores.json")
	raw, err := json.Marshal(RecordTierScores(live, "frozen-fixture"))
	if err != nil {
		t.Fatalf("marshal table: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write table: %v", err)
	}

	scorer, err := LoadReplayTierScorer(path)
	if err != nil {
		t.Fatalf("LoadReplayTierScorer: %v", err)
	}
	replayed := strengthFixtureReport()
	if err := AnnotateModelStrength(context.Background(), replayed, tiers, scorer, StrengthParams{}); err != nil {
		t.Fatalf("AnnotateModelStrength (replayed): %v", err)
	}

	if len(live.Verdicts) == 0 {
		t.Fatalf("the measured run produced no verdicts, so the round-trip proves nothing")
	}
	for i := range live.Verdicts {
		got, want := replayed.Verdicts[i], live.Verdicts[i]
		if got.ID != want.ID || got.Grade != want.Grade || got.StrongestTier != want.StrongestTier || got.StrongestDelta != want.StrongestDelta {
			t.Fatalf("replayed verdict %d = %+v, want the measured %+v", i, got, want)
		}
		if got.Grade == VerdictUnmeasured {
			t.Fatalf("verdict %d replayed as UNMEASURED, so the table lost the measurement", i)
		}
	}
}

// The cost fence: at most Budget rungs reach the wrapped scorer, and the refusal happens
// BEFORE the spend (the inner scorer is never called again once exhausted).
func TestBudgetTierScorerStopsSpending(t *testing.T) {
	inner := &countingTierScorer{inner: fakeTierScorer{scores: map[string]map[string]float64{
		TierWeak:   {"all-off": 0.4, FeatureVDSO: 0.6},
		TierStrong: {"all-off": 0.9, FeatureVDSO: 0.95},
	}}}
	fence := NewBudgetTierScorer(inner, 3)

	ctx := context.Background()
	measured := 0
	for _, rung := range [][2]string{
		{TierWeak, "all-off"}, {TierWeak, FeatureVDSO},
		{TierStrong, "all-off"}, {TierStrong, FeatureVDSO},
	} {
		out, err := fence.ScoreArm(ctx, rung[0], rung[1], nil)
		if err != nil {
			t.Fatalf("ScoreArm(%v): %v", rung, err)
		}
		if out.Measured {
			measured++
			continue
		}
		if !strings.Contains(out.Detail, "budget") {
			t.Fatalf("a fenced rung refused without naming the budget: %q", out.Detail)
		}
	}
	if measured != 3 || inner.calls != 3 || fence.Spent() != 3 {
		t.Fatalf("measured=%d inner.calls=%d spent=%d, want 3/3/3 (the 4th rung must be refused without reaching the inner scorer)",
			measured, inner.calls, fence.Spent())
	}
}

// The fence FAILS CLOSED: a zero budget spends nothing rather than meaning "unlimited",
// and the resulting report grades UNMEASURED — a refusal the classifier already handles
// honestly, not a fabricated set of zero deltas.
func TestBudgetTierScorerZeroBudgetSpendsNothing(t *testing.T) {
	inner := &countingTierScorer{inner: strengthFixtureScorer()}
	rep := strengthFixtureReport()
	if err := AnnotateModelStrength(context.Background(), rep, []string{TierWeak, TierStrong},
		NewBudgetTierScorer(inner, 0), StrengthParams{}); err != nil {
		t.Fatalf("AnnotateModelStrength: %v", err)
	}
	if inner.calls != 0 {
		t.Fatalf("a zero budget still forwarded %d calls", inner.calls)
	}
	for _, v := range rep.Verdicts {
		if v.Grade != VerdictUnmeasured {
			t.Fatalf("arm %q graded %s under a spent-nothing fence, want %s", v.ID, v.Grade, VerdictUnmeasured)
		}
	}
}

// countingTierScorer counts forwarded calls so the fence's "refuse before the spend"
// property is observable rather than inferred.
type countingTierScorer struct {
	inner TierScorer
	calls int
}

func (c *countingTierScorer) ScoreArm(ctx context.Context, tier, armID string, features map[string]string) (TierOutcome, error) {
	c.calls++
	return c.inner.ScoreArm(ctx, tier, armID, features)
}
