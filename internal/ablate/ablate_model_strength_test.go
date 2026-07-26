package ablate

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// fakeTierScorer is the deterministic, $0 stand-in for live per-tier measurement: a
// fixed score per (tier, arm). It is the model-strength twin of the fake armRunner the
// subprocess tests inject. An unknown (tier, arm) is an ERROR, not a zero — a silent
// zero would fabricate a delta and the whole point of this axis is that an unmeasured
// rung stays unmeasured.
type fakeTierScorer struct {
	scores map[string]map[string]float64
}

func (f fakeTierScorer) ScoreArm(_ context.Context, tier, armID string, _ map[string]string) (TierOutcome, error) {
	byArm, ok := f.scores[tier]
	if !ok {
		return TierOutcome{}, fmt.Errorf("fake tier scorer: no tier %q", tier)
	}
	score, ok := byArm[armID]
	if !ok {
		return TierOutcome{}, fmt.Errorf("fake tier scorer: no arm %q at tier %q", armID, tier)
	}
	return TierOutcome{Score: score, Measured: true, Detail: "fake scorer"}, nil
}

// strengthFixtureReport is a 4-arm report: the all-off baseline plus three real
// registered concepts standing in for the three trajectories the classifier must tell
// apart. Real feature tokens matter — strengthHardness only grades an arm HARD when it
// switches a REGISTERED concept on, which is what the #4414 debt router filters on.
func strengthFixtureReport() *Report {
	return &Report{
		Baseline:     "all-off",
		WorkloadHash: "hash-strength",
		Runs: []AblationRun{
			{ArmID: "all-off", Features: map[string]string{}},
			{ArmID: FeatureVDSO, Features: map[string]string{FeatureVDSO: "on"}},
			{ArmID: FeatureRadix, Features: map[string]string{FeatureRadix: "on"}},
			{ArmID: FeatureNormgate, Features: map[string]string{FeatureNormgate: "on"}},
		},
	}
}

// The three trajectories, all measured against the SAME baseline score at each tier:
//
//	vdso     +0.20 -> +0.20 -> +0.08   still pulls weight at the top     => LOAD_BEARING
//	radix    +0.20 -> +0.10 -> +0.00   the strong model erased it        => REDUNDANT
//	normgate +0.20 -> +0.05 -> -0.05   the strong model is better w/o it => HOBBLING
//
// Note radix and normgate BOTH look healthy at weak (+0.20) and would average out
// positive — the mean cannot separate them, the strongest rung can. That is the whole
// axis in one fixture.
func strengthFixtureScorer() fakeTierScorer {
	return fakeTierScorer{scores: map[string]map[string]float64{
		TierWeak:   {"all-off": 0.40, FeatureVDSO: 0.60, FeatureRadix: 0.60, FeatureNormgate: 0.60},
		TierMid:    {"all-off": 0.60, FeatureVDSO: 0.80, FeatureRadix: 0.70, FeatureNormgate: 0.65},
		TierStrong: {"all-off": 0.90, FeatureVDSO: 0.98, FeatureRadix: 0.90, FeatureNormgate: 0.85},
	}}
}

func TestAblateModelStrengthVerdict(t *testing.T) {
	tiers := []string{TierWeak, TierMid, TierStrong}

	// The witness of record: a REDUNDANT when the delta vanishes below epsilon at the
	// strong tier, a HOBBLING when it goes negative there, and a sustained LOAD_BEARING
	// control -- all three from one deterministic fake, no live model.
	t.Run("classifies each trajectory on the strongest tier", func(t *testing.T) {
		rep := strengthFixtureReport()
		if err := AnnotateModelStrength(context.Background(), rep, tiers, strengthFixtureScorer(), StrengthParams{}); err != nil {
			t.Fatalf("AnnotateModelStrength: %v", err)
		}
		for _, tc := range []struct {
			arm  string
			want string
		}{
			{FeatureVDSO, VerdictLoadBearing},
			{FeatureRadix, VerdictRedundant},
			{FeatureNormgate, VerdictHobbling},
		} {
			run := rep.ArmByID(tc.arm)
			if run == nil || run.ModelStrength == nil {
				t.Fatalf("arm %q carries no model-strength card", tc.arm)
			}
			if got := run.ModelStrength.Verdict; got != tc.want {
				t.Errorf("arm %q: verdict=%q want %q (reason=%s)", tc.arm, got, tc.want, run.ModelStrength.Reason)
			}
			// The trajectory the grade was computed from must be carried, so a reader can
			// re-derive the verdict instead of taking it on faith.
			if got := len(run.ModelStrength.Tiers); got != len(tiers) {
				t.Errorf("arm %q: %d tier rungs, want %d", tc.arm, got, len(tiers))
			}
			if got := run.ModelStrength.StrongestTier; got != TierStrong {
				t.Errorf("arm %q: strongest tier=%q want %q", tc.arm, got, TierStrong)
			}
		}
		// Sustained separates "held up at every rung" from "only survives at the top":
		// vdso clears the 0.05 bar at weak/mid/strong, so it is sustained.
		if !rep.ArmByID(FeatureVDSO).ModelStrength.Sustained {
			t.Errorf("vdso deltas +0.20/+0.20/+0.08 all clear the %.2f bar; want sustained", DefaultStrengthThreshold)
		}
		// The baseline is the reference every delta is taken against; grading it would
		// mint a fabricated REDUNDANT for the arm the table is measured from.
		if base := rep.ArmByID("all-off"); base.ModelStrength != nil {
			t.Errorf("baseline arm graded %q; want ungraded", base.ModelStrength.Verdict)
		}
	})

	// The flat verdict rows are the contract the already-shipped #4414 debt router
	// reads: it files a retirement issue per HARD scaffold graded REDUNDANT/HOBBLING.
	t.Run("publishes flat verdict rows for the debt router", func(t *testing.T) {
		rep := strengthFixtureReport()
		if err := AnnotateModelStrength(context.Background(), rep, tiers, strengthFixtureScorer(), StrengthParams{}); err != nil {
			t.Fatalf("AnnotateModelStrength: %v", err)
		}
		if len(rep.Verdicts) != 3 {
			t.Fatalf("%d verdict rows, want 3 (one per feature arm, baseline excluded)", len(rep.Verdicts))
		}
		byID := map[string]StrengthVerdict{}
		for _, v := range rep.Verdicts {
			byID[v.ID] = v
			if v.Hardness != "HARD" {
				t.Errorf("row %q: hardness=%q, want HARD (it switches a registered concept on)", v.ID, v.Hardness)
			}
			if strings.TrimSpace(v.Rationale) == "" {
				t.Errorf("row %q: empty rationale; the debt router quotes it into the issue", v.ID)
			}
		}
		if got := byID[FeatureRadix].Grade; got != VerdictRedundant {
			t.Errorf("radix row grade=%q want %q", got, VerdictRedundant)
		}
	})

	// The report JSON is what both sinks (--out and --json) carry, and what #4414 reads.
	t.Run("carries the verdict through the report JSON", func(t *testing.T) {
		rep := strengthFixtureReport()
		if err := AnnotateModelStrength(context.Background(), rep, tiers, strengthFixtureScorer(), StrengthParams{}); err != nil {
			t.Fatalf("AnnotateModelStrength: %v", err)
		}
		var doc struct {
			Runs []struct {
				ArmID         string         `json:"arm_id"`
				ModelStrength *ModelStrength `json:"model_strength"`
			} `json:"runs"`
			Verdicts []StrengthVerdict `json:"verdicts"`
		}
		if err := json.Unmarshal(rep.JSON(), &doc); err != nil {
			t.Fatalf("unmarshal report JSON: %v", err)
		}
		if len(doc.Verdicts) != 3 {
			t.Fatalf("report JSON carries %d top-level verdict rows, want 3", len(doc.Verdicts))
		}
		found := false
		for _, r := range doc.Runs {
			if r.ArmID != FeatureNormgate {
				continue
			}
			found = true
			if r.ModelStrength == nil || r.ModelStrength.Verdict != VerdictHobbling {
				t.Errorf("normgate model_strength in JSON = %+v, want verdict %q", r.ModelStrength, VerdictHobbling)
			}
		}
		if !found {
			t.Errorf("report JSON has no normgate run")
		}
	})

	// Omitting --models must leave the legacy single-tier output untouched.
	t.Run("no tiers is a no-op", func(t *testing.T) {
		rep := strengthFixtureReport()
		before := string(rep.JSON())
		if err := AnnotateModelStrength(context.Background(), rep, nil, strengthFixtureScorer(), StrengthParams{}); err != nil {
			t.Fatalf("AnnotateModelStrength: %v", err)
		}
		if after := string(rep.JSON()); after != before {
			t.Errorf("empty tier list changed the report JSON:\nbefore=%s\nafter=%s", before, after)
		}
		if len(rep.Verdicts) != 0 {
			t.Errorf("%d verdict rows without --models, want 0", len(rep.Verdicts))
		}
	})

	// The production default measures nothing, so it must refuse to grade rather than
	// hand the debt router a fabricated LOAD_BEARING (or a fabricated REDUNDANT, which
	// would file a real deletion issue off a stub).
	t.Run("the stub scorer classifies UNMEASURED", func(t *testing.T) {
		rep := strengthFixtureReport()
		if err := AnnotateModelStrength(context.Background(), rep, tiers, nil, StrengthParams{}); err != nil {
			t.Fatalf("AnnotateModelStrength: %v", err)
		}
		run := rep.ArmByID(FeatureVDSO)
		if run.ModelStrength == nil || run.ModelStrength.Verdict != VerdictUnmeasured {
			t.Fatalf("stub scorer verdict = %+v, want %q", run.ModelStrength, VerdictUnmeasured)
		}
		for _, td := range run.ModelStrength.Tiers {
			if td.Measured {
				t.Errorf("tier %q reported measured under the stub scorer", td.Tier)
			}
		}
	})

	// A scorer error must fail the whole axis loud rather than yield a partial report
	// whose ungraded arms silently read as "not debt".
	t.Run("a scorer error fails closed", func(t *testing.T) {
		rep := strengthFixtureReport()
		err := AnnotateModelStrength(context.Background(), rep, []string{"weak"},
			fakeTierScorer{scores: map[string]map[string]float64{}}, StrengthParams{})
		if err == nil {
			t.Fatalf("want an error when the scorer cannot score a tier")
		}
	})
}

func TestParseModelTiers(t *testing.T) {
	for _, tc := range []struct {
		name, spec string
		want       []string
		wantErr    bool
	}{
		{name: "empty keeps the legacy single-tier path", spec: "", want: nil},
		{name: "full ladder", spec: "weak,mid,strong", want: []string{TierWeak, TierMid, TierStrong}},
		// The classifier reads the LAST rung as "strongest", so a spec typed out of order
		// must be re-sorted or `--models strong,weak` would grade on weak.
		{name: "re-sorts into the canonical ladder", spec: "strong,weak", want: []string{TierWeak, TierStrong}},
		{name: "collapses duplicates and whitespace", spec: " mid , mid ", want: []string{TierMid}},
		{name: "a typo fails loud", spec: "weak,strng", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseModelTiers(tc.spec)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseModelTiers(%q) = %v, want an error", tc.spec, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseModelTiers(%q): %v", tc.spec, err)
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Errorf("ParseModelTiers(%q) = %v, want %v", tc.spec, got, tc.want)
			}
		})
	}
}
