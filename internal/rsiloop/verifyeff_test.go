package rsiloop

import (
	"math"
	"testing"
)

func TestScoreVerifyEffBoundsAndDeterminism(t *testing.T) {
	zeroVEI, zeroScore := ScoreVerifyEff(VerifyEffSnapshot{VerifierExposurePenalty: 1})
	if zeroVEI != 0 {
		t.Fatalf("worst-case VEI = %.2f, want 0", zeroVEI)
	}
	if zeroScore.Name != "verify_effectiveness" || zeroScore.Grade != "F" {
		t.Fatalf("unexpected worst-case scorecard: %+v", zeroScore)
	}
	if len(zeroScore.Components) != 7 {
		t.Fatalf("components = %d, want 7", len(zeroScore.Components))
	}

	perfect := VerifyEffSnapshot{
		EscapeAttributionRate:   1,
		IntroducerLinkRate:      1,
		GatePrecision:           1,
		CatcherRetentionRate:    1,
		QueueClassifiedRate:     1,
		JudgeSeparationRate:     1,
		VerifierExposurePenalty: 0,
	}
	perfectVEI, perfectScore := ScoreVerifyEff(perfect)
	if perfectVEI != 100 {
		t.Fatalf("perfect VEI = %.2f, want 100", perfectVEI)
	}
	if perfectScore.Grade != "A" {
		t.Fatalf("perfect grade = %q, want A", perfectScore.Grade)
	}

	// Out-of-range and NaN inputs must clamp safely into [0, 100].
	clampedVEI, _ := ScoreVerifyEff(VerifyEffSnapshot{
		EscapeAttributionRate:   2.5,
		IntroducerLinkRate:      math.NaN(),
		GatePrecision:           -0.5,
		VerifierExposurePenalty: -1.0,
	})
	if clampedVEI < 0 || clampedVEI > 100 {
		t.Fatalf("clamped VEI out of [0,100]: %.2f", clampedVEI)
	}

	// Determinism: repeated calls produce identical VEI.
	again, _ := ScoreVerifyEff(perfect)
	if again != perfectVEI {
		t.Fatalf("ScoreVerifyEff not deterministic: %.2f vs %.2f", perfectVEI, again)
	}
}

func TestVerifyEffHarnessRevertKeepKeep(t *testing.T) {
	h := NewVerifyEffHarness()
	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Escalated {
		t.Fatalf("expected escalated=false, got true (breaker=%v)", res.Rows)
	}
	if res.Cycles != 3 {
		t.Fatalf("cycles = %d, want 3", res.Cycles)
	}
	if res.Kept != 2 {
		t.Fatalf("kept = %d, want 2", res.Kept)
	}

	wantDecisions := []string{"REVERT", "KEEP", "KEEP"}
	wantKept := []bool{false, true, true}
	for i, row := range res.Rows {
		if row.Decision != wantDecisions[i] || row.Kept != wantKept[i] {
			t.Fatalf("cycle %d: got decision=%s kept=%v, want decision=%s kept=%v (row=%+v)",
				i+1, row.Decision, row.Kept, wantDecisions[i], wantKept[i], row)
		}
		if row.Score == nil || row.Score.Name != "verify_effectiveness" {
			t.Fatalf("cycle %d: missing verify_effectiveness scorecard: %+v", i+1, row.Score)
		}
	}
	if res.FinalBaseline <= res.Rows[0].Baseline {
		t.Fatalf("final baseline %.2f did not advance above initial %.2f", res.FinalBaseline, res.Rows[0].Baseline)
	}
}

func TestVerifyEffHarnessRejectsOutOfRangeSnapshot(t *testing.T) {
	base := VerifyEffSnapshot{
		EscapeAttributionRate: 0.2,
		GatePrecision:         0.5,
	}
	bad := VerifyEffSnapshot{
		EscapeAttributionRate: 1.5, // out of [0,1] -> SuiteGreen=false
		GatePrecision:         0.9,
	}
	h := NewVerifyEffHarnessWith(base, []VerifyEffProposal{
		{Label: "verifyeff:invalid-range", Snapshot: bad},
	})
	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Rows) != 1 || res.Rows[0].Kept || res.Rows[0].SuiteGreen {
		t.Fatalf("out-of-range snapshot must REVERT with SuiteGreen=false: %+v", res.Rows)
	}
}
