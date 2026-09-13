package mtptune

import (
	"strings"
	"testing"
)

// TestBlockVerificationSweep runs the default block-verification sweep and
// asserts the point count, monotonicity, plateau detection, peak selection, and
// formatting contract.
func TestBlockVerificationSweep(t *testing.T) {
	cfg := DefaultBlockVerifyConfig()
	report, err := RunBlockVerifySweep(cfg)
	if err != nil {
		t.Fatalf("RunBlockVerifySweep(default) error: %v", err)
	}

	// Points cover K=1..32 exactly.
	if len(report.Points) != 32 {
		t.Fatalf("expected 32 points, got %d", len(report.Points))
	}
	for i, pt := range report.Points {
		wantK := i + 1
		if pt.K != wantK {
			t.Fatalf("point %d: expected K=%d, got K=%d", i, wantK, pt.K)
		}
	}

	// Decayed profile: accepted/step is non-decreasing and strictly increases
	// until the plateau, then holds flat.
	for i := 1; i < len(report.Points); i++ {
		if report.Points[i].AcceptedTokensPerStep < report.Points[i-1].AcceptedTokensPerStep-1e-12 {
			t.Fatalf("accepted/step decreased at K=%d: %.6f -> %.6f",
				report.Points[i].K, report.Points[i-1].AcceptedTokensPerStep, report.Points[i].AcceptedTokensPerStep)
		}
	}

	// Plateau must be in [2,32], report its own series value, and every marginal
	// gain at or after PlateauK must stay below eps (the definitional property;
	// a geometric profile asymptotes rather than reaching an exact flat value).
	if report.PlateauK < 2 || report.PlateauK > 32 {
		t.Fatalf("PlateauK out of [2,32]: %d", report.PlateauK)
	}
	if report.PlateauAcceptedPerStep <= 0 {
		t.Fatalf("plateau value must be positive, got %.6f", report.PlateauAcceptedPerStep)
	}
	tail := report.Points[len(report.Points)-1].AcceptedTokensPerStep
	if report.PlateauAcceptedPerStep > tail+1e-9 {
		t.Fatalf("plateau value %.6f exceeds series tail %.6f", report.PlateauAcceptedPerStep, tail)
	}
	for i := 1; i < len(report.Points); i++ {
		if report.Points[i].K < report.PlateauK {
			continue
		}
		if report.Points[i].K == report.PlateauK {
			continue
		}
		prev := report.Points[i-1].AcceptedTokensPerStep
		gain := (report.Points[i].AcceptedTokensPerStep - prev) / prev
		if gain >= plateauEpsilon {
			t.Fatalf("marginal gain at K=%d is %.4f, not below eps %.2f after plateau K=%d",
				report.Points[i].K, gain, plateauEpsilon, report.PlateauK)
		}
	}

	// Peak in range and no worse than the baseline K=1.
	if report.PeakK < cfg.KMin || report.PeakK > cfg.KMax {
		t.Fatalf("PeakK out of range: %d", report.PeakK)
	}
	if report.PeakAcceptedPerStep < report.Points[0].AcceptedTokensPerStep-1e-12 {
		t.Fatalf("peak %.6f below K=1 baseline %.6f",
			report.PeakAcceptedPerStep, report.Points[0].AcceptedTokensPerStep)
	}

	// Formatting is panic-free and marks both plateau and peak, and shows "K=".
	out := FormatBlockVerifyReport(report)
	if !strings.Contains(out, "plateau") {
		t.Fatalf("formatted report missing 'plateau':\n%s", out)
	}
	if !strings.Contains(out, "peak") {
		t.Fatalf("formatted report missing 'peak':\n%s", out)
	}
	if !strings.Contains(out, "K=") {
		t.Fatalf("formatted report missing 'K=':\n%s", out)
	}
}

// TestBlockVerificationSweepInvalid asserts config validation rejects KMin<1 and
// KMax>64.
func TestBlockVerificationSweepInvalid(t *testing.T) {
	if _, err := RunBlockVerifySweep(BlockVerifyConfig{KMin: 0, KMax: 8, PerfectDraftAccept: 0.9, BonusEnabled: true}); err == nil {
		t.Fatal("expected error for KMin=0, got nil")
	}
	if _, err := RunBlockVerifySweep(BlockVerifyConfig{KMin: 1, KMax: 100, PerfectDraftAccept: 0.9, BonusEnabled: true}); err == nil {
		t.Fatal("expected error for KMax=100, got nil")
	}
}

// TestBlockVerificationBonusLayout asserts the perfect-draft (p=1.0) committed
// token count: K+1 under the 1+N bonus layout, K without it.
func TestBlockVerificationBonusLayout(t *testing.T) {
	withBonus, err := RunBlockVerifySweep(BlockVerifyConfig{KMin: 1, KMax: 32, PerfectDraftAccept: 1.0, BonusEnabled: true})
	if err != nil {
		t.Fatalf("bonus sweep error: %v", err)
	}
	for _, pt := range withBonus.Points {
		want := float64(pt.K + 1)
		if pt.AcceptedTokensPerStep != want {
			t.Fatalf("with bonus: K=%d expected %.4f, got %.4f", pt.K, want, pt.AcceptedTokensPerStep)
		}
	}

	noBonus, err := RunBlockVerifySweep(BlockVerifyConfig{KMin: 1, KMax: 32, PerfectDraftAccept: 1.0, BonusEnabled: false})
	if err != nil {
		t.Fatalf("no-bonus sweep error: %v", err)
	}
	for _, pt := range noBonus.Points {
		want := float64(pt.K)
		if pt.AcceptedTokensPerStep != want {
			t.Fatalf("no bonus: K=%d expected %.4f, got %.4f", pt.K, want, pt.AcceptedTokensPerStep)
		}
	}
}
