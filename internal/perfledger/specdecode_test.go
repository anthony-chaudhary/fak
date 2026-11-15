package perfledger

import (
	"errors"
	"testing"
)

// TestSpecDecode verifies the ported speculative-decoding metrics against
// hand-computed values matching the source aiperf spec_decode_metrics expected
// outputs, including the zero-step and all-rejected cases.
func TestSpecDecode(t *testing.T) {
	// accepted=7, draft=10, steps=5:
	//   mean acceptance length = 1 + 7/5 = 2.4
	//   draft acceptance rate  = 7/10 * 100 = 70.0
	//   accepted-per-verified  = (7+5)/(10+5) = 12/15 = 0.8
	//   effective speedup      = 2.4 (mean acceptance length)
	t.Run("KnownStats", func(t *testing.T) {
		s := SpecDecodeStats{NumAcceptedDraftTokens: 7, NumDraftTokens: 10, NumSpecSteps: 5}

		got, err := MeanAcceptanceLength(s)
		if err != nil {
			t.Fatalf("MeanAcceptanceLength() error = %v, want nil", err)
		}
		if !near(got, 2.4) {
			t.Errorf("MeanAcceptanceLength() = %v, want 2.4", got)
		}

		got, err = DraftAcceptanceRate(s)
		if err != nil {
			t.Fatalf("DraftAcceptanceRate() error = %v, want nil", err)
		}
		if !near(got, 70.0) {
			t.Errorf("DraftAcceptanceRate() = %v, want 70.0", got)
		}

		got, err = AcceptedPerVerified(s)
		if err != nil {
			t.Fatalf("AcceptedPerVerified() error = %v, want nil", err)
		}
		if !near(got, 0.8) {
			t.Errorf("AcceptedPerVerified() = %v, want 0.8", got)
		}

		got, err = EffectiveSpeedup(s)
		if err != nil {
			t.Fatalf("EffectiveSpeedup() error = %v, want nil", err)
		}
		if !near(got, 2.4) {
			t.Errorf("EffectiveSpeedup() = %v, want 2.4", got)
		}
	})

	// All-rejected: accepted=0, draft=10, steps=5 -> the acceptance length is
	// exactly 1.0 and the draft acceptance rate is 0.0.
	t.Run("AllRejected", func(t *testing.T) {
		s := SpecDecodeStats{NumAcceptedDraftTokens: 0, NumDraftTokens: 10, NumSpecSteps: 5}

		got, err := MeanAcceptanceLength(s)
		if err != nil {
			t.Fatalf("MeanAcceptanceLength() error = %v, want nil", err)
		}
		if !near(got, 1.0) {
			t.Errorf("MeanAcceptanceLength() = %v, want 1.0", got)
		}

		got, err = DraftAcceptanceRate(s)
		if err != nil {
			t.Fatalf("DraftAcceptanceRate() error = %v, want nil", err)
		}
		if !near(got, 0.0) {
			t.Errorf("DraftAcceptanceRate() = %v, want 0.0", got)
		}

		got, err = AcceptedPerVerified(s)
		if err != nil {
			t.Fatalf("AcceptedPerVerified() error = %v, want nil", err)
		}
		if !near(got, 5.0/15.0) {
			t.Errorf("AcceptedPerVerified() = %v, want %v", got, 5.0/15.0)
		}
	})

	// Zero-step: steps=0 -> the per-step metrics are undefined (NoMetricValue).
	t.Run("ZeroStep", func(t *testing.T) {
		s := SpecDecodeStats{NumAcceptedDraftTokens: 0, NumDraftTokens: 0, NumSpecSteps: 0}

		if _, err := MeanAcceptanceLength(s); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("MeanAcceptanceLength() error = %v, want ErrNoMetricValue", err)
		}
		if _, err := DraftAcceptanceRate(s); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("DraftAcceptanceRate() error = %v, want ErrNoMetricValue", err)
		}
		if _, err := AcceptedPerVerified(s); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("AcceptedPerVerified() error = %v, want ErrNoMetricValue", err)
		}
		if _, err := EffectiveSpeedup(s); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("EffectiveSpeedup() error = %v, want ErrNoMetricValue", err)
		}
	})

	// Run-level: two requests, one with 7/10/5 and one all-rejected 0/10/5.
	//   token-weighted acceptance length = 1 + 7/10 = 1.7
	//   overall draft acceptance rate    = 7/20 * 100 = 35.0
	t.Run("RunLevel", func(t *testing.T) {
		stats := []SpecDecodeStats{
			{NumAcceptedDraftTokens: 7, NumDraftTokens: 10, NumSpecSteps: 5},
			{NumAcceptedDraftTokens: 0, NumDraftTokens: 10, NumSpecSteps: 5},
		}

		got, err := TokenWeightedAcceptanceLength(stats)
		if err != nil {
			t.Fatalf("TokenWeightedAcceptanceLength() error = %v, want nil", err)
		}
		if !near(got, 1.7) {
			t.Errorf("TokenWeightedAcceptanceLength() = %v, want 1.7", got)
		}

		got, err = OverallDraftAcceptanceRate(stats)
		if err != nil {
			t.Fatalf("OverallDraftAcceptanceRate() error = %v, want nil", err)
		}
		if !near(got, 35.0) {
			t.Errorf("OverallDraftAcceptanceRate() = %v, want 35.0", got)
		}
	})

	// Run-level undefined: zero total steps and zero total drafts.
	t.Run("RunLevelUndefined", func(t *testing.T) {
		zero := []SpecDecodeStats{{NumAcceptedDraftTokens: 0, NumDraftTokens: 0, NumSpecSteps: 0}}
		if _, err := TokenWeightedAcceptanceLength(zero); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("TokenWeightedAcceptanceLength() error = %v, want ErrNoMetricValue", err)
		}
		if _, err := OverallDraftAcceptanceRate(zero); !errors.Is(err, ErrNoMetricValue) {
			t.Errorf("OverallDraftAcceptanceRate() error = %v, want ErrNoMetricValue", err)
		}
	})
}
