package polymodel

import (
	"math"
	"testing"
)

func TestSpecDepthConfidenceGateSelectsCalibratedPrefix(t *testing.T) {
	gate := SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.60}
	predicted := []float64{0.96, 0.92, 0.86, 0.50, 0.40}
	profile := acceptanceProfileForTest(100, 95, 90, 84, 48, 39)

	decision := gate.DraftDepth(predicted, profile, 0.021, 5, 5)
	if decision.DraftDepth != 3 || !decision.UsedConfidence {
		t.Fatalf("decision = %+v, want calibrated depth 3", decision)
	}
}

func TestSpecDepthConfidenceGateMiscalibrationFallsBack(t *testing.T) {
	gate := SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.70}
	predicted := []float64{0.98, 0.95, 0.90, 0.85}
	profile := acceptanceProfileForTest(100, 97, 93, 40, 35)

	decision := gate.DraftDepth(predicted, profile, 0.03, 4, 4)
	if decision.DraftDepth != 4 || decision.UsedConfidence {
		t.Fatalf("decision = %+v, want scalar-governor fallback depth 4", decision)
	}
}

func TestSpecDepthConfidenceGateRejectsZeroAndNaNConfidence(t *testing.T) {
	gate := SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.50}
	profile := acceptanceProfileForTest(10, 9, 8)

	for name, predicted := range map[string][]float64{
		"zero": {0.9, 0},
		"NaN":  {0.9, math.NaN()},
	} {
		t.Run(name, func(t *testing.T) {
			decision := gate.DraftDepth(predicted, profile, 0.20, 2, 2)
			if decision.DraftDepth != 2 || decision.UsedConfidence {
				t.Fatalf("decision = %+v, want fallback depth 2", decision)
			}
		})
	}
}

func TestSpecDepthConfidenceGateDefaultOffAndVerifyBudget(t *testing.T) {
	predicted := []float64{0.99, 0.99, 0.99}
	profile := acceptanceProfileForTest(100, 99, 99, 99)

	if decision := (SpecDepthConfidenceGate{}).DraftDepth(predicted, profile, 0, 3, 2); decision.DraftDepth != 2 || decision.UsedConfidence {
		t.Fatalf("zero-value decision = %+v, want unchanged fallback", decision)
	}

	gate := SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.90}
	if decision := gate.DraftDepth(predicted, profile, 0, 2, 3); decision.DraftDepth != 2 || !decision.UsedConfidence {
		t.Fatalf("budgeted decision = %+v, want confidence depth capped at 2", decision)
	}
}

func TestSpecDepthConfidenceGateNeverExpandsScalarDepth(t *testing.T) {
	gate := SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.50}
	predicted := []float64{0.99, 0.99, 0.99}
	profile := acceptanceProfileForTest(100, 99, 99, 99)

	for _, fallback := range []int{0, 2} {
		decision := gate.DraftDepth(predicted, profile, 0, 3, fallback)
		if decision.DraftDepth != fallback {
			t.Fatalf("fallback %d: decision = %+v, confidence must only shorten scalar depth", fallback, decision)
		}
	}
}

func TestSpecDepthConfidenceGateConsumesAcceptancePositionCounters(t *testing.T) {
	rate := 0.8
	profile := []AcceptancePosition{{Position: 0, Proposed: 10, Accepted: 8, Rate: &rate}}
	decision := (SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.75}).DraftDepth(
		[]float64{0.8}, profile, 0, 1, 1,
	)
	if decision.DraftDepth != 1 || !decision.UsedConfidence {
		t.Fatalf("decision = %+v, want #8258 counter-compatible depth 1", decision)
	}
}

func acceptanceProfileForTest(proposed int, accepted ...int) []AcceptancePosition {
	profile := make([]AcceptancePosition, len(accepted))
	for position, count := range accepted {
		rate := float64(count) / float64(proposed)
		profile[position] = AcceptancePosition{
			Position: position,
			Proposed: proposed,
			Accepted: count,
			Rate:     &rate,
		}
	}
	return profile
}
