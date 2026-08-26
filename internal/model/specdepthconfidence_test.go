package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

func TestDraftDepthWithConfidenceFallsBackToEconomicGovernor(t *testing.T) {
	governor := SelfSpecGovernor{MaxDraftDepth: 4, WarmupDrafts: 0, BasePageInsPerToken: govBasePageIns}
	gate := polymodel.SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.70}
	predicted := []float64{0.98, 0.95, 0.90, 0.85}
	profile := confidenceProfile(100, 97, 93, 40, 35)

	want := governor.DraftDepth(0.95, 0, 100)
	decision := governor.DraftDepthWithConfidence(0.95, 0, 100, gate, predicted, profile, 0.03, 4)
	if want != 4 || decision.DraftDepth != want || decision.UsedConfidence {
		t.Fatalf("scalar=%d confidence decision=%+v, want unchanged scalar depth 4", want, decision)
	}
}

func TestDraftDepthWithConfidenceCanShortenVerifiedBudgetOnly(t *testing.T) {
	governor := SelfSpecGovernor{MaxDraftDepth: 5, WarmupDrafts: 0, BasePageInsPerToken: govBasePageIns}
	gate := polymodel.SpecDepthConfidenceGate{Enabled: true, MinCumulativeAcceptance: 0.60}
	decision := governor.DraftDepthWithConfidence(
		0.95, 0, 100, gate,
		[]float64{0.96, 0.92, 0.86, 0.50, 0.40},
		confidenceProfile(100, 95, 90, 84, 48, 39),
		0.021, 5,
	)
	if decision.DraftDepth != 3 || !decision.UsedConfidence {
		t.Fatalf("decision = %+v, want calibrated depth 3", decision)
	}
}

func confidenceProfile(proposed int, accepted ...int) []polymodel.AcceptancePosition {
	profile := make([]polymodel.AcceptancePosition, len(accepted))
	for position, count := range accepted {
		rate := float64(count) / float64(proposed)
		profile[position] = polymodel.AcceptancePosition{Position: position, Proposed: proposed, Accepted: count, Rate: &rate}
	}
	return profile
}
