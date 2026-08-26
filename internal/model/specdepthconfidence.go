package model

import "github.com/anthony-chaudhary/fak/internal/polymodel"

// DraftDepthWithConfidence composes the existing scalar/economic governor with
// the optional calibrated per-position controller. The scalar decision is made
// first and remains authoritative whenever the confidence gate is disabled or
// cannot prove calibration, preserving the pre-existing decode path.
func (g *SelfSpecGovernor) DraftDepthWithConfidence(
	acceptRate float64,
	pageInsPerDraft float64,
	observedDrafts int,
	gate polymodel.SpecDepthConfidenceGate,
	predicted []float64,
	profile []polymodel.AcceptancePosition,
	calibrationErrorBound float64,
	maxVerifyBudget int,
) polymodel.SpecDepthConfidenceDecision {
	fallbackDepth := g.DraftDepth(acceptRate, pageInsPerDraft, observedDrafts)
	return gate.DraftDepth(predicted, profile, calibrationErrorBound, maxVerifyBudget, fallbackDepth)
}
