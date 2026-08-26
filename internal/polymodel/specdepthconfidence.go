package polymodel

import "math"

// SpecDepthConfidenceGate chooses a bounded speculative prefix from calibrated,
// per-position acceptance predictions. It is deliberately a controller seam:
// callers still perform draft/verify/accept with the existing deterministic
// decoder, and a disabled or untrusted gate returns the caller's scalar depth.
type SpecDepthConfidenceGate struct {
	// Enabled is explicit so the zero value cannot change decode behavior.
	Enabled bool
	// MinCumulativeAcceptance is the minimum conservative probability that every
	// token in the selected prefix is accepted. It must be in (0, 1].
	MinCumulativeAcceptance float64
}

// SpecDepthConfidenceDecision records whether confidence, rather than the
// existing scalar/economic governor, selected DraftDepth.
type SpecDepthConfidenceDecision struct {
	DraftDepth     int
	UsedConfidence bool
	Reason         string
}

// DraftDepth returns the longest prefix within maxVerifyBudget whose cumulative
// conservative acceptance probability remains at least the configured threshold.
//
// calibrationErrorBound serves two purposes: every empirical AcceptanceProfile
// rate must be within that absolute distance of its corresponding prediction,
// and the same bound is subtracted from each prediction before multiplying the
// prefix probability. Missing, sparse, malformed, or miscalibrated observations
// fall back rather than silently shortening a draft. fallbackDepth is expected
// to be the result of the existing scalar/economic governor.
func (g SpecDepthConfidenceGate) DraftDepth(
	predicted []float64,
	observed []AcceptancePosition,
	calibrationErrorBound float64,
	maxVerifyBudget int,
	fallbackDepth int,
) SpecDepthConfidenceDecision {
	fallback := func(reason string) SpecDepthConfidenceDecision {
		return SpecDepthConfidenceDecision{DraftDepth: fallbackDepth, Reason: reason}
	}

	if !g.Enabled {
		return fallback("disabled")
	}
	if fallbackDepth < 0 || maxVerifyBudget <= 0 {
		return fallback("invalid budget or fallback depth")
	}
	if !finiteProbability(g.MinCumulativeAcceptance) || g.MinCumulativeAcceptance == 0 {
		return fallback("invalid cumulative acceptance threshold")
	}
	if math.IsNaN(calibrationErrorBound) || math.IsInf(calibrationErrorBound, 0) || calibrationErrorBound < 0 || calibrationErrorBound > 1 {
		return fallback("invalid calibration error bound")
	}

	if fallbackDepth == 0 {
		return fallback("scalar governor disabled drafting")
	}
	limit := min(len(predicted), maxVerifyBudget, fallbackDepth)
	if limit == 0 || len(observed) < limit {
		return fallback("insufficient confidence or calibration observations")
	}

	// Validate the complete budgeted vector before selecting a prefix. A bad
	// suffix must not be hidden merely because an earlier position crossed the
	// threshold: malformed model output makes the whole confidence receipt
	// untrusted, so the scalar governor remains authoritative.
	for position := 0; position < limit; position++ {
		prediction := predicted[position]
		if !finiteProbability(prediction) || prediction == 0 {
			return fallback("invalid predicted acceptance")
		}

		sample := observed[position]
		if sample.Position != position || sample.Proposed <= 0 || sample.Accepted < 0 || sample.Accepted > sample.Proposed || sample.Rate == nil {
			return fallback("invalid acceptance profile")
		}
		rate := *sample.Rate
		empiricalRate := float64(sample.Accepted) / float64(sample.Proposed)
		if !finiteProbability(rate) || math.Abs(rate-empiricalRate) > 1e-12 {
			return fallback("acceptance rate does not match counters")
		}
		if math.Abs(prediction-empiricalRate) > calibrationErrorBound {
			return fallback("confidence is not calibrated")
		}
	}

	cumulative := 1.0
	depth := 0
	for position := 0; position < limit; position++ {
		conservative := predicted[position] - calibrationErrorBound
		if conservative <= 0 {
			break
		}
		cumulative *= conservative
		if cumulative < g.MinCumulativeAcceptance {
			break
		}
		depth = position + 1
	}

	return SpecDepthConfidenceDecision{DraftDepth: depth, UsedConfidence: true, Reason: "calibrated confidence prefix"}
}

func finiteProbability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}
