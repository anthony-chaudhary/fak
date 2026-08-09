package quality

import (
	"fmt"
	"math"
	"strconv"
)

// ThresholdAuditRequest preserves the evidence needed to decide whether a
// threshold verdict survives supported numeric round-trips and plausible input
// perturbations. Pointer fields distinguish an explicit zero from absent
// evidence.
type ThresholdAuditRequest struct {
	Threshold              float64   `json:"threshold"`
	Comparison             string    `json:"comparison"`
	Observations           []float64 `json:"observations"`
	BoundaryWidth          *float64  `json:"boundary_width"`
	RoundTripDecimalPlaces *int      `json:"round_trip_decimal_places"`
	Perturbation           *float64  `json:"perturbation"`
}

// ThresholdAuditResult is the stable, machine-readable receipt for a threshold
// audit. A conclusion is accepted only when every supplied observation keeps
// its membership under both checks.
type ThresholdAuditResult struct {
	Schema                string  `json:"schema"`
	Verdict               string  `json:"verdict"`
	RefusalCode           string  `json:"refusal_code,omitempty"`
	Reason                string  `json:"reason"`
	ObservationCount      int     `json:"observation_count"`
	BoundaryCount         int     `json:"boundary_count"`
	BoundaryMass          float64 `json:"boundary_mass"`
	RoundTripFlipCount    int     `json:"round_trip_flip_count"`
	PerturbationFlipCount int     `json:"perturbation_flip_count"`
}

const ThresholdAuditSchema = "fak-quality-threshold-audit/1"

// AuditThreshold refuses conclusions when required evidence is absent or when
// supported precision/perturbation checks can change threshold membership.
func AuditThreshold(req ThresholdAuditRequest) ThresholdAuditResult {
	result := ThresholdAuditResult{Schema: ThresholdAuditSchema, Verdict: "refused"}
	if len(req.Observations) == 0 || req.BoundaryWidth == nil || req.RoundTripDecimalPlaces == nil || req.Perturbation == nil {
		result.RefusalCode = "evidence_missing"
		result.Reason = "observations, boundary_width, round_trip_decimal_places, and perturbation are required"
		return result
	}
	if !validComparison(req.Comparison) || !finite(req.Threshold) || *req.BoundaryWidth < 0 || *req.RoundTripDecimalPlaces < 0 || *req.Perturbation < 0 || !finite(*req.BoundaryWidth) || !finite(*req.Perturbation) {
		result.RefusalCode = "invalid_contract"
		result.Reason = "comparison and all numeric audit parameters must be finite and non-negative"
		return result
	}

	result.ObservationCount = len(req.Observations)
	for _, value := range req.Observations {
		if !finite(value) {
			result.RefusalCode = "invalid_contract"
			result.Reason = "observations must be finite"
			return result
		}
		member := thresholdMember(value, req.Threshold, req.Comparison)
		if math.Abs(value-req.Threshold) <= *req.BoundaryWidth {
			result.BoundaryCount++
		}
		rounded, err := strconv.ParseFloat(strconv.FormatFloat(value, 'f', *req.RoundTripDecimalPlaces, 64), 64)
		if err != nil {
			result.RefusalCode = "invalid_contract"
			result.Reason = fmt.Sprintf("round-trip observation: %v", err)
			return result
		}
		if thresholdMember(rounded, req.Threshold, req.Comparison) != member {
			result.RoundTripFlipCount++
		}
		if thresholdMember(value-*req.Perturbation, req.Threshold, req.Comparison) != member || thresholdMember(value+*req.Perturbation, req.Threshold, req.Comparison) != member {
			result.PerturbationFlipCount++
		}
	}
	result.BoundaryMass = float64(result.BoundaryCount) / float64(result.ObservationCount)
	if result.RoundTripFlipCount > 0 || result.PerturbationFlipCount > 0 {
		result.RefusalCode = "precision_dependent"
		result.Reason = "threshold membership changes under a supported round-trip or plausible perturbation"
		return result
	}
	result.Verdict = "accepted"
	result.Reason = "threshold membership is stable under the declared precision and perturbation checks"
	return result
}

func validComparison(comparison string) bool {
	switch comparison {
	case "at_least", "greater_than", "at_most", "less_than":
		return true
	default:
		return false
	}
}

func thresholdMember(value, threshold float64, comparison string) bool {
	switch comparison {
	case "at_least":
		return value >= threshold
	case "greater_than":
		return value > threshold
	case "at_most":
		return value <= threshold
	case "less_than":
		return value < threshold
	default:
		return false
	}
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
