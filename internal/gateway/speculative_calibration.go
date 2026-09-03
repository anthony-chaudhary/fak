package gateway

import (
	"errors"
	"fmt"
	"math"
)

// ErrSyntheticQualityClaimProhibited is emitted when a synthetic acceptance rate is cited for quality claims.
var ErrSyntheticQualityClaimProhibited = errors.New("gateway: synthetic calibrated acceptance cannot be used for quality or accuracy claims")

// SpeculativeAcceptanceMode defines the explicit acceptance criteria.
type SpeculativeAcceptanceMode string

const (
	// SpecAcceptGreedy accepts draft token iff draftToken == argmax(targetLogits).
	SpecAcceptGreedy SpeculativeAcceptanceMode = "greedy"

	// SpecAcceptStochastic accepts draft token with probability min(1, p_target(x)/p_draft(x)).
	SpecAcceptStochastic SpeculativeAcceptanceMode = "stochastic"

	// SpecAcceptSynthetic applies a synthetic calibration acceptance rate [0, 1]
	// strictly for throughput benchmarking. Prohibited from supporting quality/accuracy claims.
	SpecAcceptSynthetic SpeculativeAcceptanceMode = "synthetic_calibrated"
)

// SpeculativeCalibrationReceipt records speculative acceptance metrics and quality eligibility.
type SpeculativeCalibrationReceipt struct {
	Mode                   SpeculativeAcceptanceMode `json:"mode"`
	DraftTokens            int                       `json:"draft_tokens"`
	AcceptedTokens         int                       `json:"accepted_tokens"`
	AcceptanceRate         float64                   `json:"acceptance_rate"`
	TargetLogitsEvaluated  int                       `json:"target_logits_evaluated"`
	IsSyntheticCalibration bool                      `json:"is_synthetic_calibration"`
	QualityClaimAllowed    bool                      `json:"quality_claim_allowed"`
}

// EvaluateSpeculativeAcceptance evaluates draft tokens against target logits under the declared mode.
func EvaluateSpeculativeAcceptance(
	targetLogits [][]float32, // [K][V]
	draftTokens []int, // [K]
	draftProbs []float32, // [K]
	mode SpeculativeAcceptanceMode,
	syntheticRate float64,
) (SpeculativeCalibrationReceipt, error) {
	var receipt SpeculativeCalibrationReceipt
	k := len(draftTokens)
	if k == 0 {
		return receipt, fmt.Errorf("draftTokens must not be empty")
	}
	if len(targetLogits) != k {
		return receipt, fmt.Errorf("mismatched targetLogits (%d) and draftTokens (%d)", len(targetLogits), k)
	}

	switch mode {
	case SpecAcceptGreedy:
		accepted := 0
		for i := 0; i < k; i++ {
			// Find argmax of targetLogits[i]
			logits := targetLogits[i]
			if len(logits) == 0 {
				return receipt, fmt.Errorf("target logits for step %d is empty", i)
			}
			maxIdx := 0
			maxVal := logits[0]
			for j, v := range logits {
				if v > maxVal {
					maxVal = v
					maxIdx = j
				}
			}
			if draftTokens[i] == maxIdx {
				accepted++
			} else {
				break // in speculative decode, rejection stops the accepted prefix
			}
		}

		rate := float64(accepted) / float64(k)
		return SpeculativeCalibrationReceipt{
			Mode:                   mode,
			DraftTokens:            k,
			AcceptedTokens:         accepted,
			AcceptanceRate:         rate,
			TargetLogitsEvaluated:  k,
			IsSyntheticCalibration: false,
			QualityClaimAllowed:    true,
		}, nil

	case SpecAcceptStochastic:
		accepted := 0
		for i := 0; i < k; i++ {
			logits := targetLogits[i]
			// compute softmax probability of draft token
			mx := logits[0]
			for _, v := range logits {
				if v > mx {
					mx = v
				}
			}
			var sum float64
			for _, v := range logits {
				sum += math.Exp(float64(v - mx))
			}
			pTarget := float32(math.Exp(float64(logits[draftTokens[i]]-mx)) / sum)
			pDraft := draftProbs[i]
			if pDraft <= 0 {
				pDraft = 1e-6
			}

			// Acceptance ratio = min(1, pTarget / pDraft)
			ratio := float64(pTarget / pDraft)
			if ratio >= 1.0 {
				accepted++
			} else {
				break
			}
		}

		rate := float64(accepted) / float64(k)
		return SpeculativeCalibrationReceipt{
			Mode:                   mode,
			DraftTokens:            k,
			AcceptedTokens:         accepted,
			AcceptanceRate:         rate,
			TargetLogitsEvaluated:  k,
			IsSyntheticCalibration: false,
			QualityClaimAllowed:    true,
		}, nil

	case SpecAcceptSynthetic:
		if syntheticRate < 0 || syntheticRate > 1.0 {
			return receipt, fmt.Errorf("syntheticRate must be in [0, 1], got %v", syntheticRate)
		}
		accepted := int(math.Round(float64(k) * syntheticRate))

		return SpeculativeCalibrationReceipt{
			Mode:                   mode,
			DraftTokens:            k,
			AcceptedTokens:         accepted,
			AcceptanceRate:         syntheticRate,
			TargetLogitsEvaluated:  k,
			IsSyntheticCalibration: true,
			QualityClaimAllowed:    false, // Quality claims prohibited
		}, nil

	default:
		return receipt, fmt.Errorf("unknown speculative acceptance mode: %q", mode)
	}
}

// CertifyQualityClaim validates whether a speculative receipt is eligible to support quality claims.
func CertifyQualityClaim(receipt SpeculativeCalibrationReceipt) error {
	if receipt.IsSyntheticCalibration || !receipt.QualityClaimAllowed {
		return ErrSyntheticQualityClaimProhibited
	}
	return nil
}
