package model

import (
	"errors"
	"fmt"
	"math"
)

// SpecResidualMode selects the rejection residual used by a stochastic
// speculative round. The zero value is the lossless Leviathan-Chen rule.
type SpecResidualMode uint8

const (
	SpecResidualLossless SpecResidualMode = iota
	SpecResidualPenaltyAdjusted
)

var (
	ErrSpecRejectionShape       = errors.New("model: speculative rejection evidence shape is invalid")
	ErrSpecRejectionTargetRows  = errors.New("model: speculative rejection requires exactly K+1 target rows")
	ErrSpecRejectionUnkeyed     = errors.New("model: speculative rejection requires a request key")
	ErrSpecRejectionProbability = errors.New("model: speculative rejection probability row is invalid")
)

// SpecRejectionPlanInput is the immutable evidence for one stochastic round.
// TargetLogProbs has exactly K+1 rows: one row for each draft decision and a
// final row for the bonus decision. StepOffset is the absolute request-local
// draft position, so separate rounds cannot reuse keyed draws.
type SpecRejectionPlanInput struct {
	DraftTokens             []int
	DraftEvidence           []DraftTokenEvidence
	TargetLogProbs          [][]float32
	RequestKey              uint64
	StepOffset              int
	ResidualMode            SpecResidualMode
	DraftProbabilityPenalty float32
}

// SpecRejectionPlan is a side-effect-free decision. A caller may commit
// AcceptedTokens and CorrectionToken only after its grammar/state transaction
// accepts them; otherwise it can discard this plan without rollback work here.
type SpecRejectionPlan struct {
	AcceptedTokens  []int
	CorrectionToken int
	RejectedAt      int
	RollbackKVCount int
	AcceptProb      []float32
	NextStep        int
}

// BuildSpecRejectionPlan validates complete stochastic evidence and derives one
// keyed rejection-sampling decision without mutating any input slice.
func BuildSpecRejectionPlan(in SpecRejectionPlanInput) (SpecRejectionPlan, error) {
	k := len(in.DraftTokens)
	if k == 0 || len(in.DraftEvidence) != k {
		return SpecRejectionPlan{}, ErrSpecRejectionShape
	}
	if len(in.TargetLogProbs) != k+1 {
		return SpecRejectionPlan{}, fmt.Errorf("%w: got %d rows for K=%d", ErrSpecRejectionTargetRows, len(in.TargetLogProbs), k)
	}
	if in.RequestKey == 0 {
		return SpecRejectionPlan{}, ErrSpecRejectionUnkeyed
	}
	if in.StepOffset < 0 {
		return SpecRejectionPlan{}, fmt.Errorf("%w: negative step offset", ErrSpecRejectionShape)
	}
	maxInt := int(^uint(0) >> 1)
	if in.StepOffset >= maxInt || k > maxInt-in.StepOffset-1 {
		return SpecRejectionPlan{}, fmt.Errorf("%w: step range overflows int", ErrSpecRejectionShape)
	}
	if in.ResidualMode != SpecResidualLossless && in.ResidualMode != SpecResidualPenaltyAdjusted {
		return SpecRejectionPlan{}, fmt.Errorf("%w: residual mode %d", ErrSpecRejectionShape, in.ResidualMode)
	}
	penalty := in.DraftProbabilityPenalty
	if in.ResidualMode == SpecResidualPenaltyAdjusted {
		if !(penalty > 0 && penalty <= 1) || math.IsNaN(float64(penalty)) {
			return SpecRejectionPlan{}, fmt.Errorf("%w: draft probability penalty %g", ErrSpecRejectionProbability, penalty)
		}
	} else {
		penalty = 1
	}

	proposal := DraftProposal{Tokens: append([]int(nil), in.DraftTokens...), DraftEvidence: in.DraftEvidence}
	evidence, err := proposal.StochasticDraftEvidence()
	if err != nil {
		return SpecRejectionPlan{}, err
	}
	vocab := len(in.TargetLogProbs[0])
	if vocab == 0 {
		return SpecRejectionPlan{}, ErrSpecRejectionProbability
	}
	targets := make([][]float32, len(in.TargetLogProbs))
	for i, row := range in.TargetLogProbs {
		targets[i], err = normalizedLogProbRow(row, vocab)
		if err != nil {
			return SpecRejectionPlan{}, fmt.Errorf("%w: target row %d: %v", ErrSpecRejectionProbability, i, err)
		}
	}
	for i, row := range evidence {
		if validateErr := validateDraftProbabilityCoordinates(row, vocab); validateErr != nil {
			return SpecRejectionPlan{}, fmt.Errorf("%w: draft row %d: %v", ErrSpecRejectionProbability, i, validateErr)
		}
	}

	plan := SpecRejectionPlan{
		AcceptedTokens:  make([]int, 0, k),
		CorrectionToken: -1,
		RejectedAt:      -1,
		AcceptProb:      make([]float32, k),
		NextStep:        in.StepOffset + k + 1,
	}
	for i, row := range evidence {
		q, expandErr := expandDraftProbabilityRow(row, vocab)
		if expandErr != nil {
			return SpecRejectionPlan{}, fmt.Errorf("%w: draft row %d: %v", ErrSpecRejectionProbability, i, expandErr)
		}
		p := targets[i]
		alpha := losslessAcceptProbability(in.DraftTokens[i], p, q)
		if in.ResidualMode == SpecResidualPenaltyAdjusted {
			alpha = penaltyAdjustedAcceptProbability(p, q, penalty)
		}
		plan.AcceptProb[i] = alpha
		coin := KeyedUniform(in.RequestKey, DomainAcceptCoin, in.StepOffset+i)
		if coin < alpha {
			plan.AcceptedTokens = append(plan.AcceptedTokens, in.DraftTokens[i])
			continue
		}

		residual := rejectionResidual(p, q, penalty)
		plan.CorrectionToken = sampleProbabilityRow32(residual, KeyedUniform(in.RequestKey, DomainResidual, in.StepOffset+i))
		plan.RejectedAt = i
		plan.RollbackKVCount = k - i
		return plan, nil
	}

	plan.CorrectionToken = sampleProbabilityRow32(targets[k], KeyedUniform(in.RequestKey, DomainBonusToken, in.StepOffset+k))
	return plan, nil
}

func normalizedLogProbRow(logp []float32, vocab int) ([]float32, error) {
	if len(logp) != vocab {
		return nil, fmt.Errorf("width %d, want %d", len(logp), vocab)
	}
	out := make([]float32, vocab)
	var mass float64
	for i, value32 := range logp {
		value := float64(value32)
		if math.IsNaN(value) || math.IsInf(value, 1) {
			return nil, errors.New("non-finite log probability")
		}
		if math.IsInf(value, -1) {
			continue
		}
		p := math.Exp(value)
		out[i] = float32(p)
		mass += p
	}
	if math.Abs(mass-1) > 1e-5 {
		return nil, fmt.Errorf("probability mass %.8f", mass)
	}
	return out, nil
}

func expandDraftProbabilityRow(row DraftTokenEvidence, vocab int) ([]float32, error) {
	if err := validateDraftProbabilityCoordinates(row, vocab); err != nil {
		return nil, err
	}
	out := make([]float32, vocab)
	if len(row.TokenIDs) == 0 {
		for i, logp := range row.LogProbs {
			if !math.IsInf(float64(logp), -1) {
				out[i] = float32(math.Exp(float64(logp)))
			}
		}
		return out, nil
	}
	for i, token := range row.TokenIDs {
		if !math.IsInf(float64(row.LogProbs[i]), -1) {
			out[token] = float32(math.Exp(float64(row.LogProbs[i])))
		}
	}
	return out, nil
}

func validateDraftProbabilityCoordinates(row DraftTokenEvidence, vocab int) error {
	if len(row.TokenIDs) == 0 {
		if len(row.LogProbs) != vocab {
			return fmt.Errorf("full row width %d, want %d", len(row.LogProbs), vocab)
		}
		return nil
	}
	if len(row.LogProbs) != len(row.TokenIDs) {
		return fmt.Errorf("compact row has %d coordinates and %d log probabilities", len(row.TokenIDs), len(row.LogProbs))
	}
	seen := make(map[int]struct{}, len(row.TokenIDs))
	for _, token := range row.TokenIDs {
		if token < 0 || token >= vocab {
			return fmt.Errorf("token coordinate %d outside vocabulary %d", token, vocab)
		}
		if _, ok := seen[token]; ok {
			return fmt.Errorf("duplicate token coordinate %d", token)
		}
		seen[token] = struct{}{}
	}
	return nil
}

func losslessAcceptProbability(token int, p, q []float32) float32 {
	if token < 0 || token >= len(p) || token >= len(q) || q[token] <= 0 {
		return 0
	}
	return min(float32(1), p[token]/q[token])
}

func penaltyAdjustedAcceptProbability(p, q []float32, penalty float32) float32 {
	var residualMass float64
	for i := range p {
		if diff := float64(p[i] - penalty*q[i]); diff > 0 {
			residualMass += diff
		}
	}
	return float32(max(0, min(1, 1-residualMass)))
}

func rejectionResidual(p, q []float32, penalty float32) []float32 {
	out := make([]float32, len(p))
	var mass float64
	for i := range p {
		if diff := float64(p[i] - penalty*q[i]); diff > 0 {
			out[i] = float32(diff)
			mass += diff
		}
	}
	if mass <= 1e-12 {
		copy(out, p)
		return out
	}
	inv := float32(1 / mass)
	for i := range out {
		out[i] *= inv
	}
	return out
}

func sampleProbabilityRow32(row []float32, coin float32) int {
	var cumulative float32
	last := -1
	for i, p := range row {
		if p <= 0 {
			continue
		}
		last = i
		cumulative += p
		if cumulative > coin {
			return i
		}
	}
	return last
}
