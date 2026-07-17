package model

import (
	"fmt"
	"math"
)

// NegationTransferPair is one checkpoint-specific residual minimal pair.
type NegationTransferPair struct {
	Negated  []float64 `json:"negated"`
	Positive []float64 `json:"positive"`
}

type NegationCheckpointProbe struct {
	Checkpoint string                 `json:"checkpoint"`
	Family     string                 `json:"family"`
	Pairs      []NegationTransferPair `json:"pairs"`
}

type NegationTransferRow struct {
	Checkpoint    string  `json:"checkpoint"`
	Family        string  `json:"family"`
	TransferError float64 `json:"transfer_error"`
	RefitError    float64 `json:"refit_error"`
	RefitGain     float64 `json:"refit_gain"`
	RequiresRefit bool    `json:"requires_refit"`
}

// FitNegationTransferDirection learns one additive positive-complement direction
// from an anchor checkpoint. The returned artifact is held fixed during transfer.
func FitNegationTransferDirection(pairs []NegationTransferPair) ([]float64, error) {
	if len(pairs) == 0 || len(pairs[0].Negated) == 0 {
		return nil, fmt.Errorf("empty negation transfer pairs")
	}
	width := len(pairs[0].Negated)
	direction := make([]float64, width)
	for _, pair := range pairs {
		if len(pair.Negated) != width || len(pair.Positive) != width {
			return nil, fmt.Errorf("negation transfer shape drift")
		}
		for i := range direction {
			direction[i] += pair.Positive[i] - pair.Negated[i]
		}
	}
	for i := range direction {
		direction[i] /= float64(len(pairs))
	}
	return direction, nil
}

func transferRMSE(direction []float64, pairs []NegationTransferPair) (float64, error) {
	if len(pairs) == 0 {
		return 0, fmt.Errorf("empty transfer evaluation")
	}
	var sum float64
	var n int
	for _, pair := range pairs {
		if len(pair.Negated) != len(direction) || len(pair.Positive) != len(direction) {
			return 0, fmt.Errorf("transfer evaluation shape drift")
		}
		for i := range direction {
			d := pair.Negated[i] + direction[i] - pair.Positive[i]
			sum += d * d
			n++
		}
	}
	return math.Sqrt(sum / float64(n)), nil
}

// EvaluateNegationTransfer compares one held-fixed anchor fit with each checkpoint's
// refit upper bound. Threshold is declared by the caller rather than hidden.
func EvaluateNegationTransfer(anchor []float64, probes []NegationCheckpointProbe, threshold float64) ([]NegationTransferRow, error) {
	if len(anchor) == 0 || threshold <= 0 {
		return nil, fmt.Errorf("invalid negation transfer configuration")
	}
	rows := make([]NegationTransferRow, len(probes))
	for i, probe := range probes {
		transfer, err := transferRMSE(anchor, probe.Pairs)
		if err != nil {
			return nil, err
		}
		refitDirection, err := FitNegationTransferDirection(probe.Pairs)
		if err != nil {
			return nil, err
		}
		refit, err := transferRMSE(refitDirection, probe.Pairs)
		if err != nil {
			return nil, err
		}
		rows[i] = NegationTransferRow{Checkpoint: probe.Checkpoint, Family: probe.Family, TransferError: transfer, RefitError: refit, RefitGain: transfer - refit, RequiresRefit: transfer-refit > threshold}
	}
	return rows, nil
}
