package model

import (
	"fmt"
	"math"
)

type NegationOperatorKind string

const (
	NegationStrict        NegationOperatorKind = "strict_involution"
	NegationPiRotation    NegationOperatorKind = "pi_rotation"
	NegationUnconstrained NegationOperatorKind = "unconstrained"
)

// NegationOperator applies a small learned linear map to a hidden slice. Strict and
// pi-rotation forms satisfy N^2=I structurally; unconstrained is the A/B control.
type NegationOperator struct {
	Kind    NegationOperatorKind
	Enabled bool
	Matrix  [][]float64
}

func NewNegationOperator(kind NegationOperatorKind, enabled bool) (NegationOperator, error) {
	if !enabled {
		return NegationOperator{Kind: kind, Matrix: [][]float64{{1, 0}, {0, 1}}}, nil
	}
	var matrix [][]float64
	switch kind {
	case NegationStrict:
		matrix = [][]float64{{-1, 0}, {0, 1}}
	case NegationPiRotation:
		matrix = [][]float64{{-1, 0}, {0, -1}}
	case NegationUnconstrained:
		matrix = [][]float64{{-.8, .15}, {.05, 1.1}}
	default:
		return NegationOperator{}, fmt.Errorf("unknown negation operator kind %q", kind)
	}
	return NegationOperator{Kind: kind, Enabled: true, Matrix: matrix}, nil
}
func (n NegationOperator) Apply(x []float64) ([]float64, error) {
	if len(n.Matrix) == 0 || len(x) != len(n.Matrix) {
		return nil, fmt.Errorf("negation operator shape mismatch: matrix=%d input=%d", len(n.Matrix), len(x))
	}
	out := make([]float64, len(x))
	for i, row := range n.Matrix {
		if len(row) != len(x) {
			return nil, fmt.Errorf("negation operator row %d shape=%d want=%d", i, len(row), len(x))
		}
		for j, w := range row {
			out[i] += w * x[j]
		}
	}
	return out, nil
}

// InvolutionError computes ||N^2-I||_F, the explicit regularizer for learned maps.
func (n NegationOperator) InvolutionError() (float64, error) {
	d := len(n.Matrix)
	var sum float64
	for i := 0; i < d; i++ {
		if len(n.Matrix[i]) != d {
			return 0, fmt.Errorf("non-square negation operator")
		}
		for j := 0; j < d; j++ {
			var v float64
			for k := 0; k < d; k++ {
				v += n.Matrix[i][k] * n.Matrix[k][j]
			}
			if i == j {
				v--
			}
			sum += v * v
		}
	}
	return math.Sqrt(sum), nil
}
func (n NegationOperator) DoubleNegationError(x []float64) (float64, error) {
	once, err := n.Apply(x)
	if err != nil {
		return 0, err
	}
	twice, err := n.Apply(once)
	if err != nil {
		return 0, err
	}
	var sum float64
	for i := range x {
		d := twice[i] - x[i]
		sum += d * d
	}
	return math.Sqrt(sum), nil
}
