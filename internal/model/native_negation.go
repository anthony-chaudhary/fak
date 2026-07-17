package model

import "fmt"

// NegationPrimitive is the L4 architecture seam for a cheap representation-level
// polarity flip. Implementations must preserve shape and make double negation an
// identity within their documented numeric tolerance.
type NegationPrimitive interface {
	Name() string
	Apply(state []float64) ([]float64, error)
	ParameterCount() int
	OperationCount(width int) int
}

// LearnedInvolutionPrimitive adapts the existing regularized linear operator to
// the common L4 seam. Its matrix is the learned parameter surface.
type LearnedInvolutionPrimitive struct{ Operator NegationOperator }

func (p LearnedInvolutionPrimitive) Name() string { return "learned_involution" }
func (p LearnedInvolutionPrimitive) Apply(state []float64) ([]float64, error) {
	return p.Operator.Apply(state)
}
func (p LearnedInvolutionPrimitive) ParameterCount() int {
	n := 0
	for _, row := range p.Operator.Matrix {
		n += len(row)
	}
	return n
}
func (p LearnedInvolutionPrimitive) OperationCount(width int) int { return width * width }

// PhaseNegationPrimitive represents negation as a fixed pi phase offset. Real
// coordinates encode each complex pair; multiplying by -1 is rotation by pi.
type PhaseNegationPrimitive struct{}

func (PhaseNegationPrimitive) Name() string                 { return "pi_phase" }
func (PhaseNegationPrimitive) ParameterCount() int          { return 0 }
func (PhaseNegationPrimitive) OperationCount(width int) int { return width }
func (PhaseNegationPrimitive) Apply(state []float64) ([]float64, error) {
	if len(state) == 0 || len(state)%2 != 0 {
		return nil, fmt.Errorf("phase negation requires non-empty complex coordinate pairs")
	}
	out := make([]float64, len(state))
	for i, v := range state {
		out[i] = -v
	}
	return out, nil
}

// PolarityChannelPrimitive reserves the final coordinate as a composable sign
// bit. Content remains untouched; negation flips only the polarity channel.
type PolarityChannelPrimitive struct{}

func (PolarityChannelPrimitive) Name() string           { return "polarity_channel" }
func (PolarityChannelPrimitive) ParameterCount() int    { return 0 }
func (PolarityChannelPrimitive) OperationCount(int) int { return 1 }
func (PolarityChannelPrimitive) Apply(state []float64) ([]float64, error) {
	if len(state) < 2 {
		return nil, fmt.Errorf("polarity channel requires content plus polarity")
	}
	out := append([]float64(nil), state...)
	out[len(out)-1] = -out[len(out)-1]
	return out, nil
}

// DefaultNegationPrimitives returns the three bounded candidates compared by the
// composition witness. No candidate is installed on the model automatically.
func DefaultNegationPrimitives() ([]NegationPrimitive, error) {
	op, err := NewNegationOperator(NegationStrict, true)
	if err != nil {
		return nil, err
	}
	return []NegationPrimitive{
		LearnedInvolutionPrimitive{Operator: op},
		PhaseNegationPrimitive{},
		PolarityChannelPrimitive{},
	}, nil
}
