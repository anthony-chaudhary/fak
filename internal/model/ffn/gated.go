// Package ffn contains dependency-light feed-forward network composition helpers.
package ffn

import "fmt"

// Activation maps one gate projection value to its gated activation.
type Activation func(float32) float32

// DownProject projects an activated intermediate row back to model width.
type DownProject func([]float32) ([]float32, error)

// Gated applies activate(gate) * up in place to gate, then projects the
// activated row through down exactly once.
//
// The caller owns gate and must ensure it does not overlap up. Gated validates
// its contract before mutating gate. It returns down errors unchanged.
func Gated(gate, up []float32, activate Activation, down DownProject) ([]float32, error) {
	if down == nil {
		return nil, fmt.Errorf("ffn: down projection is nil")
	}
	if err := ApplyInPlace(gate, up, activate); err != nil {
		return nil, err
	}
	return down(gate)
}

// ApplyInPlace fuses activate(gate) * up into the caller-owned gate row without a
// projection. It is the projection-free half of Gated, shared by callers that own
// both the gate/up panels and their own down-projection kernel (for example a
// batched MLP that must project rows through a resident weight).
//
// The caller owns gate and must ensure it does not overlap up. Every contract
// check runs BEFORE the first mutation, so a refused call leaves gate and up
// byte-identical: gate and up must be nonempty and of equal length, and activate
// must be non-nil. Up is read-only. On success the elements are written in
// increasing index order and nil is returned.
func ApplyInPlace(gate, up []float32, activate Activation) error {
	if len(gate) == 0 {
		return fmt.Errorf("ffn: gate and up must be nonempty")
	}
	if len(gate) != len(up) {
		return fmt.Errorf("ffn: gate length %d does not match up length %d", len(gate), len(up))
	}
	if activate == nil {
		return fmt.Errorf("ffn: activation is nil")
	}
	for i := range gate {
		gate[i] = activate(gate[i]) * up[i]
	}
	return nil
}
