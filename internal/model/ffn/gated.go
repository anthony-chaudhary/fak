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
	if len(gate) == 0 {
		return nil, fmt.Errorf("ffn: gate and up must be nonempty")
	}
	if len(gate) != len(up) {
		return nil, fmt.Errorf("ffn: gate length %d does not match up length %d", len(gate), len(up))
	}
	if activate == nil {
		return nil, fmt.Errorf("ffn: activation is nil")
	}
	if down == nil {
		return nil, fmt.Errorf("ffn: down projection is nil")
	}
	for i := range gate {
		gate[i] = activate(gate[i]) * up[i]
	}
	return down(gate)
}
