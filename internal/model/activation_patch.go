package model

import "fmt"

// ActivationPatch captures or injects the completed residual activation at one layer.
// It is deterministic and owns its captured copy, so callers may reuse it across forward
// passes without retaining model-owned residual storage.
type ActivationPatch struct {
	layer      int
	activation []float32
}

// NewActivationPatch creates a probe for one non-negative transformer layer.
func NewActivationPatch(layer int) (*ActivationPatch, error) {
	if layer < 0 {
		return nil, fmt.Errorf("activation patch layer must be non-negative: %d", layer)
	}
	return &ActivationPatch{layer: layer}, nil
}

// Layer reports the probe's selected layer.
func (p *ActivationPatch) Layer() int { return p.layer }

// Captured reports whether this probe owns a captured activation.
func (p *ActivationPatch) Captured() bool { return len(p.activation) != 0 }

// Activation returns an independent copy of the captured activation.
func (p *ActivationPatch) Activation() []float32 {
	return append([]float32(nil), p.activation...)
}

// Capture records hidden only at the selected layer. It returns true when capture occurred.
func (p *ActivationPatch) Capture(layer int, hidden []float32) bool {
	if layer != p.layer {
		return false
	}
	p.activation = append(p.activation[:0], hidden...)
	return true
}

// Inject overwrites hidden only at the selected layer. It returns true when injection
// occurred and errors rather than partially patching a shape-mismatched residual.
func (p *ActivationPatch) Inject(layer int, hidden []float32) (bool, error) {
	if layer != p.layer {
		return false, nil
	}
	if !p.Captured() {
		return false, fmt.Errorf("activation patch layer %d has no capture", p.layer)
	}
	if len(hidden) != len(p.activation) {
		return false, fmt.Errorf("activation patch shape mismatch: hidden=%d captured=%d", len(hidden), len(p.activation))
	}
	copy(hidden, p.activation)
	return true, nil
}

// CaptureHook adapts Capture to the model's gated residual-hook seam.
func (p *ActivationPatch) CaptureHook() ResidualHook {
	return func(layer int, hidden []float32) { p.Capture(layer, hidden) }
}

// InjectHook adapts Inject to the residual-hook seam. Invalid probe state is a programming
// error at forward-pass setup, so the hook panics rather than silently running unpatched.
func (p *ActivationPatch) InjectHook() ResidualHook {
	return func(layer int, hidden []float32) {
		if _, err := p.Inject(layer, hidden); err != nil {
			panic(err)
		}
	}
}
