package model

import (
	"fmt"
	"math"
)

// MaxNegAdapterRank bounds the specialist's low-rank compute. Negation adapters
// are deliberately small single-layer operators, not general replacement weights.
const MaxNegAdapterRank = 8

// NegAdapterRoute describes the routing work for one residual activation.
// AdapterMACs is zero on a detector-negative path.
type NegAdapterRoute struct {
	DetectorFired  bool
	AdapterApplied bool
	AdapterMACs    int
}

// NegAdapter routes one bounded LoRA delta from an inline negation detector at a
// single residual layer. The detector-negative path performs no allocation and
// does not write hidden, preserving its bits exactly.
type NegAdapter struct {
	Probe   NegationProbeArtifact
	Adapter *LoRAAdapter
	Layer   int
	Enabled bool
}

// NewNegAdapter validates a same-layer detector and square residual adapter.
// A same-layer route avoids carrying mutable detector state between forward passes.
func NewNegAdapter(probe NegationProbeArtifact, adapter *LoRAAdapter) (*NegAdapter, error) {
	if adapter == nil {
		return nil, fmt.Errorf("negation adapter is nil")
	}
	if err := adapter.validate(); err != nil {
		return nil, err
	}
	if probe.Version != "fak-negation-probe/1" || probe.Layer < 0 || len(probe.Weights) == 0 || probe.Threshold <= 0 || probe.Threshold >= 1 {
		return nil, fmt.Errorf("invalid negation probe artifact")
	}
	if adapter.Rank > MaxNegAdapterRank {
		return nil, fmt.Errorf("negation adapter rank %d exceeds maximum %d", adapter.Rank, MaxNegAdapterRank)
	}
	if adapter.In != len(probe.Weights) || adapter.Out != adapter.In {
		return nil, fmt.Errorf("negation adapter shape out=%d in=%d does not match residual width %d", adapter.Out, adapter.In, len(probe.Weights))
	}
	return &NegAdapter{Probe: probe, Adapter: adapter, Layer: probe.Layer, Enabled: true}, nil
}

// ParameterCount reports the trainable low-rank parameters (A and B), excluding
// the separately frozen detector.
func (a *NegAdapter) ParameterCount() int {
	if a == nil || a.Adapter == nil {
		return 0
	}
	return len(a.Adapter.A) + len(a.Adapter.B)
}

// Apply runs the detector and conditionally applies the LoRA delta in place.
// A false decision guarantees AdapterMACs==0 and leaves hidden bit-identical.
func (a *NegAdapter) Apply(layer int, hidden []float32) NegAdapterRoute {
	if a == nil || !a.Enabled || a.Adapter == nil || layer != a.Layer || len(hidden) != len(a.Probe.Weights) {
		return NegAdapterRoute{}
	}
	z := a.Probe.Bias
	for i, weight := range a.Probe.Weights {
		z += weight * float64(hidden[i])
	}
	fired := 1/(1+math.Exp(-z)) >= a.Probe.Threshold
	if !fired {
		return NegAdapterRoute{}
	}
	delta := a.Adapter.Delta(hidden)
	for i := range hidden {
		hidden[i] += delta[i]
	}
	return NegAdapterRoute{
		DetectorFired:  true,
		AdapterApplied: true,
		AdapterMACs:    a.Adapter.Rank * (a.Adapter.In + a.Adapter.Out),
	}
}

// Hook exposes the adapter through the model's gated residual-hook seam.
func (a *NegAdapter) Hook() ResidualHook {
	return func(layer int, hidden []float32) { a.Apply(layer, hidden) }
}
