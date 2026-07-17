package model

import (
	"fmt"
	"sort"
)

// NegationHeadProbe captures the two causal ingredients of a copy-and-invert
// circuit: attention mass on the negated concept and the head's OV output there.
type NegationHeadProbe struct {
	Layer       int
	Head        int
	NegatorMass float64
	ConceptMass float64
	OVOutput    []float64
	ValenceAxis []float64
}

type NegationHeadScore struct {
	Layer       int
	Head        int
	CopyScore   float64
	InvertScore float64
	Failure     float64
}

// AttributeNegationHeads ranks heads that copy the concept strongly but fail to
// invert its valence. Stable layer/head ordering breaks equal-score ties.
func AttributeNegationHeads(probes []NegationHeadProbe) ([]NegationHeadScore, error) {
	if len(probes) == 0 {
		return nil, fmt.Errorf("empty negation head probes")
	}
	out := make([]NegationHeadScore, len(probes))
	for i, probe := range probes {
		if len(probe.OVOutput) == 0 || len(probe.OVOutput) != len(probe.ValenceAxis) {
			return nil, fmt.Errorf("head L%dH%d shape mismatch", probe.Layer, probe.Head)
		}
		invert := -sparseCosine(probe.OVOutput, probe.ValenceAxis)
		copyScore := probe.ConceptMass * probe.NegatorMass
		out[i] = NegationHeadScore{Layer: probe.Layer, Head: probe.Head, CopyScore: copyScore, InvertScore: invert, Failure: copyScore * (1 - invert) / 2}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Failure != out[j].Failure {
			return out[i].Failure > out[j].Failure
		}
		if out[i].Layer != out[j].Layer {
			return out[i].Layer < out[j].Layer
		}
		return out[i].Head < out[j].Head
	})
	return out, nil
}

// OVCircuitGraft is a rank-one valence reflection installed only for a selected
// head and negated route. Clean traffic returns without allocating or writing.
type OVCircuitGraft struct {
	Layer       int
	Head        int
	ValenceAxis []float64
}

func (g OVCircuitGraft) Apply(layer, head int, negated bool, output []float64) (bool, error) {
	if !negated || layer != g.Layer || head != g.Head {
		return false, nil
	}
	if len(output) != len(g.ValenceAxis) {
		return false, fmt.Errorf("OV graft shape mismatch")
	}
	var dot, norm float64
	for i := range output {
		dot += output[i] * g.ValenceAxis[i]
		norm += g.ValenceAxis[i] * g.ValenceAxis[i]
	}
	if norm == 0 {
		return false, fmt.Errorf("zero valence axis")
	}
	for i := range output {
		output[i] -= 2 * dot / norm * g.ValenceAxis[i]
	}
	return true, nil
}
