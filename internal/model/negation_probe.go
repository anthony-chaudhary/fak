package model

import (
	"encoding/json"
	"fmt"
	"math"
)

// NegationActivationExample is one labelled activation captured at one candidate layer.
type NegationActivationExample struct {
	PairID     string    `json:"pair_id"`
	Layer      int       `json:"layer"`
	Activation []float64 `json:"activation"`
	Negated    bool      `json:"negated"`
	Split      string    `json:"split"`
}

// NegationProbeArtifact is the frozen inline detector selected by held-out separability.
type NegationProbeArtifact struct {
	Version         string    `json:"version"`
	Layer           int       `json:"layer"`
	Weights         []float64 `json:"weights"`
	Bias            float64   `json:"bias"`
	Threshold       float64   `json:"threshold"`
	HeldOutAccuracy float64   `json:"held_out_accuracy"`
}

// Detect returns whether activation carries negation and its logistic probability.
func (p NegationProbeArtifact) Detect(activation []float64) (bool, float64) {
	if len(activation) != len(p.Weights) {
		return false, 0
	}
	z := p.Bias
	for i, w := range p.Weights {
		z += w * activation[i]
	}
	score := 1 / (1 + math.Exp(-z))
	return score >= p.Threshold, score
}
func (p NegationProbeArtifact) Marshal() ([]byte, error) { return json.MarshalIndent(p, "", "  ") }
func LoadNegationProbe(data []byte) (NegationProbeArtifact, error) {
	var p NegationProbeArtifact
	if err := json.Unmarshal(data, &p); err != nil {
		return p, err
	}
	if p.Version != "fak-negation-probe/1" || p.Layer < 0 || len(p.Weights) == 0 || p.Threshold <= 0 || p.Threshold >= 1 {
		return p, fmt.Errorf("invalid negation probe artifact")
	}
	return p, nil
}

// TrainNegationProbe fits one deterministic centroid-linear probe per layer and selects the
// layer with highest held-out accuracy (lowest layer wins ties).
func TrainNegationProbe(examples []NegationActivationExample) (NegationProbeArtifact, error) {
	layers := map[int]bool{}
	for _, e := range examples {
		layers[e.Layer] = true
	}
	if len(layers) == 0 {
		return NegationProbeArtifact{}, fmt.Errorf("empty negation activation dataset")
	}
	var best NegationProbeArtifact
	best.HeldOutAccuracy = -1
	for layer := 0; layer < 10000; layer++ {
		if !layers[layer] {
			continue
		}
		p, err := fitNegationLayer(examples, layer)
		if err != nil {
			return best, err
		}
		if p.HeldOutAccuracy > best.HeldOutAccuracy {
			best = p
		}
	}
	return best, nil
}
func fitNegationLayer(examples []NegationActivationExample, layer int) (NegationProbeArtifact, error) {
	var pos, neg []float64
	var pn, nn int
	for _, e := range examples {
		if e.Layer != layer || e.Split != "train" {
			continue
		}
		if len(pos) == 0 {
			pos = make([]float64, len(e.Activation))
			neg = make([]float64, len(e.Activation))
		}
		if len(e.Activation) != len(pos) {
			return NegationProbeArtifact{}, fmt.Errorf("layer %d shape drift", layer)
		}
		dst := pos
		if !e.Negated {
			dst = neg
		}
		for i, v := range e.Activation {
			dst[i] += v
		}
		if e.Negated {
			pn++
		} else {
			nn++
		}
	}
	if pn == 0 || nn == 0 {
		return NegationProbeArtifact{}, fmt.Errorf("layer %d missing train class", layer)
	}
	weights := make([]float64, len(pos))
	var posNorm, negNorm float64
	for i := range weights {
		pos[i] /= float64(pn)
		neg[i] /= float64(nn)
		weights[i] = pos[i] - neg[i]
		posNorm += pos[i] * pos[i]
		negNorm += neg[i] * neg[i]
	}
	bias := -.5 * (posNorm - negNorm)
	p := NegationProbeArtifact{Version: "fak-negation-probe/1", Layer: layer, Weights: weights, Bias: bias, Threshold: .55}
	var correct, total int
	for _, e := range examples {
		if e.Layer != layer || e.Split != "test" {
			continue
		}
		fires, _ := p.Detect(e.Activation)
		if fires == e.Negated {
			correct++
		}
		total++
	}
	if total == 0 {
		return p, fmt.Errorf("layer %d missing held-out data", layer)
	}
	p.HeldOutAccuracy = float64(correct) / float64(total)
	return p, nil
}
