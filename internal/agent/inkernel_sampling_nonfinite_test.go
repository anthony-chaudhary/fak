package agent

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"
)

func nonFiniteRows() map[string][]float32 {
	negInf := float32(math.Inf(-1))
	return map[string][]float32{
		"nan":              {0.5, float32(math.NaN()), 0.25, 0.1},
		"positive-inf":     {0.5, float32(math.Inf(1)), 0.25, 0.1},
		"all-negative-inf": {negInf, negInf, negInf, negInf},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSamplerFlagsNonFiniteLogits(t *testing.T) {
	for name, row := range nonFiniteRows() {
		for _, temp := range []float64{0, 0.7} {
			var w samplerWorkspace
			sampleLogitsWS(row, temp, 0.9, 2, rand.New(rand.NewSource(1)), &w)
			if !w.nonFinite {
				t.Errorf("%s temp=%v: sampler did not flag the row", name, temp)
			}
			finite := []float32{0.5, 2, 0.25, float32(math.Inf(-1))}
			sampleLogitsWS(finite, temp, 0.9, 2, rand.New(rand.NewSource(1)), &w)
			if w.nonFinite {
				t.Errorf("%s temp=%v: flag survived a finite row", name, temp)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeLaneRefusesNonFiniteLogits(t *testing.T) {
	for name, row := range nonFiniteRows() {
		for _, temp := range []float64{0, 0.7} {
			emitted := false
			ln := &decodeLane{
				logits: row, temp: temp, topP: 0.9, maxNew: 4,
				rng:  rand.New(rand.NewSource(1)),
				emit: func(int) bool { emitted = true; return false },
			}
			if _, advance := ln.decodeOne(context.Background()); advance || !errors.Is(ln.err, ErrNonFiniteLogits) || emitted {
				t.Errorf("%s temp=%v: advance=%v err=%v emitted=%v, want ErrNonFiniteLogits before emit", name, temp, advance, ln.err, emitted)
			}
		}
	}
}
