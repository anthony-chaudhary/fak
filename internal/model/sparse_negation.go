package model

import (
	"fmt"
	"math"
	"sort"
)

// SparseAutoencoder is a deterministic linear dictionary with ReLU-thresholded
// coefficients. Decoder vectors are row-major [features,width].
type SparseAutoencoder struct {
	Encoder   [][]float64
	Decoder   [][]float64
	Threshold float64
}

func (s SparseAutoencoder) Encode(hidden []float64) ([]float64, error) {
	if len(s.Encoder) == 0 || len(s.Decoder) != len(s.Encoder) {
		return nil, fmt.Errorf("invalid sparse dictionary")
	}
	coeff := make([]float64, len(s.Encoder))
	for feature, row := range s.Encoder {
		if len(row) != len(hidden) {
			return nil, fmt.Errorf("encoder feature %d width=%d want=%d", feature, len(row), len(hidden))
		}
		for i, weight := range row {
			coeff[feature] += weight * hidden[i]
		}
		coeff[feature] -= s.Threshold
		if coeff[feature] < 0 {
			coeff[feature] = 0
		}
	}
	return coeff, nil
}

func (s SparseAutoencoder) Decode(coeff []float64) ([]float64, error) {
	if len(coeff) != len(s.Decoder) || len(s.Decoder) == 0 {
		return nil, fmt.Errorf("decoder coefficient shape=%d features=%d", len(coeff), len(s.Decoder))
	}
	width := len(s.Decoder[0])
	out := make([]float64, width)
	for feature, row := range s.Decoder {
		if len(row) != width {
			return nil, fmt.Errorf("decoder feature %d shape drift", feature)
		}
		for i, weight := range row {
			out[i] += coeff[feature] * weight
		}
	}
	return out, nil
}

// SparseFeaturePair is one matched affirmative/negated activation pair.
type SparseFeaturePair struct {
	Affirmative []float64
	Negated     []float64
}

type SparseFeatureScore struct {
	Feature int
	Delta   float64
}

// RankNegationFeatures sorts dictionary features by mean coefficient increase
// from affirmative to negated activations. Feature id breaks equal-delta ties.
func RankNegationFeatures(s SparseAutoencoder, pairs []SparseFeaturePair) ([]SparseFeatureScore, error) {
	if len(pairs) == 0 {
		return nil, fmt.Errorf("empty negation feature pairs")
	}
	deltas := make([]float64, len(s.Encoder))
	for _, pair := range pairs {
		aff, err := s.Encode(pair.Affirmative)
		if err != nil {
			return nil, err
		}
		neg, err := s.Encode(pair.Negated)
		if err != nil {
			return nil, err
		}
		for i := range deltas {
			deltas[i] += neg[i] - aff[i]
		}
	}
	scores := make([]SparseFeatureScore, len(deltas))
	for i, delta := range deltas {
		scores[i] = SparseFeatureScore{Feature: i, Delta: delta / float64(len(pairs))}
	}
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].Delta == scores[j].Delta {
			return scores[i].Feature < scores[j].Feature
		}
		return scores[i].Delta > scores[j].Delta
	})
	return scores, nil
}

// SuppressSubstitute removes one sparse feature and inserts a resolved positive
// direction. It projects out any already-present positive component before adding
// exactly one unit, making repeated application idempotent.
func SuppressSubstitute(s SparseAutoencoder, hidden []float64, feature int, positive []float64) ([]float64, error) {
	coeff, err := s.Encode(hidden)
	if err != nil {
		return nil, err
	}
	if feature < 0 || feature >= len(coeff) || len(positive) != len(hidden) {
		return nil, fmt.Errorf("suppress-substitute shape mismatch")
	}
	coeff[feature] = 0
	out, err := s.Decode(coeff)
	if err != nil {
		return nil, err
	}
	var dot, norm float64
	for i := range out {
		dot += out[i] * positive[i]
		norm += positive[i] * positive[i]
	}
	if norm == 0 {
		return nil, fmt.Errorf("zero positive direction")
	}
	for i := range out {
		out[i] += (1 - dot/norm) * positive[i]
	}
	return out, nil
}

func sparseCosine(a, b []float64) float64 {
	var dot, aa, bb float64
	for i := range a {
		dot += a[i] * b[i]
		aa += a[i] * a[i]
		bb += b[i] * b[i]
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}
