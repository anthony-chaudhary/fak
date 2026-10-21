package agent

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// referenceSamplingDistribution is an independent transcription of the served
// truncation contract: softmax at temp, keep the topK most probable tokens,
// then the smallest descending-order prefix whose mass reaches topP (lower
// token index first on ties), renormalized.
func referenceSamplingDistribution(logits []float32, temp, topP float64, topK int) []float64 {
	maxL := math.Inf(-1)
	for _, x := range logits {
		maxL = math.Max(maxL, float64(x))
	}
	p := make([]float64, len(logits))
	for i, x := range logits {
		p[i] = math.Exp((float64(x) - maxL) / temp)
	}
	order := make([]int, len(p))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return p[order[a]] > p[order[b]] })
	keep := len(order)
	if topK > 0 && topK < keep {
		keep = topK
	}
	if topP > 0 && topP < 1 {
		var total float64
		for _, idx := range order[:keep] {
			total += p[idx]
		}
		var cum float64
		for rank, idx := range order[:keep] {
			if rank > 0 && cum >= topP*total {
				keep = rank
				break
			}
			cum += p[idx]
		}
	}
	out := make([]float64, len(p))
	var sum float64
	for _, idx := range order[:keep] {
		out[idx] = p[idx]
		sum += p[idx]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// TestSamplerRandomizedPITIsUniform checks the drawn distribution, not just
// the kept set: with F the token-index CDF of the reference distribution, the
// randomized probability integral transform F(x-) + U*p(x) of exact draws is
// Uniform(0,1). A Kolmogorov-Smirnov test at alpha=1e-5 rejects a sampler that
// draws from the wrong distribution. Adapted from DeepInfer's randomized-PIT
// sampler test (Apache-2.0, deepseek-ai/DeepInfer tests/models/test_sampler.py
// at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no upstream source is copied.
//
// fak-test:runtime fast est=2s lane=default
func TestSamplerRandomizedPITIsUniform(t *testing.T) {
	const (
		vocab = 3000
		draws = 20000
		alpha = 1e-5
	)
	critical := math.Sqrt(-math.Log(alpha/2) / (2 * draws))
	logitRNG := rand.New(rand.NewSource(41))
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(logitRNG.NormFloat64() * 2.5)
	}
	for _, tc := range []struct {
		name string
		temp float64
		topP float64
		topK int
	}{
		{"full_softmax", 1, 0, 0},
		{"top_p", 0.7, 0.95, 0},
		{"top_k", 0.7, 0, 20},
		{"top_k_then_top_p", 0.7, 0.9, 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := referenceSamplingDistribution(logits, tc.temp, tc.topP, tc.topK)
			below := make([]float64, vocab)
			var cum float64
			for i, p := range want {
				below[i] = cum
				cum += p
			}
			var w samplerWorkspace
			draw := rand.New(rand.NewSource(43))
			jitter := rand.New(rand.NewSource(47))
			pits := make([]float64, draws)
			for n := range pits {
				tok := sampleLogitsWS(logits, tc.temp, tc.topP, tc.topK, draw, &w)
				if tok < 0 || tok >= vocab || want[tok] == 0 {
					t.Fatalf("draw %d selected token %d outside the reference support", n, tok)
				}
				pits[n] = below[tok] + jitter.Float64()*want[tok]
			}
			sort.Float64s(pits)
			var d float64
			for i, u := range pits {
				d = math.Max(d, math.Max(float64(i+1)/draws-u, u-float64(i)/draws))
			}
			if d > critical {
				t.Fatalf("KS statistic %.5f exceeds the alpha=%g critical value %.5f", d, alpha, critical)
			}
		})
	}
}
