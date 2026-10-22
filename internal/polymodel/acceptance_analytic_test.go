package polymodel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// analyticAcceptance is the exact speculative acceptance probability
// sum_x min(target_x, draft_x); upstream's 1-sum(relu(target-draft)) form
// agrees only when target sums to one.
func analyticAcceptance(target, draft []float64) float64 {
	sum := 0.0
	for x := range target {
		sum += min(target[x], draft[x])
	}
	return sum
}

func meanCalibrationError(acceptProbs, confidence []float64, rows []int) float64 {
	sum, verified := 0.0, 0
	for i := range acceptProbs {
		sum += float64(rows[i]) * math.Abs(acceptProbs[i]-confidence[i])
		verified += rows[i]
	}
	return sum / float64(verified)
}

func sampleIndex(rng *rand.Rand, probs []float64) int {
	u := rng.Float64()
	for x, p := range probs {
		if u < p {
			return x
		}
		u -= p
	}
	return len(probs) - 1
}

// fak-test:runtime fast est=10ms lane=default
func TestAnalyticAcceptanceCalibrationError(t *testing.T) {
	if got := analyticAcceptance([]float64{0.5, 0.5}, []float64{0.9, 0.1}); got != 0.6 {
		t.Fatalf("analytic acceptance = %v, want 0.6", got)
	}
	if got := meanCalibrationError([]float64{0.75, 0.25, 1, 0.5}, []float64{0.5, 0.75, 1, 0.375}, []int{1, 1, 1, 1}); got != 0.21875 {
		t.Fatalf("calibration error = %v, want 0.21875", got)
	}

	targets := [][]float64{{0.9, 0.1}, {0.5, 0.5}, {0.25, 0.75}, {1, 0}}
	drafts := [][]float64{{1, 0}, {0.9, 0.1}, {0.5, 0.5}, {0.5, 0.5}}
	alphas := make([]float64, len(targets))
	for i := range targets {
		alphas[i] = analyticAcceptance(targets[i], drafts[i])
	}
	for i, want := range []float64{0.9, 0.6, 0.75, 0.5} {
		if math.Abs(alphas[i]-want) > 1e-15 {
			t.Fatalf("position %d analytic acceptance = %v, want %v", i, alphas[i], want)
		}
	}

	rng := rand.New(rand.NewPCG(0x5eed, 0xdee9))
	var profile acceptanceProfile
	for range 40000 {
		accepted := 0
		for i := range targets {
			x := sampleIndex(rng, drafts[i])
			if rng.Float64() >= min(1, targets[i][x]/drafts[i][x]) {
				break
			}
			accepted++
		}
		profile.record(len(targets), accepted)
	}
	positions := profile.snapshot()
	conditional := make([]float64, len(positions))
	marginal := make([]float64, len(positions))
	verifyRows := make([]int, len(positions))
	for i, row := range positions {
		conditional[i], marginal[i], verifyRows[i] = *row.ConditionalRate, *row.Rate, row.Proposed
		if math.Abs(conditional[i]-alphas[i]) > 0.02 {
			t.Fatalf("position %d conditional rate = %v, want analytic %v", i, conditional[i], alphas[i])
		}
	}
	conditionalError := meanCalibrationError(alphas, conditional, verifyRows)
	marginalError := meanCalibrationError(alphas, marginal, verifyRows)
	if conditionalError > 0.02 {
		t.Fatalf("conditional calibration error = %v, want <= 0.02", conditionalError)
	}
	if math.Abs(marginalError-0.175) > 0.02 {
		t.Fatalf("marginal calibration error = %v, want 0.175", marginalError)
	}
}
