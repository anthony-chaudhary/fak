package agent

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestInkernelSamplingDistributionMatchesRealSampler(t *testing.T) {
	negInf := float32(math.Inf(-1))
	ordinary := []float32{negInf, -2.1, 0.35, 1.2, -0.8, 1.2, 0.1, -4}
	ties := make([]float32, 128)
	peaked := make([]float32, 128)
	for i := range peaked {
		peaked[i] = -20
	}
	peaked[5], peaked[60], peaked[100] = 2, 1, 0
	cases := []struct {
		name           string
		logits         []float32
		temp, topP     float64
		topK           int
		minP           float64
		bias           model.LogitBias
		freq, presence float64
		counts         []int32
	}{
		{name: "full", logits: ordinary, temp: 1},
		{name: "temperature", logits: ordinary, temp: 0.37},
		{name: "greedy-tie", logits: ordinary, topP: 0.01, topK: 1, minP: 2},
		{name: "negative-temperature", logits: ordinary, temp: -1},
		{name: "greedy-min-float-sentinel", logits: []float32{negInf, -math.MaxFloat32}},
		{name: "infinite-temperature", logits: []float32{2, 3, 4}, temp: math.Inf(1)},
		{name: "tiny-temperature", logits: ordinary, temp: math.SmallestNonzeroFloat64},
		{name: "topk-one", logits: ordinary, temp: 1, topK: 1},
		{name: "topk-small", logits: ordinary, temp: 1, topK: 4},
		{name: "topk-at-width-disabled", logits: ordinary, temp: 1, topK: len(ordinary)},
		{name: "topk-negative-disabled", logits: ordinary, temp: 1, topK: -1},
		{name: "topk-tie-64", logits: ties, temp: 1, topK: 64},
		{name: "topk-tie-heap", logits: ties, temp: 1, topK: 65},
		{name: "nucleus-full-sort-tie", logits: ties, temp: 1, topP: 0.3},
		{name: "nucleus-small-prefix", logits: peaked, temp: 1, topP: 0.9},
		{name: "nucleus-smallest", logits: ordinary, temp: 1, topP: math.SmallestNonzeroFloat64},
		{name: "nucleus-one-disabled", logits: ordinary, temp: 1, topP: 1},
		{name: "nucleus-nan-disabled", logits: ordinary, temp: 1, topP: math.NaN()},
		{name: "topk-then-nucleus", logits: ordinary, temp: 0.8, topP: 0.65, topK: 4},
		{name: "minp", logits: ordinary, temp: 0.7, minP: 0.2},
		{name: "minp-one", logits: ordinary, temp: 1, minP: 1},
		{name: "minp-above-one", logits: ordinary, temp: 1, minP: 2},
		{name: "minp-nan-disabled", logits: ordinary, temp: 1, minP: math.NaN()},
		{name: "minp-negative-disabled", logits: ordinary, temp: 1, minP: -1},
		{name: "minp-topk-nucleus", logits: ordinary, temp: 0.8, minP: 0.1, topK: 4, topP: 0.75},
		{name: "float32-subtraction-overflow", logits: []float32{-math.MaxFloat32, 0, math.MaxFloat32}, temp: 1},
		{name: "singleton", logits: []float32{0}, temp: 1, topP: 0.1, topK: 1, minP: 2},
		{name: "bias-clamps", logits: ordinary, temp: 1, bias: model.LogitBias{-1: 9, 1: 120, 3: -120, 99: 4}},
		{name: "bias-penalty-minp-topk-nucleus", logits: ordinary, temp: 0.8, topP: 0.8, topK: 4, minP: 0.05,
			bias: model.LogitBias{1: 1.7, 5: 0.25}, freq: 0.2, presence: 0.1, counts: []int32{0, 1, 0, 2, -1, 3, 0, 0, 4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			original := slices.Clone(tc.logits)
			var exported, sampled samplerWorkspace
			row, err := sampleLogitsWithPenaltyDistributionWS(tc.logits, tc.temp, tc.topP, tc.topK, tc.minP, tc.bias, tc.freq, tc.presence, tc.counts, &exported)
			if err != nil {
				t.Fatal(err)
			}
			if len(row.Weights) != len(tc.logits) || !(row.Mass > 0) {
				t.Fatalf("invalid exported row: %+v", row)
			}
			coins := []float64{0, math.SmallestNonzeroFloat64, 0.1, 0.37, 0.5, 0.9, math.Nextafter(1, 0)}
			// Probe both neighbors of each represented CDF boundary, not just
			// convenient midpoint coins. The production sampler is the oracle.
			var cumulative float64
			for _, weight := range row.Weights {
				cumulative += weight
				boundary := cumulative / row.Mass
				for _, coin := range []float64{math.Nextafter(boundary, 0), boundary, math.Nextafter(boundary, 1)} {
					if coin >= 0 && coin < 1 {
						coins = append(coins, coin)
					}
				}
			}
			for _, coin := range coins {
				var want int
				if tc.minP == 0 && len(tc.bias) == 0 && tc.freq == 0 && tc.presence == 0 {
					want = sampleLogitsCoinWS(tc.logits, tc.temp, tc.topP, tc.topK, coin, &sampled)
				} else {
					want = sampleLogitsWithPenaltyCoreWS(tc.logits, tc.temp, tc.topP, tc.topK, tc.minP, tc.bias, tc.freq, tc.presence, tc.counts, nil, coin, true, &sampled)
				}
				if got := sampleProbabilityRow(row.Weights, row.Mass, coin); got != want {
					t.Fatalf("coin %.17g: exported draw %d, real sampler %d", coin, got, want)
				}
				if tc.temp > 0 {
					if math.Float64bits(row.Mass) != math.Float64bits(sampled.mass) {
						t.Fatalf("mass changed: %.17g, real sampler %.17g", row.Mass, sampled.mass)
					}
					for i, weight := range row.Weights {
						if math.Float64bits(weight) != math.Float64bits(sampled.probs[i]) {
							t.Fatalf("token %d weight changed: %.17g, real sampler %.17g", i, weight, sampled.probs[i])
						}
					}
				}
			}
			if !slices.Equal(tc.logits, original) {
				t.Fatal("probability extraction mutated the input logits")
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkernelSamplingDistributionSupportAndPrecision(t *testing.T) {
	for _, tc := range []struct {
		name string
		topK int
		topP float64
		want []float64
	}{
		{"topk-stable-tie", 3, 0, []float64{1, 1, 1, 0}},
		{"nucleus-stable-tie", 0, 0.5, []float64{1, 1, 0, 0}},
		{"topk-then-nucleus", 3, 0.5, []float64{1, 1, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			row, err := sampleLogitsWithPenaltyDistributionWS([]float32{0, 0, 0, 0}, 1, tc.topP, tc.topK, 0, nil, 0, 0, nil, &samplerWorkspace{})
			if err != nil || !slices.Equal(row.Weights, tc.want) {
				t.Fatalf("row=%+v err=%v, want %v", row, err, tc.want)
			}
			if got := sampleProbabilityRow(row.Weights, row.Mass, 1/row.Mass); got != 1 {
				t.Fatalf("strict CDF boundary selected %d, want 1", got)
			}
		})
	}
	row, err := sampleLogitsWithPenaltyDistributionWS([]float32{-0.7, 0.2, 0}, 0.9, 0, 0, 0, nil, 0, 0, nil, &samplerWorkspace{})
	if err != nil {
		t.Fatal(err)
	}
	if row.Mass <= 1 || row.Weights[1] != 1 {
		t.Fatalf("row was normalized instead of preserving weights and mass: %+v", row)
	}
	if row.Weights[0] == float64(float32(row.Weights[0])) {
		t.Fatalf("fixture no longer distinguishes float64 weights from float32: %.17g", row.Weights[0])
	}
	if row.Weights[0] == math.Exp(float64(float32(math.Log(row.Weights[0])))) {
		t.Fatal("fixture no longer distinguishes the forbidden float32 log/exp round trip")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkernelSamplingDistributionRefusesDegenerateRows(t *testing.T) {
	negInf := float32(math.Inf(-1))
	for _, tc := range []struct {
		name   string
		logits []float32
		temp   float64
		err    error
	}{
		{"empty", nil, 1, ErrNonFiniteLogits},
		{"empty-greedy", nil, 0, ErrNonFiniteLogits},
		{"all-masked", []float32{negInf, negInf}, 1, ErrNonFiniteLogits},
		{"all-masked-greedy", []float32{negInf, negInf}, 0, ErrNonFiniteLogits},
		{"nan", []float32{0, float32(math.NaN())}, 1, ErrNonFiniteLogits},
		{"positive-inf", []float32{0, float32(math.Inf(1))}, 1, ErrNonFiniteLogits},
		{"nan-temperature", []float32{0, 1}, math.NaN(), model.ErrSpecRejectionProbability},
		{"infinite-temperature-with-mask", []float32{0, negInf}, math.Inf(1), model.ErrSpecRejectionProbability},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var w samplerWorkspace
			// A rejected row must not leak a previously valid workspace row.
			if _, err := sampleLogitsWithPenaltyDistributionWS([]float32{0, 1}, 1, 0, 0, 0, nil, 0, 0, nil, &w); err != nil {
				t.Fatal(err)
			}
			row, err := sampleLogitsWithPenaltyDistributionWS(tc.logits, tc.temp, 0, 0, 0, nil, 0, 0, nil, &w)
			if !errors.Is(err, tc.err) || len(row.Weights) != 0 || row.Mass != 0 {
				t.Fatalf("row=%+v err=%v, want empty row and %v", row, err, tc.err)
			}
			// Extraction adds refusal, not a change to the ordinary sampler's
			// historical empty-row sentinel.
			if len(tc.logits) == 0 {
				want := -1
				if tc.temp <= 0 {
					want = 0
				}
				if got := sampleLogitsCoinWS(tc.logits, tc.temp, 0, 0, 0.5, &w); got != want {
					t.Fatalf("legacy empty-row sentinel changed: got %d want %d", got, want)
				}
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkernelSamplingDistributionPreservesRNG(t *testing.T) {
	logits := []float32{-0.4, 1.2, 0.7, 0.2, -1}
	bias := model.LogitBias{0: 0.6, 4: 1.1}
	counts := []int32{0, 2, 0, 1, 0}
	legacyRNG, requestRNG := rand.New(rand.NewSource(9137)), rand.New(rand.NewSource(9137))
	var w samplerWorkspace
	for i := 0; i < 32; i++ {
		row, err := sampleLogitsWithPenaltyDistributionWS(logits, 0.8, 0.9, 4, 0.1, bias, 0.15, 0.25, counts, &w)
		if err != nil {
			t.Fatal(err)
		}
		want := sampleLogitsWithPenaltyCoreWS(logits, 0.8, 0.9, 4, 0.1, bias, 0.15, 0.25, counts, legacyRNG, 0, false, &samplerWorkspace{})
		if got := sampleProbabilityRow(row.Weights, row.Mass, requestRNG.Float64()); got != want {
			t.Fatalf("draw %d: got %d want %d", i, got, want)
		}
	}
	if got, want := requestRNG.Uint64(), legacyRNG.Uint64(); got != want {
		t.Fatalf("request RNG advanced: got %#x want %#x", got, want)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkernelSamplingDistributionWorkspaceLifetime(t *testing.T) {
	var w samplerWorkspace
	first, err := sampleLogitsWithPenaltyDistributionWS([]float32{2, 0, 1}, 1, 0, 0, 0, nil, 0, 0, nil, &w)
	if err != nil {
		t.Fatal(err)
	}
	owned := slices.Clone(first.Weights)
	second, err := sampleLogitsWithPenaltyDistributionWS([]float32{0, 2, 1}, 0, 0, 0, 0, nil, 0, 0, nil, &w)
	if err != nil {
		t.Fatal(err)
	}
	if &first.Weights[0] != &second.Weights[0] || !slices.Equal(first.Weights, []float64{0, 1, 0}) || owned[0] != 1 {
		t.Fatalf("workspace lifetime contract changed: first=%v second=%v owned=%v", first.Weights, second.Weights, owned)
	}
	if first.Mass <= 1 || second.Mass != 1 {
		t.Fatalf("mass must be a value, not aliased scratch: first=%g second=%g", first.Mass, second.Mass)
	}
	logits := []float32{2, 0, 1}
	_, _ = sampleLogitsWithPenaltyDistributionWS(logits, 0.8, 0.9, 2, 0.1, nil, 0, 0, nil, &w)
	if allocs := testing.AllocsPerRun(100, func() {
		_, _ = sampleLogitsWithPenaltyDistributionWS(logits, 0.8, 0.9, 2, 0.1, nil, 0, 0, nil, &w)
	}); allocs != 0 {
		t.Fatalf("warmed extraction allocated %g times", allocs)
	}
}
