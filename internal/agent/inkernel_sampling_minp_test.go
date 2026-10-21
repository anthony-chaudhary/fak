package agent

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"slices"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func syntheticSamplingPlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-sampling", false, nil, false)
	p.batchDecode = false
	return p
}

var samplingPlannerMessages = []Message{{Role: RoleUser, Content: "hello there"}}

// fak-test:runtime fast est=150ms lane=default
func TestInKernelPlannerRefusesNonzeroMinP(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	maxTokens := 4
	for _, minP := range []float64{0.05, 1, -0.1, math.NaN()} {
		v := minP
		opts := []SampleOpt{WithMinP(&v), WithMaxTokens(maxTokens)}
		comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, opts...)
		if comp != nil || !errors.Is(err, model.ErrSamplingModeUnsupported) {
			t.Fatalf("min_p %v: Complete completion=%t err=%v; want typed sampling-mode refusal", minP, comp != nil, err)
		}
		for _, perToken := range []bool{false, true} {
			comp, err := p.CompleteStream(context.Background(), func(string) error {
				t.Fatalf("min_p %v refusal published a token", minP)
				return nil
			}, samplingPlannerMessages, nil, append(opts, WithPerTokenStream(perToken))...)
			if comp != nil || !errors.Is(err, model.ErrSamplingModeUnsupported) {
				t.Fatalf("min_p %v stream(%t) completion=%t err=%v; want typed sampling-mode refusal", minP, perToken, comp != nil, err)
			}
		}
	}
	zero := 0.0
	if _, err := p.Complete(context.Background(), samplingPlannerMessages, nil, WithMinP(&zero), WithMaxTokens(maxTokens)); err != nil {
		t.Fatalf("explicit min_p 0 is a no-op cutoff and must be admitted: %v", err)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestInKernelExplicitTopPZeroKeepsOnlyTheMostProbableToken(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	maxTokens := 12
	complete := func(opts ...SampleOpt) *Completion {
		t.Helper()
		comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, append(opts, WithMaxTokens(maxTokens))...)
		if err != nil {
			t.Fatal(err)
		}
		return comp
	}
	zeroTemp, hot, zeroTopP, oneTopP := 0.0, 1.5, 0.0, 1.0
	greedy := complete(WithTemperature(&zeroTemp))
	if greedy.Usage.CompletionTokens < 2 {
		t.Fatalf("greedy fixture produced %d tokens; the witness needs a multi-token turn", greedy.Usage.CompletionTokens)
	}
	sampledDiffers := false
	for seed := int64(1); seed <= 4; seed++ {
		s := seed
		got := complete(WithTemperature(&hot), WithTopP(&zeroTopP), WithSeed(&s))
		if got.Message.Content != greedy.Message.Content {
			t.Fatalf("seed %d: explicit top_p 0 drew %q, want the greedy turn %q", seed, got.Message.Content, greedy.Message.Content)
		}
		omitted := complete(WithTemperature(&hot), WithSeed(&s))
		full := complete(WithTemperature(&hot), WithTopP(&oneTopP), WithSeed(&s))
		if omitted.Message.Content != full.Message.Content {
			t.Fatalf("seed %d: omitted top_p drew %q, want the full-softmax draw %q", seed, omitted.Message.Content, full.Message.Content)
		}
		sampledDiffers = sampledDiffers || omitted.Message.Content != greedy.Message.Content
	}
	if !sampledDiffers {
		t.Fatal("full-softmax draws matched greedy for every seed; the fixture cannot tell top_p 0 apart from no nucleus")
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeInferenceReceiptAdmitsExplicitTopPZero(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	zero := 0.0
	receipt := func(opts ...SampleOpt) *Completion {
		t.Helper()
		comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, append(opts, WithNativeInferenceReceipt(true), WithMaxTokens(2))...)
		var unsupported *model.NativeInferenceReceiptUnsupportedError
		if errors.As(err, &unsupported) {
			t.Fatalf("explicit top_p 0 receipt refused as modified logits: %v", err)
		}
		if err != nil || comp == nil || comp.NativeInference == nil {
			t.Fatalf("receipt completion=%t receipt=%t err=%v; want a greedy receipt", comp != nil, comp != nil && comp.NativeInference != nil, err)
		}
		return comp
	}
	withTopPZero := receipt(WithTopP(&zero))
	omitted := receipt()
	if withTopPZero.Message.Content != omitted.Message.Content {
		t.Fatalf("explicit top_p 0 receipt drew %q, want the greedy receipt turn %q", withTopPZero.Message.Content, omitted.Message.Content)
	}
}

// referenceMinPSamplingDistribution transcribes the native min_p contract
// independently: softmax at temp, keep the tokens whose probability is at least
// minP times the largest, then top-k and top-p over that set in descending
// order (lower token index first on ties), renormalized.
func referenceMinPSamplingDistribution(logits []float32, temp, topP float64, topK int, minP float64) []float64 {
	maxL := math.Inf(-1)
	for _, x := range logits {
		maxL = math.Max(maxL, float64(x))
	}
	p := make([]float64, len(logits))
	var pmax float64
	for i, x := range logits {
		p[i] = math.Exp((float64(x) - maxL) / temp)
		pmax = math.Max(pmax, p[i])
	}
	order := make([]int, 0, len(p))
	for i := range p {
		if p[i] >= minP*pmax {
			order = append(order, i)
		}
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

func minPDrawnSupport(logits []float32, temp, topP float64, topK int, minP float64) []int {
	const coins = 4096
	var w samplerWorkspace
	seen := make(map[int]bool)
	for i := 0; i < coins; i++ {
		seen[sampleLogitsMinPCoreWS(logits, temp, topP, topK, minP, nil, (float64(i)+0.5)/coins, true, &w)] = true
	}
	support := make([]int, 0, len(seen))
	for tok := range seen {
		support = append(support, tok)
	}
	sort.Ints(support)
	return support
}

func referenceSupport(dist []float64) []int {
	var support []int
	for tok, p := range dist {
		if p > 0 {
			support = append(support, tok)
		}
	}
	return support
}

// fak-test:runtime fast est=1ms lane=default
func TestMinPTruncateKeepsTokensAtOrAboveTheCutoff(t *testing.T) {
	for _, tc := range []struct {
		name string
		minP float64
		want []float64
		sum  float64
	}{
		{"boundary_inclusive", 0.25, []float64{0.5, 1, 0.25, 0, 0.25, 0}, 2},
		{"max_only", 1, []float64{0, 1, 0, 0, 0, 0}, 1},
		{"above_one_keeps_max", 4, []float64{0, 1, 0, 0, 0, 0}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			probs := []float64{0.5, 1, 0.25, 0.1, 0.25, 0.05}
			sum := minPTruncate(probs, 2.15, tc.minP)
			if sum != tc.sum || !slices.Equal(probs, tc.want) {
				t.Fatalf("minPTruncate(%g) = %v sum %g, want %v sum %g", tc.minP, probs, sum, tc.want, tc.sum)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestSamplerMinPCutoffAppliesAfterTemperature(t *testing.T) {
	logits := []float32{0, -1, -2, -3}
	for _, tc := range []struct {
		temp float64
		want []int
	}{
		{1, []int{0, 1, 2}},
		{0.5, []int{0, 1}},
		{2, []int{0, 1, 2, 3}},
	} {
		got := minPDrawnSupport(logits, tc.temp, 0, 0, 0.1)
		if !slices.Equal(got, tc.want) {
			t.Fatalf("temp %g min_p 0.1 support = %v, want %v", tc.temp, got, tc.want)
		}
		if ref := referenceSupport(referenceMinPSamplingDistribution(logits, tc.temp, 0, 0, 0.1)); !slices.Equal(got, ref) {
			t.Fatalf("temp %g support %v disagrees with the reference %v", tc.temp, got, ref)
		}
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestSamplerMinPComposesBeforeTopKAndTopP(t *testing.T) {
	weights := []float64{1, 0.8, 0.6}
	for range 20 {
		weights = append(weights, 0.05)
	}
	logits := make([]float32, len(weights))
	for i, w := range weights {
		logits[i] = float32(math.Log(w))
	}
	for _, tc := range []struct {
		name string
		topP float64
		topK int
		minP float64
		want []int
	}{
		{"min_p_only", 0, 0, 0.1, []int{0, 1, 2}},
		{"top_p_without_min_p", 0.7, 0, 0, []int{0, 1, 2}},
		{"min_p_then_top_p", 0.7, 0, 0.1, []int{0, 1}},
		{"top_k_narrower", 0, 2, 0.1, []int{0, 1}},
		{"min_p_narrower", 0, 5, 0.1, []int{0, 1, 2}},
		{"min_p_then_top_k_then_top_p", 0.7, 5, 0.1, []int{0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := minPDrawnSupport(logits, 1, tc.topP, tc.topK, tc.minP)
			if !slices.Equal(got, tc.want) {
				t.Fatalf("support = %v, want %v", got, tc.want)
			}
			if ref := referenceSupport(referenceMinPSamplingDistribution(logits, 1, tc.topP, tc.topK, tc.minP)); !slices.Equal(got, ref) {
				t.Fatalf("support %v disagrees with the reference %v", got, ref)
			}
		})
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestSamplerInertMinPDrawsBitIdenticalTokens(t *testing.T) {
	logits := samplingWitnessLogits(4096, 999)
	for _, tc := range sampleOptionsCases() {
		t.Run(tc.name, func(t *testing.T) {
			var ref, ws samplerWorkspace
			for seed := int64(0); seed < 100; seed++ {
				want := sampleLogitsWS(logits, tc.temp, tc.topP, tc.topK, rand.New(rand.NewSource(seed)), &ref)
				for _, minP := range []float64{0, 1e-300} {
					got := sampleLogitsMinPCoreWS(logits, tc.temp, tc.topP, tc.topK, minP, rand.New(rand.NewSource(seed)), 0, false, &ws)
					if got != want {
						t.Fatalf("seed %d min_p %g drew %d, historical path drew %d", seed, minP, got, want)
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestSamplerMinPZeroAllocs(t *testing.T) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	var ws samplerWorkspace
	drain := rand.New(rand.NewSource(1))
	for _, tc := range []struct {
		name string
		topP float64
		topK int
	}{
		{"min_p", 0, 0},
		{"min_p_top_p", 0.9, 0},
		{"min_p_top_k_top_p", 0.9, 50},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_ = sampleLogitsMinPCoreWS(logits, 1, tc.topP, tc.topK, 0.01, drain, 0, false, &ws)
			allocs := testing.AllocsPerRun(10, func() {
				_ = sampleLogitsMinPCoreWS(logits, 1, tc.topP, tc.topK, 0.01, drain, 0, false, &ws)
			})
			if allocs > 0 {
				t.Fatalf("min_p sampling allocated %v per call after warm-up, want 0", allocs)
			}
		})
	}
}

// Adapted from DeepInfer randomized-PIT sampler test (Apache-2.0, deepseek-ai/DeepInfer tests/models/test_sampler.py at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no upstream source is copied.
// fak-test:runtime fast est=2s lane=default
func TestSamplerMinPRandomizedPITIsUniform(t *testing.T) {
	const (
		vocab = 2000
		draws = 20000
		alpha = 1e-5
	)
	critical := math.Sqrt(-math.Log(alpha/2) / (2 * draws))
	logitRNG := rand.New(rand.NewSource(53))
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(logitRNG.NormFloat64() * 2.5)
	}
	for _, tc := range []struct {
		name string
		temp float64
		topP float64
		topK int
		minP float64
	}{
		{"min_p", 1, 0, 0, 0.05},
		{"min_p_then_top_k_top_p", 0.7, 0.9, 40, 0.02},
		{"min_p_hot_then_top_p", 1.5, 0.95, 0, 0.1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := referenceMinPSamplingDistribution(logits, tc.temp, tc.topP, tc.topK, tc.minP)
			below := make([]float64, vocab)
			var cum float64
			for i, p := range want {
				below[i] = cum
				cum += p
			}
			var w samplerWorkspace
			draw := rand.New(rand.NewSource(59))
			jitter := rand.New(rand.NewSource(61))
			pits := make([]float64, draws)
			for n := range pits {
				tok := sampleLogitsMinPCoreWS(logits, tc.temp, tc.topP, tc.topK, tc.minP, draw, 0, false, &w)
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
