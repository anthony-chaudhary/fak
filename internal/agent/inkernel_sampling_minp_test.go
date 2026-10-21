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

// fak-test:runtime fast est=350ms lane=default
func TestInKernelPlannerRefusesInvalidMinP(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	maxTokens := 4
	for _, minP := range []float64{-0.1, 1.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		v := minP
		opts := []SampleOpt{WithMinP(&v), WithMaxTokens(maxTokens)}
		comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, opts...)
		if comp != nil || !errors.Is(err, ErrNativeMinPUnsupported) || !errors.Is(err, model.ErrSamplingModeUnsupported) {
			t.Fatalf("min_p %v: Complete completion=%t err=%v; want typed sampling-mode refusal", minP, comp != nil, err)
		}
		for _, perToken := range []bool{false, true} {
			comp, err := p.CompleteStream(context.Background(), func(string) error {
				t.Fatalf("min_p %v refusal published a token", minP)
				return nil
			}, samplingPlannerMessages, nil, append(opts, WithPerTokenStream(perToken))...)
			if comp != nil || !errors.Is(err, ErrNativeMinPUnsupported) {
				t.Fatalf("min_p %v stream(%t) completion=%t err=%v; want typed sampling-mode refusal", minP, perToken, comp != nil, err)
			}
		}
	}
	hot := 1.0
	for _, minP := range []float64{0, 0.05, 1} {
		v := minP
		opts := []SampleOpt{WithMinP(&v), WithTemperature(&hot), WithMaxTokens(maxTokens)}
		if _, err := p.Complete(context.Background(), samplingPlannerMessages, nil, opts...); err != nil {
			t.Fatalf("min_p %v is in [0, 1] and must be admitted: %v", minP, err)
		}
		if _, err := p.CompleteStream(context.Background(), func(string) error { return nil }, samplingPlannerMessages, nil, append(opts, WithPerTokenStream(true))...); err != nil {
			t.Fatalf("min_p %v per-token stream must be admitted: %v", minP, err)
		}
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

func minPCompleteTokenIDs(t *testing.T, p *InKernelPlanner, opts ...SampleOpt) []int {
	t.Helper()
	comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, append(opts, WithDecodeTrace(true), WithNativeDecodeTokenIDs(true))...)
	if err != nil {
		t.Fatal(err)
	}
	if comp.NativeDecodeTokenIDs == nil {
		t.Fatal("completion carries no native decode token IDs")
	}
	return comp.NativeDecodeTokenIDs.TokenIDs
}

// minPReplayWeights replays drawn on a fresh session over the same prompt and
// returns each token's temperature-scaled weight relative to that step's most
// probable token, computed exactly as the sampler computes it.
func minPReplayWeights(t *testing.T, p *InKernelPlanner) func(drawn []int, temp float64) []float64 {
	t.Helper()
	prepared, err := p.preparePrompt(context.Background(), samplingPlannerMessages, nil, applySampleOpts())
	if err != nil {
		t.Fatal(err)
	}
	base := p.m.NewSession()
	t.Cleanup(base.Close)
	p.configureNativeSession(base)
	prefill := slices.Clone(base.Prefill(prepared.ids))
	return func(drawn []int, temp float64) []float64 {
		s := p.sessionFromPrefixClone(base.Cache.Clone())
		defer s.Close()
		p.configureNativeSession(s)
		logits := prefill
		weights := make([]float64, len(drawn))
		for i, tok := range drawn {
			maxL := float32(-math.MaxFloat32)
			for _, x := range logits {
				maxL = max(maxL, x)
			}
			weights[i] = math.Exp(float64(logits[tok]-maxL) / temp)
			logits = s.Step(tok)
		}
		return weights
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestInKernelMinPOneKeepsOnlyTheMostProbableToken(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	const tokens = 12
	zeroTemp, hot, one := 0.0, 1.5, 1.0
	greedy := minPCompleteTokenIDs(t, p, WithTemperature(&zeroTemp), WithMaxTokens(tokens))
	if len(greedy) < 2 {
		t.Fatalf("greedy fixture produced %d tokens; the witness needs a multi-token turn", len(greedy))
	}
	sampledDiffers := false
	for seed := int64(1); seed <= 4; seed++ {
		s := seed
		got := minPCompleteTokenIDs(t, p, WithTemperature(&hot), WithMinP(&one), WithSeed(&s), WithMaxTokens(tokens))
		if !slices.Equal(got, greedy) {
			t.Fatalf("seed %d: min_p 1 drew %v, want the greedy turn %v", seed, got, greedy)
		}
		unseeded := minPCompleteTokenIDs(t, p, WithTemperature(&hot), WithMinP(&one), WithMaxTokens(tokens))
		if !slices.Equal(unseeded, greedy) {
			t.Fatalf("seedless min_p 1 drew %v, want the greedy turn %v", unseeded, greedy)
		}
		sampledDiffers = sampledDiffers || !slices.Equal(minPCompleteTokenIDs(t, p, WithTemperature(&hot), WithSeed(&s), WithMaxTokens(tokens)), greedy)
	}
	if !sampledDiffers {
		t.Fatal("draws without min_p matched greedy for every seed; the fixture cannot tell min_p 1 apart from no cutoff")
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestInKernelMinPNeverSamplesBelowTheCutoff(t *testing.T) {
	t.Setenv("FAK_INKERNEL_SEED", "977")
	p := syntheticSamplingPlanner(t)
	const tokens = 16
	zeroTemp, hot, minP := 0.0, 1.5, 0.2
	replay := minPReplayWeights(t, p)
	greedy := minPCompleteTokenIDs(t, p, WithTemperature(&zeroTemp), WithMaxTokens(tokens))
	for i, w := range replay(greedy, hot) {
		if w != 1 {
			t.Fatalf("replay disagrees with the served logits: greedy token %d has relative weight %g, want 1", i, w)
		}
	}
	belowWithout, keptNonGreedy := 0, false
	for seed := int64(1); seed <= 8; seed++ {
		s := seed
		for _, opts := range [][]SampleOpt{
			{WithTemperature(&hot), WithMinP(&minP), WithSeed(&s), WithMaxTokens(tokens)},
			{WithTemperature(&hot), WithMinP(&minP), WithMaxTokens(tokens)},
		} {
			drawn := minPCompleteTokenIDs(t, p, opts...)
			for i, w := range replay(drawn, hot) {
				if w < minP {
					t.Fatalf("seed %d token %d (id %d) has relative weight %g below min_p %g", seed, i, drawn[i], w, minP)
				}
				keptNonGreedy = keptNonGreedy || w < 1
			}
		}
		drawn := minPCompleteTokenIDs(t, p, WithTemperature(&hot), WithSeed(&s), WithMaxTokens(tokens))
		for _, w := range replay(drawn, hot) {
			if w < minP {
				belowWithout++
			}
		}
	}
	if belowWithout == 0 {
		t.Fatal("draws without min_p never fell below the cutoff; the fixture cannot witness the cutoff")
	}
	if !keptNonGreedy {
		t.Fatal("every min_p draw was the most probable token; the fixture cannot tell min_p apart from greedy")
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestInKernelInertMinPDrawsBitIdenticalTurns(t *testing.T) {
	t.Setenv("FAK_INKERNEL_SEED", "4242")
	omitted, explicit := syntheticSamplingPlanner(t), syntheticSamplingPlanner(t)
	const tokens = 12
	zero, hot, topP := 0.0, 1.5, 0.9
	topK := 40
	for _, base := range [][]SampleOpt{
		{WithTemperature(&hot), WithMaxTokens(tokens)},
		{WithTemperature(&hot), WithTopP(&topP), WithTopK(&topK), WithMaxTokens(tokens)},
	} {
		for seed := int64(1); seed <= 3; seed++ {
			s := seed
			keyed := append(slices.Clone(base), WithSeed(&s))
			want := minPCompleteTokenIDs(t, omitted, keyed...)
			if got := minPCompleteTokenIDs(t, explicit, append(keyed, WithMinP(&zero))...); !slices.Equal(got, want) {
				t.Fatalf("seed %d: min_p 0 drew %v, omitted min_p drew %v", seed, got, want)
			}
		}
		want := minPCompleteTokenIDs(t, omitted, base...)
		if got := minPCompleteTokenIDs(t, explicit, append(slices.Clone(base), WithMinP(&zero))...); !slices.Equal(got, want) {
			t.Fatalf("seedless: min_p 0 drew %v, omitted min_p drew %v", got, want)
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestNativeInferenceReceiptRefusesNonzeroMinP(t *testing.T) {
	p := syntheticSamplingPlanner(t)
	zero, cut := 0.0, 0.05
	comp, err := p.Complete(context.Background(), samplingPlannerMessages, nil, WithMinP(&cut), WithNativeInferenceReceipt(true), WithMaxTokens(2))
	var unsupported *model.NativeInferenceReceiptUnsupportedError
	if comp != nil || !errors.As(err, &unsupported) {
		t.Fatalf("receipt with min_p %g: completion=%t err=%v; want a receipt refusal", cut, comp != nil, err)
	}
	comp, err = p.Complete(context.Background(), samplingPlannerMessages, nil, WithMinP(&zero), WithNativeInferenceReceipt(true), WithMaxTokens(2))
	if err != nil || comp == nil || comp.NativeInference == nil {
		t.Fatalf("receipt with min_p 0: completion=%t err=%v; want a greedy receipt", comp != nil, err)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestInKernelGreedySpeculativeRouteAdmitsInertMinP(t *testing.T) {
	plain := syntheticSamplingPlanner(t)
	spec := syntheticSamplingPlanner(t)
	probe := &speculativeRequestProbe{err: errors.New("draft unavailable")}
	spec.EnableSpeculativeDecoding(probe, 4)
	const tokens = 8
	zeroTemp, cut := 0.0, 0.5
	want := minPCompleteTokenIDs(t, plain, WithTemperature(&zeroTemp), WithMaxTokens(tokens))
	got := minPCompleteTokenIDs(t, spec, WithTemperature(&zeroTemp), WithMinP(&cut), WithMaxTokens(tokens))
	if probe.calls == 0 {
		t.Fatal("greedy min_p request did not reach the speculative route")
	}
	if !slices.Equal(got, want) {
		t.Fatalf("greedy speculative min_p turn %v, want the greedy turn %v", got, want)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestSpeculativeSerialFallbackHonorsMinP(t *testing.T) {
	t.Setenv("FAK_INKERNEL_SEED", "313")
	p := syntheticSamplingPlanner(t)
	p.EnableSpeculativeDecoding(&speculativeRequestProbe{err: errors.New("draft unavailable")}, 4)
	prepared, err := p.preparePrompt(context.Background(), samplingPlannerMessages, nil, applySampleOpts())
	if err != nil {
		t.Fatal(err)
	}
	const tokens = 10
	run := func(ctx context.Context, temp float64) []int {
		t.Helper()
		var out []int
		if _, err := p.generateReusedSpeculative(ctx, prepared.ids, tokens, temp, 0, 0, nil, 0, 0, map[int]bool{}, func(tok int) bool {
			out = append(out, tok)
			return false
		}); err != nil {
			t.Fatal(err)
		}
		return out
	}
	greedy := run(context.Background(), 0)
	one := 1.0
	sampledDiffers := false
	for range 4 {
		if got := run(withNativeMinP(context.Background(), &one), 1.5); !slices.Equal(got, greedy) {
			t.Fatalf("min_p 1 serial fallback drew %v, want the greedy turn %v", got, greedy)
		}
		sampledDiffers = sampledDiffers || !slices.Equal(run(context.Background(), 1.5), greedy)
	}
	if !sampledDiffers {
		t.Fatal("serial fallback draws without min_p matched greedy every time; the fixture cannot witness min_p")
	}
}

func minPLaneSupport(ln *decodeLane, draws int) []int {
	ln.maxNew = draws + 1
	history := slices.Clone(ln.counts)
	seen := map[int]bool{}
	for range draws {
		// decodeOne counts each draw; replaying one fixed history keeps the penalized kept set fixed.
		copy(ln.counts, history)
		next, advance := ln.decodeOne(context.Background())
		if !advance {
			break
		}
		seen[next] = true
	}
	support := make([]int, 0, len(seen))
	for tok := range seen {
		support = append(support, tok)
	}
	sort.Ints(support)
	return support
}

// fak-test:runtime fast est=30ms lane=default
func TestDecodeLaneMinPDrawsExactlyTheKeptSet(t *testing.T) {
	weights := []float64{1, 0.8, 0.6, 0.3, 0.2, 0.05, 0.05, 0.01}
	logits := make([]float32, len(weights))
	for i, w := range weights {
		logits[i] = float32(math.Log(w))
	}
	type penalty struct {
		freq, pres float64
		count0     int32
	}
	presence := penalty{pres: math.Log(4), count0: 1}
	frequency := penalty{freq: math.Log(2), count0: 2}
	for _, tc := range []struct {
		name    string
		keyed   bool
		topP    float64
		topK    int
		bias    model.LogitBias
		penalty penalty
		minP    float64
		want    []int
	}{
		{"rng_no_min_p", false, 0, 0, nil, penalty{}, 0, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"rng_min_p", false, 0, 0, nil, penalty{}, 0.25, []int{0, 1, 2, 3}},
		{"keyed_min_p", true, 0, 0, nil, penalty{}, 0.25, []int{0, 1, 2, 3}},
		{"bias_applies_before_min_p", true, 0, 0, model.LogitBias{7: math.Log(50)}, penalty{}, 0.25, []int{0, 1, 2, 3, 7}},
		{"min_p_then_top_k", false, 0, 2, nil, penalty{}, 0.25, []int{0, 1}},
		{"top_p_alone", true, 0.6, 0, nil, penalty{}, 0, []int{0, 1, 2}},
		{"min_p_then_top_p", true, 0.6, 0, nil, penalty{}, 0.25, []int{0, 1}},
		{"min_p_without_penalty", false, 0, 0, nil, penalty{}, 0.5, []int{0, 1, 2}},
		{"presence_penalty_alone", false, 0, 0, nil, presence, 0, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"rng_presence_penalty_before_min_p", false, 0, 0, nil, presence, 0.5, []int{1, 2}},
		{"keyed_presence_penalty_before_min_p", true, 0, 0, nil, presence, 0.5, []int{1, 2}},
		{"rng_frequency_penalty_before_min_p", false, 0, 0, nil, frequency, 0.5, []int{1, 2}},
		{"keyed_frequency_penalty_before_min_p", true, 0, 0, nil, frequency, 0.5, []int{1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ln := &decodeLane{
				logits:      logits,
				rng:         rand.New(rand.NewSource(71)),
				temp:        1,
				topP:        tc.topP,
				topK:        tc.topK,
				minP:        tc.minP,
				logitBias:   tc.bias,
				freqPenalty: tc.penalty.freq,
				presPenalty: tc.penalty.pres,
				sampleKey:   0x5eed,
				sampleKeyed: tc.keyed,
			}
			if tc.penalty != (penalty{}) {
				ln.counts = make([]int32, len(logits))
				ln.counts[0] = tc.penalty.count0
			}
			if got := minPLaneSupport(ln, 4000); !slices.Equal(got, tc.want) {
				t.Fatalf("support = %v, want %v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDecodeLaneGreedyMinPKeepsDeviceArgmax(t *testing.T) {
	logits := []float32{0.1, 2, -1, 1.9}
	ln := &decodeLane{logits: logits, temp: 0, minP: 0.5, rng: rand.New(rand.NewSource(1))}
	if !ln.deviceGreedy() {
		t.Fatal("min_p cannot change the argmax, so a greedy min_p lane must stay device-greedy")
	}
	if got := minPLaneSupport(ln, 64); !slices.Equal(got, []int{1}) {
		t.Fatalf("greedy min_p lane drew %v, want the argmax [1]", got)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestSamplerMinPPenaltyCoreZeroAllocs(t *testing.T) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 4321)
	bias := model.LogitBias{3: 1.5, 11: -2}
	counts := make([]int32, samplingWitnessVocab)
	counts[5], counts[9] = 2, 1
	var ws samplerWorkspace
	drain := rand.New(rand.NewSource(3))
	_ = sampleLogitsWithPenaltyCoreWS(logits, 1, 0.9, 50, 0.01, bias, 0.2, 0.1, counts, drain, 0, false, &ws)
	allocs := testing.AllocsPerRun(10, func() {
		_ = sampleLogitsWithPenaltyCoreWS(logits, 1, 0.9, 50, 0.01, bias, 0.2, 0.1, counts, drain, 0, false, &ws)
		_ = sampleLogitsWithPenaltyCoreWS(logits, 1, 0.9, 50, 0.01, bias, 0.2, 0.1, counts, nil, 0.37, true, &ws)
	})
	if allocs > 0 {
		t.Fatalf("min_p penalty sampling allocated %v per call after warm-up, want 0", allocs)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNativeMinPContextOverridesInheritedCutoff(t *testing.T) {
	base := context.Background()
	if got := nativeMinPFromContext(base); got != 0 {
		t.Fatalf("empty context min_p = %g, want 0", got)
	}
	if got := withNativeMinP(base, nil); got != base {
		t.Fatal("omitted min_p over an empty context replaced the request context")
	}
	cut := 0.3
	outer := withNativeMinP(base, &cut)
	if got := nativeMinPFromContext(outer); got != cut {
		t.Fatalf("min_p context = %g, want %g", got, cut)
	}
	if got := nativeMinPFromContext(withNativeMinP(outer, nil)); got != 0 {
		t.Fatalf("an inner request without min_p inherited %g, want 0", got)
	}
}
