package agent

// sampling_test.go proves the per-request sampling seam (#62): a SampleOpt passed
// to HTTPPlanner.Complete reaches the upstream provider wire, and an omitted option
// preserves the pre-seam default (max_tokens 1024) byte-for-byte. The test captures
// the exact JSON body the planner POSTs and asserts on the serialized fields, so it
// witnesses the whole resolve→adapter→wire path, not just the option fold.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// captureUpstream is an OpenAI-compatible stub that records the request body and
// returns a minimal valid completion. The captured body is what the assertions read.
func captureUpstream(t *testing.T, into *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("upstream got non-JSON body: %v (%s)", err, raw)
		}
		*into = body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
}

func TestHTTPPlannerHonorsPerRequestMaxTokens(t *testing.T) {
	var body map[string]any
	ts := captureUpstream(t, &body)
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "gpt-test", "")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	// A per-request max_tokens replaces the planner's fixed 1024 ceiling.
	if _, err := planner.Complete(context.Background(), msgs, nil, WithMaxTokens(4096)); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if got := jsonInt(body["max_tokens"]); got != 4096 {
		t.Fatalf("max_tokens on the wire = %d, want 4096 (the per-request override)", got)
	}
}

func TestHTTPPlannerCapsProviderMaxTokens(t *testing.T) {
	t.Setenv("FAK_PROVIDER_MAX_TOKENS", "8192")
	var body map[string]any
	ts := captureUpstream(t, &body)
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "gpt-test", "")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	if _, err := planner.Complete(context.Background(), msgs, nil, WithMaxTokens(32768)); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if got := jsonInt(body["max_tokens"]); got != 8192 {
		t.Fatalf("max_tokens on the wire = %d, want provider cap 8192", got)
	}
}

func TestHTTPPlannerDefaultsMaxTokensWhenOmitted(t *testing.T) {
	var body map[string]any
	ts := captureUpstream(t, &body)
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "gpt-test", "")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	// No opt (and a 0 max_tokens opt, which is a documented no-op) => the planner's
	// configured 1024 default, identical to the pre-seam behavior.
	if _, err := planner.Complete(context.Background(), msgs, nil, WithMaxTokens(0)); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if got := jsonInt(body["max_tokens"]); got != 1024 {
		t.Fatalf("max_tokens on the wire = %d, want 1024 (the planner default)", got)
	}
}

func TestHTTPPlannerHonorsPerRequestSamplingParams(t *testing.T) {
	var body map[string]any
	ts := captureUpstream(t, &body)
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "gpt-test", "")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	temp, topP := 0.7, 0.9
	if _, err := planner.Complete(context.Background(), msgs, nil,
		WithTemperature(&temp), WithTopP(&topP), WithStop([]string{"\n\n", "STOP"})); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if got, ok := body["temperature"].(float64); !ok || got != 0.7 {
		t.Fatalf("temperature on the wire = %v, want 0.7", body["temperature"])
	}
	if got, ok := body["top_p"].(float64); !ok || got != 0.9 {
		t.Fatalf("top_p on the wire = %v, want 0.9", body["top_p"])
	}
	stop, ok := body["stop"].([]any)
	if !ok || len(stop) != 2 || stop[0] != "\n\n" || stop[1] != "STOP" {
		t.Fatalf("stop on the wire = %v, want [\\n\\n STOP]", body["stop"])
	}
}

func TestHTTPPlannerOmitsUnsetSamplingParams(t *testing.T) {
	var body map[string]any
	ts := captureUpstream(t, &body)
	defer ts.Close()

	planner, err := NewProviderHTTPPlanner("openai", ts.URL, "gpt-test", "")
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{{Role: RoleUser, Content: "hi"}}

	// No top_p / stop opts => those keys are absent from the wire (omitempty), so an
	// existing integration's serialized body is unchanged.
	if _, err := planner.Complete(context.Background(), msgs, nil); err != nil {
		t.Fatalf("complete: %v", err)
	}
	if _, present := body["top_p"]; present {
		t.Fatalf("top_p must be omitted when unset, got %v", body["top_p"])
	}
	if _, present := body["stop"]; present {
		t.Fatalf("stop must be omitted when unset, got %v", body["stop"])
	}
}

func TestSampleLogitsWithBias(t *testing.T) {
	logits := []float32{0.1, 0.9, 0.3}
	orig := append([]float32(nil), logits...)

	if got, want := sampleLogitsWithBias(logits, 0, 0, 0, nil, nil), sampleLogits(logits, 0, 0, 0, nil); got != want {
		t.Fatalf("nil logit_bias changed selection: got %d want %d", got, want)
	}
	if got := sampleLogitsWithBias(logits, 0, 0, 0, model.LogitBias{1: -100}, nil); got != 2 {
		t.Fatalf("logit_bias -100 on winner selected %d, want runner-up 2", got)
	}
	if got := sampleLogitsWithBias(logits, 0, 0, 0, model.LogitBias{0: 1000}, nil); got != 0 {
		t.Fatalf("clamped positive logit_bias selected %d, want forced token 0", got)
	}
	for i := range logits {
		if logits[i] != orig[i] {
			t.Fatalf("sampleLogitsWithBias mutated logits[%d]: got %v want %v", i, logits[i], orig[i])
		}
	}
}

// TestSampleLogitsWithPenaltyZeroIsNoOp pins #1705: a zero (or all-nil) frequency/
// presence penalty must reproduce the EXACT pre-#1705 sampleLogitsWithBias output,
// byte-for-byte, whether or not a generation-count history is supplied. This is the
// no-regression witness for the sampler seam every existing caller (0, 0 penalties)
// falls through.
func TestSampleLogitsWithPenaltyZeroIsNoOp(t *testing.T) {
	logits := []float32{0.1, 0.9, 0.3}
	counts := []int32{5, 5, 5} // even a nonzero count history must not matter at penalty 0

	want := sampleLogitsWithBias(logits, 0, 0, 0, nil, nil)
	if got := sampleLogitsWithPenalty(logits, 0, 0, 0, nil, 0, 0, nil, nil); got != want {
		t.Fatalf("zero penalty, nil counts changed selection: got %d want %d", got, want)
	}
	if got := sampleLogitsWithPenalty(logits, 0, 0, 0, nil, 0, 0, counts, nil); got != want {
		t.Fatalf("zero penalty with a nonzero count history changed selection: got %d want %d", got, want)
	}
	// logit_bias must still apply identically alongside a zero penalty.
	wantBias := sampleLogitsWithBias(logits, 0, 0, 0, model.LogitBias{1: -100}, nil)
	if got := sampleLogitsWithPenalty(logits, 0, 0, 0, model.LogitBias{1: -100}, 0, 0, counts, nil); got != wantBias {
		t.Fatalf("zero penalty changed the logit_bias result: got %d want %d", got, wantBias)
	}
}

// TestSampleLogitsWithPenaltyFrequencySuppressesRepeatedToken pins the core #1705
// failure mode: with temp<=0 (pure argmax) a token that has already been generated
// many times this turn must lose its argmax slot to a nonzero frequency_penalty,
// exactly the effect a client needs to break a non-terminating repetition loop that
// temperature alone cannot break.
func TestSampleLogitsWithPenaltyFrequencySuppressesRepeatedToken(t *testing.T) {
	logits := []float32{1.0, 5.0, 2.0} // token 1 is the unpenalized argmax winner
	orig := append([]float32(nil), logits...)

	if got := sampleLogitsWithPenalty(logits, 0, 0, 0, nil, 0, 0, nil, nil); got != 1 {
		t.Fatalf("sanity: unpenalized argmax = %d, want 1", got)
	}

	// Token 1 already generated 10 times this turn; frequency_penalty=1.0 knocks its
	// effective logit from 5.0 down to 5.0 - 1.0*10 = -5.0, well below token 2's 2.0.
	counts := []int32{0, 10, 0}
	got := sampleLogitsWithPenalty(logits, 0, 0, 0, nil, 1.0, 0, counts, nil)
	if got != 2 {
		t.Fatalf("frequency_penalty did not suppress the repeated token: got %d, want 2 (runner-up)", got)
	}
	// The input logits slice must never be mutated (sampleLogitsWithBias's own
	// contract, preserved here).
	for i := range logits {
		if logits[i] != orig[i] {
			t.Fatalf("sampleLogitsWithPenalty mutated logits[%d]: got %v want %v", i, logits[i], orig[i])
		}
	}
}

// TestSampleLogitsWithPenaltyPresenceIsBinary pins that presence_penalty applies
// ONCE for any token seen at all (count>0), independent of how many times — unlike
// frequency_penalty, which scales with count.
func TestSampleLogitsWithPenaltyPresenceIsBinary(t *testing.T) {
	logits := []float32{1.0, 5.0, 4.9} // token 1 wins by 0.1 over token 2
	// Token 1 seen once, token 2 never seen. presence_penalty=0.5 should drop token
	// 1's effective logit to 4.5, below token 2's untouched 4.9.
	counts := []int32{0, 1, 0}
	if got := sampleLogitsWithPenalty(logits, 0, 0, 0, nil, 0, 0.5, counts, nil); got != 2 {
		t.Fatalf("presence_penalty did not flip the winner: got %d, want 2", got)
	}
}

// jsonInt reads a JSON number (decoded as float64) as an int.
func jsonInt(v any) int {
	f, _ := v.(float64)
	return int(f)
}

// referenceFullSortTopKTruncate is the golden reference implementing top-k truncation
// via full vocabulary sorting and a map-based mask.
func referenceFullSortTopKTruncate(probs []float64, sum float64, k int) float64 {
	order := descProbOrder(probs, func(i, j int) bool {
		if probs[i] != probs[j] {
			return probs[i] > probs[j]
		}
		return i < j
	})
	kept := make(map[int]bool, k)
	for rank := 0; rank < k; rank++ {
		kept[order[rank]] = true
	}
	return maskKept(probs, kept)
}

func TestTopKTruncateEquivalence(t *testing.T) {
	vocabs := []int{32768, 248000}
	kValues := []int{1, 2, 8, 32, 50, 64, 100}

	for _, vocabSize := range vocabs {
		// 1. Realistic softmax-like logit distribution
		rng := rand.New(rand.NewSource(42))
		logits := make([]float32, vocabSize)
		for i := range logits {
			logits[i] = float32(rng.NormFloat64() * 3.0)
		}
		maxL := float32(-math.MaxFloat32)
		for _, x := range logits {
			if x > maxL {
				maxL = x
			}
		}
		var baseSum float64
		baseProbs := make([]float64, vocabSize)
		for i, x := range logits {
			p := math.Exp(float64(x - maxL))
			baseProbs[i] = p
			baseSum += p
		}

		for _, k := range kValues {
			t.Run(fmt.Sprintf("softmax_vocab_%d_k_%d", vocabSize, k), func(t *testing.T) {
				fastProbs := append([]float64(nil), baseProbs...)
				refProbs := append([]float64(nil), baseProbs...)

				fastSum := topKTruncate(fastProbs, baseSum, k)
				refSum := referenceFullSortTopKTruncate(refProbs, baseSum, k)

				if math.Abs(fastSum-refSum) > 1e-12 {
					t.Fatalf("vocab %d k %d sum mismatch: got %v, want %v", vocabSize, k, fastSum, refSum)
				}
				for i := range fastProbs {
					if fastProbs[i] != refProbs[i] {
						t.Fatalf("vocab %d k %d prob mismatch at index %d: got %v, want %v", vocabSize, k, i, fastProbs[i], refProbs[i])
					}
				}
			})
		}

		// 2. Heavy ties distribution (discrete values with extensive ties across the vocabulary)
		var tieSum float64
		tieProbs := make([]float64, vocabSize)
		for i := range tieProbs {
			tieProbs[i] = float64(i%7) / 10.0
			tieSum += tieProbs[i]
		}
		for _, k := range []int{1, 8, 50, 64} {
			t.Run(fmt.Sprintf("ties_vocab_%d_k_%d", vocabSize, k), func(t *testing.T) {
				fastProbs := append([]float64(nil), tieProbs...)
				refProbs := append([]float64(nil), tieProbs...)

				fastSum := topKTruncate(fastProbs, tieSum, k)
				refSum := referenceFullSortTopKTruncate(refProbs, tieSum, k)

				if math.Abs(fastSum-refSum) > 1e-12 {
					t.Fatalf("tie vocab %d k %d sum mismatch: got %v, want %v", vocabSize, k, fastSum, refSum)
				}
				for i := range fastProbs {
					if fastProbs[i] != refProbs[i] {
						t.Fatalf("tie vocab %d k %d prob mismatch at index %d: got %v, want %v", vocabSize, k, i, fastProbs[i], refProbs[i])
					}
				}
			})
		}
	}
}

func TestTopKDeterministicTieBreak(t *testing.T) {
	// Test 1: all identical probabilities
	probs1 := []float64{0.2, 0.2, 0.2, 0.2, 0.2}
	fast1 := append([]float64(nil), probs1...)
	ref1 := append([]float64(nil), probs1...)
	topKTruncate(fast1, 1.0, 3)
	referenceFullSortTopKTruncate(ref1, 1.0, 3)
	for i := range fast1 {
		if fast1[i] != ref1[i] {
			t.Fatalf("all-ties mismatch at %d: got %v want %v", i, fast1[i], ref1[i])
		}
	}
	// Exactly indices 0, 1, 2 should be preserved (lower index breaks ties)
	if fast1[0] != 0.2 || fast1[1] != 0.2 || fast1[2] != 0.2 || fast1[3] != 0 || fast1[4] != 0 {
		t.Fatalf("unexpected kept set: %v", fast1)
	}

	// Test 2: tie right at the k boundary
	probs2 := []float64{0.5, 0.4, 0.3, 0.3, 0.3, 0.1}
	fast2 := append([]float64(nil), probs2...)
	ref2 := append([]float64(nil), probs2...)
	topKTruncate(fast2, 1.9, 3)
	referenceFullSortTopKTruncate(ref2, 1.9, 3)
	for i := range fast2 {
		if fast2[i] != ref2[i] {
			t.Fatalf("boundary-tie mismatch at %d: got %v want %v", i, fast2[i], ref2[i])
		}
	}
	// Indices 0 (0.5), 1 (0.4), and 2 (0.3) should be kept; indices 3, 4 (0.3) and 5 (0.1) zeroed
	if fast2[0] != 0.5 || fast2[1] != 0.4 || fast2[2] != 0.3 || fast2[3] != 0 || fast2[4] != 0 || fast2[5] != 0 {
		t.Fatalf("unexpected boundary kept set: %v", fast2)
	}
}

func TestTopKEdgeCases(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3}
	if sum := topKTruncate(probs, 0.6, 0); sum != 0 {
		t.Fatalf("k=0 should zero all, got sum %v", sum)
	}
	for i, p := range probs {
		if p != 0 {
			t.Fatalf("probs[%d] = %v, want 0", i, p)
		}
	}

	probs2 := []float64{0.1, 0.2, 0.3}
	if sum := topKTruncate(probs2, 0.6, 5); sum != 0.6 {
		t.Fatalf("k >= len should be no-op, got sum %v", sum)
	}
}

func BenchmarkTopKTruncate_SmallK_248k(b *testing.B) {
	const vocabSize = 248000
	const k = 50

	rng := rand.New(rand.NewSource(12345))
	orig := make([]float64, vocabSize)
	var sum float64
	for i := range orig {
		orig[i] = rng.Float64()
		sum += orig[i]
	}

	work := make([]float64, vocabSize)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		copy(work, orig)
		topKTruncate(work, sum, k)
	}
}

func BenchmarkTopKTruncate_FullSort_248k(b *testing.B) {
	const vocabSize = 248000
	const k = 50

	rng := rand.New(rand.NewSource(12345))
	orig := make([]float64, vocabSize)
	var sum float64
	for i := range orig {
		orig[i] = rng.Float64()
		sum += orig[i]
	}

	work := make([]float64, vocabSize)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		copy(work, orig)
		referenceFullSortTopKTruncate(work, sum, k)
	}
}
