package agent

// sampling_test.go proves the per-request sampling seam (#62): a SampleOpt passed
// to HTTPPlanner.Complete reaches the upstream provider wire, and an omitted option
// preserves the pre-seam default (max_tokens 1024) byte-for-byte. The test captures
// the exact JSON body the planner POSTs and asserts on the serialized fields, so it
// witnesses the whole resolve→adapter→wire path, not just the option fold.

import (
	"context"
	"encoding/json"
	"io"
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
