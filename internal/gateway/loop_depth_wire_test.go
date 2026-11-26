package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// fak-test:runtime fast est=5ms lane=default
func TestModelMaxLoopsReadsCapabilityMatrix(t *testing.T) {
	t.Setenv("FAK_MODEL_CAPABILITIES", `{"looped-env-model":{"max_loops":5}}`)
	cases := []struct {
		model string
		want  int
	}{
		{"looped-env-model", 5},
		{"vendor/looped-env-model", 5},
		{"plain-unknown-model", 1},
		{"", 1},
	}
	for _, tc := range cases {
		if got := modelMaxLoops(tc.model); got != tc.want {
			t.Fatalf("modelMaxLoops(%q) = %d, want %d", tc.model, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestResolveLoopDepthEffortAndPolicyClamp(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"looped-stub": 8})
	// A varloop checkpoint (trained 1 of max 8) keeps the effort tier table.
	stubModelTrainedLoops(t, map[string]int{"looped-stub": 1})
	capAt := func(n int) *policy.LoopDepthRule { return &policy.LoopDepthRule{MaxLoopDepth: n} }
	cases := []struct {
		name string
		body string
		rule *policy.LoopDepthRule
		want *LoopDepthReceipt
	}{
		{"effort none is shallowest", `{"model":"looped-stub","reasoning_effort":"none"}`, nil, &LoopDepthReceipt{Loops: 1, MaxLoops: 8, Source: LoopDepthSourceEffort}},
		{"effort low", `{"model":"looped-stub","reasoning_effort":"low"}`, nil, &LoopDepthReceipt{Loops: 2, MaxLoops: 8, Source: LoopDepthSourceEffort}},
		{"effort medium", `{"model":"looped-stub","reasoning_effort":"medium"}`, nil, &LoopDepthReceipt{Loops: 4, MaxLoops: 8, Source: LoopDepthSourceEffort}},
		{"effort high", `{"model":"looped-stub","reasoning_effort":"high"}`, nil, &LoopDepthReceipt{Loops: 6, MaxLoops: 8, Source: LoopDepthSourceEffort}},
		{"explicit wins over effort", `{"model":"looped-stub","loops":7,"reasoning_effort":"low"}`, nil, &LoopDepthReceipt{Loops: 7, MaxLoops: 8, Source: LoopDepthSourceRequest}},
		{"effort under cap is not clamped", `{"model":"looped-stub","reasoning_effort":"low"}`, capAt(3), &LoopDepthReceipt{Loops: 2, MaxLoops: 8, Source: LoopDepthSourceEffort}},
		{"effort above cap is clamped", `{"model":"looped-stub","reasoning_effort":"high"}`, capAt(3), &LoopDepthReceipt{Loops: 3, MaxLoops: 8, Source: LoopDepthSourcePolicyClamp}},
		{"explicit above cap is clamped", `{"model":"looped-stub","loops":8}`, capAt(5), &LoopDepthReceipt{Loops: 5, MaxLoops: 8, Source: LoopDepthSourcePolicyClamp}},
		{"default full depth is clamped", `{"model":"looped-stub"}`, capAt(5), &LoopDepthReceipt{Loops: 5, MaxLoops: 8, Source: LoopDepthSourcePolicyClamp}},
		{"cap above model max keeps model max", `{"model":"looped-stub"}`, capAt(20), &LoopDepthReceipt{Loops: 8, MaxLoops: 8, Source: LoopDepthSourceDefault}},
		{"plain model ignores cap and effort", `{"model":"plain","reasoning_effort":"high"}`, capAt(1), nil},
		{"plain model explicit 1 is not clamped", `{"model":"plain","loops":1}`, capAt(1), &LoopDepthReceipt{Loops: 1, MaxLoops: 1, Source: LoopDepthSourceRequest}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := decodeLoopDepthChatRequest(t, tc.body)
			got, msg := resolveLoopDepth(req, modelMaxLoops(req.Model), modelTrainedLoops(req.Model), tc.rule)
			if msg != "" {
				t.Fatalf("unexpected refusal %q", msg)
			}
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("receipt = %+v, want %+v", got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestSetLoopDepthRuleRejectsInvalidCap(t *testing.T) {
	s := &Server{}
	if err := s.SetLoopDepthRule(&policy.LoopDepthRule{}); err == nil {
		t.Fatal("empty rule accepted, want compile error")
	}
	if err := s.SetLoopDepthRule(&policy.LoopDepthRule{MaxLoopDepth: 3}); err != nil || s.loopDepthRule.Load() == nil {
		t.Fatalf("valid rule: err=%v rule=%v", err, s.loopDepthRule.Load())
	}
	if err := s.SetLoopDepthRule(nil); err != nil || s.loopDepthRule.Load() != nil {
		t.Fatalf("nil rule must clear: err=%v rule=%v", err, s.loopDepthRule.Load())
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthSampleOptForwardsOnlyLoopedModels(t *testing.T) {
	cases := []struct {
		name    string
		receipt *LoopDepthReceipt
		want    int
	}{
		{"no receipt", nil, 0},
		{"plain model explicit loops 1", &LoopDepthReceipt{Loops: 1, MaxLoops: 1, Source: LoopDepthSourceRequest}, 0},
		{"looped explicit", &LoopDepthReceipt{Loops: 3, MaxLoops: 5, Source: LoopDepthSourceRequest}, 3},
		{"looped shallow still forwarded", &LoopDepthReceipt{Loops: 1, MaxLoops: 5, Source: LoopDepthSourceEffort}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var sp agent.SampleParams
			loopDepthSampleOpt(tc.receipt)(&sp)
			if sp.Loops != tc.want {
				t.Fatalf("Loops = %d, want %d", sp.Loops, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthAffinityKey(t *testing.T) {
	looped := func(n int) *LoopDepthReceipt { return &LoopDepthReceipt{Loops: n, MaxLoops: 8} }
	cases := []struct {
		name    string
		key     string
		receipt *LoopDepthReceipt
		want    string
	}{
		{"no receipt keeps key", "sess", nil, "sess"},
		{"plain model keeps key", "sess", &LoopDepthReceipt{Loops: 1, MaxLoops: 1}, "sess"},
		{"empty key stays empty", "", looped(3), ""},
		{"depth 1 shallow", "sess", looped(1), "sess@d=1"},
		{"depth 2 default class", "sess", looped(2), "sess@d=3"},
		{"depth 3 default class", "sess", looped(3), "sess@d=3"},
		{"depth 5 deep class", "sess", looped(5), "sess@d=5"},
		{"depth 8 deep class", "sess", looped(8), "sess@d=5"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := depthAffinityKey(tc.key, tc.receipt); got != tc.want {
				t.Fatalf("depthAffinityKey = %q, want %q", got, tc.want)
			}
		})
	}
	if depthAffinityKey("s", looped(3)) == depthAffinityKey("s", looped(5)) {
		t.Fatal("loops=5 turn shares an affinity key with a loops=3 prefix")
	}
}
