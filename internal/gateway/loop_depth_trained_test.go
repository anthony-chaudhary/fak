package gateway

import (
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

func stubModelTrainedLoops(t *testing.T, trained map[string]int) {
	t.Helper()
	prev := modelTrainedLoops
	modelTrainedLoops = func(model string) int { return trained[model] }
	t.Cleanup(func() { modelTrainedLoops = prev })
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthTrainedLoopsFixedDepthHoldsEveryEffortTier(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"fixed-stub": 3})
	stubModelTrainedLoops(t, map[string]int{"fixed-stub": 3})
	for _, effort := range []string{"none", "low", "medium", "high"} {
		req := decodeLoopDepthChatRequest(t, `{"model":"fixed-stub","reasoning_effort":"`+effort+`"}`)
		got, msg := resolveLoopDepth(req, modelMaxLoops(req.Model), modelTrainedLoops(req.Model), nil)
		if msg != "" || got == nil || got.Loops != 3 {
			t.Fatalf("effort %s: got %+v msg %q, want 3 passes (trained)", effort, got, msg)
		}
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthTrainedLoopsVarloopEffortRaisesAboveFloor(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"var-stub": 8})
	stubModelTrainedLoops(t, map[string]int{"var-stub": 3})
	cases := map[string]int{"none": 3, "low": 3, "medium": 4, "high": 6}
	for effort, want := range cases {
		req := decodeLoopDepthChatRequest(t, `{"model":"var-stub","reasoning_effort":"`+effort+`"}`)
		got, msg := resolveLoopDepth(req, modelMaxLoops(req.Model), modelTrainedLoops(req.Model), nil)
		if msg != "" || got == nil || got.Loops != want {
			t.Fatalf("effort %s: got %+v msg %q, want %d passes", effort, got, msg, want)
		}
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthTrainedLoopsUnknownIsFixedDepth(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"unknown-stub": 8})
	stubModelTrainedLoops(t, map[string]int{})
	req := decodeLoopDepthChatRequest(t, `{"model":"unknown-stub","reasoning_effort":"none"}`)
	got, msg := resolveLoopDepth(req, modelMaxLoops(req.Model), modelTrainedLoops(req.Model), nil)
	if msg != "" || got == nil || got.Loops != 8 {
		t.Fatalf("unknown trained count: got %+v msg %q, want 8 passes", got, msg)
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthTrainedLoopsNoEffortRunsFullDepth(t *testing.T) {
	stubModelMaxLoops(t, map[string]int{"fixed-stub": 3})
	stubModelTrainedLoops(t, map[string]int{"fixed-stub": 3})
	rec := httptest.NewRecorder()
	req := decodeLoopDepthChatRequest(t, `{"model":"fixed-stub"}`)
	receipt, ok := resolveChatLoopDepth(rec, req, (*policy.LoopDepthRule)(nil))
	if !ok || receipt == nil || receipt.Loops != 3 || receipt.Source != LoopDepthSourceDefault {
		t.Fatalf("no-effort receipt = %+v ok=%v, want 3 default passes", receipt, ok)
	}
}
