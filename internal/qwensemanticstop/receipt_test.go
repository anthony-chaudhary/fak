package qwensemanticstop

import (
	"strings"
	"testing"
)

func validReceipt() Receipt {
	r := Receipt{
		Backend: "vllm", BackendVersion: "0.27.1", Model: exactModel,
		CancellationSupported: true, CancellationBoundMS: 100,
		NextLatencyTolerance:   0.10,
		PromotionEvidence:      "ten exact-model pairs correlate disconnect to scheduler release",
		DemotionEvidence:       "any missing correlation or post-cancel token demotes to client_latency_only",
		InvalidatingAssumption: "vLLM request IDs and cancellation metrics retain per-request semantics",
	}
	for i := 0; i < 10; i++ {
		r.Pairs = append(r.Pairs, Pair{
			Controls:   Controls{PromptSHA256: "sha256:prompt", Seed: 0, Temperature: 0, MaxTokens: 256},
			Control:    Arm{RequestID: "control-" + string(rune('a'+i)), GeneratedTokens: 256, ServerBusyMS: 3000, GPUUtilizationPercent: 90, NextRequestLatencyMS: 100, FinishState: "length"},
			Comparison: Arm{RequestID: "cancel-" + string(rune('a'+i)), GeneratedTokens: 24, ServerBusyMS: 300, GPUUtilizationPercent: 45, NextRequestLatencyMS: 101, FinishState: "cancelled", DisconnectObserved: true, CancelObserved: true, CancellationLatencyMS: 12, TokensAfterCancel: 0, SchedulerReleased: true},
		})
	}
	return r
}

func TestEvaluatePromotesOnlyServerReclamation(t *testing.T) {
	r := validReceipt()
	if err := Evaluate(&r); err != nil {
		t.Fatal(err)
	}
	if r.Semantics != ComputeReclaimed || r.Schema != Schema || r.EvaluatedAt.IsZero() {
		t.Fatalf("unexpected promoted receipt: %#v", r)
	}
}

func TestEvaluateRejectsClientOnlyDisconnect(t *testing.T) {
	r := validReceipt()
	r.Pairs[4].Comparison.CancelObserved = false
	if err := Evaluate(&r); err == nil || !strings.Contains(err.Error(), "CANCELLATION_UNPROVEN") {
		t.Fatalf("got %v, want cancellation rejection", err)
	}
	if r.Semantics != ClientLatencyOnly {
		t.Fatalf("semantics = %q, want %q", r.Semantics, ClientLatencyOnly)
	}
}

func TestEvaluateRejectsPostCancelWorkAndContamination(t *testing.T) {
	for _, mutate := range []func(*Receipt){
		func(r *Receipt) { r.Pairs[0].Comparison.TokensAfterCancel = 1 },
		func(r *Receipt) { r.Pairs[0].Comparison.NextRequestLatencyMS = 150 },
	} {
		r := validReceipt()
		mutate(&r)
		if err := Evaluate(&r); err == nil {
			t.Fatal("accepted receipt without reclaimed compute")
		}
		if r.Semantics != ClientLatencyOnly {
			t.Fatalf("semantics = %q, want fail-closed label", r.Semantics)
		}
	}
}

func TestEvaluateTypesUnsupportedCancellationAsHold(t *testing.T) {
	r := validReceipt()
	r.CancellationSupported = false
	if err := Evaluate(&r); err == nil || err.Error() != HoldUnsupported {
		t.Fatalf("got %v, want %s", err, HoldUnsupported)
	}
	if r.HoldReason != HoldUnsupported || r.Semantics != ClientLatencyOnly {
		t.Fatalf("unexpected HOLD receipt: %#v", r)
	}
}
