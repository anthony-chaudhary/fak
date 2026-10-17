package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestPrefillRouteFromReceiptRecordsDeclineReason pins that a declined
// whole-sequence prefill reaches the perf row with its stable decline reason, so
// a scalar token-replay turn is attributable from the ledger alone.
func TestPrefillRouteFromReceiptRecordsDeclineReason(t *testing.T) {
	receipt := &model.NativeInferenceReceipt{
		Qwen35SequencePrefillRoute: &model.NativeSequencePrefillRouteReceipt{
			PrefillCalls:  1,
			ObservedCalls: 1,
			Complete:      true,
			Status: &model.Qwen35SequencePrefillRouteStatus{
				RequestedPath:  compute.Qwen35SequencePrefillPath,
				EffectivePath:  model.Qwen35SequencePrefillFallbackPath,
				DeclineReason:  model.Qwen35SequencePrefillDeclineEmbeddingCap,
				FallbackActive: true,
			},
		},
	}
	got := prefillRouteFromReceipt(receipt)
	if got == nil {
		t.Fatal("declined route produced no perf-row prefill route")
	}
	if got.EffectivePath != model.Qwen35SequencePrefillFallbackPath || got.DeclineReason != model.Qwen35SequencePrefillDeclineEmbeddingCap || !got.FallbackActive || got.NativePerformanceQualifying {
		t.Fatalf("perf-row route=%+v, want the recorded decline", got)
	}
}

// TestPrefillRouteFromReceiptOmitsAbsentRoute pins that a receipt with no route
// block yields no perf-row field, rather than a fabricated route.
func TestPrefillRouteFromReceiptOmitsAbsentRoute(t *testing.T) {
	for name, receipt := range map[string]*model.NativeInferenceReceipt{
		"nil-receipt": nil,
		"no-route":    {},
		"no-status":   {Qwen35SequencePrefillRoute: &model.NativeSequencePrefillRouteReceipt{PrefillCalls: 1}},
	} {
		if got := prefillRouteFromReceipt(receipt); got != nil {
			t.Fatalf("%s: perf-row route=%+v, want nil", name, got)
		}
	}
}
