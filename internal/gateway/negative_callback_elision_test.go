package gateway

import (
	"errors"
	"testing"
)

func TestNominalHostCallbackRetentionWitness(t *testing.T) {
	// First witness requirements (#9924):
	// 1. Run 4 combinations: structured on/off x speculative on/off.
	// 2. Peak-memory accounting across all arms.
	// 3. Zero post-settlement leak when nominal callbacks are retained.
	// 4. Refuse promotion / configuration attempting to skip host callbacks without an OOM proof.

	variants := []ExecutionVariant{
		{StructuredOutput: false, SpeculativeDecoding: false, SkipHostCallback: false},
		{StructuredOutput: true, SpeculativeDecoding: false, SkipHostCallback: false},
		{StructuredOutput: false, SpeculativeDecoding: true, SkipHostCallback: false},
		{StructuredOutput: true, SpeculativeDecoding: true, SkipHostCallback: false},
	}

	receipt, err := VerifyNominalHostCallbackRetention(variants, false)
	if err != nil {
		t.Fatalf("VerifyNominalHostCallbackRetention failed: %v", err)
	}

	if receipt.ModesEvaluated != 4 {
		t.Fatalf("expected 4 modes evaluated, got %d", receipt.ModesEvaluated)
	}
	if receipt.PostSettlementLeakBytes != 0 {
		t.Fatalf("expected 0 leak bytes with nominal callbacks, got %d", receipt.PostSettlementLeakBytes)
	}
	if receipt.PeakAllocatedBytes <= 0 {
		t.Fatalf("expected positive peak allocation, got %d", receipt.PeakAllocatedBytes)
	}

	badVariants := []ExecutionVariant{
		{StructuredOutput: true, SpeculativeDecoding: true, SkipHostCallback: true},
	}
	_, err = VerifyNominalHostCallbackRetention(badVariants, false)
	if !errors.Is(err, ErrDisallowedCallbackElision) {
		t.Fatalf("expected ErrDisallowedCallbackElision, got %v", err)
	}
}
