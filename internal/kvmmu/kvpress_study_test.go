package kvmmu

import (
	"errors"
	"slices"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestKVPressPolicyBudgetAndInvariants(t *testing.T) {
	scores := []float64{0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.1, 0.2}
	plan, err := PlanKVCompression(KVPressPolicy{
		Family:       ScorerSnapKV,
		KeepRatio:    0.5, // 4 of 8 tokens
		SinkTokens:   1,   // index 0 pinned
		RecentWindow: 1,   // index 7 pinned
	}, scores)
	if err != nil {
		t.Fatal(err)
	}
	// Remaining 2 slots go to highest-scoring middle tokens: indices 2 (0.9) and 4 (0.8).
	want := []int{0, 2, 4, 7}
	if !slices.Equal(plan.KeptIndices, want) || plan.EvictedCount != 4 || plan.BudgetTokens != 4 {
		t.Fatalf("got %+v, want kept=%v", plan, want)
	}

	// Invalid family or out-of-range ratio fails closed.
	if _, err := PlanKVCompression(KVPressPolicy{Family: "unknown", KeepRatio: 0.5}, scores); !errors.Is(err, ErrInvalidKVPressPolicy) {
		t.Fatalf("expected ErrInvalidKVPressPolicy, got %v", err)
	}
	if _, err := PlanKVCompression(KVPressPolicy{Family: ScorerStreamingLLM, KeepRatio: 0}, scores); !errors.Is(err, ErrInvalidKVPressPolicy) {
		t.Fatalf("expected ErrInvalidKVPressPolicy for zero ratio, got %v", err)
	}
}
