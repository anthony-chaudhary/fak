package ctxmmu

import (
	"bytes"
	"math"
	"testing"
)

func TestAssessHeadroom_Below70Percent(t *testing.T) {
	policy := DefaultHeadroomPolicy()
	maxCapacity := 10000

	testCases := []struct {
		name          string
		currentTokens int
		wantRatio     float64
	}{
		{name: "50% capacity", currentTokens: 5000, wantRatio: 0.50},
		{name: "20% capacity", currentTokens: 2000, wantRatio: 0.20},
		{name: "69.9% capacity", currentTokens: 6990, wantRatio: 0.699},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assessment := AssessHeadroom(tc.currentTokens, maxCapacity, policy)

			if math.Abs(assessment.Ratio-tc.wantRatio) > 1e-4 {
				t.Errorf("Ratio = %v, want %v", assessment.Ratio, tc.wantRatio)
			}
			if assessment.ExceedsEscalation {
				t.Errorf("ExceedsEscalation = true, want false")
			}
			if assessment.RecommendedAction != ActionNone {
				t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionNone)
			}
			if assessment.ReclaimTargetTokens != 0 {
				t.Errorf("ReclaimTargetTokens = %d, want 0", assessment.ReclaimTargetTokens)
			}
		})
	}
}

func TestAssessHeadroom_Between70And85Percent(t *testing.T) {
	policy := DefaultHeadroomPolicy()
	maxCapacity := 10000

	testCases := []struct {
		name          string
		currentTokens int
		wantRatio     float64
		wantReclaim   int
	}{
		{name: "exactly 70%", currentTokens: 7000, wantRatio: 0.70, wantReclaim: 0},
		{name: "75% capacity", currentTokens: 7500, wantRatio: 0.75, wantReclaim: 500},
		{name: "80% capacity", currentTokens: 8000, wantRatio: 0.80, wantReclaim: 1000},
		{name: "84.9% capacity", currentTokens: 8490, wantRatio: 0.849, wantReclaim: 1490},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assessment := AssessHeadroom(tc.currentTokens, maxCapacity, policy)

			if math.Abs(assessment.Ratio-tc.wantRatio) > 1e-4 {
				t.Errorf("Ratio = %v, want %v", assessment.Ratio, tc.wantRatio)
			}
			if assessment.ExceedsEscalation {
				t.Errorf("ExceedsEscalation = true, want false")
			}
			if assessment.RecommendedAction != ActionTier1Tombstone {
				t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionTier1Tombstone)
			}
			if assessment.ReclaimTargetTokens != tc.wantReclaim {
				t.Errorf("ReclaimTargetTokens = %d, want %d", assessment.ReclaimTargetTokens, tc.wantReclaim)
			}
		})
	}
}

func TestAssessHeadroom_AtOrAbove85Percent(t *testing.T) {
	policy := DefaultHeadroomPolicy()
	maxCapacity := 10000

	testCases := []struct {
		name          string
		currentTokens int
		wantRatio     float64
		wantReclaim   int
	}{
		{name: "exactly 85%", currentTokens: 8500, wantRatio: 0.85, wantReclaim: 1500},
		{name: "90% capacity", currentTokens: 9000, wantRatio: 0.90, wantReclaim: 2000},
		{name: "95% capacity", currentTokens: 9500, wantRatio: 0.95, wantReclaim: 2500},
		{name: "100% capacity", currentTokens: 10000, wantRatio: 1.00, wantReclaim: 3000},
		{name: "110% over capacity", currentTokens: 11000, wantRatio: 1.10, wantReclaim: 4000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assessment := AssessHeadroom(tc.currentTokens, maxCapacity, policy)

			if math.Abs(assessment.Ratio-tc.wantRatio) > 1e-4 {
				t.Errorf("Ratio = %v, want %v", assessment.Ratio, tc.wantRatio)
			}
			if !assessment.ExceedsEscalation {
				t.Errorf("ExceedsEscalation = false, want true")
			}
			if assessment.RecommendedAction != ActionTier2Compaction {
				t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionTier2Compaction)
			}
			if assessment.ReclaimTargetTokens != tc.wantReclaim {
				t.Errorf("ReclaimTargetTokens = %d, want %d", assessment.ReclaimTargetTokens, tc.wantReclaim)
			}
		})
	}
}

func TestAssessHeadroom_EdgeCases(t *testing.T) {
	policy := DefaultHeadroomPolicy()

	t.Run("zero capacity", func(t *testing.T) {
		assessment := AssessHeadroom(5000, 0, policy)
		if assessment.Ratio != 0 {
			t.Errorf("Ratio = %v, want 0", assessment.Ratio)
		}
		if assessment.ExceedsEscalation {
			t.Errorf("ExceedsEscalation = true, want false")
		}
		if assessment.RecommendedAction != ActionNone {
			t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionNone)
		}
		if assessment.ReclaimTargetTokens != 0 {
			t.Errorf("ReclaimTargetTokens = %d, want 0", assessment.ReclaimTargetTokens)
		}
	})

	t.Run("negative capacity", func(t *testing.T) {
		assessment := AssessHeadroom(5000, -100, policy)
		if assessment.Ratio != 0 {
			t.Errorf("Ratio = %v, want 0", assessment.Ratio)
		}
		if assessment.ExceedsEscalation {
			t.Errorf("ExceedsEscalation = true, want false")
		}
		if assessment.RecommendedAction != ActionNone {
			t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionNone)
		}
	})

	t.Run("zero current tokens", func(t *testing.T) {
		assessment := AssessHeadroom(0, 10000, policy)
		if assessment.Ratio != 0 {
			t.Errorf("Ratio = %v, want 0", assessment.Ratio)
		}
		if assessment.ExceedsEscalation {
			t.Errorf("ExceedsEscalation = true, want false")
		}
		if assessment.RecommendedAction != ActionNone {
			t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionNone)
		}
	})

	t.Run("100% capacity", func(t *testing.T) {
		assessment := AssessHeadroom(10000, 10000, policy)
		if assessment.Ratio != 1.0 {
			t.Errorf("Ratio = %v, want 1.0", assessment.Ratio)
		}
		if !assessment.ExceedsEscalation {
			t.Errorf("ExceedsEscalation = false, want true")
		}
		if assessment.RecommendedAction != ActionTier2Compaction {
			t.Errorf("RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionTier2Compaction)
		}
		if assessment.ReclaimTargetTokens != 3000 {
			t.Errorf("ReclaimTargetTokens = %d, want 3000", assessment.ReclaimTargetTokens)
		}
	})

	t.Run("custom escalation threshold 90%", func(t *testing.T) {
		customPolicy := HeadroomPolicy{
			EscalationThreshold: 0.90,
		}
		maxCapacity := 10000

		// At 87% (8700 tokens): below custom 90% threshold, should be Tier 1 tombstone and not escalate
		assessment87 := AssessHeadroom(8700, maxCapacity, customPolicy)
		if assessment87.ExceedsEscalation {
			t.Errorf("at 87%%: ExceedsEscalation = true, want false for 90%% threshold")
		}
		if assessment87.RecommendedAction != ActionTier1Tombstone {
			t.Errorf("at 87%%: RecommendedAction = %q, want %q", assessment87.RecommendedAction, ActionTier1Tombstone)
		}

		// At 90% (9000 tokens): reaches custom threshold, should trigger Tier 2 compaction and escalation
		assessment90 := AssessHeadroom(9000, maxCapacity, customPolicy)
		if !assessment90.ExceedsEscalation {
			t.Errorf("at 90%%: ExceedsEscalation = false, want true for 90%% threshold")
		}
		if assessment90.RecommendedAction != ActionTier2Compaction {
			t.Errorf("at 90%%: RecommendedAction = %q, want %q", assessment90.RecommendedAction, ActionTier2Compaction)
		}
	})

	t.Run("custom escalation threshold 80%", func(t *testing.T) {
		customPolicy := HeadroomPolicy{
			EscalationThreshold: 0.80,
		}
		maxCapacity := 10000

		// At 81% (8100 tokens): above custom 80% threshold, should escalate
		assessment := AssessHeadroom(8100, maxCapacity, customPolicy)
		if !assessment.ExceedsEscalation {
			t.Errorf("at 81%%: ExceedsEscalation = false, want true for 80%% threshold")
		}
		if assessment.RecommendedAction != ActionTier2Compaction {
			t.Errorf("at 81%%: RecommendedAction = %q, want %q", assessment.RecommendedAction, ActionTier2Compaction)
		}
	})

	t.Run("zero-initialized policy defaults to 0.85 and 0.70", func(t *testing.T) {
		var emptyPolicy HeadroomPolicy
		assessment := AssessHeadroom(8500, 10000, emptyPolicy)
		if !assessment.ExceedsEscalation {
			t.Errorf("zero policy should default to 0.85 escalation: got ExceedsEscalation=false")
		}
		if assessment.RecommendedAction != ActionTier2Compaction {
			t.Errorf("zero policy should trigger tier2_compaction at 85%%: got %q", assessment.RecommendedAction)
		}
	})
}

func TestPrefixPreservingByteSplice(t *testing.T) {
	prefixSystem := []byte("You are an autonomous AI coding agent with strict safety invariants.")
	prefixTools := []byte(`[{"name":"bash","desc":"run shell"},{"name":"read","desc":"read file"}]`)

	pages := []TokenPage{
		// Turn 0: Pinned immutable prefix
		{
			ID:        1,
			TurnIndex: 0,
			Kind:      PageKindPrefixSystem,
			Role:      "system",
			Content:   prefixSystem,
			Tokens:    EstimateTokens(prefixSystem),
			Resident:  true,
			Pinned:    true,
		},
		{
			ID:        2,
			TurnIndex: 0,
			Kind:      PageKindPrefixTools,
			Role:      "system",
			Content:   prefixTools,
			Tokens:    EstimateTokens(prefixTools),
			Resident:  true,
			Pinned:    true,
		},
		// Turn 1: Middle turn user + tool result (should be compacted)
		{
			ID:        3,
			TurnIndex: 1,
			Kind:      PageKindUser,
			Role:      "user",
			Content:   []byte("Please run the diagnostic tests."),
			Tokens:    10,
			Resident:  true,
		},
		{
			ID:        4,
			TurnIndex: 1,
			Kind:      PageKindToolResult,
			Role:      "tool",
			ToolName:  "bash",
			Content:   []byte("Large diagnostic trace output: PASS 1, PASS 2, PASS 3 ... (1000 lines of logs)"),
			Tokens:    250,
			Resident:  true,
		},
		// Turn 2: Middle turn assistant + tool result (should be compacted)
		{
			ID:        5,
			TurnIndex: 2,
			Kind:      PageKindAssistant,
			Role:      "assistant",
			Content:   []byte("Analyzing diagnostics..."),
			Tokens:    15,
			Resident:  true,
		},
		{
			ID:        6,
			TurnIndex: 2,
			Kind:      PageKindToolResult,
			Role:      "tool",
			ToolName:  "read",
			Content:   []byte("package main\n\nfunc main() { /* hundreds of lines of file content */ }\n"),
			Tokens:    400,
			Resident:  true,
		},
		// Turn 3: Active window (recent turn, should be kept verbatim)
		{
			ID:        7,
			TurnIndex: 3,
			Kind:      PageKindUser,
			Role:      "user",
			Content:   []byte("What are the next steps?"),
			Tokens:    10,
			Resident:  true,
		},
		{
			ID:        8,
			TurnIndex: 3,
			Kind:      PageKindToolResult,
			Role:      "tool",
			ToolName:  "bash",
			Content:   []byte("active recent output that must stay intact"),
			Tokens:    20,
			Resident:  true,
		},
	}

	beforeSnapshot := make([]TokenPage, len(pages))
	copy(beforeSnapshot, pages)

	// Keep last 1 turn as active window (Turn 3), compact middle turns (Turns 1 and 2)
	compacted, reclaimed := PrefixPreservingByteSplice(pages, 1)

	if reclaimed <= 0 {
		t.Fatalf("expected positive tokens reclaimed, got %d", reclaimed)
	}

	// 1. Prefix Warmth Verification: Token 0 system & tool pages must be bit-for-bit preserved
	if !VerifyPrefixWarmth(beforeSnapshot, compacted) {
		t.Fatalf("VerifyPrefixWarmth failed: prefix was modified during byte-splice compaction!")
	}
	if !bytes.Equal(compacted[0].Content, prefixSystem) {
		t.Fatalf("System prefix bytes corrupted")
	}
	if !bytes.Equal(compacted[1].Content, prefixTools) {
		t.Fatalf("Tools prefix bytes corrupted")
	}

	// 2. Middle turn tool results must have been replaced with Tier 1 tombstones
	page4 := compacted[3] // Turn 1 tool result
	if !page4.Tombstone.Active {
		t.Errorf("Turn 1 tool result tombstone not active")
	}
	if string(page4.Content) != Tier1TombstoneMarker {
		t.Errorf("Turn 1 tool result content = %q, want %q", string(page4.Content), Tier1TombstoneMarker)
	}
	if page4.Tombstone.OriginalTokens != 250 {
		t.Errorf("Turn 1 tool result OriginalTokens = %d, want 250", page4.Tombstone.OriginalTokens)
	}

	page6 := compacted[5] // Turn 2 tool result
	if !page6.Tombstone.Active {
		t.Errorf("Turn 2 tool result tombstone not active")
	}
	if string(page6.Content) != Tier1TombstoneMarker {
		t.Errorf("Turn 2 tool result content = %q, want %q", string(page6.Content), Tier1TombstoneMarker)
	}

	// 3. Active window (Turn 3) tool result must remain unmodified
	page8 := compacted[7]
	if page8.Tombstone.Active {
		t.Errorf("Turn 3 active tool result was tombstoned; should be kept verbatim")
	}
	if string(page8.Content) != "active recent output that must stay intact" {
		t.Errorf("Turn 3 active tool result content modified: %q", string(page8.Content))
	}
}
