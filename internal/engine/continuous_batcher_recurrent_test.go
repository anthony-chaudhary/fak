package engine

import (
	"context"
	"testing"
)

// TestRecurrentAdmissionUsesExecutionBudget verifies that under a fixed KV memory budget:
// 1. An ordinary non-recurrent request (depth=1) fits within the budget and is admitted into an active slot.
// 2. An equivalent recurrent request (depth > 1, requiring larger KV cache state) exceeds the budget and is
//    queued into waitingQueue rather than admitted immediately.
func TestRecurrentAdmissionUsesExecutionBudget(t *testing.T) {
	ctx := context.Background()

	t.Run("recurrent_exceeds_remaining_budget_and_promotes_on_drain", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4
		cfg.MaxKVCacheBytes = 10_000 // Fixed budget of 10,000 bytes
		cfg.KVBytesPerToken = 100    // 100 bytes per token

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// Prompt = 20 tokens, Target = 30 tokens -> 50 tokens total
		// depth = 1 -> required KV = 50 tokens * 100 bytes/tok * 1 = 5,000 bytes (fits in 10,000 budget)
		reqOrdinary := &SubagentRequest{
			SessionID:      "sub-ordinary",
			PromptTokens:   make([]int, 20),
			TargetTokens:   5, // short generation for quick completion
			ExecutionDepth: 1,
		}

		id1, err := cb.Submit(reqOrdinary)
		if err != nil {
			t.Fatalf("Submit(reqOrdinary) failed: %v", err)
		}
		if id1 != "sub-ordinary" {
			t.Fatalf("Submit returned %s, want sub-ordinary", id1)
		}

		// Ordinary request fits within budget and is admitted immediately to an active slot
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength = %d, want 0", got)
		}
		if got, want := cb.CurrentKVCacheBytes(), int64(25*100*1); got != want {
			// (20 prompt + 5 target) * 100 * 1 = 2,500 bytes
			t.Fatalf("CurrentKVCacheBytes = %d, want %d", got, want)
		}

		// Equivalent recurrent request: same prompt (20) and target (5), but ExecutionDepth = 4
		// required KV = 25 tokens * 100 bytes/tok * 4 = 10,000 bytes.
		// Current used = 2,500. Total required = 2,500 + 10,000 = 12,500 > 10,000 budget.
		reqRecurrent := &SubagentRequest{
			SessionID:      "sub-recurrent",
			PromptTokens:   make([]int, 20),
			TargetTokens:   5,
			ExecutionDepth: 4,
		}

		id2, err := cb.Submit(reqRecurrent)
		if err != nil {
			t.Fatalf("Submit(reqRecurrent) failed: %v", err)
		}
		if id2 != "sub-recurrent" {
			t.Fatalf("Submit returned %s, want sub-recurrent", id2)
		}

		// Recurrent request exceeds the budget, so it must NOT be admitted immediately to an active slot;
		// it must be queued into waitingQueue.
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount after recurrent submit = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 1 {
			t.Fatalf("WaitingQueueLength after recurrent submit = %d, want 1", got)
		}
		if _, ok := cb.GetSlot("sub-recurrent"); ok {
			t.Fatalf("sub-recurrent should NOT be in active session map while queued")
		}

		// Step until the ordinary request finishes and retires
		maxSteps := 20
		stepsRun := 0
		for stepsRun < maxSteps && cb.WaitingQueueLength() > 0 {
			res, err := cb.Step(ctx)
			if err != nil {
				t.Fatalf("Step failed at iteration %d: %v", stepsRun, err)
			}
			stepsRun++
			_ = res
		}

		// After ordinary request retires, its KV memory is freed, and the recurrent request
		// (10,000 bytes <= 10,000 budget) is admitted into an active slot.
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength after ordinary retired = %d, want 0", got)
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount after recurrent promotion = %d, want 1", got)
		}
		slotRec, ok := cb.GetSlot("sub-recurrent")
		if !ok {
			t.Fatalf("sub-recurrent not found in active session map after promotion")
		}
		if slotRec.State != SlotStateActiveDecode {
			t.Fatalf("sub-recurrent slot state = %s, want %s", slotRec.State, SlotStateActiveDecode)
		}
		if got, want := slotRec.KVCacheBytes, int64(25*100*4); got != want {
			t.Fatalf("sub-recurrent KVCacheBytes = %d, want %d", got, want)
		}
	})

	t.Run("recurrent_exceeds_total_budget_alone_and_is_queued", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4
		cfg.MaxKVCacheBytes = 8_000 // Fixed budget of 8,000 bytes
		cfg.KVBytesPerToken = 100   // 100 bytes per token

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// 1. Non-recurrent request (depth=1):
		// tokens = 20 prompt + 30 target = 50 tokens
		// required KV = 50 * 100 * 1 = 5,000 <= 8,000 -> Admitted!
		reqNonRecurrent := &SubagentRequest{
			SessionID:      "ord-fits",
			PromptTokens:   make([]int, 20),
			TargetTokens:   30,
			ExecutionDepth: 1,
		}

		if _, err := cb.Submit(reqNonRecurrent); err != nil {
			t.Fatalf("Submit(reqNonRecurrent) failed: %v", err)
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength = %d, want 0", got)
		}

		// Cancel ord-fits to clear the batcher completely
		if err := cb.Cancel("ord-fits"); err != nil {
			t.Fatalf("Cancel failed: %v", err)
		}
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount after cancel = %d, want 0", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 0 {
			t.Fatalf("CurrentKVCacheBytes after cancel = %d, want 0", got)
		}

		// 2. Equivalent recurrent request (depth=2):
		// tokens = 20 prompt + 30 target = 50 tokens
		// required KV = 50 * 100 * 2 = 10,000 bytes > 8,000 budget.
		// Even with all 4 slots empty, it exceeds the budget and is queued into waitingQueue.
		reqRecurrent := &SubagentRequest{
			SessionID:      "rec-exceeds",
			PromptTokens:   make([]int, 20),
			TargetTokens:   30,
			ExecutionDepth: 2,
		}

		if _, err := cb.Submit(reqRecurrent); err != nil {
			t.Fatalf("Submit(reqRecurrent) failed: %v", err)
		}

		// Must NOT be admitted immediately to an active slot
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount = %d, want 0", got)
		}
		// Must be queued in waitingQueue
		if got := cb.WaitingQueueLength(); got != 1 {
			t.Fatalf("WaitingQueueLength = %d, want 1", got)
		}
		if _, ok := cb.GetSlot("rec-exceeds"); ok {
			t.Fatalf("rec-exceeds should NOT be in active session map")
		}
	})

	t.Run("recurrent_loops_and_custom_bytes_per_token", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.MaxKVCacheBytes = 5_000

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// Using RecurrentLoops and KVBytesPerToken on the request
		// tokens = 10 + 10 = 20
		// bytesPerToken = 50
		// RecurrentLoops = 3 -> depth = 3
		// Required = 20 * 50 * 3 = 3,000 <= 5,000 -> Admitted!
		req1 := &SubagentRequest{
			SessionID:       "loop-fits",
			PromptTokens:    make([]int, 10),
			TargetTokens:    10,
			RecurrentLoops:  3,
			KVBytesPerToken: 50,
		}

		if _, err := cb.Submit(req1); err != nil {
			t.Fatalf("Submit(req1) failed: %v", err)
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength = %d, want 0", got)
		}

		// Second request: same dimensions, but RecurrentLoops = 3 (requires 3,000)
		// Total required = 3,000 + 3,000 = 6,000 > 5,000 -> Queued!
		req2 := &SubagentRequest{
			SessionID:       "loop-queued",
			PromptTokens:    make([]int, 10),
			TargetTokens:    10,
			RecurrentLoops:  3,
			KVBytesPerToken: 50,
		}

		if _, err := cb.Submit(req2); err != nil {
			t.Fatalf("Submit(req2) failed: %v", err)
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 1 {
			t.Fatalf("WaitingQueueLength = %d, want 1", got)
		}
	})
}
