package engine

import (
	"context"
	"testing"
)

// TestHeterogeneousRecurrentDepthIntegration proves:
// 1. Yield/resume at explicit iteration boundaries for two requests with different depth limits
//    (heterogeneous depths, e.g. Request A depth=2, Request B depth=4).
// 2. Real-path integration reproduction proves bounded completion, independent state, and cancellation progress:
//    - Submit Request A (depth=2, targetTokens=10) and Request B (depth=4, targetTokens=15).
//    - Prove that both advance across iteration boundaries with heterogeneous depth limits.
//    - Demonstrate yield/resume on one request (e.g. tool execution pause on Request A) while the deeper request (Request B) continues progressing.
//    - Demonstrate cancellation progress on a slot: cancel Request B midway; verify Request B transitions to canceled state while Request A finishes cleanly to completion.
//    - Prove that KV memory capacity freed by the canceled slot is properly reclaimed by the batcher.
func TestHeterogeneousRecurrentDepthIntegration(t *testing.T) {
	ctx := context.Background()

	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 4
	cfg.KVBytesPerToken = 100
	cfg.MaxKVCacheBytes = 100_000 // Ensure sufficient headroom for initial admission

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	// Request A: depth=2, targetTokens=10, promptTokens=2 tokens
	reqA := &SubagentRequest{
		SessionID:      "sub-A-depth-2",
		PromptTokens:   []int{101, 102},
		TargetTokens:   10,
		ExecutionDepth: 2,
	}

	// Request B: depth=4, targetTokens=15, promptTokens=3 tokens
	reqB := &SubagentRequest{
		SessionID:      "sub-B-depth-4",
		PromptTokens:   []int{201, 202, 203},
		TargetTokens:   15,
		ExecutionDepth: 4,
	}

	expectedKVA := cb.RequiredKVCacheBytes(reqA) // (2 + 10) * 100 * 2 = 2,400 bytes
	expectedKVB := cb.RequiredKVCacheBytes(reqB) // (3 + 15) * 100 * 4 = 7,200 bytes

	if expectedKVA != 2400 {
		t.Fatalf("expectedKVA = %d, want 2400", expectedKVA)
	}
	if expectedKVB != 7200 {
		t.Fatalf("expectedKVB = %d, want 7200", expectedKVB)
	}

	// Submit both requests
	if _, err := cb.Submit(reqA); err != nil {
		t.Fatalf("Submit(reqA) failed: %v", err)
	}
	if _, err := cb.Submit(reqB); err != nil {
		t.Fatalf("Submit(reqB) failed: %v", err)
	}

	// Verify both slots are active with heterogeneous depths and isolated KV footprints
	if got := cb.ActiveSlotCount(); got != 2 {
		t.Fatalf("initial ActiveSlotCount = %d, want 2", got)
	}
	slotA, okA := cb.GetSlot(reqA.SessionID)
	if !okA || slotA.State != SlotStateActiveDecode {
		t.Fatalf("slotA not active")
	}
	if slotA.KVCacheBytes != expectedKVA {
		t.Fatalf("slotA KVCacheBytes = %d, want %d", slotA.KVCacheBytes, expectedKVA)
	}

	slotB, okB := cb.GetSlot(reqB.SessionID)
	if !okB || slotB.State != SlotStateActiveDecode {
		t.Fatalf("slotB not active")
	}
	if slotB.KVCacheBytes != expectedKVB {
		t.Fatalf("slotB KVCacheBytes = %d, want %d", slotB.KVCacheBytes, expectedKVB)
	}

	totalInitialKV := expectedKVA + expectedKVB
	if got := cb.CurrentKVCacheBytes(); got != totalInitialKV {
		t.Fatalf("initial CurrentKVCacheBytes = %d, want %d", got, totalInitialKV)
	}

	// Step 3 iterations: both advance across iteration boundaries with heterogeneous depth limits
	var tokensA []int
	var tokensB []int
	for i := 0; i < 3; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		tokA, okA := res.GeneratedTokens[reqA.SessionID]
		tokB, okB := res.GeneratedTokens[reqB.SessionID]
		if !okA || !okB {
			t.Fatalf("Step %d: expected both requests to generate tokens, got okA=%v, okB=%v", i, okA, okB)
		}
		tokensA = append(tokensA, tokA)
		tokensB = append(tokensB, tokB)
	}

	if len(tokensA) != 3 || len(tokensB) != 3 {
		t.Fatalf("expected 3 tokens each, got len(tokensA)=%d, len(tokensB)=%d", len(tokensA), len(tokensB))
	}

	// Yield Request A for simulated tool execution pause (transitions to SlotStateYieldedIO)
	if err := cb.YieldSlot(reqA.SessionID); err != nil {
		t.Fatalf("YieldSlot(reqA) failed: %v", err)
	}
	if slotA.State != SlotStateYieldedIO {
		t.Fatalf("slotA state after yield = %s, want %s", slotA.State, SlotStateYieldedIO)
	}
	// KV cache remains stationary during yield
	if got := cb.CurrentKVCacheBytes(); got != totalInitialKV {
		t.Fatalf("CurrentKVCacheBytes during yield = %d, want %d", got, totalInitialKV)
	}

	// Step 2 iterations while Request A is yielded: deeper Request B (depth=4) continues progressing alone
	for i := 0; i < 2; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step during yield %d failed: %v", i, err)
		}
		if _, ok := res.GeneratedTokens[reqA.SessionID]; ok {
			t.Fatalf("Request A unexpectedly generated token while yielded")
		}
		tokB, ok := res.GeneratedTokens[reqB.SessionID]
		if !ok {
			t.Fatalf("Request B failed to generate token while Request A yielded")
		}
		tokensB = append(tokensB, tokB)
	}
	if len(tokensB) != 5 {
		t.Fatalf("expected 5 tokens for Request B, got %d", len(tokensB))
	}

	// Resume Request A after tool execution completes
	if err := cb.ResumeSlot(reqA.SessionID); err != nil {
		t.Fatalf("ResumeSlot(reqA) failed: %v", err)
	}
	if slotA.State != SlotStateActiveDecode {
		t.Fatalf("slotA state after resume = %s, want %s", slotA.State, SlotStateActiveDecode)
	}

	// Step 1 iteration with both active again
	res, err := cb.Step(ctx)
	if err != nil {
		t.Fatalf("Step after resume failed: %v", err)
	}
	tokA, okA := res.GeneratedTokens[reqA.SessionID]
	tokB, okB := res.GeneratedTokens[reqB.SessionID]
	if !okA || !okB {
		t.Fatalf("expected both active after resume, got okA=%v, okB=%v", okA, okB)
	}
	tokensA = append(tokensA, tokA) // 4 tokens now
	tokensB = append(tokensB, tokB) // 6 tokens now

	// Demonstrate cancellation progress: cancel Request B midway
	// Request B is at 6 / 15 tokens. Cancel it midway.
	if err := cb.Cancel(reqB.SessionID); err != nil {
		t.Fatalf("Cancel(reqB) failed: %v", err)
	}

	// Verify Request B transitions to canceled state (SlotStateFinished with context.Canceled err)
	slotBCanceled, okBCanceled := cb.GetSlot(reqB.SessionID)
	if !okBCanceled {
		t.Fatalf("slotB should be retrievable from completedMap")
	}
	if slotBCanceled.State != SlotStateFinished {
		t.Fatalf("slotB state after cancel = %s, want %s", slotBCanceled.State, SlotStateFinished)
	}
	if slotBCanceled.Err() != context.Canceled {
		t.Fatalf("slotB Err() = %v, want %v", slotBCanceled.Err(), context.Canceled)
	}

	// Prove that KV memory capacity freed by the canceled slot (7,200 bytes) is properly reclaimed by the batcher
	if got, want := cb.CurrentKVCacheBytes(), expectedKVA; got != want {
		t.Fatalf("CurrentKVCacheBytes after canceling reqB = %d, want %d (reclaimed %d)", got, want, expectedKVB)
	}
	if got := cb.ActiveSlotCount(); got != 1 {
		t.Fatalf("ActiveSlotCount after canceling reqB = %d, want 1", got)
	}

	// Step until Request A completes to its target tokens (10 tokens)
	stepSafety := 20
	for cb.ActiveSlotCount() > 0 && stepSafety > 0 {
		stepSafety--
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step draining reqA failed: %v", err)
		}
		if tok, ok := res.GeneratedTokens[reqA.SessionID]; ok {
			tokensA = append(tokensA, tok)
		}
	}

	// Verify Request A finished cleanly to completion
	if len(tokensA) != reqA.TargetTokens {
		t.Fatalf("tokensA count = %d, want targetTokens=%d", len(tokensA), reqA.TargetTokens)
	}
	slotAFinished, okAFinished := cb.GetSlot(reqA.SessionID)
	if !okAFinished {
		t.Fatalf("slotA not found after completion")
	}
	if slotAFinished.State != SlotStateFinished {
		t.Fatalf("slotA state = %s, want %s", slotAFinished.State, SlotStateFinished)
	}
	if slotAFinished.Err() != nil {
		t.Fatalf("slotA finished with unexpected error: %v", slotAFinished.Err())
	}

	// Final state: all slots empty, KV cache completely reclaimed to 0 bytes
	if got := cb.ActiveSlotCount(); got != 0 {
		t.Fatalf("final ActiveSlotCount = %d, want 0", got)
	}
	if got := cb.CurrentKVCacheBytes(); got != 0 {
		t.Fatalf("final CurrentKVCacheBytes = %d, want 0", got)
	}
}
