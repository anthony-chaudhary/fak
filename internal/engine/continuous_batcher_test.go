package engine

import (
	"context"
	"fmt"
	"testing"
)

// TestContinuousBatcher_BasicScheduling verifies submission, iteration-level continuous batching,
// and completion of 4 concurrent subagent requests with irregular turn lengths.
func TestContinuousBatcher_BasicScheduling(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 4

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx := context.Background()

	// 4 subagent requests with irregular turn lengths
	requests := []*SubagentRequest{
		{SessionID: "sub-1", PromptTokens: []int{101, 102}, TargetTokens: 5},
		{SessionID: "sub-2", PromptTokens: []int{201, 202}, TargetTokens: 12},
		{SessionID: "sub-3", PromptTokens: []int{301, 302}, TargetTokens: 8},
		{SessionID: "sub-4", PromptTokens: []int{401, 402}, TargetTokens: 15},
	}

	for _, req := range requests {
		id, err := cb.Submit(req)
		if err != nil {
			t.Fatalf("Submit(%s) failed: %v", req.SessionID, err)
		}
		if id != req.SessionID {
			t.Fatalf("Submit returned %s, want %s", id, req.SessionID)
		}
	}

	if got := cb.ActiveSlotCount(); got != 4 {
		t.Fatalf("ActiveSlotCount = %d, want 4", got)
	}

	// Step until all 4 complete
	maxSteps := 50
	stepsRun := 0
	for stepsRun < maxSteps {
		allDone := true
		for _, req := range requests {
			slot, ok := cb.GetSlot(req.SessionID)
			if !ok || slot.TokensGenerated() < req.TargetTokens {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}

		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step failed at iteration %d: %v", stepsRun, err)
		}
		if res.ActiveSlots > 4 {
			t.Fatalf("ActiveSlots %d exceeded MaxSlots 4", res.ActiveSlots)
		}
		stepsRun++
	}

	// Verify all 4 completed with exact target tokens
	for _, req := range requests {
		slot, ok := cb.GetSlot(req.SessionID)
		if !ok {
			t.Fatalf("session %s not found", req.SessionID)
		}
		select {
		case <-slot.Done():
		default:
			t.Fatalf("session %s doneCh not closed", req.SessionID)
		}
		if got, want := slot.TokensGenerated(), req.TargetTokens; got != want {
			t.Fatalf("session %s tokens = %d, want %d", req.SessionID, got, want)
		}
		if slot.State != SlotStateFinished {
			t.Fatalf("session %s state = %s, want %s", req.SessionID, slot.State, SlotStateFinished)
		}
	}
}

// TestContinuousBatcher_YieldAndResumeNoStall verifies slot yield during subagent tool execution
// and immediate resume without recomputing context or causing compute stalls.
func TestContinuousBatcher_YieldAndResumeNoStall(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 4

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx := context.Background()

	reqA := &SubagentRequest{SessionID: "sub-A", PromptTokens: []int{10, 20}, TargetTokens: 10}
	reqB := &SubagentRequest{SessionID: "sub-B", PromptTokens: []int{30, 40}, TargetTokens: 10}

	if _, err := cb.Submit(reqA); err != nil {
		t.Fatalf("Submit reqA failed: %v", err)
	}
	if _, err := cb.Submit(reqB); err != nil {
		t.Fatalf("Submit reqB failed: %v", err)
	}

	// Step 3 times: both sub-A and sub-B generate 3 tokens
	for i := 0; i < 3; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		if res.ActiveSlots != 2 {
			t.Fatalf("Step %d: ActiveSlots = %d, want 2", i, res.ActiveSlots)
		}
	}

	slotA, _ := cb.GetSlot("sub-A")
	slotB, _ := cb.GetSlot("sub-B")
	if slotA.TokensGenerated() != 3 || slotB.TokensGenerated() != 3 {
		t.Fatalf("expected 3 tokens each, got A=%d, B=%d", slotA.TokensGenerated(), slotB.TokensGenerated())
	}

	// Subagent A yields for external tool execution
	if err := cb.YieldSlot("sub-A"); err != nil {
		t.Fatalf("YieldSlot(sub-A) failed: %v", err)
	}

	// Verify slot state and UMA residency invariants (zero eviction, zero byte swaps)
	if slotA.State != SlotStateYieldedIO {
		t.Fatalf("slotA.State = %s, want %s", slotA.State, SlotStateYieldedIO)
	}
	if slotA.EvictionDuration != 0 {
		t.Fatalf("slotA.EvictionDuration = %v, want 0 in UMA", slotA.EvictionDuration)
	}
	if slotA.BytesSwapped != 0 {
		t.Fatalf("slotA.BytesSwapped = %d, want 0 in UMA", slotA.BytesSwapped)
	}
	if !slotA.KVCacheStationary {
		t.Fatalf("slotA.KVCacheStationary = false, want true")
	}

	// Step 2 times while sub-A is yielded:
	// Only sub-B should be active and generate tokens; sub-A should remain stationary.
	for i := 0; i < 2; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step during yield failed: %v", err)
		}
		if res.ActiveSlots != 1 {
			t.Fatalf("ActiveSlots = %d, want 1 during yield", res.ActiveSlots)
		}
		if res.YieldedSlots != 1 {
			t.Fatalf("YieldedSlots = %d, want 1", res.YieldedSlots)
		}
		if res.StallDuration != 0 {
			t.Fatalf("StallDuration = %v, want 0 (zero compute stalls)", res.StallDuration)
		}
	}

	if slotA.TokensGenerated() != 3 {
		t.Fatalf("slotA generated tokens while yielded: %d, want 3", slotA.TokensGenerated())
	}
	if slotB.TokensGenerated() != 5 {
		t.Fatalf("slotB generated tokens: %d, want 5", slotB.TokensGenerated())
	}

	// Resume sub-A immediately when tool completes
	if err := cb.ResumeSlot("sub-A"); err != nil {
		t.Fatalf("ResumeSlot(sub-A) failed: %v", err)
	}
	if slotA.State != SlotStateActiveDecode {
		t.Fatalf("slotA.State after resume = %s, want %s", slotA.State, SlotStateActiveDecode)
	}
	if slotA.ReprefillTokens != 0 {
		t.Fatalf("slotA.ReprefillTokens = %d, want 0 (KV cache preserved in UMA)", slotA.ReprefillTokens)
	}

	// Next step: both are active again with zero compute stalls
	res, err := cb.Step(ctx)
	if err != nil {
		t.Fatalf("Step after resume failed: %v", err)
	}
	if res.ActiveSlots != 2 {
		t.Fatalf("ActiveSlots after resume = %d, want 2", res.ActiveSlots)
	}
	if res.StallDuration != 0 {
		t.Fatalf("StallDuration after resume = %v, want 0", res.StallDuration)
	}
	if slotA.TokensGenerated() != 4 {
		t.Fatalf("slotA tokens = %d, want 4", slotA.TokensGenerated())
	}
	if slotB.TokensGenerated() != 6 {
		t.Fatalf("slotB tokens = %d, want 6", slotB.TokensGenerated())
	}

	// Step to completion
	for cb.ActiveSlotCount() > 0 {
		if _, err := cb.Step(ctx); err != nil {
			t.Fatalf("Step failed: %v", err)
		}
	}

	if slotA.TokensGenerated() != 10 || slotB.TokensGenerated() != 10 {
		t.Fatalf("Both should reach 10 tokens: A=%d, B=%d", slotA.TokensGenerated(), slotB.TokensGenerated())
	}
}

// TestContinuousBatcher_DynamicSlotInsertion verifies that new subagents can enter mid-flight
// into open slots or queued until finished slots are retired, without stalling existing decodes.
func TestContinuousBatcher_DynamicSlotInsertion(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 3

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx := context.Background()

	// Submit 2 initial subagents (slots 0 and 1 occupied, slot 2 empty)
	req1 := &SubagentRequest{SessionID: "agent-1", TargetTokens: 20}
	req2 := &SubagentRequest{SessionID: "agent-2", TargetTokens: 25}

	cb.Submit(req1)
	cb.Submit(req2)

	// Step 5 iterations
	for i := 0; i < 5; i++ {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step %d failed: %v", i, err)
		}
		if res.ActiveSlots != 2 {
			t.Fatalf("ActiveSlots = %d, want 2", res.ActiveSlots)
		}
	}

	// Mid-flight: dynamically submit agent-3 (enters empty slot 2 immediately)
	req3 := &SubagentRequest{SessionID: "agent-3", TargetTokens: 5}
	if _, err := cb.Submit(req3); err != nil {
		t.Fatalf("Submit agent-3 failed: %v", err)
	}

	// Also submit agent-4: capacity is 3, so agent-4 enters waitingQueue
	req4 := &SubagentRequest{SessionID: "agent-4", TargetTokens: 8}
	if _, err := cb.Submit(req4); err != nil {
		t.Fatalf("Submit agent-4 failed: %v", err)
	}

	if cb.WaitingQueueLength() != 1 {
		t.Fatalf("WaitingQueueLength = %d, want 1", cb.WaitingQueueLength())
	}

	// Step 1 iteration: agent-1, agent-2, agent-3 all decode simultaneously
	res, err := cb.Step(ctx)
	if err != nil {
		t.Fatalf("Step failed: %v", err)
	}
	if res.ActiveSlots != 3 {
		t.Fatalf("ActiveSlots = %d, want 3", res.ActiveSlots)
	}

	slot1, _ := cb.GetSlot("agent-1")
	slot2, _ := cb.GetSlot("agent-2")
	slot3, _ := cb.GetSlot("agent-3")

	if slot1.TokensGenerated() != 6 || slot2.TokensGenerated() != 6 || slot3.TokensGenerated() != 1 {
		t.Fatalf("Tokens after dynamic insertion: 1=%d, 2=%d, 3=%d",
			slot1.TokensGenerated(), slot2.TokensGenerated(), slot3.TokensGenerated())
	}

	// Step until agent-3 completes (it had target 5, already did 1, so 4 more steps)
	for i := 0; i < 4; i++ {
		res, err = cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step failed: %v", err)
		}
	}

	// In the step where agent-3 completed, it should be retired and agent-4 promoted!
	if slot3.TokensGenerated() != 5 {
		t.Fatalf("agent-3 tokens = %d, want 5", slot3.TokensGenerated())
	}

	slot4, ok := cb.GetSlot("agent-4")
	if !ok {
		t.Fatalf("agent-4 was not promoted into active slots")
	}
	if slot4.State != SlotStateActiveDecode {
		t.Fatalf("slot4.State = %s, want %s", slot4.State, SlotStateActiveDecode)
	}

	// Step until all complete
	maxSteps := 50
	for i := 0; i < maxSteps && (cb.ActiveSlotCount() > 0 || cb.WaitingQueueLength() > 0); i++ {
		cb.Step(ctx)
	}

	for _, id := range []string{"agent-1", "agent-2", "agent-3", "agent-4"} {
		s, ok := cb.GetSlot(id)
		if !ok || s.TokensGenerated() != s.TargetTokens {
			t.Fatalf("agent %s not completed: ok=%v, tokens=%d, target=%d",
				id, ok, s.TokensGenerated(), s.TargetTokens)
		}
	}
}

// TestContinuousBatcher_8SubagentThroughputBand verifies aggregate throughput across 8 subagents
// exceeds >80 tok/s and operational intensity is maintained in the 50..150 FLOPs/byte band.
func TestContinuousBatcher_8SubagentThroughputBand(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 8

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx := context.Background()

	// Verify single-agent baseline throughput (~19 tok/s)
	singleTPS := cb.AggregateThroughput(1)
	if singleTPS < 18.0 || singleTPS > 20.0 {
		t.Fatalf("Single-agent throughput = %f, expected ~19 tok/s", singleTPS)
	}

	// Verify 8-subagent roofline aggregate throughput (> 80 tok/s)
	tps8 := cb.AggregateThroughput(8)
	if tps8 <= 80.0 {
		t.Fatalf("8-subagent throughput = %f tok/s, want > 80.0 tok/s", tps8)
	}

	// Verify 8-subagent operational intensity in the 50..150 FLOPs/byte band
	intensity8 := cb.OperationalIntensity(8)
	if intensity8 < 50.0 || intensity8 > 150.0 {
		t.Fatalf("8-subagent operational intensity = %f FLOPs/byte, want between 50 and 150", intensity8)
	}

	arithmeticIntensity8 := cb.ArithmeticIntensity(8)
	if arithmeticIntensity8 != intensity8 {
		t.Fatalf("ArithmeticIntensity (%f) != OperationalIntensity (%f)", arithmeticIntensity8, intensity8)
	}

	// Submit 8 concurrent subagents
	for i := 0; i < 8; i++ {
		req := &SubagentRequest{
			SessionID:    fmt.Sprintf("subagent-%d", i),
			PromptTokens: []int{100 + i, 200 + i},
			TargetTokens: 10,
		}
		if _, err := cb.Submit(req); err != nil {
			t.Fatalf("Submit subagent-%d failed: %v", i, err)
		}
	}

	if got := cb.ActiveSlotCount(); got != 8 {
		t.Fatalf("ActiveSlotCount = %d, want 8", got)
	}

	// Execute an iteration step with all 8 subagents active
	res, err := cb.Step(ctx)
	if err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	if res.ActiveSlots != 8 {
		t.Fatalf("res.ActiveSlots = %d, want 8", res.ActiveSlots)
	}
	if res.AggregateThroughput <= 80.0 {
		t.Fatalf("res.AggregateThroughput = %f, want > 80 tok/s", res.AggregateThroughput)
	}
	if res.OperationalIntensity < 50.0 || res.OperationalIntensity > 150.0 {
		t.Fatalf("res.OperationalIntensity = %f, want in [50, 150]", res.OperationalIntensity)
	}
	if res.TokensGenerated != 8 {
		t.Fatalf("res.TokensGenerated = %d, want 8", res.TokensGenerated)
	}

	// Complete all 8 subagents
	for cb.ActiveSlotCount() > 0 {
		if _, err := cb.Step(ctx); err != nil {
			t.Fatalf("Step failed: %v", err)
		}
	}

	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("subagent-%d", i)
		slot, ok := cb.GetSlot(id)
		if !ok || slot.TokensGenerated() != 10 {
			t.Fatalf("subagent %s did not complete 10 tokens", id)
		}
	}
}

// TestContinuousBatcher_ValidationAndErrors verifies edge cases and parameter checks.
func TestContinuousBatcher_ValidationAndErrors(t *testing.T) {
	// Invalid slot sizes
	if _, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 0}); err != nil {
		// Default takes over if MaxSlots is 0 in NewContinuousBatcher
	}
	if _, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: -1}); err != ErrInvalidSlots {
		t.Fatalf("expected ErrInvalidSlots for -1, got %v", err)
	}
	if _, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 33}); err != ErrInvalidSlots {
		t.Fatalf("expected ErrInvalidSlots for 33, got %v", err)
	}

	cb, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 2})
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	// Nil request
	if _, err := cb.Submit(nil); err != ErrNilRequest {
		t.Fatalf("expected ErrNilRequest, got %v", err)
	}

	// Invalid target tokens
	if _, err := cb.Submit(&SubagentRequest{TargetTokens: 0}); err != ErrInvalidTargetTokens {
		t.Fatalf("expected ErrInvalidTargetTokens, got %v", err)
	}

	// Submit valid
	id, err := cb.Submit(&SubagentRequest{SessionID: "sess-1", TargetTokens: 10})
	if err != nil || id != "sess-1" {
		t.Fatalf("Submit failed: %v", err)
	}

	// Duplicate session ID
	if _, err := cb.Submit(&SubagentRequest{SessionID: "sess-1", TargetTokens: 10}); err != ErrSessionAlreadyExists {
		t.Fatalf("expected ErrSessionAlreadyExists, got %v", err)
	}

	// Yield non-existent
	if err := cb.YieldSlot("unknown"); err != ErrSessionNotFound {
		t.Fatalf("expected ErrSessionNotFound, got %v", err)
	}

	// Resume non-yielded
	if err := cb.ResumeSlot("sess-1"); err != ErrSlotNotYielded {
		t.Fatalf("expected ErrSlotNotYielded, got %v", err)
	}

	// Yield active
	if err := cb.YieldSlot("sess-1"); err != nil {
		t.Fatalf("YieldSlot failed: %v", err)
	}

	// Double yield
	if err := cb.YieldSlot("sess-1"); err != ErrSlotNotActive {
		t.Fatalf("expected ErrSlotNotActive on double yield, got %v", err)
	}

	// Cancel session
	if err := cb.Cancel("sess-1"); err != nil {
		t.Fatalf("Cancel failed: %v", err)
	}

	// Cancel non-existent
	if err := cb.Cancel("sess-1"); err != ErrSessionNotFound {
		t.Fatalf("expected ErrSessionNotFound, got %v", err)
	}
}

// TestContinuousBatcher_IrregularTurnLengths tests multi-agent scheduling with irregular turn lengths (10..200 tokens).
func TestContinuousBatcher_IrregularTurnLengths(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 8

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx := context.Background()

	// 12 subagents with lengths ranging from 10 to 200 tokens
	turnLengths := []int{10, 45, 12, 200, 25, 80, 15, 120, 30, 150, 18, 90}
	sessions := make([]string, len(turnLengths))

	for i, length := range turnLengths {
		id := fmt.Sprintf("irregular-sub-%d", i)
		sessions[i] = id
		req := &SubagentRequest{
			SessionID:    id,
			PromptTokens: []int{1000 + i},
			TargetTokens: length,
		}
		if _, err := cb.Submit(req); err != nil {
			t.Fatalf("Submit(%s) failed: %v", id, err)
		}
	}

	// Step until all complete or max steps reached
	maxSteps := 300
	steps := 0
	for steps < maxSteps && (cb.ActiveSlotCount() > 0 || cb.WaitingQueueLength() > 0) {
		res, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step failed at %d: %v", steps, err)
		}
		if res.ActiveSlots > 8 {
			t.Fatalf("ActiveSlots %d exceeded capacity 8", res.ActiveSlots)
		}
		steps++
	}

	// Verify all completed
	for i, id := range sessions {
		slot, ok := cb.GetSlot(id)
		if !ok {
			t.Fatalf("session %s not found", id)
		}
		if slot.TokensGenerated() != turnLengths[i] {
			t.Fatalf("session %s generated %d tokens, want %d", id, slot.TokensGenerated(), turnLengths[i])
		}
	}
}

// TestContinuousBatcher_ContextCancellation verifies that Step cancels when ctx is canceled.
func TestContinuousBatcher_ContextCancellation(t *testing.T) {
	cb, err := NewContinuousBatcher()
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}
	defer cb.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = cb.Step(ctx)
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}
