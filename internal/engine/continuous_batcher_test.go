package engine

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
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

// TestContinuousBatcher_ClosePreservesBufferedToken verifies that Close() closes
// the token channel WITHOUT draining an already-buffered token. A reader must
// still observe the in-flight token before the channel reports closed.
func TestContinuousBatcher_ClosePreservesBufferedToken(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 1

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}

	ctx := context.Background()
	if _, err := cb.Submit(&SubagentRequest{
		SessionID:    "close-buffered",
		PromptTokens: []int{7},
		TargetTokens: 4,
	}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	if _, err := cb.Step(ctx); err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	slot, ok := cb.GetSlot("close-buffered")
	if !ok {
		t.Fatal("session close-buffered not found")
	}
	want := slot.LastToken

	if err := cb.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	select {
	case got, open := <-slot.Tokens():
		if !open {
			t.Fatal("Close() drained the buffered token and closed tokenCh without delivering it")
		}
		if got != want {
			t.Fatalf("buffered token = %d, want %d", got, want)
		}
	default:
		t.Fatal("Close() did not deliver the buffered token")
	}

	if _, open := <-slot.Tokens(); open {
		t.Fatal("tokenCh still open after Close(); Close() failed to close the channel")
	}
}

func TestContinuousBatcher_CloseTerminatesOwnedWork(t *testing.T) {
	cfg := DefaultContinuousBatcherConfig()
	cfg.MaxSlots = 3
	cfg.PrefillBudget = 1

	cb, err := NewContinuousBatcher(cfg)
	if err != nil {
		t.Fatalf("NewContinuousBatcher failed: %v", err)
	}

	requests := []*SubagentRequest{
		{SessionID: "prefilling", PromptTokens: []int{1, 2, 3}, TargetTokens: 4, ChunkedPrefill: true},
		{SessionID: "active", PromptTokens: []int{4}, TargetTokens: 4},
		{SessionID: "yielded", PromptTokens: []int{5}, TargetTokens: 4},
		{SessionID: "queued", PromptTokens: []int{6}, TargetTokens: 4},
	}
	for _, req := range requests {
		if _, err := cb.Submit(req); err != nil {
			t.Fatalf("Submit(%s) failed: %v", req.SessionID, err)
		}
	}
	if err := cb.YieldSlot("yielded"); err != nil {
		t.Fatalf("YieldSlot failed: %v", err)
	}

	wantStates := map[string]SlotState{
		"prefilling": SlotStatePrefilling,
		"active":     SlotStateActiveDecode,
		"yielded":    SlotStateYieldedIO,
	}
	owned := make(map[string]*Slot, len(wantStates))
	for id, want := range wantStates {
		slot, ok := cb.GetSlot(id)
		if !ok {
			t.Fatalf("GetSlot(%s) failed", id)
		}
		if slot.State != want {
			t.Fatalf("%s state = %s, want %s", id, slot.State, want)
		}
		owned[id] = slot
	}
	if got := cb.WaitingQueueLength(); got != 1 {
		t.Fatalf("WaitingQueueLength before close = %d, want 1", got)
	}

	if err := cb.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	if got := cb.WaitingQueueLength(); got != 0 {
		t.Fatalf("WaitingQueueLength after close = %d, want 0", got)
	}
	if got := cb.EmptySlotCount(); got != cfg.MaxSlots {
		t.Fatalf("EmptySlotCount after close = %d, want %d", got, cfg.MaxSlots)
	}
	if _, ok := cb.GetSlot("queued"); ok {
		t.Fatal("queued request was promoted after close")
	}
	for id, slot := range owned {
		if slot.State != SlotStateFinished {
			t.Errorf("%s state after close = %s, want %s", id, slot.State, SlotStateFinished)
		}
		if !errors.Is(slot.Err(), ErrBatcherClosed) {
			t.Errorf("%s error after close = %v, want ErrBatcherClosed", id, slot.Err())
		}
		select {
		case <-slot.Done():
		default:
			t.Errorf("%s Done still open after close", id)
		}
		if _, open := <-slot.Tokens(); open {
			t.Errorf("%s Tokens still open after close", id)
		}
		got, ok := cb.GetSlot(id)
		if !ok || got != slot {
			t.Errorf("GetSlot(%s) after close = (%p, %v), want completed slot %p", id, got, ok, slot)
		}
	}

	if err := cb.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}
	if got := cb.WaitingQueueLength(); got != 0 {
		t.Fatalf("WaitingQueueLength after second close = %d, want 0", got)
	}
	if _, err := cb.Submit(&SubagentRequest{SessionID: "late", TargetTokens: 1}); !errors.Is(err, ErrBatcherClosed) {
		t.Fatalf("Submit after close = %v, want ErrBatcherClosed", err)
	}
}

func TestContinuousBatcher_ModelSessionCleanup(t *testing.T) {
	newOwned := func(t *testing.T) (*ContinuousBatcher, *Slot, *atomic.Int32) {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 1
		cfg.Model = model.NewSynthetic(wireModelConfig())
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		if _, err := cb.Submit(&SubagentRequest{SessionID: "owned", PromptTokens: []int{1}, TargetTokens: 1}); err != nil {
			t.Fatalf("Submit failed: %v", err)
		}
		slot, ok := cb.GetSlot("owned")
		if !ok || slot.sess == nil {
			t.Fatal("model-backed slot has no owned session")
		}
		var releases atomic.Int32
		slot.sess.SetHostPathBypass("batcher-cleanup-test", func() { releases.Add(1) })
		return cb, slot, &releases
	}

	for _, tc := range []struct {
		name   string
		retire func(*ContinuousBatcher) error
	}{
		{name: "normal", retire: func(cb *ContinuousBatcher) error {
			_, err := cb.Step(context.Background())
			return err
		}},
		{name: "cancel", retire: func(cb *ContinuousBatcher) error { return cb.Cancel("owned") }},
		{name: "close", retire: func(cb *ContinuousBatcher) error { return cb.Close() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cb, slot, releases := newOwned(t)
			if err := tc.retire(cb); err != nil {
				t.Fatalf("retire failed: %v", err)
			}
			if got := releases.Load(); got != 1 {
				t.Fatalf("session release count = %d, want 1", got)
			}
			if slot.sess != nil {
				t.Fatal("terminal slot retains model session")
			}
			if err := cb.Close(); err != nil {
				t.Fatalf("Close after retirement failed: %v", err)
			}
			if err := cb.Close(); err != nil {
				t.Fatalf("repeated Close failed: %v", err)
			}
			if got := releases.Load(); got != 1 {
				t.Fatalf("session release count after repeated Close = %d, want 1", got)
			}
		})
	}

	t.Run("already-completed", func(t *testing.T) {
		cb, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 1})
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		sess := model.NewSynthetic(wireModelConfig()).NewSession()
		var releases atomic.Int32
		sess.SetHostPathBypass("batcher-cleanup-test", func() { releases.Add(1) })
		slot := &Slot{SessionID: "completed", State: SlotStateFinished, sess: sess}
		cb.completedMap[slot.SessionID] = slot

		if err := cb.Close(); err != nil {
			t.Fatalf("Close failed: %v", err)
		}
		if got := releases.Load(); got != 1 {
			t.Fatalf("completed session release count = %d, want 1", got)
		}
		if slot.sess != nil {
			t.Fatal("completed slot retains model session")
		}
	})
}

func TestContinuousBatcher_CloseCancelRace(t *testing.T) {
	mdl := model.NewSynthetic(wireModelConfig())
	for i := 0; i < 100; i++ {
		cb, err := NewContinuousBatcher(ContinuousBatcherConfig{MaxSlots: 1, Model: mdl})
		if err != nil {
			t.Fatalf("iteration %d: NewContinuousBatcher failed: %v", i, err)
		}
		if _, err := cb.Submit(&SubagentRequest{SessionID: "race", TargetTokens: 4}); err != nil {
			t.Fatalf("iteration %d: Submit failed: %v", i, err)
		}
		slot, ok := cb.GetSlot("race")
		if !ok {
			t.Fatalf("iteration %d: slot not found", i)
		}
		if slot.sess == nil {
			t.Fatalf("iteration %d: model-backed slot has no session", i)
		}
		var releases atomic.Int32
		slot.sess.SetHostPathBypass("batcher-race-test", func() { releases.Add(1) })

		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(2)
		var cancelErr error
		go func() {
			defer wg.Done()
			<-start
			cancelErr = cb.Cancel("race")
		}()
		go func() {
			defer wg.Done()
			<-start
			_ = cb.Close()
		}()
		close(start)
		wg.Wait()

		if cancelErr != nil && !errors.Is(cancelErr, ErrBatcherClosed) {
			t.Fatalf("iteration %d: Cancel error = %v, want nil or ErrBatcherClosed", i, cancelErr)
		}
		select {
		case <-slot.Done():
		default:
			t.Fatalf("iteration %d: Done still open", i)
		}
		if _, open := <-slot.Tokens(); open {
			t.Fatalf("iteration %d: Tokens still open", i)
		}
		if got := cb.EmptySlotCount(); got != 1 {
			t.Fatalf("iteration %d: EmptySlotCount = %d, want 1", i, got)
		}
		if got := releases.Load(); got != 1 {
			t.Fatalf("iteration %d: session release count = %d, want 1", i, got)
		}
		if slot.sess != nil {
			t.Fatalf("iteration %d: terminal slot retains model session", i)
		}
	}
}

// TestStepPhaseBoundsDecodeStarvationByWaitingServedRatio pins the TGI-style
// waiting_served_ratio / max_waiting_tokens refill gate and its zero-config
// prefill-first default.
func TestStepPhaseBoundsDecodeStarvationByWaitingServedRatio(t *testing.T) {
	t.Parallel()
	run := func(t *testing.T, ratio float64, maxWait, steps int) []BatchPhase {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 8
		cfg.PrefillBudget = 2
		cfg.WaitingServedRatio = ratio
		cfg.MaxWaitingDecodeSteps = maxWait
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		for i := 0; i < 4; i++ {
			if _, err := cb.Submit(&SubagentRequest{SessionID: fmt.Sprintf("dec-%d", i), PromptTokens: []int{i + 1}, TargetTokens: 100}); err != nil {
				t.Fatalf("Submit decoder %d: %v", i, err)
			}
		}
		if _, err := cb.Submit(&SubagentRequest{SessionID: "waiting", PromptTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, TargetTokens: 1, ChunkedPrefill: true}); err != nil {
			t.Fatalf("Submit waiting: %v", err)
		}
		phases := make([]BatchPhase, 0, steps)
		for i := 0; i < steps; i++ {
			res, err := cb.StepPhase(context.Background())
			if err != nil {
				t.Fatalf("StepPhase %d: %v", i, err)
			}
			if res.Phase == PhasePrefill && res.PrefillTokens != 2 {
				t.Fatalf("step %d PrefillTokens = %d, want 2", i, res.PrefillTokens)
			}
			phases = append(phases, res.Phase)
		}
		return phases
	}
	equal := func(got, want []BatchPhase) bool {
		if len(got) != len(want) {
			return false
		}
		for i := range got {
			if got[i] != want[i] {
				return false
			}
		}
		return true
	}
	D, P := PhaseDecode, PhasePrefill

	// ratio 1.0, 4 decoding vs 1 prefilling: decode until the counter forces exactly one prefill.
	if got, want := run(t, 1.0, 3, 8), []BatchPhase{D, D, D, P, D, D, D, P}; !equal(got, want) {
		t.Fatalf("bounded phases = %v, want %v", got, want)
	}
	// Zero config reproduces prefill-first: 12 prompt tokens / budget 2 = 6 prefill steps.
	if got, want := run(t, 0, 0, 7), []BatchPhase{P, P, P, P, P, P, D}; !equal(got, want) {
		t.Fatalf("zero-config phases = %v, want %v", got, want)
	}
	// Enough waiting slots satisfies the ratio without waiting on the counter.
	if got, want := run(t, 0.25, 100, 2), []BatchPhase{P, P}; !equal(got, want) {
		t.Fatalf("ratio-satisfied phases = %v, want %v", got, want)
	}
}
