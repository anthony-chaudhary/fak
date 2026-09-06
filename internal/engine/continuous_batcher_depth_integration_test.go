package engine

import (
	"context"
	"testing"
)

// TestHeterogeneousRecurrentDepthIntegration verifies continuous batcher scheduling,
// bounded execution, cancellation isolation, and queue promotion for concurrent requests
// with heterogeneous recurrent execution depths (e.g. depth 2 vs depth 4).
func TestHeterogeneousRecurrentDepthIntegration(t *testing.T) {
	ctx := context.Background()

	t.Run("bounded_heterogeneous_depth_scheduling", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// 1. Submit two concurrent requests with distinct ExecutionDepth limits:
		// reqA with ExecutionDepth=2, reqB with ExecutionDepth=4.
		reqA := &SubagentRequest{
			SessionID:      "subagent-depth-2",
			PromptTokens:   []int{101, 102},
			ExecutionDepth: 2,
		}
		reqB := &SubagentRequest{
			SessionID:      "subagent-depth-4",
			PromptTokens:   []int{201, 202, 203},
			ExecutionDepth: 4,
		}

		idA, err := cb.Submit(reqA)
		if err != nil {
			t.Fatalf("Submit(reqA) failed: %v", err)
		}
		if idA != "subagent-depth-2" {
			t.Fatalf("Submit(reqA) returned %q, want %q", idA, "subagent-depth-2")
		}

		idB, err := cb.Submit(reqB)
		if err != nil {
			t.Fatalf("Submit(reqB) failed: %v", err)
		}
		if idB != "subagent-depth-4" {
			t.Fatalf("Submit(reqB) returned %q, want %q", idB, "subagent-depth-4")
		}

		// Both requests must be admitted immediately into active decode slots
		if got := cb.ActiveSlotCount(); got != 2 {
			t.Fatalf("ActiveSlotCount = %d, want 2", got)
		}

		slotA, ok := cb.GetSlot("subagent-depth-2")
		if !ok {
			t.Fatalf("session subagent-depth-2 not found in batcher")
		}
		slotB, ok := cb.GetSlot("subagent-depth-4")
		if !ok {
			t.Fatalf("session subagent-depth-4 not found in batcher")
		}

		if slotA.CurrentDepth != 0 || slotB.CurrentDepth != 0 {
			t.Fatalf("initial depths must be 0, got A=%d, B=%d", slotA.CurrentDepth, slotB.CurrentDepth)
		}
		if slotA.ExecutionDepth != 2 {
			t.Fatalf("slotA.ExecutionDepth = %d, want 2", slotA.ExecutionDepth)
		}
		if slotB.ExecutionDepth != 4 {
			t.Fatalf("slotB.ExecutionDepth = %d, want 4", slotB.ExecutionDepth)
		}

		// 2. Exercise Step(ctx) iteration 1:
		// Both reqA and reqB execute their first recurrent pass.
		res1, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 1 failed: %v", err)
		}
		if res1.ActiveSlots != 2 {
			t.Fatalf("Step 1 ActiveSlots = %d, want 2", res1.ActiveSlots)
		}
		if res1.TokensGenerated != 2 {
			t.Fatalf("Step 1 TokensGenerated = %d, want 2", res1.TokensGenerated)
		}
		if slotA.CurrentDepth != 1 || slotB.CurrentDepth != 1 {
			t.Fatalf("after Step 1: want depth 1 each, got A=%d, B=%d", slotA.CurrentDepth, slotB.CurrentDepth)
		}
		if slotA.State != SlotStateActiveDecode || slotB.State != SlotStateActiveDecode {
			t.Fatalf("after Step 1: both slots must be active, got A=%s, B=%s", slotA.State, slotB.State)
		}
		if len(res1.RetiredSessionIDs) != 0 {
			t.Fatalf("Step 1: no sessions should retire, got %v", res1.RetiredSessionIDs)
		}

		// 3. Exercise Step(ctx) iteration 2:
		// reqA reaches depth 2 (its depth limit) and must complete.
		// reqB reaches depth 2 (limit 4) and must remain active.
		res2, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 2 failed: %v", err)
		}
		if res2.ActiveSlots != 2 {
			t.Fatalf("Step 2 ActiveSlots during step = %d, want 2", res2.ActiveSlots)
		}
		if slotA.CurrentDepth != 2 {
			t.Fatalf("after Step 2: slotA depth = %d, want 2", slotA.CurrentDepth)
		}
		if slotB.CurrentDepth != 2 {
			t.Fatalf("after Step 2: slotB depth = %d, want 2", slotB.CurrentDepth)
		}

		// Proves bounded completion: reqA completes its recurrent passes when its depth limit (2) is reached
		if slotA.State != SlotStateFinished {
			t.Fatalf("slotA.State after depth 2 = %s, want %s", slotA.State, SlotStateFinished)
		}
		select {
		case <-slotA.Done():
		default:
			t.Fatalf("slotA.Done() must be closed after reaching depth limit 2")
		}
		if slotA.Err() != nil {
			t.Fatalf("slotA.Err() = %v, want nil", slotA.Err())
		}
		if len(res2.RetiredSessionIDs) != 1 || res2.RetiredSessionIDs[0] != "subagent-depth-2" {
			t.Fatalf("Step 2: RetiredSessionIDs = %v, want [subagent-depth-2]", res2.RetiredSessionIDs)
		}

		// reqB must continue iterating independently (depth 2 of 4 reached)
		if slotB.State != SlotStateActiveDecode {
			t.Fatalf("slotB.State after Step 2 = %s, want %s", slotB.State, SlotStateActiveDecode)
		}
		select {
		case <-slotB.Done():
			t.Fatalf("slotB.Done() must NOT be closed at depth 2 (limit 4)")
		default:
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("after Step 2 retire: ActiveSlotCount = %d, want 1 (reqB only)", got)
		}

		// 4. Exercise Step(ctx) iteration 3:
		// reqB iterates independently to depth 3; reqA remains retired.
		res3, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 3 failed: %v", err)
		}
		if res3.ActiveSlots != 1 {
			t.Fatalf("Step 3 ActiveSlots = %d, want 1", res3.ActiveSlots)
		}
		if slotB.CurrentDepth != 3 {
			t.Fatalf("after Step 3: slotB depth = %d, want 3", slotB.CurrentDepth)
		}
		if slotB.State != SlotStateActiveDecode {
			t.Fatalf("after Step 3: slotB must still be active, got %s", slotB.State)
		}
		if len(res3.RetiredSessionIDs) != 0 {
			t.Fatalf("Step 3: no sessions should retire, got %v", res3.RetiredSessionIDs)
		}

		// 5. Exercise Step(ctx) iteration 4:
		// reqB reaches depth 4 and completes.
		res4, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 4 failed: %v", err)
		}
		if slotB.CurrentDepth != 4 {
			t.Fatalf("after Step 4: slotB depth = %d, want 4", slotB.CurrentDepth)
		}
		if slotB.State != SlotStateFinished {
			t.Fatalf("slotB.State after depth 4 = %s, want %s", slotB.State, SlotStateFinished)
		}
		select {
		case <-slotB.Done():
		default:
			t.Fatalf("slotB.Done() must be closed after reaching depth limit 4")
		}
		if slotB.Err() != nil {
			t.Fatalf("slotB.Err() = %v, want nil", slotB.Err())
		}
		if len(res4.RetiredSessionIDs) != 1 || res4.RetiredSessionIDs[0] != "subagent-depth-4" {
			t.Fatalf("Step 4: RetiredSessionIDs = %v, want [subagent-depth-4]", res4.RetiredSessionIDs)
		}
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("after Step 4: ActiveSlotCount = %d, want 0", got)
		}

		// 6. Verify token streams and generated counts
		if got, want := slotA.TokensGenerated(), 2; got != want {
			t.Fatalf("slotA tokens = %d, want %d", got, want)
		}
		if got, want := slotB.TokensGenerated(), 4; got != want {
			t.Fatalf("slotB tokens = %d, want %d", got, want)
		}

		tokensA := make([]int, 0, 2)
		for tok := range slotA.Tokens() {
			tokensA = append(tokensA, tok)
		}
		if len(tokensA) != 2 {
			t.Fatalf("drained slotA tokens count = %d, want 2", len(tokensA))
		}

		tokensB := make([]int, 0, 4)
		for tok := range slotB.Tokens() {
			tokensB = append(tokensB, tok)
		}
		if len(tokensB) != 4 {
			t.Fatalf("drained slotB tokens count = %d, want 4", len(tokensB))
		}

		if got, want := cb.TotalTokensGenerated(), int64(6); got != want {
			t.Fatalf("cb.TotalTokensGenerated = %d, want %d", got, want)
		}
	})

	t.Run("cancellation_and_independent_state_across_heterogeneous_depths", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// Submit reqCancel (depth=3) and reqSurvivor (depth=5)
		reqCancel := &SubagentRequest{
			SessionID:      "sub-cancel",
			PromptTokens:   []int{1, 2, 3},
			ExecutionDepth: 3,
		}
		reqSurvivor := &SubagentRequest{
			SessionID:      "sub-survivor",
			PromptTokens:   []int{4, 5, 6},
			ExecutionDepth: 5,
		}

		if _, err := cb.Submit(reqCancel); err != nil {
			t.Fatalf("Submit(reqCancel) failed: %v", err)
		}
		if _, err := cb.Submit(reqSurvivor); err != nil {
			t.Fatalf("Submit(reqSurvivor) failed: %v", err)
		}

		// Step 1: both reach depth 1
		if _, err := cb.Step(ctx); err != nil {
			t.Fatalf("Step 1 failed: %v", err)
		}
		slotCancel, ok := cb.GetSlot("sub-cancel")
		if !ok {
			t.Fatalf("sub-cancel not found")
		}
		slotSurvivor, ok := cb.GetSlot("sub-survivor")
		if !ok {
			t.Fatalf("sub-survivor not found")
		}

		if slotCancel.CurrentDepth != 1 || slotSurvivor.CurrentDepth != 1 {
			t.Fatalf("after Step 1: want depth 1 each, got cancel=%d, survivor=%d",
				slotCancel.CurrentDepth, slotSurvivor.CurrentDepth)
		}

		// Cancel reqCancel mid-flight at depth 1
		if err := cb.Cancel("sub-cancel"); err != nil {
			t.Fatalf("Cancel(sub-cancel) failed: %v", err)
		}

		// Verify cancelled slot state
		if slotCancel.State != SlotStateFinished {
			t.Fatalf("slotCancel.State = %s, want %s", slotCancel.State, SlotStateFinished)
		}
		if slotCancel.Err() != context.Canceled {
			t.Fatalf("slotCancel.Err() = %v, want context.Canceled", slotCancel.Err())
		}
		select {
		case <-slotCancel.Done():
		default:
			t.Fatalf("slotCancel.Done() channel must be closed on cancellation")
		}
		if slotCancel.CurrentDepth != 1 {
			t.Fatalf("slotCancel.CurrentDepth must freeze at 1, got %d", slotCancel.CurrentDepth)
		}

		// Verify reqSurvivor is completely unaffected by reqCancel's cancellation
		if slotSurvivor.State != SlotStateActiveDecode {
			t.Fatalf("slotSurvivor.State = %s, want %s", slotSurvivor.State, SlotStateActiveDecode)
		}
		if slotSurvivor.Err() != nil {
			t.Fatalf("slotSurvivor.Err() = %v, want nil", slotSurvivor.Err())
		}
		select {
		case <-slotSurvivor.Done():
			t.Fatalf("slotSurvivor.Done() must NOT be closed")
		default:
		}
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount after cancel = %d, want 1", got)
		}

		// Continue stepping: reqSurvivor iterates independently to completion (depth 2..5)
		for step := 2; step <= 5; step++ {
			res, err := cb.Step(ctx)
			if err != nil {
				t.Fatalf("Step %d failed: %v", step, err)
			}
			if res.ActiveSlots != 1 {
				t.Fatalf("Step %d ActiveSlots = %d, want 1", step, res.ActiveSlots)
			}
			if slotSurvivor.CurrentDepth != step {
				t.Fatalf("Step %d: slotSurvivor depth = %d, want %d", step, slotSurvivor.CurrentDepth, step)
			}
		}

		// reqSurvivor reaches depth 5 and finishes cleanly
		if slotSurvivor.State != SlotStateFinished {
			t.Fatalf("slotSurvivor.State after depth 5 = %s, want %s", slotSurvivor.State, SlotStateFinished)
		}
		if slotSurvivor.Err() != nil {
			t.Fatalf("slotSurvivor.Err() = %v, want nil", slotSurvivor.Err())
		}
		select {
		case <-slotSurvivor.Done():
		default:
			t.Fatalf("slotSurvivor.Done() must be closed after reaching depth limit 5")
		}
		if slotSurvivor.TokensGenerated() != 5 {
			t.Fatalf("slotSurvivor TokensGenerated = %d, want 5", slotSurvivor.TokensGenerated())
		}
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount after survivor completion = %d, want 0", got)
		}
	})

	t.Run("heterogeneous_mixed_recurrent_and_standard_decode", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// reqRec has explicit recurrent depth limit of 3
		reqRec := &SubagentRequest{
			SessionID:      "sub-recurrent-3",
			PromptTokens:   []int{10, 20},
			ExecutionDepth: 3,
		}
		// reqStd is standard non-recurrent decode targeting 6 tokens
		reqStd := &SubagentRequest{
			SessionID:    "sub-standard-6",
			PromptTokens: []int{30, 40},
			TargetTokens: 6,
		}

		if _, err := cb.Submit(reqRec); err != nil {
			t.Fatalf("Submit(reqRec) failed: %v", err)
		}
		if _, err := cb.Submit(reqStd); err != nil {
			t.Fatalf("Submit(reqStd) failed: %v", err)
		}

		// Steps 1..3: both generate tokens
		for i := 1; i <= 3; i++ {
			if _, err := cb.Step(ctx); err != nil {
				t.Fatalf("Step %d failed: %v", i, err)
			}
		}

		slotRec, _ := cb.GetSlot("sub-recurrent-3")
		slotStd, _ := cb.GetSlot("sub-standard-6")

		// Recurrent request reaches depth 3 and finishes
		if slotRec.State != SlotStateFinished {
			t.Fatalf("slotRec.State = %s, want %s", slotRec.State, SlotStateFinished)
		}
		if slotRec.TokensGenerated() != 3 {
			t.Fatalf("slotRec tokens = %d, want 3", slotRec.TokensGenerated())
		}

		// Standard request is at 3 of 6 tokens, continues active
		if slotStd.State != SlotStateActiveDecode {
			t.Fatalf("slotStd.State = %s, want %s", slotStd.State, SlotStateActiveDecode)
		}
		if slotStd.TokensGenerated() != 3 {
			t.Fatalf("slotStd tokens = %d, want 3", slotStd.TokensGenerated())
		}

		// Steps 4..6: standard request finishes alone
		for i := 4; i <= 6; i++ {
			if _, err := cb.Step(ctx); err != nil {
				t.Fatalf("Step %d failed: %v", i, err)
			}
		}

		if slotStd.State != SlotStateFinished {
			t.Fatalf("slotStd.State after 6 tokens = %s, want %s", slotStd.State, SlotStateFinished)
		}
		if slotStd.TokensGenerated() != 6 {
			t.Fatalf("slotStd tokens = %d, want 6", slotStd.TokensGenerated())
		}
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount = %d, want 0", got)
		}
	})

	t.Run("waiting_queue_promotes_on_recurrent_depth_completion", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 1 // Only 1 active slot allowed

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		req1 := &SubagentRequest{
			SessionID:      "sub-depth-2",
			PromptTokens:   []int{1},
			ExecutionDepth: 2,
		}
		req2 := &SubagentRequest{
			SessionID:      "sub-depth-3",
			PromptTokens:   []int{2},
			ExecutionDepth: 3,
		}

		if _, err := cb.Submit(req1); err != nil {
			t.Fatalf("Submit(req1) failed: %v", err)
		}
		if _, err := cb.Submit(req2); err != nil {
			t.Fatalf("Submit(req2) failed: %v", err)
		}

		// req1 admitted, req2 queued
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount = %d, want 1", got)
		}
		if got := cb.WaitingQueueLength(); got != 1 {
			t.Fatalf("WaitingQueueLength = %d, want 1", got)
		}

		// Step 1: req1 at depth 1
		if _, err := cb.Step(ctx); err != nil {
			t.Fatalf("Step 1 failed: %v", err)
		}
		if got := cb.WaitingQueueLength(); got != 1 {
			t.Fatalf("WaitingQueueLength after step 1 = %d, want 1", got)
		}

		// Step 2: req1 reaches depth 2, completes, and req2 is promoted immediately!
		res2, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 2 failed: %v", err)
		}
		if len(res2.RetiredSessionIDs) != 1 || res2.RetiredSessionIDs[0] != "sub-depth-2" {
			t.Fatalf("Step 2 retired = %v, want [sub-depth-2]", res2.RetiredSessionIDs)
		}
		if len(res2.PromotedSessionIDs) != 1 || res2.PromotedSessionIDs[0] != "sub-depth-3" {
			t.Fatalf("Step 2 promoted = %v, want [sub-depth-3]", res2.PromotedSessionIDs)
		}
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength after promotion = %d, want 0", got)
		}

		slot2, ok := cb.GetSlot("sub-depth-3")
		if !ok {
			t.Fatalf("sub-depth-3 not found after promotion")
		}
		if slot2.State != SlotStateActiveDecode {
			t.Fatalf("slot2.State = %s, want %s", slot2.State, SlotStateActiveDecode)
		}

		// Steps 3, 4, 5: req2 steps to depth 3 and completes
		for i := 1; i <= 3; i++ {
			if _, err := cb.Step(ctx); err != nil {
				t.Fatalf("req2 step %d failed: %v", i, err)
			}
		}

		if slot2.State != SlotStateFinished {
			t.Fatalf("slot2.State = %s, want %s", slot2.State, SlotStateFinished)
		}
		if slot2.CurrentDepth != 3 {
			t.Fatalf("slot2.CurrentDepth = %d, want 3", slot2.CurrentDepth)
		}
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount = %d, want 0", got)
		}
	})
}
