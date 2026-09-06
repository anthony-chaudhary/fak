package engine

import (
	"context"
	"reflect"
	"testing"
)

// TestRecurrentBatchIsolationIntegration verifies that concurrent fixed-depth recurrent requests
// make interleaved progress through the ContinuousBatcher without cross-request state contamination,
// that their prompt and generated token histories remain strictly isolated, and that slot states
// and KV cache memory transition correctly upon individual completion.
func TestRecurrentBatchIsolationIntegration(t *testing.T) {
	ctx := context.Background()

	t.Run("heterogeneous_fixed_depth_interleaved_isolation", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4
		cfg.MaxKVCacheBytes = 50_000
		cfg.KVBytesPerToken = 100

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		// Construct two distinct fixed-depth recurrent requests with distinct prompts and targets:
		// session-A: depth=2, prompt length=3, target=5
		// session-B: depth=3, prompt length=5, target=8
		promptA := []int{101, 102, 103}
		promptB := []int{201, 202, 203, 204, 205}

		reqA := &SubagentRequest{
			SessionID:      "session-A",
			PromptTokens:   append([]int(nil), promptA...),
			TargetTokens:   5,
			ExecutionDepth: 2,
		}
		reqB := &SubagentRequest{
			SessionID:      "session-B",
			PromptTokens:   append([]int(nil), promptB...),
			TargetTokens:   8,
			ExecutionDepth: 3,
		}

		// Expected KV cache footprint:
		// reqA: (3 + 5) * 100 * 2 = 1,600 bytes
		// reqB: (5 + 8) * 100 * 3 = 3,900 bytes
		// Total: 5,500 bytes <= 50,000 budget
		idA, err := cb.Submit(reqA)
		if err != nil {
			t.Fatalf("Submit(reqA) failed: %v", err)
		}
		if idA != "session-A" {
			t.Fatalf("Submit(reqA) id = %q, want session-A", idA)
		}

		idB, err := cb.Submit(reqB)
		if err != nil {
			t.Fatalf("Submit(reqB) failed: %v", err)
		}
		if idB != "session-B" {
			t.Fatalf("Submit(reqB) id = %q, want session-B", idB)
		}

		// Verify initial admission into concurrent active decode slots
		if got := cb.ActiveSlotCount(); got != 2 {
			t.Fatalf("ActiveSlotCount = %d, want 2", got)
		}
		if got := cb.WaitingQueueLength(); got != 0 {
			t.Fatalf("WaitingQueueLength = %d, want 0", got)
		}
		if got, want := cb.CurrentKVCacheBytes(), int64(5500); got != want {
			t.Fatalf("CurrentKVCacheBytes = %d, want %d", got, want)
		}

		slotA, okA := cb.GetSlot("session-A")
		if !okA {
			t.Fatalf("GetSlot(session-A) not found")
		}
		slotB, okB := cb.GetSlot("session-B")
		if !okB {
			t.Fatalf("GetSlot(session-B) not found")
		}

		if slotA.Index == slotB.Index {
			t.Fatalf("slotA and slotB must occupy distinct slot indices, got both %d", slotA.Index)
		}
		if slotA.State != SlotStateActiveDecode || slotB.State != SlotStateActiveDecode {
			t.Fatalf("both slots must be ActiveDecode, got A=%s, B=%s", slotA.State, slotB.State)
		}
		if !reflect.DeepEqual(slotA.PromptTokens, promptA) {
			t.Fatalf("slotA.PromptTokens = %v, want %v", slotA.PromptTokens, promptA)
		}
		if !reflect.DeepEqual(slotB.PromptTokens, promptB) {
			t.Fatalf("slotB.PromptTokens = %v, want %v", slotB.PromptTokens, promptB)
		}

		// --- Step 1: Both sessions execute pass 1 concurrently ---
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
		if len(res1.RetiredSessionIDs) != 0 {
			t.Fatalf("Step 1 RetiredSessionIDs = %v, want empty", res1.RetiredSessionIDs)
		}

		tokA1, okA1 := res1.GeneratedTokens["session-A"]
		tokB1, okB1 := res1.GeneratedTokens["session-B"]
		if !okA1 || !okB1 {
			t.Fatalf("Step 1 generated tokens missing session: got %v", res1.GeneratedTokens)
		}

		// Verify streaming channel outputs
		select {
		case tok := <-slotA.Tokens():
			if tok != tokA1 {
				t.Fatalf("slotA token channel received %d, want %d", tok, tokA1)
			}
		default:
			t.Fatalf("slotA token channel had no token available after Step 1")
		}
		select {
		case tok := <-slotB.Tokens():
			if tok != tokB1 {
				t.Fatalf("slotB token channel received %d, want %d", tok, tokB1)
			}
		default:
			t.Fatalf("slotB token channel had no token available after Step 1")
		}

		// Both sessions advance to depth 1 and remain active
		if slotA.CurrentDepth != 1 || slotB.CurrentDepth != 1 {
			t.Fatalf("Step 1 depths: want A=1, B=1; got A=%d, B=%d", slotA.CurrentDepth, slotB.CurrentDepth)
		}
		if slotA.State != SlotStateActiveDecode || slotB.State != SlotStateActiveDecode {
			t.Fatalf("Step 1 states: want ActiveDecode; got A=%s, B=%s", slotA.State, slotB.State)
		}

		// Verify prompt isolation after step 1
		if !reflect.DeepEqual(slotA.PromptTokens, promptA) {
			t.Fatalf("slotA prompt corrupted after Step 1: got %v, want %v", slotA.PromptTokens, promptA)
		}
		if !reflect.DeepEqual(slotB.PromptTokens, promptB) {
			t.Fatalf("slotB prompt corrupted after Step 1: got %v, want %v", slotB.PromptTokens, promptB)
		}

		// --- Step 2: Both sessions execute pass 2; session-A completes, session-B remains active ---
		res2, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 2 failed: %v", err)
		}
		if res2.ActiveSlots != 2 {
			t.Fatalf("Step 2 ActiveSlots during step = %d, want 2", res2.ActiveSlots)
		}
		if res2.TokensGenerated != 2 {
			t.Fatalf("Step 2 TokensGenerated = %d, want 2", res2.TokensGenerated)
		}

		tokA2, okA2 := res2.GeneratedTokens["session-A"]
		tokB2, okB2 := res2.GeneratedTokens["session-B"]
		if !okA2 || !okB2 {
			t.Fatalf("Step 2 generated tokens missing session: got %v", res2.GeneratedTokens)
		}

		select {
		case tok := <-slotA.Tokens():
			if tok != tokA2 {
				t.Fatalf("slotA token channel received %d, want %d", tok, tokA2)
			}
		default:
			t.Fatalf("slotA token channel had no token available after Step 2")
		}
		select {
		case tok := <-slotB.Tokens():
			if tok != tokB2 {
				t.Fatalf("slotB token channel received %d, want %d", tok, tokB2)
			}
		default:
			t.Fatalf("slotB token channel had no token available after Step 2")
		}

		// session-A has reached ExecutionDepth=2 and transitions to Finished
		if slotA.CurrentDepth != 2 {
			t.Fatalf("slotA CurrentDepth = %d, want 2", slotA.CurrentDepth)
		}
		if slotA.State != SlotStateFinished {
			t.Fatalf("slotA State = %s, want %s", slotA.State, SlotStateFinished)
		}
		select {
		case <-slotA.Done():
		default:
			t.Fatalf("slotA Done channel must be closed after reaching depth 2")
		}
		if slotA.Err() != nil {
			t.Fatalf("slotA Err = %v, want nil", slotA.Err())
		}
		if len(res2.RetiredSessionIDs) != 1 || res2.RetiredSessionIDs[0] != "session-A" {
			t.Fatalf("Step 2 RetiredSessionIDs = %v, want [session-A]", res2.RetiredSessionIDs)
		}

		// session-B has reached depth 2 of 3 and remains ActiveDecode
		if slotB.CurrentDepth != 2 {
			t.Fatalf("slotB CurrentDepth = %d, want 2", slotB.CurrentDepth)
		}
		if slotB.State != SlotStateActiveDecode {
			t.Fatalf("slotB State = %s, want %s", slotB.State, SlotStateActiveDecode)
		}
		select {
		case <-slotB.Done():
			t.Fatalf("slotB Done channel must NOT be closed at depth 2 (limit 3)")
		default:
		}

		// KV cache memory of session-A (1,600 bytes) is freed; session-B (3,900 bytes) remains
		if got := cb.ActiveSlotCount(); got != 1 {
			t.Fatalf("ActiveSlotCount after session-A retired = %d, want 1", got)
		}
		if got, want := cb.CurrentKVCacheBytes(), int64(3900); got != want {
			t.Fatalf("CurrentKVCacheBytes after session-A retired = %d, want %d", got, want)
		}

		// --- Step 3: session-B executes pass 3 alone and completes ---
		res3, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 3 failed: %v", err)
		}
		if res3.ActiveSlots != 1 {
			t.Fatalf("Step 3 ActiveSlots = %d, want 1", res3.ActiveSlots)
		}
		if res3.TokensGenerated != 1 {
			t.Fatalf("Step 3 TokensGenerated = %d, want 1", res3.TokensGenerated)
		}

		tokB3, okB3 := res3.GeneratedTokens["session-B"]
		if !okB3 {
			t.Fatalf("Step 3 generated tokens missing session-B: got %v", res3.GeneratedTokens)
		}
		if _, okA3 := res3.GeneratedTokens["session-A"]; okA3 {
			t.Fatalf("retired session-A must not generate tokens in Step 3")
		}

		select {
		case tok := <-slotB.Tokens():
			if tok != tokB3 {
				t.Fatalf("slotB token channel received %d, want %d", tok, tokB3)
			}
		default:
			t.Fatalf("slotB token channel had no token available after Step 3")
		}

		// session-B has reached ExecutionDepth=3 and transitions to Finished
		if slotB.CurrentDepth != 3 {
			t.Fatalf("slotB CurrentDepth = %d, want 3", slotB.CurrentDepth)
		}
		if slotB.State != SlotStateFinished {
			t.Fatalf("slotB State = %s, want %s", slotB.State, SlotStateFinished)
		}
		select {
		case <-slotB.Done():
		default:
			t.Fatalf("slotB Done channel must be closed after reaching depth 3")
		}
		if slotB.Err() != nil {
			t.Fatalf("slotB Err = %v, want nil", slotB.Err())
		}
		if len(res3.RetiredSessionIDs) != 1 || res3.RetiredSessionIDs[0] != "session-B" {
			t.Fatalf("Step 3 RetiredSessionIDs = %v, want [session-B]", res3.RetiredSessionIDs)
		}

		// All slots free and KV cache fully drained
		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount after session-B retired = %d, want 0", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 0 {
			t.Fatalf("CurrentKVCacheBytes after all retired = %d, want 0", got)
		}

		// --- Step 4: Empty step advances cleanly without error ---
		res4, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 4 failed: %v", err)
		}
		if res4.ActiveSlots != 0 || res4.TokensGenerated != 0 {
			t.Fatalf("Step 4 on idle batcher: ActiveSlots=%d, TokensGenerated=%d", res4.ActiveSlots, res4.TokensGenerated)
		}

		// --- Verification of final isolated states ---
		completedA, okA := cb.GetSlot("session-A")
		if !okA {
			t.Fatalf("GetSlot(session-A) failed to find completed slot")
		}
		completedB, okB := cb.GetSlot("session-B")
		if !okB {
			t.Fatalf("GetSlot(session-B) failed to find completed slot")
		}

		if completedA.State != SlotStateFinished || completedB.State != SlotStateFinished {
			t.Fatalf("final slot states must be Finished, got A=%s, B=%s", completedA.State, completedB.State)
		}
		if completedA.TargetTokens != 5 || completedB.TargetTokens != 8 {
			t.Fatalf("target tokens corrupted: got A=%d, B=%d", completedA.TargetTokens, completedB.TargetTokens)
		}
		if completedA.ExecutionDepth != 2 || completedB.ExecutionDepth != 3 {
			t.Fatalf("execution depths corrupted: got A=%d, B=%d", completedA.ExecutionDepth, completedB.ExecutionDepth)
		}
		if completedA.CurrentDepth != 2 || completedB.CurrentDepth != 3 {
			t.Fatalf("current depths corrupted: got A=%d, B=%d", completedA.CurrentDepth, completedB.CurrentDepth)
		}
		if completedA.RecurrentPasses != 2 || completedB.RecurrentPasses != 3 {
			t.Fatalf("recurrent passes corrupted: got A=%d, B=%d", completedA.RecurrentPasses, completedB.RecurrentPasses)
		}

		// Verify prompt tokens never suffered cross-contamination
		if !reflect.DeepEqual(completedA.PromptTokens, promptA) {
			t.Fatalf("final session-A PromptTokens = %v, want %v", completedA.PromptTokens, promptA)
		}
		if !reflect.DeepEqual(completedB.PromptTokens, promptB) {
			t.Fatalf("final session-B PromptTokens = %v, want %v", completedB.PromptTokens, promptB)
		}

		// Verify output token isolation
		expectedTokensA := []int{tokA1, tokA2}
		expectedTokensB := []int{tokB1, tokB2, tokB3}
		if !reflect.DeepEqual(completedA.GeneratedTokens, expectedTokensA) {
			t.Fatalf("session-A GeneratedTokens = %v, want %v", completedA.GeneratedTokens, expectedTokensA)
		}
		if !reflect.DeepEqual(completedB.GeneratedTokens, expectedTokensB) {
			t.Fatalf("session-B GeneratedTokens = %v, want %v", completedB.GeneratedTokens, expectedTokensB)
		}

		if got, want := cb.TotalTokensGenerated(), int64(5); got != want {
			t.Fatalf("TotalTokensGenerated = %d, want %d", got, want)
		}
	})

	t.Run("equal_fixed_depth_concurrent_isolation", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4
		cfg.MaxKVCacheBytes = 50_000
		cfg.KVBytesPerToken = 100

		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cb.Close()

		promptC := []int{301, 302}
		promptD := []int{401, 402, 403}

		reqC := &SubagentRequest{
			SessionID:      "session-C",
			PromptTokens:   append([]int(nil), promptC...),
			TargetTokens:   4,
			ExecutionDepth: 2,
		}
		reqD := &SubagentRequest{
			SessionID:      "session-D",
			PromptTokens:   append([]int(nil), promptD...),
			TargetTokens:   6,
			ExecutionDepth: 2,
		}

		if _, err := cb.Submit(reqC); err != nil {
			t.Fatalf("Submit(reqC) failed: %v", err)
		}
		if _, err := cb.Submit(reqD); err != nil {
			t.Fatalf("Submit(reqD) failed: %v", err)
		}

		if got := cb.ActiveSlotCount(); got != 2 {
			t.Fatalf("ActiveSlotCount = %d, want 2", got)
		}

		slotC, _ := cb.GetSlot("session-C")
		slotD, _ := cb.GetSlot("session-D")

		// Step 1: Both advance to depth 1
		res1, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 1 failed: %v", err)
		}
		if res1.ActiveSlots != 2 || len(res1.RetiredSessionIDs) != 0 {
			t.Fatalf("Step 1 unexpected result: active=%d, retired=%v", res1.ActiveSlots, res1.RetiredSessionIDs)
		}
		if slotC.CurrentDepth != 1 || slotD.CurrentDepth != 1 {
			t.Fatalf("Step 1 depths: got C=%d, D=%d, want 1 each", slotC.CurrentDepth, slotD.CurrentDepth)
		}

		// Step 2: Both reach depth 2 and retire simultaneously
		res2, err := cb.Step(ctx)
		if err != nil {
			t.Fatalf("Step 2 failed: %v", err)
		}
		if res2.ActiveSlots != 2 {
			t.Fatalf("Step 2 ActiveSlots = %d, want 2", res2.ActiveSlots)
		}
		if len(res2.RetiredSessionIDs) != 2 {
			t.Fatalf("Step 2 RetiredSessionIDs = %v, want 2 sessions retired", res2.RetiredSessionIDs)
		}

		if slotC.State != SlotStateFinished || slotD.State != SlotStateFinished {
			t.Fatalf("both slots must be Finished, got C=%s, D=%s", slotC.State, slotD.State)
		}
		select {
		case <-slotC.Done():
		default:
			t.Fatalf("slotC Done channel must be closed")
		}
		select {
		case <-slotD.Done():
		default:
			t.Fatalf("slotD Done channel must be closed")
		}

		// Verify prompt and generated token isolation
		if !reflect.DeepEqual(slotC.PromptTokens, promptC) {
			t.Fatalf("slotC prompt contaminated: got %v, want %v", slotC.PromptTokens, promptC)
		}
		if !reflect.DeepEqual(slotD.PromptTokens, promptD) {
			t.Fatalf("slotD prompt contaminated: got %v, want %v", slotD.PromptTokens, promptD)
		}
		if len(slotC.GeneratedTokens) != 2 || len(slotD.GeneratedTokens) != 2 {
			t.Fatalf("generated token count mismatch: C=%d, D=%d, want 2 each",
				len(slotC.GeneratedTokens), len(slotD.GeneratedTokens))
		}

		if got := cb.ActiveSlotCount(); got != 0 {
			t.Fatalf("ActiveSlotCount after both retired = %d, want 0", got)
		}
		if got := cb.CurrentKVCacheBytes(); got != 0 {
			t.Fatalf("CurrentKVCacheBytes after both retired = %d, want 0", got)
		}
	})

	t.Run("concurrent_matches_isolated_execution", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4
		cfg.MaxKVCacheBytes = 50_000
		cfg.KVBytesPerToken = 100

		promptIso := []int{501, 502, 503}
		reqIso := &SubagentRequest{
			SessionID:      "session-iso",
			PromptTokens:   append([]int(nil), promptIso...),
			TargetTokens:   6,
			ExecutionDepth: 3,
		}

		// 1. Run session-iso in isolation
		cbIsolated, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cbIsolated.Close()

		if _, err := cbIsolated.Submit(reqIso); err != nil {
			t.Fatalf("Submit(reqIso) failed: %v", err)
		}

		var isolatedTokens []int
		for step := 1; step <= 3; step++ {
			res, err := cbIsolated.Step(ctx)
			if err != nil {
				t.Fatalf("cbIsolated Step %d failed: %v", step, err)
			}
			tok, ok := res.GeneratedTokens["session-iso"]
			if !ok {
				t.Fatalf("cbIsolated Step %d missing token", step)
			}
			isolatedTokens = append(isolatedTokens, tok)
		}

		slotIso, ok := cbIsolated.GetSlot("session-iso")
		if !ok || slotIso.State != SlotStateFinished {
			t.Fatalf("cbIsolated slot not finished cleanly")
		}

		// 2. Run identical request concurrently alongside another active recurrent request
		cbConcurrent, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cbConcurrent.Close()

		reqConcurrentA := &SubagentRequest{
			SessionID:      "session-iso",
			PromptTokens:   append([]int(nil), promptIso...),
			TargetTokens:   6,
			ExecutionDepth: 3,
		}
		reqConcurrentB := &SubagentRequest{
			SessionID:      "session-noise",
			PromptTokens:   []int{901, 902},
			TargetTokens:   4,
			ExecutionDepth: 2,
		}

		if _, err := cbConcurrent.Submit(reqConcurrentA); err != nil {
			t.Fatalf("Submit(reqConcurrentA) failed: %v", err)
		}
		if _, err := cbConcurrent.Submit(reqConcurrentB); err != nil {
			t.Fatalf("Submit(reqConcurrentB) failed: %v", err)
		}

		var concurrentTokens []int
		for step := 1; step <= 3; step++ {
			res, err := cbConcurrent.Step(ctx)
			if err != nil {
				t.Fatalf("cbConcurrent Step %d failed: %v", step, err)
			}
			tokA, okA := res.GeneratedTokens["session-iso"]
			if !okA {
				t.Fatalf("cbConcurrent Step %d missing session-iso", step)
			}
			concurrentTokens = append(concurrentTokens, tokA)

			if step <= 2 {
				if _, okB := res.GeneratedTokens["session-noise"]; !okB {
					t.Fatalf("cbConcurrent Step %d missing session-noise", step)
				}
			} else {
				if _, okB := res.GeneratedTokens["session-noise"]; okB {
					t.Fatalf("session-noise must have retired before Step 3")
				}
			}
		}

		slotConcA, ok := cbConcurrent.GetSlot("session-iso")
		if !ok || slotConcA.State != SlotStateFinished {
			t.Fatalf("cbConcurrent slot not finished cleanly")
		}

		// Verify exact token-for-token parity with isolated execution
		if !reflect.DeepEqual(concurrentTokens, isolatedTokens) {
			t.Fatalf("concurrent tokens %v do not match isolated execution %v", concurrentTokens, isolatedTokens)
		}
		if !reflect.DeepEqual(slotConcA.GeneratedTokens, slotIso.GeneratedTokens) {
			t.Fatalf("slot GeneratedTokens %v do not match isolated %v", slotConcA.GeneratedTokens, slotIso.GeneratedTokens)
		}
		if !reflect.DeepEqual(slotConcA.PromptTokens, promptIso) {
			t.Fatalf("slotConcA PromptTokens corrupted: got %v, want %v", slotConcA.PromptTokens, promptIso)
		}
		if slotConcA.CurrentDepth != slotIso.CurrentDepth {
			t.Fatalf("slotConcA CurrentDepth %d != isolated %d", slotConcA.CurrentDepth, slotIso.CurrentDepth)
		}
		if slotConcA.RecurrentPasses != slotIso.RecurrentPasses {
			t.Fatalf("slotConcA RecurrentPasses %d != isolated %d", slotConcA.RecurrentPasses, slotIso.RecurrentPasses)
		}
	})

	t.Run("interleaved_yield_and_resume_isolation", func(t *testing.T) {
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 4

		// Request A: depth=5, targetTokens=5, promptTokens=[101, 102]
		newReqA := func() *SubagentRequest {
			return &SubagentRequest{
				SessionID:      "req-A",
				ExecutionDepth: 5,
				TargetTokens:   5,
				PromptTokens:   []int{101, 102},
			}
		}

		// Request B: depth=7, targetTokens=7, promptTokens=[201, 202, 203]
		newReqB := func() *SubagentRequest {
			return &SubagentRequest{
				SessionID:      "req-B",
				ExecutionDepth: 7,
				TargetTokens:   7,
				PromptTokens:   []int{201, 202, 203},
			}
		}

		// 1. Run Request A in isolation and record generated tokens
		cbA, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher(cfg) failed: %v", err)
		}
		defer cbA.Close()

		reqAIso := newReqA()
		if _, err := cbA.Submit(reqAIso); err != nil {
			t.Fatalf("Submit reqAIso failed: %v", err)
		}

		var isolatedTokensA []int
		for cbA.ActiveSlotCount() > 0 {
			res, err := cbA.Step(ctx)
			if err != nil {
				t.Fatalf("cbA Step failed: %v", err)
			}
			if tok, ok := res.GeneratedTokens[reqAIso.SessionID]; ok {
				isolatedTokensA = append(isolatedTokensA, tok)
			}
		}

		// 2. Run Request B in isolation and record generated tokens
		cbB, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher(cfg) failed: %v", err)
		}
		defer cbB.Close()

		reqBIso := newReqB()
		if _, err := cbB.Submit(reqBIso); err != nil {
			t.Fatalf("Submit reqBIso failed: %v", err)
		}

		var isolatedTokensB []int
		for cbB.ActiveSlotCount() > 0 {
			res, err := cbB.Step(ctx)
			if err != nil {
				t.Fatalf("cbB Step failed: %v", err)
			}
			if tok, ok := res.GeneratedTokens[reqBIso.SessionID]; ok {
				isolatedTokensB = append(isolatedTokensB, tok)
			}
		}

		// 3. Interleaved execution with yield and resume in a concurrent batcher
		cbInterleaved, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher failed: %v", err)
		}
		defer cbInterleaved.Close()

		reqAInter := newReqA()
		reqBInter := newReqB()

		if _, err := cbInterleaved.Submit(reqAInter); err != nil {
			t.Fatalf("Submit reqAInter failed: %v", err)
		}
		if _, err := cbInterleaved.Submit(reqBInter); err != nil {
			t.Fatalf("Submit reqBInter failed: %v", err)
		}

		var tokensA []int
		var tokensB []int

		// Step 3 times while both are active
		for i := 0; i < 3; i++ {
			res, err := cbInterleaved.Step(ctx)
			if err != nil {
				t.Fatalf("Step %d failed: %v", i, err)
			}
			if tok, ok := res.GeneratedTokens[reqAInter.SessionID]; ok {
				tokensA = append(tokensA, tok)
			}
			if tok, ok := res.GeneratedTokens[reqBInter.SessionID]; ok {
				tokensB = append(tokensB, tok)
			}
		}

		// Request A yields for external tool I/O
		if err := cbInterleaved.YieldSlot(reqAInter.SessionID); err != nil {
			t.Fatalf("YieldSlot(reqAInter) failed: %v", err)
		}

		// Step 2 times while Request A is yielded: Request B generates tokens alone
		for i := 0; i < 2; i++ {
			res, err := cbInterleaved.Step(ctx)
			if err != nil {
				t.Fatalf("Step during yield failed: %v", err)
			}
			if _, ok := res.GeneratedTokens[reqAInter.SessionID]; ok {
				t.Fatalf("Request A produced a token while in YieldedIO state")
			}
			if tok, ok := res.GeneratedTokens[reqBInter.SessionID]; ok {
				tokensB = append(tokensB, tok)
			}
		}

		// Resume Request A
		if err := cbInterleaved.ResumeSlot(reqAInter.SessionID); err != nil {
			t.Fatalf("ResumeSlot(reqAInter) failed: %v", err)
		}

		// Drain remaining tokens until both finish
		for cbInterleaved.ActiveSlotCount() > 0 {
			res, err := cbInterleaved.Step(ctx)
			if err != nil {
				t.Fatalf("Step after resume failed: %v", err)
			}
			if tok, ok := res.GeneratedTokens[reqAInter.SessionID]; ok {
				tokensA = append(tokensA, tok)
			}
			if tok, ok := res.GeneratedTokens[reqBInter.SessionID]; ok {
				tokensB = append(tokensB, tok)
			}
		}

		// Verify zero KV cross contamination and exact match with isolated tokens
		if len(tokensA) != len(isolatedTokensA) {
			t.Fatalf("interleaved tokensA len = %d, want %d", len(tokensA), len(isolatedTokensA))
		}
		for i := range isolatedTokensA {
			if tokensA[i] != isolatedTokensA[i] {
				t.Fatalf("interleaved Request A token mismatch at index %d: got=%d, want=%d", i, tokensA[i], isolatedTokensA[i])
			}
		}

		if len(tokensB) != len(isolatedTokensB) {
			t.Fatalf("interleaved tokensB len = %d, want %d", len(tokensB), len(isolatedTokensB))
		}
		for i := range isolatedTokensB {
			if tokensB[i] != isolatedTokensB[i] {
				t.Fatalf("interleaved Request B token mismatch at index %d: got=%d, want=%d", i, tokensB[i], isolatedTokensB[i])
			}
		}
	})
}
