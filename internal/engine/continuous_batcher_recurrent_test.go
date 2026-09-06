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

// TestRecurrentBatchIsolationIntegration verifies that two fixed-depth recurrent requests
// executing concurrently through continuous batching with interleaved progress:
// 1. Maintain complete KV state and token isolation matching their isolated runs token-for-token.
// 2. Start with ActiveSlotCount() == 2 and finish with ActiveSlotCount() == 0.
// 3. Maintain independent KV cache memory footprints matching their respective depth-scaled formulas:
//    KV capacity = (len(PromptTokens) + TargetTokens) * KVBytesPerToken * ExecutionDepth.
func TestRecurrentBatchIsolationIntegration(t *testing.T) {
	ctx := context.Background()

	// Request A: depth=2, targetTokens=10, promptTokens=[101, 102]
	newReqA := func() *SubagentRequest {
		return &SubagentRequest{
			SessionID:      "req-A",
			ExecutionDepth: 2,
			TargetTokens:   10,
			PromptTokens:   []int{101, 102},
		}
	}

	// Request B: depth=3, targetTokens=10, promptTokens=[201, 202, 203]
	newReqB := func() *SubagentRequest {
		return &SubagentRequest{
			SessionID:      "req-B",
			ExecutionDepth: 3,
			TargetTokens:   10,
			PromptTokens:   []int{201, 202, 203},
		}
	}

	// 1. Run Request A in isolation and record generated tokens
	cfgA := DefaultContinuousBatcherConfig()
	cfgA.MaxSlots = 4
	cbA, err := NewContinuousBatcher(cfgA)
	if err != nil {
		t.Fatalf("NewContinuousBatcher(cfgA) failed: %v", err)
	}
	defer cbA.Close()

	reqAIso := newReqA()
	expectedKVA := int64(len(reqAIso.PromptTokens)+reqAIso.TargetTokens) * cfgA.KVBytesPerToken * int64(reqAIso.ExecutionDepth)

	if _, err := cbA.Submit(reqAIso); err != nil {
		t.Fatalf("Submit reqAIso failed: %v", err)
	}
	if got := cbA.ActiveSlotCount(); got != 1 {
		t.Fatalf("cbA ActiveSlotCount = %d, want 1", got)
	}
	if got := cbA.CurrentKVCacheBytes(); got != expectedKVA {
		t.Fatalf("cbA CurrentKVCacheBytes = %d, want %d", got, expectedKVA)
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

	if len(isolatedTokensA) != reqAIso.TargetTokens {
		t.Fatalf("isolatedTokensA len = %d, want %d", len(isolatedTokensA), reqAIso.TargetTokens)
	}
	if got := cbA.ActiveSlotCount(); got != 0 {
		t.Fatalf("cbA final ActiveSlotCount = %d, want 0", got)
	}
	if got := cbA.CurrentKVCacheBytes(); got != 0 {
		t.Fatalf("cbA final CurrentKVCacheBytes = %d, want 0", got)
	}

	// 2. Run Request B in isolation and record generated tokens
	cfgB := DefaultContinuousBatcherConfig()
	cfgB.MaxSlots = 4
	cbB, err := NewContinuousBatcher(cfgB)
	if err != nil {
		t.Fatalf("NewContinuousBatcher(cfgB) failed: %v", err)
	}
	defer cbB.Close()

	reqBIso := newReqB()
	expectedKVB := int64(len(reqBIso.PromptTokens)+reqBIso.TargetTokens) * cfgB.KVBytesPerToken * int64(reqBIso.ExecutionDepth)

	if _, err := cbB.Submit(reqBIso); err != nil {
		t.Fatalf("Submit reqBIso failed: %v", err)
	}
	if got := cbB.ActiveSlotCount(); got != 1 {
		t.Fatalf("cbB ActiveSlotCount = %d, want 1", got)
	}
	if got := cbB.CurrentKVCacheBytes(); got != expectedKVB {
		t.Fatalf("cbB CurrentKVCacheBytes = %d, want %d", got, expectedKVB)
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

	if len(isolatedTokensB) != reqBIso.TargetTokens {
		t.Fatalf("isolatedTokensB len = %d, want %d", len(isolatedTokensB), reqBIso.TargetTokens)
	}
	if got := cbB.ActiveSlotCount(); got != 0 {
		t.Fatalf("cbB final ActiveSlotCount = %d, want 0", got)
	}
	if got := cbB.CurrentKVCacheBytes(); got != 0 {
		t.Fatalf("cbB final CurrentKVCacheBytes = %d, want 0", got)
	}

	// 3. Run both concurrently in a single ContinuousBatcher instance with interleaved batch stepping
	cfgConcurrent := DefaultContinuousBatcherConfig()
	cfgConcurrent.MaxSlots = 4
	cbConcurrent, err := NewContinuousBatcher(cfgConcurrent)
	if err != nil {
		t.Fatalf("NewContinuousBatcher(cfgConcurrent) failed: %v", err)
	}
	defer cbConcurrent.Close()

	reqAConc := newReqA()
	reqBConc := newReqB()

	if _, err := cbConcurrent.Submit(reqAConc); err != nil {
		t.Fatalf("Submit reqAConc failed: %v", err)
	}
	if _, err := cbConcurrent.Submit(reqBConc); err != nil {
		t.Fatalf("Submit reqBConc failed: %v", err)
	}

	// Verify ActiveSlotCount() starts at 2
	if got := cbConcurrent.ActiveSlotCount(); got != 2 {
		t.Fatalf("initial concurrent ActiveSlotCount = %d, want 2", got)
	}

	// Verify that KV cache memory footprints remain independent and match their respective depth-scaled formulas:
	// KV = (len(PromptTokens) + TargetTokens) * KVBytesPerToken * ExecutionDepth
	slotA, okA := cbConcurrent.GetSlot(reqAConc.SessionID)
	if !okA {
		t.Fatalf("slotA not found in sessionMap")
	}
	slotB, okB := cbConcurrent.GetSlot(reqBConc.SessionID)
	if !okB {
		t.Fatalf("slotB not found in sessionMap")
	}

	if got, want := slotA.KVCacheBytes, expectedKVA; got != want {
		t.Fatalf("slotA KVCacheBytes = %d, want %d (formula: (2+10)*%d*2)", got, want, cfgConcurrent.KVBytesPerToken)
	}
	if got, want := slotB.KVCacheBytes, expectedKVB; got != want {
		t.Fatalf("slotB KVCacheBytes = %d, want %d (formula: (3+10)*%d*3)", got, want, cfgConcurrent.KVBytesPerToken)
	}
	if got, want := cbConcurrent.CurrentKVCacheBytes(), expectedKVA+expectedKVB; got != want {
		t.Fatalf("concurrent CurrentKVCacheBytes = %d, want %d (sum of independent footprints)", got, want)
	}

	// Interleaved batch stepping
	var concurrentTokensA []int
	var concurrentTokensB []int
	stepCount := 0
	for cbConcurrent.ActiveSlotCount() > 0 {
		res, err := cbConcurrent.Step(ctx)
		if err != nil {
			t.Fatalf("cbConcurrent Step at iteration %d failed: %v", stepCount, err)
		}
		stepCount++

		if tok, ok := res.GeneratedTokens[reqAConc.SessionID]; ok {
			concurrentTokensA = append(concurrentTokensA, tok)
		}
		if tok, ok := res.GeneratedTokens[reqBConc.SessionID]; ok {
			concurrentTokensB = append(concurrentTokensB, tok)
		}
	}

	// Verify ActiveSlotCount() finishes at 0
	if got := cbConcurrent.ActiveSlotCount(); got != 0 {
		t.Fatalf("final concurrent ActiveSlotCount = %d, want 0", got)
	}
	if got := cbConcurrent.CurrentKVCacheBytes(); got != 0 {
		t.Fatalf("final concurrent CurrentKVCacheBytes = %d, want 0", got)
	}

	// Prove that generated tokens for Request A and Request B match their isolated execution token-for-token
	if len(concurrentTokensA) != len(isolatedTokensA) {
		t.Fatalf("concurrentTokensA length = %d, want %d", len(concurrentTokensA), len(isolatedTokensA))
	}
	for i := range isolatedTokensA {
		if concurrentTokensA[i] != isolatedTokensA[i] {
			t.Fatalf("Request A token mismatch at index %d: concurrent=%d, isolated=%d (cross-request contamination detected)",
				i, concurrentTokensA[i], isolatedTokensA[i])
		}
	}

	if len(concurrentTokensB) != len(isolatedTokensB) {
		t.Fatalf("concurrentTokensB length = %d, want %d", len(concurrentTokensB), len(isolatedTokensB))
	}
	for i := range isolatedTokensB {
		if concurrentTokensB[i] != isolatedTokensB[i] {
			t.Fatalf("Request B token mismatch at index %d: concurrent=%d, isolated=%d (cross-request contamination detected)",
				i, concurrentTokensB[i], isolatedTokensB[i])
		}
	}

	// Also verify completed slot records
	completedSlotA, okCA := cbConcurrent.GetSlot(reqAConc.SessionID)
	if !okCA {
		t.Fatalf("completedSlotA not found")
	}
	if completedSlotA.State != SlotStateFinished {
		t.Fatalf("completedSlotA State = %s, want %s", completedSlotA.State, SlotStateFinished)
	}
	if len(completedSlotA.GeneratedTokens) != reqAConc.TargetTokens {
		t.Fatalf("completedSlotA tokens = %d, want %d", len(completedSlotA.GeneratedTokens), reqAConc.TargetTokens)
	}

	completedSlotB, okCB := cbConcurrent.GetSlot(reqBConc.SessionID)
	if !okCB {
		t.Fatalf("completedSlotB not found")
	}
	if completedSlotB.State != SlotStateFinished {
		t.Fatalf("completedSlotB State = %s, want %s", completedSlotB.State, SlotStateFinished)
	}
	if len(completedSlotB.GeneratedTokens) != reqBConc.TargetTokens {
		t.Fatalf("completedSlotB tokens = %d, want %d", len(completedSlotB.GeneratedTokens), reqBConc.TargetTokens)
	}

	// 4. Additionally verify interleaved progress when one request yields for external I/O and resumes
	t.Run("interleaved_yield_and_resume_isolation", func(t *testing.T) {
		cbInterleaved, err := NewContinuousBatcher(cfgConcurrent)
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
