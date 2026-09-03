package agentopt

import (
	"testing"
)

// TestChunkedPrefillScheduling demonstrates smooth decode pacing and bounded decode
// latency under heavy multi-agent prefill load using continuous batching chunk interleaving.
func TestChunkedPrefillScheduling(t *testing.T) {
	cfg := ChunkedInterleaverConfig{
		MaxBatchTokens: 512,
		ChunkSize:      256,
		MaxBatchSeqs:   16,
	}
	interleaver := NewChunkedInterleaver(cfg)

	// Two active agent decode streams generating 20 tokens each.
	if err := interleaver.AddDecode("agent-decode-1", 20); err != nil {
		t.Fatalf("failed to add decode 1: %v", err)
	}
	if err := interleaver.AddDecode("agent-decode-2", 20); err != nil {
		t.Fatalf("failed to add decode 2: %v", err)
	}

	// Heavy multi-agent prefill load arriving concurrently:
	// Three large prompts totaling 10,752 tokens (21x the max batch budget of 512).
	prefills := map[string]int{
		"agent-prefill-large-1": 3584,
		"agent-prefill-large-2": 3584,
		"agent-prefill-large-3": 3584,
	}
	for id, tokens := range prefills {
		if err := interleaver.AddPrefill(id, tokens); err != nil {
			t.Fatalf("failed to add prefill %s: %v", id, err)
		}
	}

	var batches []ScheduledBatch
	stepCount := 0
	maxSteps := 100 // safety limit

	interleavedSteps := 0

	for interleaver.HasPending() && stepCount < maxSteps {
		stepCount++
		batch := interleaver.Step()
		if batch.IsEmpty() {
			t.Fatalf("unexpected empty batch at step %d while requests pending", stepCount)
		}
		batches = append(batches, batch)

		// Verification 1: Batch token budget invariant
		if batch.TotalTokens > cfg.MaxBatchTokens {
			t.Fatalf("step %d exceeded MaxBatchTokens: %d > %d",
				stepCount, batch.TotalTokens, cfg.MaxBatchTokens)
		}
		if batch.TotalTokens != batch.PrefillTokens+batch.DecodeTokens {
			t.Fatalf("step %d token sum mismatch: %d != %d + %d",
				stepCount, batch.TotalTokens, batch.PrefillTokens, batch.DecodeTokens)
		}

		// Verification 2: Sequence count budget invariant
		if len(batch.Items) > cfg.MaxBatchSeqs {
			t.Fatalf("step %d exceeded MaxBatchSeqs: %d > %d",
				stepCount, len(batch.Items), cfg.MaxBatchSeqs)
		}

		// Verification 3: Prefill chunk size bounds
		for _, item := range batch.Items {
			if item.Kind == KindPrefill {
				if item.Tokens > cfg.ChunkSize {
					t.Fatalf("step %d prefill chunk for %s exceeded ChunkSize: %d > %d",
						stepCount, item.RequestID, item.Tokens, cfg.ChunkSize)
				}
			}
			if item.Kind == KindDecode {
				if item.Tokens != 1 {
					t.Fatalf("step %d decode item for %s expected 1 token, got %d",
						stepCount, item.RequestID, item.Tokens)
				}
			}
		}

		// Track interleaved steps where both decode and prefill work were performed
		if batch.DecodeTokens > 0 && batch.PrefillTokens > 0 {
			interleavedSteps++
		}

		// While decodes are active (first 20 steps), both decodes must be scheduled in EVERY step
		if stepCount <= 20 {
			if !batch.HasDecode("agent-decode-1") {
				t.Fatalf("step %d: agent-decode-1 was starved by prefill load", stepCount)
			}
			if !batch.HasDecode("agent-decode-2") {
				t.Fatalf("step %d: agent-decode-2 was starved by prefill load", stepCount)
			}
		}
	}

	if interleaver.HasPending() {
		t.Fatalf("interleaver still has pending requests after %d steps", stepCount)
	}

	// Verification 4: Decode pacing and bounded decode latency
	intervals1 := interleaver.DecodeIntervals("agent-decode-1")
	if len(intervals1) != 19 {
		t.Fatalf("expected 19 decode intervals for 20 tokens, got %d", len(intervals1))
	}
	for i, iv := range intervals1 {
		if iv != 1 {
			t.Fatalf("agent-decode-1 interval %d was %d (expected 1 for smooth pacing)", i, iv)
		}
	}

	intervals2 := interleaver.DecodeIntervals("agent-decode-2")
	if len(intervals2) != 19 {
		t.Fatalf("expected 19 decode intervals for 20 tokens, got %d", len(intervals2))
	}
	for i, iv := range intervals2 {
		if iv != 1 {
			t.Fatalf("agent-decode-2 interval %d was %d (expected 1 for smooth pacing)", i, iv)
		}
	}

	stats := interleaver.Stats()
	if stats.MaxDecodeInterval != 1 {
		t.Fatalf("expected MaxDecodeInterval = 1, got %d", stats.MaxDecodeInterval)
	}

	// Verification 5: Cumulative token delivery
	expectedPrefillTokens := 3584 * 3
	if stats.TotalPrefillTokens != expectedPrefillTokens {
		t.Fatalf("expected %d total prefill tokens, got %d",
			expectedPrefillTokens, stats.TotalPrefillTokens)
	}
	if stats.TotalDecodeTokens != 40 {
		t.Fatalf("expected 40 total decode tokens, got %d", stats.TotalDecodeTokens)
	}

	// Verification 6: Interleaving occurred during all active decode steps
	if interleavedSteps < 20 {
		t.Fatalf("expected at least 20 interleaved steps, got %d", interleavedSteps)
	}

	t.Logf("PASS: executed %d steps, %d interleaved, max decode interval = %d (perfect pacing)",
		stepCount, interleavedSteps, stats.MaxDecodeInterval)
}

func TestChunkedPrefillDynamicChunkSize(t *testing.T) {
	// Scenario: Batch token budget is 300, ChunkSize is 256.
	// 50 decode requests take 50 tokens, leaving 250 tokens for prefill.
	// The prefill prompt (1000 tokens) must dynamically slice a 250-token chunk.
	cfg := ChunkedInterleaverConfig{
		MaxBatchTokens: 300,
		ChunkSize:      256,
		MaxBatchSeqs:   64,
	}
	interleaver := NewChunkedInterleaver(cfg)

	for i := 0; i < 50; i++ {
		id := "dec-" + string(rune('a'+i))
		if err := interleaver.AddDecode(id, 1); err != nil {
			t.Fatalf("failed to add decode %s: %v", id, err)
		}
	}

	if err := interleaver.AddPrefill("p-large", 1000); err != nil {
		t.Fatalf("failed to add prefill: %v", err)
	}

	batch := interleaver.Step()
	if batch.DecodeTokens != 50 {
		t.Fatalf("expected 50 decode tokens, got %d", batch.DecodeTokens)
	}
	// Available tokens was 300 - 50 = 250.
	// Chunk size was 256, but clamped to 250 available tokens.
	if batch.PrefillTokens != 250 {
		t.Fatalf("expected 250 dynamically sliced prefill tokens, got %d", batch.PrefillTokens)
	}
	if batch.TotalTokens != 300 {
		t.Fatalf("expected 300 total tokens, got %d", batch.TotalTokens)
	}
}

func TestChunkedPrefillAutoTransition(t *testing.T) {
	cfg := ChunkedInterleaverConfig{
		MaxBatchTokens:         512,
		ChunkSize:              256,
		DefaultDecodeTokens:    2,
		AutoTransitionToDecode: true,
	}
	interleaver := NewChunkedInterleaver(cfg)

	// Prefill of 512 tokens requires 2 chunks of 256 tokens.
	if err := interleaver.AddPrefill("agent-req-1", 512); err != nil {
		t.Fatalf("failed to add prefill: %v", err)
	}

	// Step 1: First prefill chunk
	b1 := interleaver.Step()
	if b1.PrefillTokens != 256 || len(b1.CompletedPrefill) != 0 {
		t.Fatalf("step 1 unexpected batch: %+v", b1)
	}

	// Step 2: Second prefill chunk (completes prefill)
	b2 := interleaver.Step()
	if b2.PrefillTokens != 256 || len(b2.CompletedPrefill) != 1 || b2.CompletedPrefill[0] != "agent-req-1" {
		t.Fatalf("step 2 unexpected batch: %+v", b2)
	}

	// Step 3: Auto-transitioned to decode
	b3 := interleaver.Step()
	if b3.DecodeTokens != 1 || !b3.HasDecode("agent-req-1") {
		t.Fatalf("step 3 expected decode step for agent-req-1, got %+v", b3)
	}

	// Step 4: Second decode token (completes decode)
	b4 := interleaver.Step()
	if b4.DecodeTokens != 1 || len(b4.CompletedDecode) != 1 || b4.CompletedDecode[0] != "agent-req-1" {
		t.Fatalf("step 4 expected decode completion, got %+v", b4)
	}

	if interleaver.HasPending() {
		t.Fatalf("expected all requests completed")
	}
}

func TestChunkedPrefillValidation(t *testing.T) {
	interleaver := NewChunkedInterleaver(ChunkedInterleaverConfig{})

	// Empty ID
	if err := interleaver.AddPrefill("", 100); err == nil {
		t.Fatalf("expected error for empty prefill ID")
	}
	if err := interleaver.AddDecode(""); err == nil {
		t.Fatalf("expected error for empty decode ID")
	}

	// Non-positive prompt tokens
	if err := interleaver.AddPrefill("p1", 0); err == nil {
		t.Fatalf("expected error for 0 prompt tokens")
	}
	if err := interleaver.AddPrefill("p1", -10); err == nil {
		t.Fatalf("expected error for negative prompt tokens")
	}

	// Duplicate ID
	if err := interleaver.AddPrefill("p1", 100); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := interleaver.AddPrefill("p1", 100); err == nil {
		t.Fatalf("expected error for duplicate prefill ID")
	}
	if err := interleaver.AddDecode("p1"); err == nil {
		t.Fatalf("expected error for duplicate decode ID matching existing prefill")
	}

	// Empty step
	emptySched := NewChunkedInterleaver(ChunkedInterleaverConfig{})
	batch := emptySched.Step()
	if !batch.IsEmpty() {
		t.Fatalf("expected empty batch when no requests pending")
	}
}

func TestChunkedPrefillCancellation(t *testing.T) {
	interleaver := NewChunkedInterleaver(ChunkedInterleaverConfig{})

	if err := interleaver.AddPrefill("p1", 500); err != nil {
		t.Fatalf("failed to add prefill: %v", err)
	}
	if err := interleaver.AddDecode("d1", 10); err != nil {
		t.Fatalf("failed to add decode: %v", err)
	}

	if !interleaver.RemoveRequest("p1") {
		t.Fatalf("failed to remove prefill p1")
	}
	if interleaver.PendingPrefillCount() != 0 {
		t.Fatalf("expected 0 pending prefills, got %d", interleaver.PendingPrefillCount())
	}

	if !interleaver.RemoveRequest("d1") {
		t.Fatalf("failed to remove decode d1")
	}
	if interleaver.PendingDecodeCount() != 0 {
		t.Fatalf("expected 0 pending decodes, got %d", interleaver.PendingDecodeCount())
	}

	if interleaver.RemoveRequest("non-existent") {
		t.Fatalf("expected false for removing non-existent request")
	}
}
