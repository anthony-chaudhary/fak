package model

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestSpeculativeBatchConfigValidation(t *testing.T) {
	cfg := DefaultSpeculativeBatchConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultSpeculativeBatchConfig invalid: %v", err)
	}

	invalidCases := []struct {
		name string
		mod  func(c *SpeculativeBatchConfig)
	}{
		{
			name: "zero UBatchSize",
			mod:  func(c *SpeculativeBatchConfig) { c.UBatchSize = 0 },
		},
		{
			name: "negative SpecDraftUBatchSize",
			mod:  func(c *SpeculativeBatchConfig) { c.SpecDraftUBatchSize = -1 },
		},
		{
			name: "zero MaxContextLength",
			mod:  func(c *SpeculativeBatchConfig) { c.MaxContextLength = 0 },
		},
		{
			name: "MaxContextLength smaller than UBatchSize",
			mod: func(c *SpeculativeBatchConfig) {
				c.UBatchSize = 1024
				c.MaxContextLength = 512
			},
		},
	}

	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			c := DefaultSpeculativeBatchConfig()
			tc.mod(&c)
			if err := c.Validate(); err == nil {
				t.Fatalf("expected error for case %s, got nil", tc.name)
			}
		})
	}
}

func TestSpeculativeBatchDispatchesDecoupled(t *testing.T) {
	cfg := SpeculativeBatchConfig{
		UBatchSize:          1024,
		SpecDraftUBatchSize: 512,
		MaxContextLength:    262144,
	}

	testCases := []struct {
		draftTokens   int
		basePos       int
		wantBatches   int
		wantLastBatch int
		wantLastPad   int
	}{
		{draftTokens: 5, basePos: 100, wantBatches: 1, wantLastBatch: 5, wantLastPad: 507},
		{draftTokens: 512, basePos: 1000, wantBatches: 1, wantLastBatch: 512, wantLastPad: 0},
		{draftTokens: 1024, basePos: 2000, wantBatches: 2, wantLastBatch: 512, wantLastPad: 0},
		{draftTokens: 1200, basePos: 3000, wantBatches: 3, wantLastBatch: 176, wantLastPad: 336},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("drafts_%d", tc.draftTokens), func(t *testing.T) {
			dispatches, err := cfg.PlanSpeculativeVerification(tc.draftTokens, tc.basePos)
			if err != nil {
				t.Fatalf("PlanSpeculativeVerification failed: %v", err)
			}
			if len(dispatches) != tc.wantBatches {
				t.Fatalf("got %d batches, want %d", len(dispatches), tc.wantBatches)
			}

			totalTokens := 0
			for i, d := range dispatches {
				if d.Index != i {
					t.Errorf("batch %d has unexpected Index %d", i, d.Index)
				}
				if d.FixedDimension != cfg.SpecDraftUBatchSize {
					t.Errorf("batch %d has FixedDimension %d, want %d", i, d.FixedDimension, cfg.SpecDraftUBatchSize)
				}
				if d.NumTokens+d.PaddedTokens != cfg.SpecDraftUBatchSize {
					t.Errorf("batch %d: NumTokens (%d) + PaddedTokens (%d) != FixedDimension (%d)",
						i, d.NumTokens, d.PaddedTokens, cfg.SpecDraftUBatchSize)
				}
				if d.CaptureKey.Kind != compute.GraphCaptureSpeculative {
					t.Errorf("batch %d has CaptureKey kind %s, want %s", i, d.CaptureKey.Kind, compute.GraphCaptureSpeculative)
				}
				if d.CaptureKey.BatchSize != cfg.SpecDraftUBatchSize {
					t.Errorf("batch %d has CaptureKey batch size %d, want %d", i, d.CaptureKey.BatchSize, cfg.SpecDraftUBatchSize)
				}
				if d.BasePosition != tc.basePos+totalTokens {
					t.Errorf("batch %d has BasePosition %d, want %d", i, d.BasePosition, tc.basePos+totalTokens)
				}
				totalTokens += d.NumTokens
			}

			if totalTokens != tc.draftTokens {
				t.Fatalf("total scheduled tokens %d != expected %d", totalTokens, tc.draftTokens)
			}

			last := dispatches[len(dispatches)-1]
			if last.NumTokens != tc.wantLastBatch {
				t.Errorf("last batch NumTokens = %d, want %d", last.NumTokens, tc.wantLastBatch)
			}
			if last.PaddedTokens != tc.wantLastPad {
				t.Errorf("last batch PaddedTokens = %d, want %d", last.PaddedTokens, tc.wantLastPad)
			}
		})
	}
}

func TestSpeculativeBatchDeepContextDecomposition(t *testing.T) {
	cfg := SpeculativeBatchConfig{
		UBatchSize:          1024,
		SpecDraftUBatchSize: 512,
		MaxContextLength:    262144, // 256K tokens
	}

	// Deep-context prompt execution (>200K tokens: 210,000 tokens)
	promptTokens := 210000
	draftTokens := 128
	startPos := 0

	plan, err := cfg.PlanDeepContextExecution(promptTokens, draftTokens, startPos)
	if err != nil {
		t.Fatalf("PlanDeepContextExecution failed: %v", err)
	}

	if plan.TotalPromptTokens != promptTokens {
		t.Errorf("TotalPromptTokens = %d, want %d", plan.TotalPromptTokens, promptTokens)
	}
	if plan.TotalDraftTokens != draftTokens {
		t.Errorf("TotalDraftTokens = %d, want %d", plan.TotalDraftTokens, draftTokens)
	}

	// 210,000 tokens / 1024 = 205 full chunks + 1 chunk of 80 tokens = 206 chunks total.
	wantChunks := 206
	if len(plan.PromptChunks) != wantChunks {
		t.Fatalf("len(plan.PromptChunks) = %d, want %d", len(plan.PromptChunks), wantChunks)
	}

	accumTokens := 0
	for i, chunk := range plan.PromptChunks {
		if chunk.Index != i {
			t.Errorf("chunk %d has Index %d", i, chunk.Index)
		}
		if chunk.NumTokens > cfg.UBatchSize {
			t.Errorf("chunk %d exceeded UBatchSize: %d > %d", i, chunk.NumTokens, cfg.UBatchSize)
		}
		if chunk.StartToken != accumTokens {
			t.Errorf("chunk %d StartToken = %d, want %d", i, chunk.StartToken, accumTokens)
		}
		if chunk.EndToken != chunk.StartToken+chunk.NumTokens {
			t.Errorf("chunk %d EndToken %d != StartToken %d + NumTokens %d",
				i, chunk.EndToken, chunk.StartToken, chunk.NumTokens)
		}
		if i == wantChunks-1 {
			if !chunk.IsLast {
				t.Errorf("last chunk should have IsLast = true")
			}
			if chunk.NumTokens != 80 {
				t.Errorf("last chunk NumTokens = %d, want 80", chunk.NumTokens)
			}
		} else {
			if chunk.IsLast {
				t.Errorf("intermediate chunk %d has IsLast = true", i)
			}
			if chunk.NumTokens != cfg.UBatchSize {
				t.Errorf("chunk %d NumTokens = %d, want %d", i, chunk.NumTokens, cfg.UBatchSize)
			}
		}
		accumTokens += chunk.NumTokens
	}

	if accumTokens != promptTokens {
		t.Fatalf("total chunked prompt tokens %d != %d", accumTokens, promptTokens)
	}

	// Verify speculative batches
	if len(plan.SpeculativeBatches) != 1 {
		t.Fatalf("expected 1 speculative batch, got %d", len(plan.SpeculativeBatches))
	}
	specBatch := plan.SpeculativeBatches[0]
	if specBatch.NumTokens != draftTokens {
		t.Errorf("specBatch NumTokens = %d, want %d", specBatch.NumTokens, draftTokens)
	}
	if specBatch.FixedDimension != cfg.SpecDraftUBatchSize {
		t.Errorf("specBatch FixedDimension = %d, want %d", specBatch.FixedDimension, cfg.SpecDraftUBatchSize)
	}
	if specBatch.PaddedTokens != cfg.SpecDraftUBatchSize-draftTokens {
		t.Errorf("specBatch PaddedTokens = %d, want %d", specBatch.PaddedTokens, cfg.SpecDraftUBatchSize-draftTokens)
	}
	if specBatch.BasePosition != promptTokens {
		t.Errorf("specBatch BasePosition = %d, want %d", specBatch.BasePosition, promptTokens)
	}

	wantRemaining := cfg.MaxContextLength - (promptTokens + draftTokens)
	if plan.ContextRemaining != wantRemaining {
		t.Errorf("ContextRemaining = %d, want %d", plan.ContextRemaining, wantRemaining)
	}

	// Verify context bound guard
	_, err = cfg.PlanDeepContextExecution(262140, 10, 0)
	if err == nil {
		t.Fatalf("expected context overflow error for 262140 + 10 > 262144, got nil")
	}
}

func TestSpeculativeBatchGraphCaptureKeyStability(t *testing.T) {
	cfg := SpeculativeBatchConfig{
		UBatchSize:          1024,
		SpecDraftUBatchSize: 512,
		MaxContextLength:    262144,
		DeviceTag:           "sm90",
	}

	planner, err := cfg.GraphPlanner()
	if err != nil {
		t.Fatalf("GraphPlanner creation failed: %v", err)
	}

	specKey := planner.SpeculativeCaptureKey()
	expectedSpecKeyStr := "speculative_draft:b512:sm90"
	if specKey.String() != expectedSpecKeyStr {
		t.Fatalf("SpeculativeCaptureKey = %s, want %s", specKey.String(), expectedSpecKeyStr)
	}

	// Varying primary prompt chunk sizes must NOT alter the speculative capture key
	varyingChunkSizes := []int{1024, 768, 512, 256, 128, 64, 32, 16, 1}
	for _, size := range varyingChunkSizes {
		primaryKey := planner.PrimaryCaptureKey(size)
		expectedPrimaryStr := fmt.Sprintf("primary_prompt:b%d:sm90", size)
		if primaryKey.String() != expectedPrimaryStr {
			t.Errorf("PrimaryCaptureKey(%d) = %s, want %s", size, primaryKey.String(), expectedPrimaryStr)
		}

		// Speculative capture key remains invariant
		currentSpecKey := planner.SpeculativeCaptureKey()
		if currentSpecKey.String() != expectedSpecKeyStr {
			t.Errorf("for prompt chunk size %d, SpeculativeCaptureKey changed to %s", size, currentSpecKey.String())
		}
	}

	// Test GraphRunner execution with varying dynamic draft lengths
	runner := compute.NewSpeculativeGraphRunner(planner)

	dynamicDraftLengths := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for i, length := range dynamicDraftLengths {
		err := runner.ExecuteSpeculativeDraft(length, func(effectiveBatch int) error {
			if effectiveBatch != cfg.SpecDraftUBatchSize {
				return fmt.Errorf("unexpected effective batch %d, want %d", effectiveBatch, cfg.SpecDraftUBatchSize)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("run %d with draft len %d failed: %v", i, length, err)
		}
	}

	stats := runner.Stats()
	// The very first run captures the graph (1 miss).
	// All subsequent 7 runs are hits because the fixed speculative dimension is reused.
	if stats.SpeculativeMisses != 1 {
		t.Errorf("SpeculativeMisses = %d, want 1", stats.SpeculativeMisses)
	}
	if stats.SpeculativeHits != len(dynamicDraftLengths)-1 {
		t.Errorf("SpeculativeHits = %d, want %d", stats.SpeculativeHits, len(dynamicDraftLengths)-1)
	}
	if stats.CapturesTotal != 1 {
		t.Errorf("CapturesTotal = %d, want 1", stats.CapturesTotal)
	}

	// Now execute primary prompt chunks with varying sizes
	for _, chunkSize := range []int{1024, 512, 1024} {
		err := runner.ExecutePrimaryChunk(chunkSize, func(size int) error {
			return nil
		})
		if err != nil {
			t.Fatalf("primary chunk %d failed: %v", chunkSize, err)
		}
	}

	statsAfterPrimary := runner.Stats()
	// Primary chunking must not invalidate speculative draft capture key or stats
	if statsAfterPrimary.SpeculativeHits != len(dynamicDraftLengths)-1 {
		t.Errorf("post-primary SpeculativeHits = %d, want %d",
			statsAfterPrimary.SpeculativeHits, len(dynamicDraftLengths)-1)
	}
	if statsAfterPrimary.SpeculativeMisses != 1 {
		t.Errorf("post-primary SpeculativeMisses = %d, want 1", statsAfterPrimary.SpeculativeMisses)
	}

	// And another speculative draft run should still be a hit!
	err = runner.ExecuteSpeculativeDraft(42, func(effectiveBatch int) error {
		return nil
	})
	if err != nil {
		t.Fatalf("speculative execution after primary failed: %v", err)
	}

	finalStats := runner.Stats()
	if finalStats.SpeculativeHits != len(dynamicDraftLengths) {
		t.Errorf("final SpeculativeHits = %d, want %d", finalStats.SpeculativeHits, len(dynamicDraftLengths))
	}
	if finalStats.SpeculativeMisses != 1 {
		t.Errorf("final SpeculativeMisses = %d, want 1", finalStats.SpeculativeMisses)
	}
}
