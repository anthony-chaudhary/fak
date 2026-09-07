package compute

import (
	"context"
	"errors"
	"math"
	"testing"
)

// TestWatchdogPrefillScheduler runs the comprehensive test suite for Issue #11904:
// watchdog-paced chunked prefill submission scheduler for deep context on AMD APUs.
func TestWatchdogPrefillScheduler(t *testing.T) {
	t.Run("BoundaryPartitioning", testWatchdogPrefillScheduler_BoundaryPartitioning)
	t.Run("SafetyAssertions", testWatchdogPrefillScheduler_SafetyAssertions)
	t.Run("FlopAndByteCalculations", testWatchdogPrefillScheduler_FlopAndByteCalculations)
	t.Run("PacedExecutionSimulation", testWatchdogPrefillScheduler_PacedExecutionSimulation)
	t.Run("ContextCancellation", testWatchdogPrefillScheduler_ContextCancellation)
	t.Run("BatchedMultiSequence", testWatchdogPrefillScheduler_BatchedMultiSequence)
	t.Run("ProfilesAndHardwareProfiles", testWatchdogPrefillScheduler_ProfilesAndHardware)
	t.Run("AdaptiveDownsizingAtExtremeDepth", testWatchdogPrefillScheduler_AdaptiveDownsizingAtExtremeDepth)
}

func testWatchdogPrefillScheduler_BoundaryPartitioning(t *testing.T) {
	sched, err := NewDeepContextPrefillScheduler(ProfileSingleSequenceDeepContext)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	testCases := []struct {
		name         string
		promptTokens int
		expectChunks int
	}{
		{
			name:         "sub-chunk 1000 tokens",
			promptTokens: 1000,
			expectChunks: 1,
		},
		{
			name:         "exact chunk 1024 tokens",
			promptTokens: 1024,
			expectChunks: 1,
		},
		{
			name:         "boundary plus one 1025 tokens",
			promptTokens: 1025,
			expectChunks: 2,
		},
		{
			name:         "exact double 2048 tokens",
			promptTokens: 2048,
			expectChunks: 2,
		},
		{
			name:         "deep context 32k tokens",
			promptTokens: 32768,
			expectChunks: 32, // 32768 / 1024
		},
		{
			name:         "deep context 64k tokens",
			promptTokens: 65536,
			expectChunks: 64, // 65536 / 1024
		},
		{
			name:         "deep context 131072 tokens (128k)",
			promptTokens: 131072,
		},
		{
			name:         "deep context 262144 tokens (256k)",
			promptTokens: 262144,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			schedule, err := sched.PlanSchedule(tc.promptTokens)
			if err != nil {
				t.Fatalf("PlanSchedule(%d) error = %v", tc.promptTokens, err)
			}

			if schedule.TotalTokens != tc.promptTokens {
				t.Errorf("TotalTokens = %d, want %d", schedule.TotalTokens, tc.promptTokens)
			}

			if tc.expectChunks > 0 && len(schedule.Chunks) != tc.expectChunks {
				t.Errorf("Chunk count = %d, want %d", len(schedule.Chunks), tc.expectChunks)
			}

			// Validate partition continuity and invariants
			tokensCounted := 0
			for i, chunk := range schedule.Chunks {
				if chunk.Index != i {
					t.Errorf("chunk %d has index %d", i, chunk.Index)
				}
				if chunk.StartToken != tokensCounted {
					t.Errorf("chunk %d starts at %d, want %d (gap or overlap)", i, chunk.StartToken, tokensCounted)
				}
				if chunk.TokenCount <= 0 {
					t.Errorf("chunk %d has non-positive token count %d", i, chunk.TokenCount)
				}
				if chunk.TokenCount > sched.profile.MaxChunkTokens {
					t.Errorf("chunk %d token count %d exceeds max %d", i, chunk.TokenCount, sched.profile.MaxChunkTokens)
				}

				isLast := (i == len(schedule.Chunks)-1)
				if chunk.IsLastChunk != isLast {
					t.Errorf("chunk %d IsLastChunk = %v, want %v", i, chunk.IsLastChunk, isLast)
				}

				if isLast {
					if chunk.Yield.Type != FenceTypeFullYield {
						t.Errorf("final chunk yield = %v, want %v", chunk.Yield.Type, FenceTypeFullYield)
					}
					if !chunk.Yield.YieldHostCPU {
						t.Errorf("final chunk yield should require host CPU yield")
					}
				} else {
					if chunk.Yield.Type != FenceTypeSignalFence {
						t.Errorf("intermediate chunk %d yield = %v, want %v", i, chunk.Yield.Type, FenceTypeSignalFence)
					}
					if !chunk.Yield.HostInterruptWait {
						t.Errorf("intermediate chunk %d should request host interrupt wait for watchdog reset", i)
					}
				}

				tokensCounted += chunk.TokenCount
			}

			if tokensCounted != tc.promptTokens {
				t.Errorf("Sum of chunk tokens = %d, want %d", tokensCounted, tc.promptTokens)
			}

			if err := schedule.Validate(sched.profile.MaxExecutionCeilingMs); err != nil {
				t.Errorf("schedule validation failed: %v", err)
			}
		})
	}
}

func testWatchdogPrefillScheduler_SafetyAssertions(t *testing.T) {
	sched, err := NewDeepContextPrefillScheduler(ProfileSingleSequenceDeepContext)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	// 1. Verify safety on deep context prompts (32k, 64k, 136k, 262k)
	deepPrompts := []int{32768, 65536, 136000, 262144}
	for _, promptLen := range deepPrompts {
		schedule, err := sched.PlanSchedule(promptLen)
		if err != nil {
			t.Fatalf("PlanSchedule(%d) error: %v", promptLen, err)
		}

		if !schedule.IsWatchdogSafe {
			t.Errorf("promptLen %d marked not watchdog safe (max chunk ms = %.2f)", promptLen, schedule.MaxChunkDurationMs)
		}

		for _, chunk := range schedule.Chunks {
			if chunk.EstimatedDurationMs > sched.profile.MaxExecutionCeilingMs {
				t.Errorf("chunk %d duration %.2f ms exceeds ceiling %.2f ms",
					chunk.Index, chunk.EstimatedDurationMs, sched.profile.MaxExecutionCeilingMs)
			}
			if chunk.EstimatedDurationMs > sched.profile.WatchdogTimeoutMs {
				t.Fatalf("CRITICAL: chunk %d duration %.2f ms exceeds driver watchdog %.2f ms",
					chunk.Index, chunk.EstimatedDurationMs, sched.profile.WatchdogTimeoutMs)
			}
		}
	}

	// 2. Test safety validation rejection when a chunk duration exceeds threshold
	schedule, err := sched.PlanSchedule(1024)
	if err != nil {
		t.Fatalf("PlanSchedule(1024) error: %v", err)
	}

	// Artificially inflate chunk duration to exceed ceiling
	schedule.Chunks[0].EstimatedDurationMs = 2500.0 // > 2000.0 ms ceiling
	err = schedule.Validate(2000.0)
	if err == nil {
		t.Fatalf("expected validation error for chunk duration 2500 ms with ceiling 2000 ms, got nil")
	}

	var safetyErr *WatchdogSafetyViolationError
	if !errors.As(err, &safetyErr) {
		t.Errorf("expected error to be *WatchdogSafetyViolationError, got %T: %v", err, err)
	} else {
		if safetyErr.ChunkIndex != 0 {
			t.Errorf("safetyErr.ChunkIndex = %d, want 0", safetyErr.ChunkIndex)
		}
		if safetyErr.EstimatedTimeMs != 2500.0 {
			t.Errorf("safetyErr.EstimatedTimeMs = %.2f, want 2500.0", safetyErr.EstimatedTimeMs)
		}
	}

	if !errors.Is(err, ErrWatchdogSafetyViolation) {
		t.Errorf("expected error to unwrap to ErrWatchdogSafetyViolation, got %v", err)
	}

	// 3. Test non-positive prompt tokens error
	_, err = sched.PlanSchedule(0)
	if !errors.Is(err, ErrInvalidPromptTokens) {
		t.Errorf("PlanSchedule(0) error = %v, want ErrInvalidPromptTokens", err)
	}
	_, err = sched.PlanSchedule(-100)
	if !errors.Is(err, ErrInvalidPromptTokens) {
		t.Errorf("PlanSchedule(-100) error = %v, want ErrInvalidPromptTokens", err)
	}
}

func testWatchdogPrefillScheduler_FlopAndByteCalculations(t *testing.T) {
	geom := ModelGeometry27B()
	if err := geom.Validate(); err != nil {
		t.Fatalf("ModelGeometry27B() validation failed: %v", err)
	}

	// 1. Model geometry assertions
	if geom.Layers != 34 {
		t.Errorf("geom.Layers = %d, want 34", geom.Layers)
	}
	if geom.HeadDim != 128 {
		t.Errorf("geom.HeadDim = %d, want 128", geom.HeadDim)
	}
	if geom.NumKVHeads != 8 {
		t.Errorf("geom.NumKVHeads = %d, want 8", geom.NumKVHeads)
	}

	params := geom.TotalActiveParameters()
	if params < 10_000_000_000 || params > 35_000_000_000 {
		t.Errorf("geom.TotalActiveParameters() = %d, expected in 10B-35B range", params)
	}

	weightBytes := geom.WeightSizeBytes()
	gib := float64(weightBytes) / (1024 * 1024 * 1024)
	if gib < 5.0 || gib > 22.0 {
		t.Errorf("geom.WeightSizeBytes() = %.2f GiB, expected in 5-22 GiB range", gib)
	}

	kvPerToken := geom.KVBytesPerToken()
	// 2 * 34 layers * 8 kvHeads * 128 headDim * 2 bytes = 139,264 bytes/token
	expectedKVPerToken := int64(2 * 34 * 8 * 128 * 2)
	if kvPerToken != expectedKVPerToken {
		t.Errorf("KVBytesPerToken = %d, want %d", kvPerToken, expectedKVPerToken)
	}

	sched, err := NewDefaultAPUPrefillScheduler(geom)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	// 2. FLOP & Byte scaling with context depth
	// Attention FLOPs grow with start token pos due to causal attention history
	chunkLen := 512
	flopsEarly, readEarly, writeEarly, durEarly, compEarly, memEarly, _ := sched.EstimateChunkPacing(0, chunkLen, 1)
	flopsDeep, readDeep, writeDeep, durDeep, compDeep, memDeep, _ := sched.EstimateChunkPacing(136000, chunkLen, 1)

	if flopsDeep <= flopsEarly {
		t.Errorf("flops at 136k (%d) should exceed flops at pos 0 (%d)", flopsDeep, flopsEarly)
	}
	if readDeep <= readEarly {
		t.Errorf("bytesRead at 136k (%d) should exceed bytesRead at pos 0 (%d)", readDeep, readEarly)
	}
	if durDeep <= durEarly {
		t.Errorf("duration at 136k (%.2f ms) should exceed duration at pos 0 (%.2f ms)", durDeep, durEarly)
	}

	// 3. Verify roofline timing model: duration == max(compute, memory) + overhead
	expectedDurEarly := math.Max(compEarly, memEarly) + sched.hardware.KernelLaunchOverheadMs
	if math.Abs(durEarly-expectedDurEarly) > 1e-4 {
		t.Errorf("durEarly = %.4f, want %.4f", durEarly, expectedDurEarly)
	}

	expectedDurDeep := math.Max(compDeep, memDeep) + sched.hardware.KernelLaunchOverheadMs
	if math.Abs(durDeep-expectedDurDeep) > 1e-4 {
		t.Errorf("durDeep = %.4f, want %.4f", durDeep, expectedDurDeep)
	}

	// 4. Zero-FLOP operations verification: F16 KV contiguization
	// Compare context < ContiguizationMinContext (32768) vs context >= 32768
	geomNoContig := geom
	geomNoContig.ContiguizationPass = false
	schedNoContig, err := NewDefaultAPUPrefillScheduler(geomNoContig)
	if err != nil {
		t.Fatalf("failed to create schedNoContig: %v", err)
	}

	flopsWithContig, bytesReadWithContig, bytesWriteWithContig, _, _, _, _ := sched.EstimateChunkPacing(40000, 512, 1)
	flopsNoContig, bytesReadNoContig, bytesWriteNoContig, _, _, _, _ := schedNoContig.EstimateChunkPacing(40000, 512, 1)

	// Contiguization adds memory read/write traffic with 0 FLOPs!
	if flopsWithContig != flopsNoContig {
		t.Errorf("contiguization should add ZERO FLOPs: with=%d, without=%d", flopsWithContig, flopsNoContig)
	}
	if (bytesReadWithContig + bytesWriteWithContig) <= (bytesReadNoContig + bytesWriteNoContig) {
		t.Errorf("contiguization should add memory traffic: with=%d, without=%d",
			bytesReadWithContig+bytesWriteWithContig, bytesReadNoContig+bytesWriteNoContig)
	}
	_ = writeEarly
	_ = writeDeep
}

func testWatchdogPrefillScheduler_PacedExecutionSimulation(t *testing.T) {
	sched, err := NewDeepContextPrefillScheduler(ProfileSingleSequenceDeepContext)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	promptLen := 8192 // 8 chunks of 1024
	schedule, err := sched.PlanSchedule(promptLen)
	if err != nil {
		t.Fatalf("PlanSchedule(%d) error: %v", promptLen, err)
	}

	var executedChunks []int
	var progressReports []ProgressReport

	mockExecutor := func(ctx context.Context, chunk PrefillChunk) error {
		executedChunks = append(executedChunks, chunk.Index)
		return nil
	}

	mockHook := func(ctx context.Context, chunk PrefillChunk, report ProgressReport) error {
		progressReports = append(progressReports, report)
		return nil
	}

	ctx := context.Background()
	receipt, err := sched.Execute(ctx, schedule, mockExecutor, mockHook)
	if err != nil {
		t.Fatalf("Execute error = %v", err)
	}

	if !receipt.CompletedClean {
		t.Errorf("receipt.CompletedClean = false, want true")
	}
	if receipt.ChunksExecuted != len(schedule.Chunks) {
		t.Errorf("receipt.ChunksExecuted = %d, want %d", receipt.ChunksExecuted, len(schedule.Chunks))
	}
	if receipt.TotalTokens != promptLen {
		t.Errorf("receipt.TotalTokens = %d, want %d", receipt.TotalTokens, promptLen)
	}
	if receipt.YieldPointsHit != len(schedule.Chunks)-1 {
		t.Errorf("receipt.YieldPointsHit = %d, want %d", receipt.YieldPointsHit, len(schedule.Chunks)-1)
	}

	// Verify executor was invoked for all chunks in sequence
	if len(executedChunks) != len(schedule.Chunks) {
		t.Fatalf("executedChunks len = %d, want %d", len(executedChunks), len(schedule.Chunks))
	}
	for i, chunkIdx := range executedChunks {
		if chunkIdx != i {
			t.Errorf("executed chunk at step %d was chunkIdx %d", i, chunkIdx)
		}
	}

	// Verify progress reports
	if len(progressReports) != len(schedule.Chunks) {
		t.Fatalf("progressReports len = %d, want %d", len(progressReports), len(schedule.Chunks))
	}
	for i, report := range progressReports {
		if report.ChunkIndex != i {
			t.Errorf("report %d ChunkIndex = %d", i, report.ChunkIndex)
		}
		expectedTokens := (i + 1) * 1024
		if report.CompletedTokens != expectedTokens {
			t.Errorf("report %d CompletedTokens = %d, want %d", i, report.CompletedTokens, expectedTokens)
		}
	}

	finalReport := progressReports[len(progressReports)-1]
	if finalReport.PercentComplete != 100.0 {
		t.Errorf("final PercentComplete = %.2f, want 100.0", finalReport.PercentComplete)
	}
}

func testWatchdogPrefillScheduler_ContextCancellation(t *testing.T) {
	sched, err := NewDeepContextPrefillScheduler(ProfileSingleSequenceDeepContext)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	schedule, err := sched.PlanSchedule(4096) // 4 chunks of 1024
	if err != nil {
		t.Fatalf("PlanSchedule error: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	chunkCount := 0

	mockExecutor := func(c context.Context, chunk PrefillChunk) error {
		chunkCount++
		if chunkCount == 2 {
			cancel() // cancel context after chunk 2
		}
		return nil
	}

	receipt, err := sched.Execute(ctx, schedule, mockExecutor, nil)
	if err == nil {
		t.Fatalf("expected error on cancelled context, got nil")
	}
	if !errors.Is(err, ErrExecutionCancelled) {
		t.Errorf("expected ErrExecutionCancelled, got: %v", err)
	}
	if !receipt.Cancelled {
		t.Errorf("receipt.Cancelled = false, want true")
	}
	if receipt.CompletedClean {
		t.Errorf("receipt.CompletedClean = true, want false")
	}
	if receipt.ChunksExecuted != 2 {
		t.Errorf("receipt.ChunksExecuted = %d, want 2", receipt.ChunksExecuted)
	}
}

func testWatchdogPrefillScheduler_BatchedMultiSequence(t *testing.T) {
	sched, err := NewDeepContextPrefillScheduler(ProfileBatchedMultiSequence)
	if err != nil {
		t.Fatalf("failed to create scheduler: %v", err)
	}

	sequences := []int{1024, 2048, 4096} // total = 7168 tokens
	schedule, err := sched.PlanBatchedSchedule(sequences)
	if err != nil {
		t.Fatalf("PlanBatchedSchedule error: %v", err)
	}

	if schedule.TotalTokens != 7168 {
		t.Errorf("schedule.TotalTokens = %d, want 7168", schedule.TotalTokens)
	}
	if schedule.BatchSize != 3 {
		t.Errorf("schedule.BatchSize = %d, want 3", schedule.BatchSize)
	}

	// Batched multi-sequence defaults to 512 tokens
	for i, chunk := range schedule.Chunks {
		if !chunk.IsLastChunk && chunk.TokenCount > 512 {
			t.Errorf("chunk %d token count = %d, exceeds batched default 512", i, chunk.TokenCount)
		}
	}

	if err := schedule.Validate(2000.0); err != nil {
		t.Errorf("schedule validation failed: %v", err)
	}
}

func testWatchdogPrefillScheduler_ProfilesAndHardware(t *testing.T) {
	// Profiles validation
	pSingle := DefaultSingleSequenceProfile()
	if err := pSingle.Validate(); err != nil {
		t.Errorf("DefaultSingleSequenceProfile validation failed: %v", err)
	}
	if pSingle.DefaultChunkTokens != 1024 {
		t.Errorf("DefaultChunkTokens = %d, want 1024", pSingle.DefaultChunkTokens)
	}
	if pSingle.MaxExecutionCeilingMs != 2000.0 {
		t.Errorf("MaxExecutionCeilingMs = %.2f, want 2000.0", pSingle.MaxExecutionCeilingMs)
	}
	if pSingle.WatchdogTimeoutMs != 10000.0 {
		t.Errorf("WatchdogTimeoutMs = %.2f, want 10000.0", pSingle.WatchdogTimeoutMs)
	}

	pBatch := DefaultBatchedMultiSequenceProfile(4)
	if err := pBatch.Validate(); err != nil {
		t.Errorf("DefaultBatchedMultiSequenceProfile validation failed: %v", err)
	}
	if pBatch.DefaultChunkTokens != 512 {
		t.Errorf("DefaultChunkTokens = %d, want 512", pBatch.DefaultChunkTokens)
	}
	if pBatch.BatchSize != 4 {
		t.Errorf("BatchSize = %d, want 4", pBatch.BatchSize)
	}

	pCustom, err := NewCustomChunkingProfile(768, 1536, 128, 1500.0, 8000.0, 2, true)
	if err != nil {
		t.Fatalf("NewCustomChunkingProfile failed: %v", err)
	}
	if pCustom.DefaultChunkTokens != 768 {
		t.Errorf("pCustom.DefaultChunkTokens = %d, want 768", pCustom.DefaultChunkTokens)
	}

	// Hardware profiles validation
	hwHalo := StrixHaloHardwareProfile()
	if err := hwHalo.Validate(); err != nil {
		t.Errorf("StrixHaloHardwareProfile validation failed: %v", err)
	}
	if hwHalo.Architecture != "gfx1151" {
		t.Errorf("hwHalo.Architecture = %s, want gfx1151", hwHalo.Architecture)
	}
	if hwHalo.ComputeUnits != 40 {
		t.Errorf("hwHalo.ComputeUnits = %d, want 40", hwHalo.ComputeUnits)
	}

	hwPoint := StrixPointHardwareProfile()
	if err := hwPoint.Validate(); err != nil {
		t.Errorf("StrixPointHardwareProfile validation failed: %v", err)
	}
	if hwPoint.Architecture != "gfx1150" {
		t.Errorf("hwPoint.Architecture = %s, want gfx1150", hwPoint.Architecture)
	}

	hwPhoenix := PhoenixHardwareProfile()
	if err := hwPhoenix.Validate(); err != nil {
		t.Errorf("PhoenixHardwareProfile validation failed: %v", err)
	}
	if hwPhoenix.Architecture != "gfx1103" {
		t.Errorf("hwPhoenix.Architecture = %s, want gfx1103", hwPhoenix.Architecture)
	}
}

func testWatchdogPrefillScheduler_AdaptiveDownsizingAtExtremeDepth(t *testing.T) {
	// Under a constrained execution ceiling (e.g. 500 ms) or very deep context,
	// the adaptive scheduler should downsize chunk tokens below the default 1024
	// to ensure every chunk stays under the ceiling.
	geom := ModelGeometry27B()
	profile, err := NewCustomChunkingProfile(1024, 2048, 64, 500.0, 10000.0, 1, true)
	if err != nil {
		t.Fatalf("NewCustomChunkingProfile failed: %v", err)
	}

	hw := StrixHaloHardwareProfile()
	sched, err := NewWatchdogPrefillScheduler(profile, geom, hw)
	if err != nil {
		t.Fatalf("NewWatchdogPrefillScheduler failed: %v", err)
	}

	promptLen := 16384
	schedule, err := sched.PlanSchedule(promptLen)
	if err != nil {
		t.Fatalf("PlanSchedule(%d) error: %v", promptLen, err)
	}

	if !schedule.IsWatchdogSafe {
		t.Errorf("schedule should be watchdog safe with adaptive downsizing")
	}

	// Verify all chunks respect the 500 ms ceiling
	hasDownsizedChunk := false
	for _, chunk := range schedule.Chunks {
		if chunk.EstimatedDurationMs > 500.0 {
			t.Errorf("chunk %d duration %.2f ms exceeds 500 ms ceiling", chunk.Index, chunk.EstimatedDurationMs)
		}
		if chunk.TokenCount < 1024 {
			hasDownsizedChunk = true
		}
	}

	if !hasDownsizedChunk {
		t.Errorf("expected at least one chunk to be adaptively downsized under 500 ms ceiling")
	}

	summary := schedule.Summary()
	if len(summary) == 0 {
		t.Errorf("expected non-empty schedule summary")
	}
}
