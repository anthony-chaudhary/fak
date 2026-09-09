// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"sync"
	"testing"
	"time"
)

// TestMALLTemporalPinningSpeculativeTreeSpine validates Issue #506 / TICKET-12:
//   - PinSpeculativeTreeMasks pins within 32 MiB budget with CacheHintTemporal.
//   - Dual cache policy: weights stream with CacheHintStreamingBypass (SLC=1, GLC=0, NT=1),
//     while KV cache + tree masks stay pinned with CacheHintTemporal (SLC=0, GLC=0, NT=0).
//   - EvaluateSpeculativeVerification reports >= 50% DRAM bandwidth reduction.
//   - UnpinSpeculativeTreeMasks cleanly releases pinned quota.
//   - Budget overflow fails closed (rejects allocations > 32 MiB).
func TestMALLTemporalPinningTreeAttentionSpine(t *testing.T) {
	t.Run("PinSpeculativeTreeMasksWithinBudget", func(t *testing.T) {
		tiler := NewMALLTiler()

		treeDepth := 4
		numBranches := 4
		alloc, err := tiler.PinSpeculativeTreeMasks(treeDepth, numBranches)
		if err != nil {
			t.Fatalf("PinSpeculativeTreeMasks failed: %v", err)
		}
		if alloc == nil {
			t.Fatal("expected non-nil allocation")
		}

		if alloc.AllocID == "" {
			t.Error("expected non-empty AllocID")
		}
		if alloc.TreeDepth != treeDepth {
			t.Errorf("expected TreeDepth %d, got %d", treeDepth, alloc.TreeDepth)
		}
		if alloc.NumBranches != numBranches {
			t.Errorf("expected NumBranches %d, got %d", numBranches, alloc.NumBranches)
		}

		expectedMaskBytes := int64(numBranches * treeDepth * treeDepth * 4) // 4 * 4 * 4 * 4 = 256
		if alloc.MaskBytes != expectedMaskBytes {
			t.Errorf("expected MaskBytes %d, got %d", expectedMaskBytes, alloc.MaskBytes)
		}

		expectedKVCacheBytes := int64(numBranches*treeDepth) * StrixHaloBytesPerToken // 16 * 4096 = 65,536
		if alloc.KVCacheBytes != expectedKVCacheBytes {
			t.Errorf("expected KVCacheBytes %d, got %d", expectedKVCacheBytes, alloc.KVCacheBytes)
		}

		expectedTotal := expectedMaskBytes + expectedKVCacheBytes // 65,792
		if alloc.TotalPinnedBytes != expectedTotal {
			t.Errorf("expected TotalPinnedBytes %d, got %d", expectedTotal, alloc.TotalPinnedBytes)
		}
		if alloc.TotalPinnedBytes > StrixHaloMALLSizeBytes {
			t.Fatalf("TotalPinnedBytes %d exceeds 32 MiB capacity %d", alloc.TotalPinnedBytes, StrixHaloMALLSizeBytes)
		}

		// Cache policy must be CacheHintTemporal (SLC=0, GLC=0, NT=0)
		if alloc.CachePolicy.SLC != 0 || alloc.CachePolicy.GLC != 0 || alloc.CachePolicy.NT != 0 {
			t.Errorf("expected SLC=0, GLC=0, NT=0, got SLC=%d GLC=%d NT=%d",
				alloc.CachePolicy.SLC, alloc.CachePolicy.GLC, alloc.CachePolicy.NT)
		}
		if !alloc.CachePolicy.Temporal || alloc.CachePolicy.Bypass {
			t.Errorf("expected Temporal=true, Bypass=false, got Temporal=%v Bypass=%v",
				alloc.CachePolicy.Temporal, alloc.CachePolicy.Bypass)
		}
		if alloc.CachePolicy.PolicyName != "TEMPORAL_PINNED" {
			t.Errorf("expected policy name 'TEMPORAL_PINNED', got %q", alloc.CachePolicy.PolicyName)
		}

		if alloc.PinnedAt.IsZero() || alloc.PinnedAt.After(time.Now()) {
			t.Errorf("invalid PinnedAt timestamp: %v", alloc.PinnedAt)
		}

		if tiler.TreeMaskBytesPinnedCount() != expectedMaskBytes {
			t.Errorf("expected TreeMaskBytesPinnedCount %d, got %d", expectedMaskBytes, tiler.TreeMaskBytesPinnedCount())
		}

		allocs := tiler.PinnedMaskAllocations()
		if len(allocs) != 1 {
			t.Fatalf("expected 1 active allocation, got %d", len(allocs))
		}
		if allocs[0].AllocID != alloc.AllocID {
			t.Errorf("allocation ID mismatch: %q != %q", allocs[0].AllocID, alloc.AllocID)
		}
	})

	t.Run("DualCachePolicyWeightsBypassKVPinned", func(t *testing.T) {
		tiler := NewMALLTiler()

		// 1. Model weights stream with CacheHintStreamingBypass (SLC=1, GLC=0, NT=1)
		weightHint := tiler.ClassifyStreamingWeights()
		if weightHint.SLC != 1 {
			t.Errorf("streaming weights: expected SLC=1 (MALL bypass), got %d", weightHint.SLC)
		}
		if weightHint.GLC != 0 {
			t.Errorf("streaming weights: expected GLC=0, got %d", weightHint.GLC)
		}
		if weightHint.NT != 1 {
			t.Errorf("streaming weights: expected NT=1 (non-temporal streaming), got %d", weightHint.NT)
		}
		if weightHint.Temporal {
			t.Error("streaming weights: expected Temporal=false, got true")
		}
		if !weightHint.Bypass {
			t.Error("streaming weights: expected Bypass=true, got false")
		}
		if weightHint.PolicyName != "STREAMING_BYPASS" {
			t.Errorf("streaming weights: expected PolicyName 'STREAMING_BYPASS', got %q", weightHint.PolicyName)
		}

		// 2. Active KV cache stays pinned with CacheHintTemporal (SLC=0, GLC=0, NT=0)
		kvHint := tiler.ClassifyTokenSpan(0, StrixHaloMALLCapacityTokens)
		if kvHint.SLC != 0 {
			t.Errorf("pinned KV cache: expected SLC=0 (MALL allocate), got %d", kvHint.SLC)
		}
		if kvHint.GLC != 0 {
			t.Errorf("pinned KV cache: expected GLC=0, got %d", kvHint.GLC)
		}
		if kvHint.NT != 0 {
			t.Errorf("pinned KV cache: expected NT=0 (temporal reuse), got %d", kvHint.NT)
		}
		if !kvHint.Temporal {
			t.Error("pinned KV cache: expected Temporal=true, got false")
		}
		if kvHint.Bypass {
			t.Error("pinned KV cache: expected Bypass=false, got true")
		}
		if kvHint.PolicyName != "TEMPORAL_PINNED" {
			t.Errorf("pinned KV cache: expected PolicyName 'TEMPORAL_PINNED', got %q", kvHint.PolicyName)
		}

		// 3. Speculative tree mask allocation carries CacheHintTemporal (SLC=0, GLC=0, NT=0)
		treeAlloc, err := tiler.PinSpeculativeTreeMasks(8, 4)
		if err != nil {
			t.Fatalf("PinSpeculativeTreeMasks failed: %v", err)
		}
		if treeAlloc.CachePolicy.SLC != 0 || treeAlloc.CachePolicy.GLC != 0 || treeAlloc.CachePolicy.NT != 0 {
			t.Errorf("tree mask allocation: expected SLC=0, GLC=0, NT=0, got SLC=%d GLC=%d NT=%d",
				treeAlloc.CachePolicy.SLC, treeAlloc.CachePolicy.GLC, treeAlloc.CachePolicy.NT)
		}
		if !treeAlloc.CachePolicy.Temporal || treeAlloc.CachePolicy.Bypass {
			t.Errorf("tree mask allocation: expected Temporal=true, Bypass=false")
		}

		// 4. Weight blocks scheduled for streaming have streaming bypass hint
		weightBlocks := tiler.ScheduleWeightBlocks(1024*1024, StrixHaloBlockSizeBytes)
		if len(weightBlocks) == 0 {
			t.Fatal("expected non-empty weight blocks")
		}
		for i, wb := range weightBlocks {
			if wb.Hint.SLC != 1 || wb.Hint.NT != 1 || !wb.Hint.Bypass || wb.Hint.Temporal {
				t.Errorf("weight block %d hint invalid: %+v", i, wb.Hint)
			}
		}
	})

	t.Run("EvaluateSpeculativeVerificationBandwidthReduction", func(t *testing.T) {
		tiler := NewMALLTiler()

		for _, tc := range []struct {
			tokens      int
			numBranches int
			treeDepth   int
		}{
			{tokens: 4096, numBranches: 4, treeDepth: 8},
			{tokens: 8192, numBranches: 8, treeDepth: 4},
			{tokens: 2048, numBranches: 2, treeDepth: 6},
			{tokens: 0, numBranches: 1, treeDepth: 1}, // edge case
		} {
			profile := tiler.EvaluateSpeculativeVerification(tc.tokens, tc.numBranches, tc.treeDepth)

			// Invariant 1: Bandwidth reduction ratio >= 0.50 (>= 50% memory traffic avoided)
			if profile.BandwidthReductionRatio < 0.50 {
				t.Errorf("tokens=%d branches=%d depth=%d: expected BandwidthReductionRatio >= 0.50, got %.4f",
					tc.tokens, tc.numBranches, tc.treeDepth, profile.BandwidthReductionRatio)
			}

			// Invariant 2: Positive DRAM reads bypassed
			if profile.DRAMReadsBypassedBytes <= 0 {
				t.Errorf("tokens=%d branches=%d depth=%d: expected positive DRAMReadsBypassedBytes, got %d",
					tc.tokens, tc.numBranches, tc.treeDepth, profile.DRAMReadsBypassedBytes)
			}

			// Invariant 3: WeightCachePolicy is CacheHintStreamingBypass (SLC=1, GLC=0, NT=1)
			if profile.WeightCachePolicy.SLC != 1 || profile.WeightCachePolicy.GLC != 0 || profile.WeightCachePolicy.NT != 1 {
				t.Errorf("expected WeightCachePolicy SLC=1, GLC=0, NT=1, got SLC=%d GLC=%d NT=%d",
					profile.WeightCachePolicy.SLC, profile.WeightCachePolicy.GLC, profile.WeightCachePolicy.NT)
			}
			if profile.WeightCachePolicy.Temporal || !profile.WeightCachePolicy.Bypass {
				t.Errorf("expected WeightCachePolicy Temporal=false, Bypass=true")
			}

			// Invariant 4: KVCachePolicy is CacheHintTemporal (SLC=0, GLC=0, NT=0)
			if profile.KVCachePolicy.SLC != 0 || profile.KVCachePolicy.GLC != 0 || profile.KVCachePolicy.NT != 0 {
				t.Errorf("expected KVCachePolicy SLC=0, GLC=0, NT=0, got SLC=%d GLC=%d NT=%d",
					profile.KVCachePolicy.SLC, profile.KVCachePolicy.GLC, profile.KVCachePolicy.NT)
			}
			if !profile.KVCachePolicy.Temporal || profile.KVCachePolicy.Bypass {
				t.Errorf("expected KVCachePolicy Temporal=true, Bypass=false")
			}

			// Invariant 5: Effective bandwidth is >= 273.0 GB/s (DRAM floor) and up to 1200.0 GB/s (MALL peak)
			if profile.EffectiveBandwidthGBs < StrixHaloPhysicalDRAMBandwidthGBs {
				t.Errorf("expected EffectiveBandwidthGBs >= %.2f GB/s, got %.2f",
					StrixHaloPhysicalDRAMBandwidthGBs, profile.EffectiveBandwidthGBs)
			}
		}
	})

	t.Run("UnpinSpeculativeTreeMasksCleanlyReleasesQuota", func(t *testing.T) {
		tiler := NewMALLTiler()

		alloc, err := tiler.PinSpeculativeTreeMasks(16, 4)
		if err != nil {
			t.Fatalf("PinSpeculativeTreeMasks failed: %v", err)
		}
		if tiler.TreeMaskBytesPinnedCount() != alloc.MaskBytes {
			t.Fatalf("expected %d pinned mask bytes, got %d", alloc.MaskBytes, tiler.TreeMaskBytesPinnedCount())
		}

		err = tiler.UnpinSpeculativeTreeMasks(alloc.AllocID)
		if err != nil {
			t.Fatalf("UnpinSpeculativeTreeMasks failed: %v", err)
		}

		if tiler.TreeMaskBytesPinnedCount() != 0 {
			t.Errorf("expected 0 pinned mask bytes after unpin, got %d", tiler.TreeMaskBytesPinnedCount())
		}
		if len(tiler.PinnedMaskAllocations()) != 0 {
			t.Errorf("expected 0 active allocations after unpin, got %d", len(tiler.PinnedMaskAllocations()))
		}

		// Re-unpinning the same ID must return an error
		errAgain := tiler.UnpinSpeculativeTreeMasks(alloc.AllocID)
		if errAgain == nil {
			t.Error("expected error when unpinning already-removed allocation ID")
		}

		// Unpinning a non-existent ID must return an error
		errNonExistent := tiler.UnpinSpeculativeTreeMasks("non-existent-id")
		if errNonExistent == nil {
			t.Error("expected error when unpinning non-existent allocation ID")
		}
	})

	t.Run("BudgetOverflowFailsClosed", func(t *testing.T) {
		tiler := NewMALLTiler()

		// 1. Single allocation exceeding 32 MiB capacity:
		// treeDepth = 100, numBranches = 100 -> kvCacheBytes = 100 * 100 * 4096 = 40,960,000 bytes > 32 MiB
		allocHuge, errHuge := tiler.PinSpeculativeTreeMasks(100, 100)
		if errHuge == nil {
			t.Error("expected error for allocation exceeding 32 MiB capacity")
		}
		if allocHuge != nil {
			t.Errorf("expected nil allocation on capacity overflow, got %+v", allocHuge)
		}
		if tiler.TreeMaskBytesPinnedCount() != 0 {
			t.Errorf("expected 0 pinned bytes after rejected allocation, got %d", tiler.TreeMaskBytesPinnedCount())
		}

		// 2. Cumulative budget overflow:
		// Pin allocation near 32 MiB capacity:
		// treeDepth = 64, numBranches = 120:
		// kvCacheBytes = 64 * 120 * 4096 = 31,457,280 bytes
		// maskBytes = 120 * 64 * 64 * 4 = 1,966,080 bytes
		// total = 33,423,360 bytes (within 33,554,432 bytes)
		alloc1, err1 := tiler.PinSpeculativeTreeMasks(64, 120)
		if err1 != nil {
			t.Fatalf("first allocation within budget failed: %v", err1)
		}
		if alloc1 == nil {
			t.Fatal("expected non-nil alloc1")
		}

		// Attempt second allocation that exceeds remaining headroom:
		// treeDepth = 8, numBranches = 8:
		// total = 8 * 8 * 4096 + 8 * 8 * 8 * 4 = 262,144 + 2,048 = 264,192 bytes
		// 33,423,360 + 264,192 = 33,687,552 > 33,554,432
		alloc2, err2 := tiler.PinSpeculativeTreeMasks(8, 8)
		if err2 == nil {
			t.Error("expected cumulative budget overflow error")
		}
		if alloc2 != nil {
			t.Errorf("expected nil alloc2 on overflow, got %+v", alloc2)
		}

		// Unpin alloc1 to release quota
		if err := tiler.UnpinSpeculativeTreeMasks(alloc1.AllocID); err != nil {
			t.Fatalf("unpin alloc1 failed: %v", err)
		}

		// Now alloc2 must succeed
		alloc2Retry, errRetry := tiler.PinSpeculativeTreeMasks(8, 8)
		if errRetry != nil {
			t.Fatalf("alloc2 retry after unpinning failed: %v", errRetry)
		}
		if alloc2Retry == nil {
			t.Fatal("expected non-nil alloc2Retry")
		}

		// 3. Invalid inputs fail closed
		if _, err := tiler.PinSpeculativeTreeMasks(0, 4); err == nil {
			t.Error("expected error for treeDepth <= 0")
		}
		if _, err := tiler.PinSpeculativeTreeMasks(-5, 4); err == nil {
			t.Error("expected error for treeDepth < 0")
		}
		if _, err := tiler.PinSpeculativeTreeMasks(4, 0); err == nil {
			t.Error("expected error for numBranches <= 0")
		}
		if _, err := tiler.PinSpeculativeTreeMasks(4, -3); err == nil {
			t.Error("expected error for numBranches < 0")
		}
	})

	t.Run("ConcurrentPinUnpinUnderRace", func(t *testing.T) {
		tiler := NewMALLTiler()

		const workers = 8
		const iters = 20

		var wg sync.WaitGroup
		errCh := make(chan error, workers*iters)

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for i := 0; i < iters; i++ {
					alloc, err := tiler.PinSpeculativeTreeMasks(4, 2)
					if err != nil {
						errCh <- err
						continue
					}
					if alloc.CachePolicy.SLC != 0 || alloc.CachePolicy.NT != 0 {
						t.Errorf("worker %d iter %d: expected CacheHintTemporal", workerID, i)
					}

					profile := tiler.EvaluateSpeculativeVerification(2048, 2, 4)
					if profile.BandwidthReductionRatio < 0.50 {
						t.Errorf("worker %d iter %d: bandwidth reduction ratio %.2f < 0.50",
							workerID, i, profile.BandwidthReductionRatio)
					}

					if err := tiler.UnpinSpeculativeTreeMasks(alloc.AllocID); err != nil {
						errCh <- err
					}
				}
			}(w)
		}

		wg.Wait()
		close(errCh)

		for err := range errCh {
			t.Fatalf("concurrent worker error: %v", err)
		}

		if tiler.TreeMaskBytesPinnedCount() != 0 {
			t.Errorf("expected 0 pinned mask bytes after all workers finished, got %d", tiler.TreeMaskBytesPinnedCount())
		}
		if len(tiler.PinnedMaskAllocations()) != 0 {
			t.Errorf("expected 0 active allocations after all workers finished, got %d", len(tiler.PinnedMaskAllocations()))
		}
	})
}
