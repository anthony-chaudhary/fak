package strix

import (
	"errors"
	"strings"
	"sync"
	"testing"
)

// TestStreamingBypass validates the non-temporal weight streaming pipeline against
// all scoped acceptance criteria, ablation arms, and hardware invariants.
func TestStreamingBypass(t *testing.T) {
	t.Run("Criterion2_MultiGigabyteAlignedScheduling", func(t *testing.T) {
		cfg := DefaultStreamingBypassConfig()
		cfg.ChunkSizeBytes = 2 * 1024 * 1024 // 2MB chunks

		scheduler, err := NewStreamingBypassScheduler(cfg, nil, nil)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		// Simulate scheduling a 17.5 GB (17,500 MB) 35B quantized model weight footprint
		baseAddr := uint64(0x00007F0000000000) // 128-byte aligned
		sizeBytes := Default35BWeightBytes

		chunks, err := scheduler.ScheduleWeightStream(baseAddr, sizeBytes)
		if err != nil {
			t.Fatalf("ScheduleWeightStream failed: %v", err)
		}

		expectedChunks := int((sizeBytes + cfg.ChunkSizeBytes - 1) / cfg.ChunkSizeBytes)
		if len(chunks) != expectedChunks {
			t.Fatalf("expected %d chunks, got %d", expectedChunks, len(chunks))
		}

		for idx, chunk := range chunks {
			// Verify 128-byte alignment
			if chunk.BaseAddress%uint64(DefaultStreamingBurstStrideBytes) != 0 {
				t.Fatalf("chunk %d address 0x%x not 128-byte aligned", idx, chunk.BaseAddress)
			}
			if chunk.SizeBytes%int64(DefaultStreamingBurstStrideBytes) != 0 && idx < len(chunks)-1 {
				t.Fatalf("chunk %d size %d not 128-byte aligned", idx, chunk.SizeBytes)
			}

			// Verify SLC=1 bypass bitmask
			if chunk.Modifier.SLC != 1 {
				t.Fatalf("chunk %d expected SLC=1, got SLC=%d", idx, chunk.Modifier.SLC)
			}
			if chunk.Modifier.GLC != 0 || chunk.Modifier.DLC != 0 {
				t.Fatalf("chunk %d expected GLC=0, DLC=0, got GLC=%d DLC=%d", idx, chunk.Modifier.GLC, chunk.Modifier.DLC)
			}
			if !strings.Contains(chunk.Modifier.AssemblySuffix, "slc") {
				t.Fatalf("chunk %d missing 'slc' in assembly suffix: %q", idx, chunk.Modifier.AssemblySuffix)
			}
		}

		telemetry := scheduler.Telemetry()
		if telemetry.TotalBytesScheduled != sizeBytes {
			t.Fatalf("expected TotalBytesScheduled=%d, got %d", sizeBytes, telemetry.TotalBytesScheduled)
		}
		if telemetry.TotalChunksScheduled != int64(expectedChunks) {
			t.Fatalf("expected TotalChunksScheduled=%d, got %d", expectedChunks, telemetry.TotalChunksScheduled)
		}
	})

	t.Run("Criterion3_PrefetchInterleavingLookahead", func(t *testing.T) {
		cfg := DefaultStreamingBypassConfig()
		cfg.ChunkSizeBytes = 64 * 1024 // 64 KB chunks for granular lookahead verification
		cfg.LookaheadWavefronts = 2
		cfg.EnablePrefetch = true
		cfg.AblationArm = Arm1NonTemporalWithPrefetch

		scheduler, err := NewStreamingBypassScheduler(cfg, nil, nil)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		baseAddr := uint64(0x00007F1000000000)
		sizeBytes := int64(10 * 1024 * 1024) // 10 MB

		chunks, err := scheduler.ScheduleWeightStream(baseAddr, sizeBytes)
		if err != nil {
			t.Fatalf("ScheduleWeightStream failed: %v", err)
		}

		prefetchesSeen := 0
		for idx, chunk := range chunks {
			if chunk.PrefetchIssued {
				prefetchesSeen++
				expectedPrefetchAddr := chunk.BaseAddress + uint64(cfg.LookaheadWavefronts)*uint64(cfg.ChunkSizeBytes)
				if chunk.PrefetchAddress != expectedPrefetchAddr {
					t.Fatalf("chunk %d prefetch address mismatch: expected 0x%x, got 0x%x",
						idx, expectedPrefetchAddr, chunk.PrefetchAddress)
				}

				// Verify s_prefetch_data token in assembly
				hasPrefetchToken := false
				for _, token := range chunk.AssemblyTokens {
					if strings.HasPrefix(token, PrefetchInstructionOpcode) {
						hasPrefetchToken = true
						break
					}
				}
				if !hasPrefetchToken {
					t.Fatalf("chunk %d marked prefetch_issued but missing %s in assembly tokens",
						idx, PrefetchInstructionOpcode)
				}
			}
		}

		if prefetchesSeen == 0 {
			t.Fatal("expected prefetch instructions to be interleaved, got 0")
		}

		telemetry := scheduler.Telemetry()
		if telemetry.TotalPrefetchIssued != int64(prefetchesSeen) {
			t.Fatalf("telemetry mismatch: expected %d prefetches, recorded %d",
				prefetchesSeen, telemetry.TotalPrefetchIssued)
		}
	})

	t.Run("Criterion4_MALLRootResidencyPreservationSimulation", func(t *testing.T) {
		cfg := DefaultStreamingBypassConfig()
		cfg.ChunkSizeBytes = 2 * 1024 * 1024
		cfg.AblationArm = Arm1NonTemporalWithPrefetch

		mall := NewMALLCacheModel()
		// Pin 16 MB of hot root KV cache in the 32MB MALL Infinity Cache
		rootKVAddr := uint64(0x00007F2000000000)
		rootKVBytes := int64(16 * 1024 * 1024)
		mall.PinKVRange(rootKVAddr, rootKVBytes)

		initialResidency := mall.PinnedKVResidencyPct()
		if initialResidency < 99.9 {
			t.Fatalf("expected initial root KV residency ~100%%, got %.1f%%", initialResidency)
		}

		scheduler, err := NewStreamingBypassScheduler(cfg, nil, mall)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		// Stream 17.5 GB of model weights with SLC=1
		weightAddr := uint64(0x00007F3000000000)
		weightSize := Default35BWeightBytes

		chunks, err := scheduler.ScheduleWeightStream(weightAddr, weightSize)
		if err != nil {
			t.Fatalf("ScheduleWeightStream failed: %v", err)
		}

		telemetry, err := scheduler.ExecuteSimulatedStream(chunks)
		if err != nil {
			t.Fatalf("ExecuteSimulatedStream failed: %v", err)
		}

		// Invariant: Non-temporal weight streaming must preserve >= 95% (here 100%) of MALL root KV
		if telemetry.MALLRootResidencyPercent < MinimalMALLResidencyThreshold {
			t.Fatalf("MALL root KV residency dropped below threshold: expected >= %.1f%%, got %.1f%%",
				MinimalMALLResidencyThreshold, telemetry.MALLRootResidencyPercent)
		}

		if mall.PinnedKVEvictionCount() != 0 {
			t.Fatalf("expected 0 pinned KV evictions under SLC=1, observed %d", mall.PinnedKVEvictionCount())
		}

		if telemetry.SustainedDRAMBandwidthGBps < TargetStreamingBWGBs {
			t.Fatalf("expected sustained DRAM bandwidth >= %.1f GB/s, got %.1f GB/s",
				TargetStreamingBWGBs, telemetry.SustainedDRAMBandwidthGBps)
		}
	})

	t.Run("AblationArms_Comparison", func(t *testing.T) {
		baseWeightAddr := uint64(0x00007F4000000000)
		weightSize := int64(64 * 1024 * 1024) // 64 MB stream (> 32MB MALL)

		// Arm 1: Non-temporal streaming with prefetch
		cfgArm1 := DefaultStreamingBypassConfig()
		cfgArm1.AblationArm = Arm1NonTemporalWithPrefetch
		cfgArm1.ChunkSizeBytes = 2 * 1024 * 1024

		mall1 := NewMALLCacheModel()
		mall1.PinKVRange(0x00007F5000000000, 16*1024*1024)

		sched1, _ := NewStreamingBypassScheduler(cfgArm1, nil, mall1)
		chunks1, _ := sched1.ScheduleWeightStream(baseWeightAddr, weightSize)
		telem1, _ := sched1.ExecuteSimulatedStream(chunks1)
		sched1.Close()

		if telem1.MALLRootResidencyPercent < 95.0 {
			t.Fatalf("Arm 1 expected MALL residency >= 95%%, got %.1f%%", telem1.MALLRootResidencyPercent)
		}
		if telem1.TotalPrefetchIssued == 0 {
			t.Fatal("Arm 1 expected prefetch instructions issued")
		}

		// Arm 2: Non-temporal streaming without prefetch
		cfgArm2 := DefaultStreamingBypassConfig()
		cfgArm2.AblationArm = Arm2NonTemporalNoPrefetch
		cfgArm2.ChunkSizeBytes = 2 * 1024 * 1024

		mall2 := NewMALLCacheModel()
		mall2.PinKVRange(0x00007F5000000000, 16*1024*1024)

		sched2, _ := NewStreamingBypassScheduler(cfgArm2, nil, mall2)
		chunks2, _ := sched2.ScheduleWeightStream(baseWeightAddr, weightSize)
		telem2, _ := sched2.ExecuteSimulatedStream(chunks2)
		sched2.Close()

		if telem2.MALLRootResidencyPercent < 95.0 {
			t.Fatalf("Arm 2 expected MALL residency >= 95%%, got %.1f%%", telem2.MALLRootResidencyPercent)
		}
		if telem2.TotalPrefetchIssued != 0 {
			t.Fatalf("Arm 2 expected 0 prefetches, got %d", telem2.TotalPrefetchIssued)
		}
		if telem2.SustainedDRAMBandwidthGBps >= telem1.SustainedDRAMBandwidthGBps {
			t.Fatalf("Arm 2 without prefetch should have lower bandwidth than Arm 1 (got %.1f vs %.1f)",
				telem2.SustainedDRAMBandwidthGBps, telem1.SustainedDRAMBandwidthGBps)
		}

		// Arm 3: Cached weights baseline (SLC=0, thrashing MALL)
		cfgArm3 := DefaultStreamingBypassConfig()
		cfgArm3.AblationArm = Arm3CachedWeightsBaseline
		cfgArm3.ChunkSizeBytes = 2 * 1024 * 1024

		mall3 := NewMALLCacheModel()
		mall3.PinKVRange(0x00007F5000000000, 16*1024*1024)

		sched3, _ := NewStreamingBypassScheduler(cfgArm3, nil, mall3)
		chunks3, _ := sched3.ScheduleWeightStream(baseWeightAddr, weightSize)
		telem3, _ := sched3.ExecuteSimulatedStream(chunks3)
		sched3.Close()

		// Invariant: Arm 3 cached weights thrash the 32MB MALL, evicting pinned KV
		if telem3.MALLRootResidencyPercent >= 50.0 {
			t.Fatalf("Arm 3 expected severe MALL eviction (< 50%%), got %.1f%%", telem3.MALLRootResidencyPercent)
		}
		if mall3.PinnedKVEvictionCount() == 0 {
			t.Fatal("Arm 3 expected non-zero pinned KV evictions")
		}
	})

	t.Run("QueueStall_BackpressureAndFallback", func(t *testing.T) {
		cfg := DefaultStreamingBypassConfig()
		cfg.ChunkSizeBytes = 1024 // Tiny chunks to rapidly fill queue
		cfg.MaxPrefetchQueueDepth = 4
		cfg.LookaheadWavefronts = 20 // Large lookahead triggers queue backpressure
		cfg.FallbackOnBackpressure = true

		scheduler, err := NewStreamingBypassScheduler(cfg, nil, nil)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		chunks, err := scheduler.ScheduleWeightStream(0x00007F6000000000, 1024*1024)
		if err != nil {
			t.Fatalf("ScheduleWeightStream failed: %v", err)
		}

		if len(chunks) == 0 {
			t.Fatal("expected chunks to be scheduled")
		}

		telem := scheduler.Telemetry()
		if telem.BackpressureEvents == 0 {
			t.Fatal("expected backpressure events with constrained queue depth")
		}
		if telem.FallbackSynchronousEmitted == 0 {
			t.Fatal("expected fallback synchronous emissions during queue backpressure")
		}
	})

	t.Run("DescriptorAuditor_Validation", func(t *testing.T) {
		scheduler, err := NewStreamingBypassScheduler(DefaultStreamingBypassConfig(), nil, nil)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		// Valid descriptors with SLC=1
		validDescriptors := []StreamingDescriptor{
			{Word0: 0x1000, Word1: 0x0001, Word2: 0x2000, Word3: DescriptorWord3ResourceTypeBuffer | DescriptorWord3SLCBit},
			{Word0: 0x3000, Word1: 0x0001, Word2: 0x2000, Word3: DescriptorWord3ResourceTypeBuffer | DescriptorWord3SLCBit},
		}

		if err := scheduler.AuditDescriptors(validDescriptors); err != nil {
			t.Fatalf("expected valid descriptors to pass audit: %v", err)
		}

		// Invalid descriptor missing SLC bit
		invalidDescriptors := []StreamingDescriptor{
			{Word0: 0x1000, Word1: 0x0001, Word2: 0x2000, Word3: DescriptorWord3ResourceTypeBuffer | DescriptorWord3SLCBit},
			{Word0: 0x3000, Word1: 0x0001, Word2: 0x2000, Word3: DescriptorWord3ResourceTypeBuffer}, // Missing SLC
		}

		err = scheduler.AuditDescriptors(invalidDescriptors)
		if err == nil {
			t.Fatal("expected auditor to reject descriptor missing SLC bit")
		}
		if !errors.Is(err, ErrDescriptorNonBypassViolation) {
			t.Fatalf("expected ErrDescriptorNonBypassViolation, got: %v", err)
		}
	})

	t.Run("ErrorCasesAndValidation", func(t *testing.T) {
		scheduler, _ := NewStreamingBypassScheduler(DefaultStreamingBypassConfig(), nil, nil)
		defer scheduler.Close()

		// 1. Unaligned address
		_, err := scheduler.ScheduleWeightStream(0x1001, 1024*1024)
		if !errors.Is(err, ErrUnalignedWeightAddress) {
			t.Fatalf("expected ErrUnalignedWeightAddress, got: %v", err)
		}

		// 2. Zero/negative size
		_, err = scheduler.ScheduleWeightStream(0x1000, 0)
		if !errors.Is(err, ErrInvalidWeightSize) {
			t.Fatalf("expected ErrInvalidWeightSize, got: %v", err)
		}

		// 3. Closed scheduler
		scheduler.Close()
		_, err = scheduler.ScheduleWeightStream(0x1000, 1024*1024)
		if !errors.Is(err, ErrStreamingPipelineClosed) {
			t.Fatalf("expected ErrStreamingPipelineClosed, got: %v", err)
		}
	})

	t.Run("ConcurrencyAndDataRaces", func(t *testing.T) {
		scheduler, err := NewStreamingBypassScheduler(DefaultStreamingBypassConfig(), nil, nil)
		if err != nil {
			t.Fatalf("unexpected scheduler creation error: %v", err)
		}
		defer scheduler.Close()

		var wg sync.WaitGroup
		workers := 8
		iterations := 10

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				base := uint64(0x00007F7000000000 + uint64(workerID)*0x10000000)
				for i := 0; i < iterations; i++ {
					chunks, err := scheduler.ScheduleWeightStream(base, 4*1024*1024)
					if err != nil {
						t.Errorf("worker %d iteration %d ScheduleWeightStream failed: %v", workerID, i, err)
						return
					}
					_, err = scheduler.ExecuteSimulatedStream(chunks)
					if err != nil {
						t.Errorf("worker %d iteration %d ExecuteSimulatedStream failed: %v", workerID, i, err)
						return
					}
					_ = scheduler.Telemetry()
				}
			}(w)
		}

		wg.Wait()
	})
}
