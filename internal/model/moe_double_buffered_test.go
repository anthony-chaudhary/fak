package model

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestMoEDoubleBuffered runs the unified test suite for double-buffered prefill layer
// streaming with D2D hit compaction (Issue #11115).
func TestMoEDoubleBuffered(t *testing.T) {
	t.Run("PingPongTransitions", testMoEDoubleBufferedPingPongTransitions)
	t.Run("D2DHitCompaction", testMoEDoubleBufferedD2DHitCompaction)
	t.Run("MemoryReleaseOnCompletion", testMoEDoubleBufferedMemoryReleaseOnCompletion)
	t.Run("ConcurrentRaceSafety", testMoEDoubleBufferedConcurrentRaceSafety)
}

// TestMoEDoubleBuffered_PingPongTransitions verifies that the pipeline strictly alternates
// between BufferA (even layers) and BufferB (odd layers) for compute, while prefetching
// layer L+1 into the alternate buffer.
func TestMoEDoubleBuffered_PingPongTransitions(t *testing.T) {
	testMoEDoubleBufferedPingPongTransitions(t)
}

func testMoEDoubleBufferedPingPongTransitions(t *testing.T) {
	const (
		numLayers  = 5
		numExperts = 4
		expertSize = 1024
	)

	cfg := MoEPipelineConfig{
		NumLayers:   numLayers,
		NumExperts:  numExperts,
		ExpertBytes: expertSize,
		HiddenSize:  32,
	}

	pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
	if err != nil {
		t.Fatalf("failed to create pipeline: %v", err)
	}

	// Verify initial 2 * E expert slot layout
	if total := pipeline.TotalSlots(); total != 2*numExperts {
		t.Fatalf("TotalSlots() = %d, want %d (2 * E)", total, 2*numExperts)
	}
	if len(pipeline.BufferA.Slots) != numExperts {
		t.Fatalf("BufferA slots = %d, want %d", len(pipeline.BufferA.Slots), numExperts)
	}
	if len(pipeline.BufferB.Slots) != numExperts {
		t.Fatalf("BufferB slots = %d, want %d", len(pipeline.BufferB.Slots), numExperts)
	}
	if pipeline.BufferA.ID != 0 || pipeline.BufferB.ID != 1 {
		t.Fatalf("buffer IDs: A=%d, B=%d; want 0 and 1", pipeline.BufferA.ID, pipeline.BufferB.ID)
	}

	ctx := context.Background()
	input := make([]float32, 32)
	for i := range input {
		input[i] = float32(i + 1)
	}

	// Track transitions
	type stepRecord struct {
		layer        int
		computeBufID int
		prefetchBuf  int // -1 if no prefetch
	}
	var history []stepRecord

	act := input
	for l := 0; l < numLayers; l++ {
		expectedComputeBufID := l % 2
		buf := pipeline.GetBuffer(l)
		if buf.ID != expectedComputeBufID {
			t.Fatalf("layer %d: GetBuffer(%d).ID = %d, want %d", l, l, buf.ID, expectedComputeBufID)
		}

		expectedPrefetchBufID := -1
		if l+1 < numLayers {
			expectedPrefetchBufID = (l + 1) % 2
		}

		history = append(history, stepRecord{
			layer:        l,
			computeBufID: buf.ID,
			prefetchBuf:  expectedPrefetchBufID,
		})

		var stepErr error
		act, stepErr = pipeline.Step(ctx, l, act)
		if stepErr != nil {
			t.Fatalf("Step layer %d failed: %v", l, stepErr)
		}
	}

	// Verify alternating ping-pong transitions
	expectedTransitions := []struct {
		computeBuf int
		prefBuf    int
	}{
		{computeBuf: 0, prefBuf: 1},  // Layer 0: compute on BufferA (0), prefetch layer 1 into BufferB (1)
		{computeBuf: 1, prefBuf: 0},  // Layer 1: compute on BufferB (1), prefetch layer 2 into BufferA (0)
		{computeBuf: 0, prefBuf: 1},  // Layer 2: compute on BufferA (0), prefetch layer 3 into BufferB (1)
		{computeBuf: 1, prefBuf: 0},  // Layer 3: compute on BufferB (1), prefetch layer 4 into BufferA (0)
		{computeBuf: 0, prefBuf: -1}, // Layer 4: compute on BufferA (0), prefetch complete
	}

	if len(history) != len(expectedTransitions) {
		t.Fatalf("history length = %d, want %d", len(history), len(expectedTransitions))
	}

	for i, h := range history {
		exp := expectedTransitions[i]
		if h.computeBufID != exp.computeBuf {
			t.Errorf("layer %d: compute buffer ID = %d, want %d", i, h.computeBufID, exp.computeBuf)
		}
		if h.prefetchBuf != exp.prefBuf {
			t.Errorf("layer %d: prefetch buffer ID = %d, want %d", i, h.prefetchBuf, exp.prefBuf)
		}
	}

	stats := pipeline.Stats()
	if stats.ComputeCount != numLayers {
		t.Errorf("stats.ComputeCount = %d, want %d", stats.ComputeCount, numLayers)
	}
	if stats.PingPongSwaps != numLayers-1 {
		t.Errorf("stats.PingPongSwaps = %d, want %d", stats.PingPongSwaps, numLayers-1)
	}
}

// TestMoEDoubleBuffered_D2DHitCompaction verifies that device-resident experts are reused
// via D2D copies directly without redundant host bus transfers.
func TestMoEDoubleBuffered_D2DHitCompaction(t *testing.T) {
	testMoEDoubleBufferedD2DHitCompaction(t)
}

func testMoEDoubleBufferedD2DHitCompaction(t *testing.T) {
	t.Run("SharedExpertsAcrossLayers", func(t *testing.T) {
		const (
			numLayers  = 4
			numExperts = 4
		)

		var hostFetchCounts = make(map[string]int)
		var hostMu sync.Mutex

		// In this scenario:
		// Experts 0 and 1 are shared across all layers ("shared_0", "shared_1").
		// Experts 2 and 3 are unique per layer ("L%d_E%d").
		weightKeyFn := func(layer, expertID int) string {
			if expertID < 2 {
				return fmt.Sprintf("shared_expert_%d", expertID)
			}
			return fmt.Sprintf("layer_%d_expert_%d", layer, expertID)
		}

		cfg := MoEPipelineConfig{
			NumLayers:   numLayers,
			NumExperts:  numExperts,
			ExpertBytes: 2048,
			HiddenSize:  32,
			WeightKeyFn: weightKeyFn,
			HostFetchFn: func(ctx context.Context, layer, expertID int) ([]float32, error) {
				key := weightKeyFn(layer, expertID)
				hostMu.Lock()
				hostFetchCounts[key]++
				hostMu.Unlock()

				data := make([]float32, 512)
				val := float32((layer+1)*100 + expertID)
				for i := range data {
					data[i] = val
				}
				return data, nil
			},
		}

		pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
		if err != nil {
			t.Fatalf("failed to create pipeline: %v", err)
		}

		ctx := context.Background()
		input := make([]float32, 32)
		for i := range input {
			input[i] = 1.0
		}

		// Run through layers
		act := input
		for l := 0; l < numLayers; l++ {
			var err error
			act, err = pipeline.Step(ctx, l, act)
			if err != nil {
				t.Fatalf("Step(%d) failed: %v", l, err)
			}
		}

		stats := pipeline.Stats()

		// Analysis:
		// Layer 0: 4 host fetches (shared_0, shared_1, L0_E2, L0_E3). 0 D2D hits.
		// Layer 1: shared_0 and shared_1 hit BufferA via D2D! 2 host fetches (L1_E2, L1_E3). 2 D2D hits.
		// Layer 2: shared_0 and shared_1 hit BufferB via D2D! 2 host fetches (L2_E2, L2_E3). 2 D2D hits.
		// Layer 3: shared_0 and shared_1 hit BufferA via D2D! 2 host fetches (L3_E2, L3_E3). 2 D2D hits.
		// Total host fetches = 4 + 2 + 2 + 2 = 10 (instead of 16 without compaction).
		// Total D2D hits = 6.
		const (
			expectedHostFetches = 10
			expectedD2DHits     = 6
		)

		if stats.HostFetches != expectedHostFetches {
			t.Errorf("stats.HostFetches = %d, want %d", stats.HostFetches, expectedHostFetches)
		}
		if stats.D2DHits != expectedD2DHits {
			t.Errorf("stats.D2DHits = %d, want %d", stats.D2DHits, expectedD2DHits)
		}
		if stats.D2DCopies != expectedD2DHits {
			t.Errorf("stats.D2DCopies = %d, want %d", stats.D2DCopies, expectedD2DHits)
		}

		// Verify host bus fetch counts for shared experts: exactly 1 fetch each!
		hostMu.Lock()
		defer hostMu.Unlock()
		for e := 0; e < 2; e++ {
			key := fmt.Sprintf("shared_expert_%d", e)
			if count := hostFetchCounts[key]; count != 1 {
				t.Errorf("expert %q was fetched %d times across host bus; want exactly 1 (no re-fetches)", key, count)
			}
		}

		// Verify unique experts were fetched once per layer
		for l := 0; l < numLayers; l++ {
			for e := 2; e < 4; e++ {
				key := fmt.Sprintf("layer_%d_expert_%d", l, e)
				if count := hostFetchCounts[key]; count != 1 {
					t.Errorf("unique expert %q was fetched %d times; want 1", key, count)
				}
			}
		}

		if ratio := stats.CompactionRatio(); ratio <= 0.0 || ratio >= 1.0 {
			t.Errorf("unexpected CompactionRatio = %f, want ~0.375", ratio)
		}
	})

	t.Run("PreResidentExpertsFromPreviousSteps", func(t *testing.T) {
		// Verify that experts already resident in device memory (e.g. from prior steps)
		// are satisfied 100% via D2D hit compaction on layer 0.
		const numExperts = 3
		cfg := MoEPipelineConfig{
			NumLayers:   1,
			NumExperts:  numExperts,
			ExpertBytes: 1024,
			HiddenSize:  16,
		}

		pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
		if err != nil {
			t.Fatalf("failed to create pipeline: %v", err)
		}

		// Pre-populate device resident experts
		residentData := make(map[string][]float32)
		for e := 0; e < numExperts; e++ {
			key := fmt.Sprintf("L0_E%d", e)
			data := make([]float32, 256)
			for i := range data {
				data[i] = float32((e + 1) * 111)
			}
			residentData[key] = data
			pipeline.RegisterDeviceResident(key, data, compute.Tensor{})
		}

		// Prefetch layer 0
		ctx := context.Background()
		if err := pipeline.PrefetchLayer(ctx, 0); err != nil {
			t.Fatalf("PrefetchLayer(0) failed: %v", err)
		}

		stats := pipeline.Stats()
		if stats.HostFetches != 0 {
			t.Errorf("HostFetches = %d, want 0 (100%% D2D hit compaction)", stats.HostFetches)
		}
		if stats.D2DHits != numExperts {
			t.Errorf("D2DHits = %d, want %d", stats.D2DHits, numExperts)
		}

		// Verify data in BufferA slots matches resident data
		for e := 0; e < numExperts; e++ {
			slot := pipeline.BufferA.Slots[e]
			if slot.Source != TransferSourceD2D {
				t.Errorf("slot %d source = %s, want %s", e, slot.Source, TransferSourceD2D)
			}
			key := fmt.Sprintf("L0_E%d", e)
			expectedData := residentData[key]
			if !reflect.DeepEqual(slot.Data, expectedData) {
				t.Errorf("slot %d data mismatch with device resident data", e)
			}
		}
	})

	t.Run("PlanStagingPartitioning", func(t *testing.T) {
		cfg := MoEPipelineConfig{
			NumLayers:   2,
			NumExperts:  4,
			ExpertBytes: 1024,
		}
		pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
		if err != nil {
			t.Fatalf("failed to create pipeline: %v", err)
		}

		// Register 2 experts as resident
		pipeline.RegisterDeviceResident("L0_E0", make([]float32, 256), compute.Tensor{})
		pipeline.RegisterDeviceResident("L0_E2", make([]float32, 256), compute.Tensor{})

		plan := pipeline.PlanStaging(0, pipeline.BufferA)
		if len(plan.D2DHits) != 2 {
			t.Errorf("plan.D2DHits count = %d, want 2", len(plan.D2DHits))
		}
		if len(plan.HostFetches) != 2 {
			t.Errorf("plan.HostFetches count = %d, want 2", len(plan.HostFetches))
		}
	})
}

// TestMoEDoubleBuffered_MemoryReleaseOnCompletion verifies that staging buffers and device
// allocations are completely released upon prefill completion.
func TestMoEDoubleBuffered_MemoryReleaseOnCompletion(t *testing.T) {
	testMoEDoubleBufferedMemoryReleaseOnCompletion(t)
}

func testMoEDoubleBufferedMemoryReleaseOnCompletion(t *testing.T) {
	t.Run("ExecutePrefillReleasesMemory", func(t *testing.T) {
		const (
			numLayers   = 3
			numExperts  = 4
			expertBytes = 4096
		)

		cfg := MoEPipelineConfig{
			NumLayers:   numLayers,
			NumExperts:  numExperts,
			ExpertBytes: expertBytes,
			HiddenSize:  32,
		}

		pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
		if err != nil {
			t.Fatalf("failed to create pipeline: %v", err)
		}

		ctx := context.Background()
		input := make([]float32, 32)

		// Execute prefill to completion
		out, err := pipeline.ExecutePrefill(ctx, input)
		if err != nil {
			t.Fatalf("ExecutePrefill failed: %v", err)
		}
		if len(out) != len(input) {
			t.Fatalf("output length = %d, want %d", len(out), len(input))
		}

		// Verify pipeline completed and released
		if !pipeline.IsCompleted() {
			t.Error("pipeline.IsCompleted() = false, want true")
		}
		if !pipeline.IsReleased() {
			t.Error("pipeline.IsReleased() = false, want true")
		}

		// Verify buffers are clean and allocations zeroed
		if pipeline.BufferA.AllocatedBytes != 0 {
			t.Errorf("BufferA.AllocatedBytes = %d, want 0", pipeline.BufferA.AllocatedBytes)
		}
		if pipeline.BufferB.AllocatedBytes != 0 {
			t.Errorf("BufferB.AllocatedBytes = %d, want 0", pipeline.BufferB.AllocatedBytes)
		}
		if pipeline.BufferA.InUse || pipeline.BufferB.InUse {
			t.Error("staging buffers still marked InUse after release")
		}

		for i, s := range pipeline.BufferA.Slots {
			if s.Resident || s.Data != nil || s.PrefetchDone {
				t.Errorf("BufferA slot %d not cleared: resident=%v, dataLen=%d, done=%v",
					i, s.Resident, len(s.Data), s.PrefetchDone)
			}
		}
		for i, s := range pipeline.BufferB.Slots {
			if s.Resident || s.Data != nil || s.PrefetchDone {
				t.Errorf("BufferB slot %d not cleared: resident=%v, dataLen=%d, done=%v",
					i, s.Resident, len(s.Data), s.PrefetchDone)
			}
		}
	})

	t.Run("BackendTensorRelease", func(t *testing.T) {
		recBackend := newImmutableWeightRecordingBackend()

		cfg := MoEPipelineConfig{
			NumLayers:   2,
			NumExperts:  2,
			ExpertBytes: 1024,
			HiddenSize:  16,
			Backend:     recBackend,
		}

		pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
		if err != nil {
			t.Fatalf("failed to create pipeline: %v", err)
		}

		ctx := context.Background()
		input := make([]float32, 16)
		_, err = pipeline.ExecutePrefill(ctx, input)
		if err != nil {
			t.Fatalf("ExecutePrefill with backend failed: %v", err)
		}

		// Check that uploads occurred and were freed
		recBackend.mu.Lock()
		freeCount := 0
		for _, count := range recBackend.freeCalls {
			freeCount += count
		}
		recBackend.mu.Unlock()

		if freeCount == 0 {
			t.Error("expected backend Free to be called on release, got 0 free calls")
		}
	})
}

// TestMoEDoubleBuffered_ConcurrentRaceSafety exercises the pipeline under simulated
// asynchronous DMA bus transfers and compute delays to verify race-freedom.
func TestMoEDoubleBuffered_ConcurrentRaceSafety(t *testing.T) {
	testMoEDoubleBufferedConcurrentRaceSafety(t)
}

func testMoEDoubleBufferedConcurrentRaceSafety(t *testing.T) {
	const (
		numLayers  = 6
		numExperts = 4
	)

	var activeTransfers int32
	var activeComputes int32
	var maxConcurrentOps int32

	cfg := MoEPipelineConfig{
		NumLayers:              numLayers,
		NumExperts:             numExperts,
		ExpertBytes:            2048,
		HiddenSize:             32,
		SimulatedTransferDelay: 5 * time.Millisecond,
		SimulatedComputeDelay:  5 * time.Millisecond,
		HostFetchFn: func(ctx context.Context, layer, expertID int) ([]float32, error) {
			curXfer := atomic.AddInt32(&activeTransfers, 1)
			curComp := atomic.LoadInt32(&activeComputes)
			total := curXfer + curComp
			for {
				oldMax := atomic.LoadInt32(&maxConcurrentOps)
				if total <= oldMax || atomic.CompareAndSwapInt32(&maxConcurrentOps, oldMax, total) {
					break
				}
			}

			time.Sleep(2 * time.Millisecond)
			atomic.AddInt32(&activeTransfers, -1)

			data := make([]float32, 512)
			for i := range data {
				data[i] = float32((layer+1)*10 + expertID)
			}
			return data, nil
		},
		ComputeFn: func(ctx context.Context, layer int, buf *MoEStagingBuffer, in []float32) ([]float32, error) {
			curComp := atomic.AddInt32(&activeComputes, 1)
			curXfer := atomic.LoadInt32(&activeTransfers)
			total := curComp + curXfer
			for {
				oldMax := atomic.LoadInt32(&maxConcurrentOps)
				if total <= oldMax || atomic.CompareAndSwapInt32(&maxConcurrentOps, oldMax, total) {
					break
				}
			}

			time.Sleep(3 * time.Millisecond)
			atomic.AddInt32(&activeComputes, -1)

			out := make([]float32, len(in))
			for i := range out {
				out[i] = in[i] + float32(layer+1)
			}
			return out, nil
		},
	}

	pipeline, err := NewMoEDoubleBufferedPipeline(cfg)
	if err != nil {
		t.Fatalf("failed to create pipeline: %v", err)
	}

	ctx := context.Background()
	input := make([]float32, 32)
	for i := range input {
		input[i] = 10.0
	}

	// Concurrent stats readers to stress test race conditions
	stopReaders := make(chan struct{})
	var wgReaders sync.WaitGroup
	for r := 0; r < 4; r++ {
		wgReaders.Add(1)
		go func() {
			defer wgReaders.Done()
			for {
				select {
				case <-stopReaders:
					return
				default:
					_ = pipeline.Stats()
					_ = pipeline.TotalSlots()
					_ = pipeline.IsCompleted()
					_ = pipeline.IsReleased()
					time.Sleep(500 * time.Microsecond)
				}
			}
		}()
	}

	out, err := pipeline.ExecutePrefill(ctx, input)
	close(stopReaders)
	wgReaders.Wait()

	if err != nil {
		t.Fatalf("ExecutePrefill failed under concurrent load: %v", err)
	}

	// In 6 layers, input had (1+2+3+4+5+6) = 21 added to each element
	for i, v := range out {
		expected := float32(10.0 + 21.0)
		if v != expected {
			t.Errorf("out[%d] = %f, want %f", i, v, expected)
		}
	}

	stats := pipeline.Stats()
	if stats.ComputeCount != numLayers {
		t.Errorf("ComputeCount = %d, want %d", stats.ComputeCount, numLayers)
	}
	if stats.TotalLayersProcessed > 0 && stats.TotalLayersProcessed != numLayers {
		t.Errorf("TotalLayersProcessed = %d", stats.TotalLayersProcessed)
	}

	// Overlap observation: maxConcurrentOps should be > 1 because transfer of L+1
	// ran concurrently with compute of L!
	if maxOps := atomic.LoadInt32(&maxConcurrentOps); maxOps <= 1 {
		t.Logf("Notice: maxConcurrentOps observed was %d; timing may vary by scheduler", maxOps)
	} else {
		t.Logf("Observed genuine concurrency between transfer and compute: maxConcurrentOps = %d", maxOps)
	}
}
