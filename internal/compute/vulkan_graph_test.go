package compute

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"
)

// ---- Reusable Layer Command Graph Tests (#9834) ---------------------------------

func TestVulkanGraphRecordingAndValidation(t *testing.T) {
	g := NewVulkanCommandGraph("test_layer_graph", 0)
	if g.ID() != "test_layer_graph" || g.LayerID() != 0 {
		t.Fatalf("unexpected id=%s layer=%d", g.ID(), g.LayerID())
	}
	if g.State() != VulkanGraphUnrecorded {
		t.Fatalf("expected initial state %s, got %s", VulkanGraphUnrecorded, g.State())
	}

	// Cannot add node before BeginRecording
	if err := g.AddNode(VulkanGraphNode{ID: 0, OpType: VulkanOpRMSNorm}); err == nil {
		t.Fatal("expected error adding node before BeginRecording, got nil")
	}

	if err := g.BeginRecording(); err != nil {
		t.Fatalf("BeginRecording failed: %v", err)
	}
	if g.State() != VulkanGraphRecording {
		t.Fatalf("expected state %s, got %s", VulkanGraphRecording, g.State())
	}

	// Calling BeginRecording again while recording should fail
	if err := g.BeginRecording(); err == nil {
		t.Fatal("expected error for duplicate BeginRecording, got nil")
	}

	// Add nodes: Node 0 (RMSNorm) -> Node 1 (MatMul) -> Node 2 (RoPE)
	if err := g.AddRMSNormNode(0, "norm", 10, 11, 12, 1, 4096, 1e-5, nil); err != nil {
		t.Fatalf("AddRMSNormNode failed: %v", err)
	}
	if err := g.AddMatMulNode(1, "qkv", 12, 13, 14, 1, 4096, 4096, []int{0}); err != nil {
		t.Fatalf("AddMatMulNode failed: %v", err)
	}
	if err := g.AddRoPENode(2, "rope", 14, 1, 32, 128, 10000.0, []int{1}); err != nil {
		t.Fatalf("AddRoPENode failed: %v", err)
	}

	// Test missing predecessor error
	err := g.AddNode(VulkanGraphNode{
		ID:           3,
		OpType:       VulkanOpAttention,
		Name:         "attn",
		Predecessors: []int{99}, // doesn't exist
	})
	if err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for missing predecessor, got: %v", err)
	}

	// Test self-referencing predecessor
	err = g.AddNode(VulkanGraphNode{
		ID:           3,
		OpType:       VulkanOpAttention,
		Name:         "attn",
		Predecessors: []int{3}, // self-loop
	})
	if err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for self-predecessor, got: %v", err)
	}

	// EndRecording successfully
	if err := g.EndRecording(); err != nil {
		t.Fatalf("EndRecording failed: %v", err)
	}
	if g.State() != VulkanGraphRecorded {
		t.Fatalf("expected state %s, got %s", VulkanGraphRecorded, g.State())
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	nodes := g.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}
	if g.DeviceSyncEvents() < 2 {
		t.Fatalf("expected at least 2 device sync barriers, got %d", g.DeviceSyncEvents())
	}
}

func TestVulkanGraphCycleDetection(t *testing.T) {
	g := NewVulkanCommandGraph("cyclic_graph", 1)
	if err := g.BeginRecording(); err != nil {
		t.Fatalf("BeginRecording failed: %v", err)
	}

	// Node 0
	if err := g.AddNode(VulkanGraphNode{ID: 0, OpType: VulkanOpRMSNorm}); err != nil {
		t.Fatalf("AddNode 0 failed: %v", err)
	}
	// Node 1 depends on 0
	if err := g.AddNode(VulkanGraphNode{ID: 1, OpType: VulkanOpMatMul, Predecessors: []int{0}}); err != nil {
		t.Fatalf("AddNode 1 failed: %v", err)
	}

	// In order to test cycle detection in EndRecording/Validate, inject a cycle directly:
	// Let node 0 also depend on node 1
	g.nodes[0].Predecessors = []int{1}

	err := g.Validate()
	if err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for cycle, got: %v", err)
	}
}

func TestVulkanGraphEmptyGraphValidation(t *testing.T) {
	g := NewVulkanCommandGraph("empty_graph", 0)
	if err := g.BeginRecording(); err != nil {
		t.Fatalf("BeginRecording failed: %v", err)
	}
	err := g.EndRecording()
	if err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for empty graph, got: %v", err)
	}
}

func TestVulkanGraphReuseAndSubmissionSavings(t *testing.T) {
	cfg := Qwen38LayerGraphConfig{
		HiddenDim:       3584,
		IntermediateDim: 18944,
		NumHeads:        28,
		NumKVHeads:      4,
		HeadDim:         128,
		Eps:             1e-6,
		Theta:           1000000.0,
		BatchSize:       1,
		SeqLen:          1,
	}

	g, err := BuildQwen38LayerGraph(0, cfg)
	if err != nil {
		t.Fatalf("BuildQwen38LayerGraph failed: %v", err)
	}

	// Qwen3.8 layer forward graph consists of 11 sequential operations:
	// Norm -> QKV -> RoPE -> Attn -> OProj -> Residual -> Norm -> FFN GateUp -> SwiGLU -> Down -> Residual
	if len(g.Nodes()) != 11 {
		t.Fatalf("expected 11 nodes in Qwen3.8 layer graph, got %d", len(g.Nodes()))
	}

	if g.NaiveSubmits() != 11 {
		t.Fatalf("expected 11 naive submissions, got %d", g.NaiveSubmits())
	}
	if g.Submissions() != 1 {
		t.Fatalf("expected 1 batched submission, got %d", g.Submissions())
	}
	if g.SubmissionReduction() != 10 {
		t.Fatalf("expected 10 submissions reduced per replay, got %d", g.SubmissionReduction())
	}

	// Replay the graph 50 times (simulating 50 decode steps)
	const replays = 50
	for i := 0; i < replays; i++ {
		if err := g.Execute(); err != nil {
			t.Fatalf("Execute failed at replay %d: %v", i, err)
		}
	}

	if g.ReplayCount() != replays {
		t.Fatalf("expected %d replays, got %d", replays, g.ReplayCount())
	}
	expectedSaved := replays * 10
	if g.SubmissionsSaved() != expectedSaved {
		t.Fatalf("expected %d submissions saved, got %d", expectedSaved, g.SubmissionsSaved())
	}

	// Device sync events should be present connecting dependent nodes
	if g.DeviceSyncEvents() < 10 {
		t.Fatalf("expected at least 10 device sync barriers, got %d", g.DeviceSyncEvents())
	}
}

func TestVulkanGraphDeviceLossTeardown(t *testing.T) {
	cfg := Qwen38LayerGraphConfig{
		HiddenDim:       1024,
		IntermediateDim: 2048,
		NumHeads:        8,
		NumKVHeads:      2,
		HeadDim:         128,
		Eps:             1e-5,
		Theta:           10000.0,
		BatchSize:       1,
		SeqLen:          1,
	}
	g, err := BuildQwen38LayerGraph(0, cfg)
	if err != nil {
		t.Fatalf("BuildQwen38LayerGraph failed: %v", err)
	}

	// Trigger device loss
	if err := g.OnDeviceLost(); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("expected ErrVulkanDeviceLost, got: %v", err)
	}

	if g.State() != VulkanGraphDeviceLost {
		t.Fatalf("expected state %s, got %s", VulkanGraphDeviceLost, g.State())
	}

	// Execute must fail with ErrVulkanDeviceLost
	if err := g.Execute(); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("Execute after device loss must return ErrVulkanDeviceLost, got: %v", err)
	}

	// BeginRecording must fail with ErrVulkanDeviceLost
	if err := g.BeginRecording(); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("BeginRecording after device loss must return ErrVulkanDeviceLost, got: %v", err)
	}
}

func TestVulkanGraphCache(t *testing.T) {
	cache := NewVulkanLayerGraphCache()

	// Initial get: miss
	if _, ok := cache.Get(0); ok {
		t.Fatal("expected cache miss for layer 0")
	}

	cfg := Qwen38LayerGraphConfig{
		HiddenDim:       512,
		IntermediateDim: 1024,
		NumHeads:        4,
		NumKVHeads:      2,
		HeadDim:         128,
		Eps:             1e-5,
		Theta:           10000.0,
	}

	g0, err := BuildQwen38LayerGraph(0, cfg)
	if err != nil {
		t.Fatalf("BuildQwen38LayerGraph failed: %v", err)
	}
	g1, err := BuildQwen38LayerGraph(1, cfg)
	if err != nil {
		t.Fatalf("BuildQwen38LayerGraph failed: %v", err)
	}

	if err := cache.Put(0, g0); err != nil {
		t.Fatalf("cache.Put failed: %v", err)
	}
	if err := cache.Put(1, g1); err != nil {
		t.Fatalf("cache.Put failed: %v", err)
	}

	// Retrieve
	hitG0, ok := cache.Get(0)
	if !ok || hitG0 != g0 {
		t.Fatal("expected cache hit for layer 0")
	}

	stats := cache.Stats()
	if stats.Hits != 1 || stats.Misses != 1 || stats.CachedLayers != 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// Invalidate layer 0
	cache.Invalidate(0)
	if _, ok := cache.Get(0); ok {
		t.Fatal("expected layer 0 to be invalidated")
	}

	// Device loss marks all graphs lost and closes cache
	cache.OnDeviceLost()
	if _, ok := cache.Get(1); ok {
		t.Fatal("expected miss on closed cache")
	}
	if err := cache.Put(2, g1); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("expected ErrVulkanDeviceLost on Put after device loss, got: %v", err)
	}
}

// ---- Bounded Staging Pool Tests (#9835) -----------------------------------------

func TestVulkanStagingPoolAllocationAndBoundary(t *testing.T) {
	// Invalid slot count (< 2)
	if _, err := NewVulkanStagingPool(1, 1024, 64); err == nil {
		t.Fatal("expected error for numSlots < 2, got nil")
	}

	// Invalid capacity
	if _, err := NewVulkanStagingPool(2, 0, 64); err == nil {
		t.Fatal("expected error for slotCapacity <= 0, got nil")
	}

	const numSlots = 2
	const capBytes = 16 * 1024 * 1024 // 16 MiB per slot
	pool, err := NewVulkanStagingPool(numSlots, capBytes, 256)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	if pool.NumSlots() != numSlots {
		t.Fatalf("expected %d slots, got %d", numSlots, pool.NumSlots())
	}
	if pool.SlotCapacity() != capBytes {
		t.Fatalf("expected %d capacity, got %d", capBytes, pool.SlotCapacity())
	}

	// Negative or zero byte transfer request
	if _, err := pool.AcquireSlot(0, 0); err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for 0 bytes, got: %v", err)
	}

	// Request exceeding capacity -> ErrVulkanResourceExhausted
	if _, err := pool.AcquireSlot(0, capBytes+1); err == nil || !errors.Is(err, ErrVulkanResourceExhausted) {
		t.Fatalf("expected ErrVulkanResourceExhausted for oversize transfer, got: %v", err)
	}
}

func TestVulkanStagingPoolSlotCycling(t *testing.T) {
	const numSlots = 2
	const capBytes = 8 * 1024 * 1024 // 8 MiB
	pool, err := NewVulkanStagingPool(numSlots, capBytes, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	// 1. Acquire slot 0 for layer 0 transfer
	slot0, err := pool.AcquireSlot(0, 4*1024*1024)
	if err != nil {
		t.Fatalf("AcquireSlot(0) failed: %v", err)
	}
	if slot0.ID != 0 || slot0.State != StagingSlotTransferring {
		t.Fatalf("unexpected slot0: %+v", slot0)
	}
	if pool.ActiveCount() != 1 {
		t.Fatalf("expected 1 active slot, got %d", pool.ActiveCount())
	}

	// 2. Mark transfer complete -> ready
	if err := pool.MarkTransferComplete(slot0.ID, 1); err != nil {
		t.Fatalf("MarkTransferComplete failed: %v", err)
	}

	// 3. Acquire slot 0 for compute
	if err := pool.AcquireForCompute(slot0.ID); err != nil {
		t.Fatalf("AcquireForCompute failed: %v", err)
	}

	// 4. Concurrently acquire slot 1 for layer 1 transfer while slot 0 computes!
	slot1, err := pool.AcquireSlot(1, 6*1024*1024)
	if err != nil {
		t.Fatalf("AcquireSlot(1) failed: %v", err)
	}
	if slot1.ID != 1 || slot1.State != StagingSlotTransferring {
		t.Fatalf("unexpected slot1: %+v", slot1)
	}
	if pool.ActiveCount() != 2 {
		t.Fatalf("expected 2 active slots, got %d", pool.ActiveCount())
	}

	// 5. Release compute on slot 0 -> becomes free
	if err := pool.ReleaseCompute(slot0.ID); err != nil {
		t.Fatalf("ReleaseCompute(0) failed: %v", err)
	}
	if pool.ActiveCount() != 1 {
		t.Fatalf("expected 1 active slot after release, got %d", pool.ActiveCount())
	}

	// 6. Mark transfer complete and compute on slot 1
	if err := pool.MarkTransferComplete(slot1.ID, 2); err != nil {
		t.Fatalf("MarkTransferComplete(1) failed: %v", err)
	}
	if err := pool.AcquireForCompute(slot1.ID); err != nil {
		t.Fatalf("AcquireForCompute(1) failed: %v", err)
	}

	// 7. Acquire slot 0 again for layer 2 transfer (cycling back)
	slot0Recycled, ok, err := pool.TryAcquireSlot(2, 5*1024*1024)
	if err != nil || !ok || slot0Recycled.ID != 0 {
		t.Fatalf("expected slot 0 recycled, got ok=%v slot=%+v err=%v", ok, slot0Recycled, err)
	}

	// Clean up compute on remaining slots
	_ = pool.ReleaseCompute(slot1.ID)
	_ = pool.MarkTransferComplete(slot0Recycled.ID, 3)
	_ = pool.AcquireForCompute(slot0Recycled.ID)
	_ = pool.ReleaseCompute(slot0Recycled.ID)

	if pool.ActiveCount() != 0 {
		t.Fatalf("expected 0 active slots, got %d", pool.ActiveCount())
	}
}

func TestVulkanStagingPoolConcurrentSafety(t *testing.T) {
	const numSlots = 2
	const capBytes = 4 * 1024 * 1024
	pool, err := NewVulkanStagingPool(numSlots, capBytes, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	const numLayers = 30
	var wg sync.WaitGroup

	// Pipeline: producer transfers layers sequentially, consumer processes compute
	transferQueue := make(chan int, numSlots)

	// Transfer producer
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(transferQueue)
		for layer := 0; layer < numLayers; layer++ {
			slot, err := pool.AcquireSlot(layer, 1024*1024)
			if err != nil {
				t.Errorf("AcquireSlot failed at layer %d: %v", layer, err)
				return
			}
			// Simulate transfer time
			time.Sleep(100 * time.Microsecond)
			if err := pool.MarkTransferComplete(slot.ID, uint64(layer+1)); err != nil {
				t.Errorf("MarkTransferComplete failed at layer %d: %v", layer, err)
				return
			}
			transferQueue <- slot.ID
		}
	}()

	// Compute consumer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for slotID := range transferQueue {
			if err := pool.AcquireForCompute(slotID); err != nil {
				t.Errorf("AcquireForCompute failed on slot %d: %v", slotID, err)
				return
			}
			// Simulate compute time
			time.Sleep(150 * time.Microsecond)
			if err := pool.ReleaseCompute(slotID); err != nil {
				t.Errorf("ReleaseCompute failed on slot %d: %v", slotID, err)
				return
			}
		}
	}()

	wg.Wait()
	if pool.ActiveCount() != 0 {
		t.Fatalf("expected all slots free after concurrent pipeline, got %d active", pool.ActiveCount())
	}
}

func TestVulkanStagingPoolDeviceLossAndCancellation(t *testing.T) {
	pool, err := NewVulkanStagingPool(2, 1024*1024, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	// Trigger device loss
	if err := pool.OnDeviceLost(); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("expected ErrVulkanDeviceLost, got: %v", err)
	}

	// Subsequent acquisition must immediately return ErrVulkanDeviceLost
	if _, err := pool.AcquireSlot(0, 1024); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("expected ErrVulkanDeviceLost on AcquireSlot after device lost, got: %v", err)
	}
	if _, _, err := pool.TryAcquireSlot(0, 1024); !errors.Is(err, ErrVulkanDeviceLost) {
		t.Fatalf("expected ErrVulkanDeviceLost on TryAcquireSlot after device lost, got: %v", err)
	}
}

func TestVulkanStagingOverlapPipeline(t *testing.T) {
	pool, err := NewVulkanStagingPool(2, 64*1024*1024, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	stages := []OverlapStage{
		{Name: "layer_0", TransferCost: 1.2, ComputeCost: 2.5},
		{Name: "layer_1", TransferCost: 1.2, ComputeCost: 2.5},
		{Name: "layer_2", TransferCost: 1.2, ComputeCost: 2.5},
		{Name: "layer_3", TransferCost: 1.2, ComputeCost: 2.5},
	}

	res := pool.OverlapSchedule(stages)

	// Serial: 4 * (1.2 + 2.5) = 14.8
	wantSerial := 14.8
	if math.Abs(res.Serial-wantSerial) > 1e-4 {
		t.Fatalf("res.Serial = %.4f, want %.4f", res.Serial, wantSerial)
	}

	// Overlapped: Transfer 0 completes at 1.2 -> Compute 0 completes at 3.7.
	// Transfer 1 finishes at 2.4 (hidden behind Compute 0). Compute 1 starts at 3.7 -> completes at 6.2.
	// Transfer 2 finishes at 3.6 (hidden). Compute 2 starts at 6.2 -> completes at 8.7.
	// Transfer 3 finishes at 4.8 (hidden). Compute 3 starts at 8.7 -> completes at 11.2.
	wantOverlapped := 11.2
	if math.Abs(res.Overlapped-wantOverlapped) > 1e-4 {
		t.Fatalf("res.Overlapped = %.4f, want %.4f", res.Overlapped, wantOverlapped)
	}

	// Speedup: 14.8 / 11.2 ≈ 1.3214x
	if res.Speedup <= 1.0 {
		t.Fatalf("expected speedup > 1.0, got %.4f", res.Speedup)
	}
}

// ---- AMD RDNA3 Wave32 Rowtile Tests (#9677) -------------------------------------

func TestVulkanRowtileDimensionCalculation(t *testing.T) {
	targets := []struct {
		gfx         string
		m, n, k     int
		quantFormat string
		wantWaves   int
		wantThreads int
		wantFmt     string
		wantEnc     string
	}{
		// gfx1102 (RX 7600) Decode
		{
			gfx:         "gfx1102",
			m:           1,
			n:           3584,
			k:           3584,
			quantFormat: "Q8_0",
			wantWaves:   2,
			wantThreads: 64,
			wantFmt:     "Q8_0",
			wantEnc:     "symmetric-int8",
		},
		// gfx1102 (RX 7600) Prefill
		{
			gfx:         "gfx1102",
			m:           128,
			n:           3584,
			k:           3584,
			quantFormat: "Q4_K",
			wantWaves:   2,
			wantThreads: 64,
			wantFmt:     "Q4_K",
			wantEnc:     "unequal-packed-pair",
		},
		// gfx1151 (Strix Halo APU) Decode
		{
			gfx:         "gfx1151",
			m:           1,
			n:           5120,
			k:           5120,
			quantFormat: "Q8_0",
			wantWaves:   2,
			wantThreads: 64,
			wantFmt:     "Q8_0",
			wantEnc:     "symmetric-int8",
		},
		// gfx1151 (Strix Halo APU) Prefill (wider workgroup)
		{
			gfx:         "gfx1151",
			m:           256,
			n:           5120,
			k:           5120,
			quantFormat: "Q4_K",
			wantWaves:   4,
			wantThreads: 128,
			wantFmt:     "Q4_K",
			wantEnc:     "unequal-packed-pair",
		},
		// Q5_K format
		{
			gfx:         "gfx1151",
			m:           64,
			n:           4096,
			k:           4096,
			quantFormat: "Q5_K",
			wantWaves:   4,
			wantThreads: 128,
			wantFmt:     "Q5_K",
			wantEnc:     "unequal-packed-pair",
		},
	}

	for _, tc := range targets {
		name := fmt.Sprintf("%s_%s_m%d", tc.gfx, tc.quantFormat, tc.m)
		t.Run(name, func(t *testing.T) {
			params, err := CalculateRowtileParams(tc.gfx, tc.m, tc.n, tc.k, tc.quantFormat)
			if err != nil {
				t.Fatalf("CalculateRowtileParams failed: %v", err)
			}

			if params.WavefrontSize != 32 {
				t.Fatalf("expected Wave32 (32), got %d", params.WavefrontSize)
			}
			if params.WavesPerWorkgroup != tc.wantWaves {
				t.Fatalf("expected %d waves, got %d", tc.wantWaves, params.WavesPerWorkgroup)
			}
			if params.WorkgroupThreads != tc.wantThreads {
				t.Fatalf("expected %d threads, got %d", tc.wantThreads, params.WorkgroupThreads)
			}
			if params.QuantFormat != tc.wantFmt {
				t.Fatalf("expected format %s, got %s", tc.wantFmt, params.QuantFormat)
			}
			if params.PackedPairEncoding != tc.wantEnc {
				t.Fatalf("expected encoding %s, got %s", tc.wantEnc, params.PackedPairEncoding)
			}
			if params.LDSBytes <= 0 || params.LDSBytes > 65536 {
				t.Fatalf("invalid LDS bytes %d (must be in (0, 65536])", params.LDSBytes)
			}
			if params.EstimatedOccupancy <= 0.0 || params.EstimatedOccupancy > 1.0 {
				t.Fatalf("invalid estimated occupancy %.4f", params.EstimatedOccupancy)
			}
		})
	}
}

func TestVulkanRowtileGridCalculation(t *testing.T) {
	params, err := CalculateRowtileParams("gfx1102", 1, 3584, 3584, "Q8_0")
	if err != nil {
		t.Fatalf("CalculateRowtileParams failed: %v", err)
	}

	grid := RowtileGrid(params, 1, 3584)
	// TileM=1, TileN=64. GridX = ceil(3584 / 64) = 56, GridY = ceil(1 / 1) = 1, GridZ = 1
	if grid[0] != 56 || grid[1] != 1 || grid[2] != 1 {
		t.Fatalf("unexpected grid dimensions: %+v, want [56, 1, 1]", grid)
	}

	// Prefill grid
	prefillParams, err := CalculateRowtileParams("gfx1151", 256, 5120, 5120, "Q4_K")
	if err != nil {
		t.Fatalf("CalculateRowtileParams failed: %v", err)
	}
	gridPrefill := RowtileGrid(prefillParams, 256, 5120)
	// TileM=64, TileN=64. GridX = ceil(5120 / 64) = 80, GridY = ceil(256 / 64) = 4, GridZ = 1
	if gridPrefill[0] != 80 || gridPrefill[1] != 4 || gridPrefill[2] != 1 {
		t.Fatalf("unexpected prefill grid: %+v, want [80, 4, 1]", gridPrefill)
	}
}

func TestVulkanRowtileValidationAndErrors(t *testing.T) {
	// Unknown GFX arch
	if _, err := CalculateRowtileParams("gfx906", 1, 1024, 1024, "Q8_0"); err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for unsupported arch gfx906, got: %v", err)
	}

	// Non-positive dimensions
	if _, err := CalculateRowtileParams("gfx1102", 0, 1024, 1024, "Q8_0"); err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for m=0, got: %v", err)
	}

	// K not aligned to block size (32)
	if _, err := CalculateRowtileParams("gfx1102", 1, 1024, 1025, "Q8_0"); err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for K not divisible by 32, got: %v", err)
	}

	// Unknown quant format
	if _, err := CalculateRowtileParams("gfx1102", 1, 1024, 1024, "UNKNOWN"); err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected ErrVulkanInvalidGeometry for unknown quant format, got: %v", err)
	}

	// Target string normalization: e.g. "gfx1102:xnack-" should resolve to gfx1102
	arch, ok := LookupRowtileArch("gfx1102:xnack-")
	if !ok || arch.GFX != "gfx1102" {
		t.Fatalf("LookupRowtileArch with feature suffix failed: ok=%v, arch=%+v", ok, arch)
	}
}

func TestVulkanGraphDuplicateNodeAndReset(t *testing.T) {
	g := NewVulkanCommandGraph("dup_reset_graph", 0)
	if err := g.BeginRecording(); err != nil {
		t.Fatalf("BeginRecording failed: %v", err)
	}

	if err := g.AddRMSNormNode(1, "norm", 1, 2, 3, 1, 256, 1e-5, nil); err != nil {
		t.Fatalf("AddRMSNormNode failed: %v", err)
	}

	// Adding duplicate node ID should fail
	err := g.AddRMSNormNode(1, "norm_dup", 1, 2, 3, 1, 256, 1e-5, nil)
	if err == nil || !errors.Is(err, ErrVulkanInvalidGeometry) {
		t.Fatalf("expected duplicate node error, got: %v", err)
	}

	// Reset graph
	g.Reset()
	if g.State() != VulkanGraphUnrecorded {
		t.Fatalf("expected state unrecorded after reset, got: %s", g.State())
	}
	if len(g.Nodes()) != 0 {
		t.Fatalf("expected 0 nodes after reset, got: %d", len(g.Nodes()))
	}
}

func TestVulkanGraphIndividualNodeConstructors(t *testing.T) {
	g := NewVulkanCommandGraph("nodes_test", 0)
	if err := g.BeginRecording(); err != nil {
		t.Fatalf("BeginRecording failed: %v", err)
	}

	if err := g.AddSwiGLUNode(1, "swiglu", 10, 11, 12, 1024, nil); err != nil {
		t.Fatalf("AddSwiGLUNode failed: %v", err)
	}
	if err := g.AddAddInPlaceNode(2, "add", 12, 13, 1024, []int{1}); err != nil {
		t.Fatalf("AddAddInPlaceNode failed: %v", err)
	}
	if err := g.AddAttentionNode(3, "attn", 14, 15, 16, 17, 1, 4, 2, 64, 0.125, []int{2}); err != nil {
		t.Fatalf("AddAttentionNode failed: %v", err)
	}
	barrier := VulkanMemoryBarrier{
		SrcStageMask:  VulkanStageComputeShader,
		DstStageMask:  VulkanStageTransfer,
		SrcAccessMask: VulkanAccessShaderWrite,
		DstAccessMask: VulkanAccessTransferRead,
	}
	if err := g.AddBarrierNode(4, "barrier", barrier, []int{3}); err != nil {
		t.Fatalf("AddBarrierNode failed: %v", err)
	}

	if err := g.EndRecording(); err != nil {
		t.Fatalf("EndRecording failed: %v", err)
	}
	if len(g.Nodes()) != 4 {
		t.Fatalf("expected 4 nodes, got %d", len(g.Nodes()))
	}
}

func TestVulkanLayerGraphCacheClear(t *testing.T) {
	cache := NewVulkanLayerGraphCache()
	cfg := Qwen38LayerGraphConfig{
		HiddenDim:       256,
		IntermediateDim: 512,
		NumHeads:        4,
		NumKVHeads:      2,
		HeadDim:         64,
		Eps:             1e-5,
		Theta:           10000.0,
	}
	g, err := BuildQwen38LayerGraph(0, cfg)
	if err != nil {
		t.Fatalf("BuildQwen38LayerGraph failed: %v", err)
	}
	if err := cache.Put(0, g); err != nil {
		t.Fatalf("cache.Put failed: %v", err)
	}
	if cache.Stats().CachedLayers != 1 {
		t.Fatalf("expected 1 cached layer, got: %d", cache.Stats().CachedLayers)
	}
	cache.Clear()
	if cache.Stats().CachedLayers != 0 {
		t.Fatalf("expected 0 cached layers after clear, got: %d", cache.Stats().CachedLayers)
	}
}

func TestVulkanStagingPoolTelemetryAndClose(t *testing.T) {
	pool, err := NewVulkanStagingPool(2, 1024*1024, 64)
	if err != nil {
		t.Fatalf("NewVulkanStagingPool failed: %v", err)
	}

	if pool.TotalStaged() != 0 || pool.Transfers() != 0 {
		t.Fatalf("expected 0 initial telemetry, got staged=%d transfers=%d", pool.TotalStaged(), pool.Transfers())
	}

	slot, err := pool.AcquireSlot(0, 4096)
	if err != nil {
		t.Fatalf("AcquireSlot failed: %v", err)
	}
	if pool.TotalStaged() != 4096 || pool.Transfers() != 1 {
		t.Fatalf("unexpected telemetry: staged=%d transfers=%d", pool.TotalStaged(), pool.Transfers())
	}

	_ = pool.MarkTransferComplete(slot.ID, 1)
	_ = pool.AcquireForCompute(slot.ID)
	_ = pool.ReleaseCompute(slot.ID)

	pool.Close()
	// Acquire after close returns error
	if _, err := pool.AcquireSlot(1, 4096); err == nil {
		t.Fatal("expected error on AcquireSlot after Close")
	}
}

func TestVulkanRowtileExtendedFormatsAndGFX1100(t *testing.T) {
	// GFX1100 (RX 7900 XTX)
	p1100, err := CalculateRowtileParams("gfx1100", 256, 4096, 4096, "Q8_0")
	if err != nil {
		t.Fatalf("CalculateRowtileParams for gfx1100 failed: %v", err)
	}
	if p1100.WavesPerWorkgroup != 4 {
		t.Fatalf("expected 4 waves per wg for gfx1100 prefill, got %d", p1100.WavesPerWorkgroup)
	}

	// Q6_K format
	pQ6K, err := CalculateRowtileParams("gfx1151", 1, 4096, 4096, "Q6_K")
	if err != nil {
		t.Fatalf("CalculateRowtileParams for Q6_K failed: %v", err)
	}
	if pQ6K.QuantFormat != "Q6_K" {
		t.Fatalf("expected Q6_K format, got %s", pQ6K.QuantFormat)
	}

	// BF16 format
	pBF16, err := CalculateRowtileParams("gfx1151", 1, 4096, 4096, "BF16")
	if err != nil {
		t.Fatalf("CalculateRowtileParams for BF16 failed: %v", err)
	}
	if pBF16.PackedPairEncoding != "packed-bf16_2" {
		t.Fatalf("expected packed-bf16_2, got %s", pBF16.PackedPairEncoding)
	}

	// FP16 format
	pFP16, err := CalculateRowtileParams("gfx1102", 1, 4096, 4096, "FP16")
	if err != nil {
		t.Fatalf("CalculateRowtileParams for FP16 failed: %v", err)
	}
	if pFP16.PackedPairEncoding != "packed-half2" {
		t.Fatalf("expected packed-half2, got %s", pFP16.PackedPairEncoding)
	}

	// FP32 format
	pFP32, err := CalculateRowtileParams("gfx1102", 1, 4096, 4096, "FP32")
	if err != nil {
		t.Fatalf("CalculateRowtileParams for FP32 failed: %v", err)
	}
	if pFP32.PackedPairEncoding != "float32-direct" {
		t.Fatalf("expected float32-direct, got %s", pFP32.PackedPairEncoding)
	}
}
