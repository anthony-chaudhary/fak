package compute

import (
	"errors"
	"math/rand"
	"testing"
	"time"
)

func newTestHAL(t *testing.T) *AMDGPUDirectHAL {
	t.Helper()
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{DefaultPageSize: 4096})
	node := AMDDeviceNode{
		NodeID:         0,
		GPUID:          0,
		DeviceName:     "AMD Instinct MI300X",
		Architecture:   "gfx942",
		TotalVRAMBytes: 192 * 1024 * 1024 * 1024, // 192 GiB
		BAR1SizeBytes:  192 * 1024 * 1024 * 1024, // Large BAR
		IsLargeBAR:     true,
		DMABUFCapable:  true,
		KeepVRAMMapped: true,
	}
	if err := hal.RegisterNode(node); err != nil {
		t.Fatalf("RegisterNode failed: %v", err)
	}
	return hal
}

func TestAMDGPUDirect_MoEExpertP2PStreamer(t *testing.T) {
	hal := newTestHAL(t)

	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          2,
		NumExpertsPerLayer: 256,
		ExpertSizeBytes:    64 * 1024, // 64 KiB per expert for fast verification
		SectorSizeBytes:    4096,      // 4 KiB aligned sectors
		BaseLBA:            0x20000,
		HotTableCapacity:   51, // ~20% of 256
		PrefetchBuffers:    4,
		SimulatedIOLatency: 15 * time.Microsecond,
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	// 1. Verify 4KiB-aligned physical NVMe LBA mappings for all 256 experts across layers
	for l := 0; l < cfg.NumLayers; l++ {
		for e := 0; e < cfg.NumExpertsPerLayer; e++ {
			key := MoEExpertKey{LayerIndex: l, ExpertID: e}
			lbaMap, err := streamer.GetExpertLBAMap(key)
			if err != nil {
				t.Fatalf("missing lba mapping for expert %s: %v", key, err)
			}
			if (lbaMap.StartingLBA*lbaMap.SectorSize)%MoE4KiBAlignment != 0 {
				t.Fatalf("expert %s starting LBA %d is not 4KiB aligned", key, lbaMap.StartingLBA)
			}
			if lbaMap.SizeBytes != cfg.ExpertSizeBytes {
				t.Fatalf("expert %s size mismatch: got %d, want %d", key, lbaMap.SizeBytes, cfg.ExpertSizeBytes)
			}
		}
	}

	// 2. Verify Zero-Copy Invariant at initialization
	if streamer.StagingCopyCount() != 0 {
		t.Fatalf("zero-copy invariant violated: StagingCopyCount=%d (must be 0)", streamer.StagingCopyCount())
	}

	// 3. Hot-Expert VRAM Table Setup
	// Pin shared experts across layers (Expert 0 & 1)
	for l := 0; l < cfg.NumLayers; l++ {
		for e := 0; e < 2; e++ {
			shKey := MoEExpertKey{LayerIndex: l, ExpertID: e}
			if err := streamer.PinSharedExpert(shKey); err != nil {
				t.Fatalf("PinSharedExpert %s failed: %v", shKey, err)
			}
		}
	}

	// Simulate Zipfian activation frequencies: experts 2..40 heavily activated, 41..255 sparse
	rng := rand.New(rand.NewSource(42))
	for l := 0; l < cfg.NumLayers; l++ {
		for e := 2; e < 45; e++ {
			k := MoEExpertKey{LayerIndex: l, ExpertID: e}
			acts := rng.Intn(50) + 10
			for a := 0; a < acts; a++ {
				streamer.RecordActivation(k)
			}
		}
	}

	// Auto-pin top hot routed experts into resident VRAM table
	if err := streamer.AutoPinTopKHotExperts(45); err != nil {
		t.Fatalf("AutoPinTopKHotExperts failed: %v", err)
	}

	// Verify shared experts remain pinned
	shared0 := MoEExpertKey{LayerIndex: 0, ExpertID: 0}
	entry, ok := streamer.GetHotTableEntry(shared0)
	if !ok || !entry.Pinned || !entry.IsShared {
		t.Fatalf("shared expert %s was evicted or not pinned", shared0)
	}

	// 4. Router-Driven Predictive Prefetch
	// Build synthetic router logits for Layer 0 tokens
	numTokens := 8
	routingLogits := make([][]float32, numTokens)
	for tIdx := 0; tIdx < numTokens; tIdx++ {
		routingLogits[tIdx] = make([]float32, cfg.NumExpertsPerLayer)
		// Heavily activate upcoming Layer 1 experts 50, 60, 70, 80 (not in hot table)
		routingLogits[tIdx][50] = 5.5
		routingLogits[tIdx][60] = 4.8
		routingLogits[tIdx][70] = 3.9
		routingLogits[tIdx][80] = 3.2
	}

	dispatched, err := streamer.DispatchPredictivePrefetch(0, routingLogits, 4)
	if err != nil {
		t.Fatalf("DispatchPredictivePrefetch failed: %v", err)
	}
	if len(dispatched) == 0 {
		t.Fatalf("expected predictive prefetch dispatches, got 0")
	}

	// Allow speculative asynchronous DMA reads to settle in VRAM prefetch buffers
	time.Sleep(30 * time.Millisecond)

	// Verify prefetched expert is resident
	prefKey := MoEExpertKey{LayerIndex: 1, ExpertID: 50}
	if !streamer.IsPrefetched(prefKey) {
		t.Fatalf("expected expert %s to be resident in prefetch buffer", prefKey)
	}

	// 5. Execute Pipelined MoE Layer (Layer 1)
	// Active sequence includes: shared experts (hot hit), prefetched experts (prefetch hit), and sparse cold experts
	layer1Active := []int{0, 1, 50, 60, 70, 90, 100}
	gemmTime := 20 * time.Microsecond
	gemmFn := func(key MoEExpertKey, slab *VRAMWeightSlab) error {
		if slab.VRAMAddress == 0 {
			return errors.New("zero VRAM target address during GEMM execution")
		}
		time.Sleep(gemmTime)
		return nil
	}

	layerStats, err := streamer.ExecutePipelinedLayer(1, layer1Active, gemmFn)
	if err != nil {
		t.Fatalf("ExecutePipelinedLayer failed: %v", err)
	}

	// Telemetry validations
	if layerStats.HotTableHits == 0 {
		t.Fatalf("expected hot table hits > 0, got %d", layerStats.HotTableHits)
	}
	if layerStats.HotTableHitRate <= 0 {
		t.Fatalf("expected hot table hit rate > 0, got %.4f", layerStats.HotTableHitRate)
	}
	if layerStats.PrefetchHits == 0 {
		t.Fatalf("expected prefetch hits > 0, got %d", layerStats.PrefetchHits)
	}
	if layerStats.PrefetchAccuracy <= 0 {
		t.Fatalf("expected prefetch accuracy > 0, got %.4f", layerStats.PrefetchAccuracy)
	}

	// 6. Compute-I/O Overlap Benchmark: 256 simulated experts
	benchStats, err := streamer.BenchmarkComputeIOOverlap(256, 30*time.Microsecond, 18*time.Microsecond)
	if err != nil {
		t.Fatalf("BenchmarkComputeIOOverlap failed: %v", err)
	}

	// Verify >85% latency hiding of NVMe streaming latency behind concurrent GEMM
	if benchStats.HidingRatio <= 0.85 {
		t.Fatalf("compute-I/O overlap latency hiding ratio %.4f <= 0.85 (>85%% required)", benchStats.HidingRatio)
	}

	// Verify zero-copy invariant on all executed commands
	cmds := streamer.CompletedCommands()
	if len(cmds) == 0 {
		t.Fatalf("expected completed NVMe commands, got 0")
	}
	for _, cmd := range cmds {
		if cmd.StagingCopyCount() != 0 {
			t.Fatalf("command %d violated zero-copy invariant: StagingCopyCount=%d", cmd.CommandID, cmd.StagingCopyCount())
		}
		if cmd.TargetVRAMAddr == 0 {
			t.Fatalf("command %d target VRAM address is 0", cmd.CommandID)
		}
	}

	t.Logf("=== MoE Expert P2P Streamer Test Summary ===")
	t.Logf("Total Accesses:       %d", layerStats.TotalAccesses)
	t.Logf("Hot Table Hits:       %d (HitRate: %.2f%%)", layerStats.HotTableHits, layerStats.HotTableHitRate*100)
	t.Logf("Prefetch Hits:        %d (Accuracy: %.2f%%)", layerStats.PrefetchHits, layerStats.PrefetchAccuracy*100)
	t.Logf("Benchmark Hiding:     %.2f%% (>85%% required)", benchStats.HidingRatio*100)
	t.Logf("Completed DMA Cmds:   %d", len(cmds))
	t.Logf("Staging Copy Count:   %d (Zero-Copy Verified)", streamer.StagingCopyCount())
}

func TestMoEExpertP2PStreamer_DoubleBufferRotation(t *testing.T) {
	hal := newTestHAL(t)
	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          1,
		NumExpertsPerLayer: 8,
		ExpertSizeBytes:    32 * 1024,
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	// Initial buffer assignments
	active := streamer.ActiveBuffer()
	pref := streamer.PrefetchBuffer()

	if active.ID != BufferA {
		t.Errorf("initial active buffer should be BufferA, got %s", active.ID)
	}
	if pref.ID != BufferB {
		t.Errorf("initial prefetch buffer should be BufferB, got %s", pref.ID)
	}
	if active.VRAMAddress == pref.VRAMAddress {
		t.Errorf("double-buffer slabs must have disjoint VRAM addresses: %x == %x", active.VRAMAddress, pref.VRAMAddress)
	}
	if streamer.Rotations() != 0 {
		t.Errorf("initial rotations should be 0, got %d", streamer.Rotations())
	}

	// First rotation: Buffer B becomes active, Buffer A becomes prefetch
	rActive, rPref := streamer.RotateBuffers()
	if rActive.ID != BufferB || rPref.ID != BufferA {
		t.Errorf("after 1 rotation: active=%s, pref=%s (want BufferB/BufferA)", rActive.ID, rPref.ID)
	}
	if streamer.Rotations() != 1 {
		t.Errorf("expected 1 rotation, got %d", streamer.Rotations())
	}

	// Second rotation: Buffer A becomes active, Buffer B becomes prefetch
	rActive2, rPref2 := streamer.RotateBuffers()
	if rActive2.ID != BufferA || rPref2.ID != BufferB {
		t.Errorf("after 2 rotations: active=%s, pref=%s (want BufferA/BufferB)", rActive2.ID, rPref2.ID)
	}
	if streamer.Rotations() != 2 {
		t.Errorf("expected 2 rotations, got %d", streamer.Rotations())
	}
}

func TestMoEExpertP2PStreamer_HotTableEviction(t *testing.T) {
	hal := newTestHAL(t)
	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          1,
		NumExpertsPerLayer: 8,
		ExpertSizeBytes:    32 * 1024,
		HotTableCapacity:   3, // Fixed small capacity to test eviction boundary
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	// 1. Pin shared expert L0-E0 (must never be evicted)
	sharedKey := MoEExpertKey{LayerIndex: 0, ExpertID: 0}
	if err := streamer.PinSharedExpert(sharedKey); err != nil {
		t.Fatalf("PinSharedExpert failed: %v", err)
	}

	// 2. Add unpinned routed expert L0-E1 with access count 10
	e1Key := MoEExpertKey{LayerIndex: 0, ExpertID: 1}
	for i := 0; i < 10; i++ {
		streamer.RecordActivation(e1Key)
	}
	if err := streamer.putHotTableEntryLocked(e1Key, false, false); err != nil {
		t.Fatalf("putHotTableEntryLocked e1 failed: %v", err)
	}

	// 3. Add unpinned routed expert L0-E2 with access count 3 (selected for eviction)
	e2Key := MoEExpertKey{LayerIndex: 0, ExpertID: 2}
	for i := 0; i < 3; i++ {
		streamer.RecordActivation(e2Key)
	}
	if err := streamer.putHotTableEntryLocked(e2Key, false, false); err != nil {
		t.Fatalf("putHotTableEntryLocked e2 failed: %v", err)
	}

	// Table is now at full capacity (3/3: E0 pinned, E1 unpinned count 11, E2 unpinned count 4)
	// 4. Insert 4th expert L0-E3 with count 20
	e3Key := MoEExpertKey{LayerIndex: 0, ExpertID: 3}
	for i := 0; i < 20; i++ {
		streamer.RecordActivation(e3Key)
	}
	if err := streamer.putHotTableEntryLocked(e3Key, false, false); err != nil {
		t.Fatalf("putHotTableEntryLocked e3 failed: %v", err)
	}

	// E2 (lowest access count among unpinned) must have been evicted
	if _, ok := streamer.GetHotTableEntry(e2Key); ok {
		t.Errorf("expected expert %s to be evicted, but still present in hot table", e2Key)
	}

	// E0 (shared/pinned) must still reside in table
	if entry, ok := streamer.GetHotTableEntry(sharedKey); !ok || !entry.Pinned {
		t.Errorf("pinned shared expert %s was evicted or lost pinned status", sharedKey)
	}

	// E1 and E3 must be in table
	if _, ok := streamer.GetHotTableEntry(e1Key); !ok {
		t.Errorf("expected expert %s to be in table", e1Key)
	}
	if _, ok := streamer.GetHotTableEntry(e3Key); !ok {
		t.Errorf("expected expert %s to be in table", e3Key)
	}

	// 5. Pin all entries and verify ErrHotTableFull
	if err := streamer.PinHotExpert(e1Key); err != nil {
		t.Fatalf("PinHotExpert e1 failed: %v", err)
	}
	if err := streamer.PinHotExpert(e3Key); err != nil {
		t.Fatalf("PinHotExpert e3 failed: %v", err)
	}

	e4Key := MoEExpertKey{LayerIndex: 0, ExpertID: 4}
	err = streamer.putHotTableEntryLocked(e4Key, false, false)
	if !errors.Is(err, ErrHotTableFull) {
		t.Fatalf("expected ErrHotTableFull when all entries are pinned, got: %v", err)
	}
}

func TestMoEExpertP2PStreamer_MissingExpertHandling(t *testing.T) {
	hal := newTestHAL(t)
	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          2,
		NumExpertsPerLayer: 16,
		ExpertSizeBytes:    32 * 1024,
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	// Invalid layer
	invalidLayerKey := MoEExpertKey{LayerIndex: 99, ExpertID: 0}
	if _, err := streamer.GetExpertLBAMap(invalidLayerKey); !errors.Is(err, ErrExpertNotFound) {
		t.Errorf("expected ErrExpertNotFound for invalid layer, got: %v", err)
	}
	if _, err := streamer.LoadExpert(invalidLayerKey); !errors.Is(err, ErrExpertNotFound) {
		t.Errorf("expected LoadExpert ErrExpertNotFound for invalid layer, got: %v", err)
	}

	// Invalid expert ID
	invalidExpertKey := MoEExpertKey{LayerIndex: 0, ExpertID: 999}
	if _, err := streamer.GetExpertLBAMap(invalidExpertKey); !errors.Is(err, ErrExpertNotFound) {
		t.Errorf("expected ErrExpertNotFound for invalid expert ID, got: %v", err)
	}

	// Execution pipeline with missing expert
	_, err = streamer.ExecutePipelinedLayer(0, []int{0, 999}, func(key MoEExpertKey, slab *VRAMWeightSlab) error {
		return nil
	})
	if !errors.Is(err, ErrExpertNotFound) {
		t.Errorf("expected ExecutePipelinedLayer ErrExpertNotFound, got: %v", err)
	}
}

func TestMoEExpertP2PStreamer_PredictivePrefetch(t *testing.T) {
	hal := newTestHAL(t)
	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          3,
		NumExpertsPerLayer: 32,
		ExpertSizeBytes:    32 * 1024,
		PrefetchBuffers:    4,
		SimulatedIOLatency: 5 * time.Microsecond,
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	// Custom predictor test
	streamer.SetPredictorFunc(func(currentLayer int, routingLogits [][]float32, topK int) []int {
		return []int{7, 14, 21}
	})

	pred := streamer.PredictUpcomingExperts(0, nil, 3)
	if len(pred) != 3 || pred[0] != 7 || pred[1] != 14 || pred[2] != 21 {
		t.Fatalf("custom predictor failed, got: %+v", pred)
	}

	// Reset custom predictor to test default router logit aggregation
	streamer.SetPredictorFunc(nil)

	routingLogits := [][]float32{
		{0.1, 0.0, 4.5, 0.2},
		{0.0, 1.2, 3.8, 0.0},
	}
	defaultPred := streamer.PredictUpcomingExperts(0, routingLogits, 2)
	if len(defaultPred) != 2 || defaultPred[0] != 2 || defaultPred[1] != 1 {
		t.Fatalf("default predictor failed, got: %+v", defaultPred)
	}

	// Last layer prefetch should return nil (no upcoming layer)
	dispatched, err := streamer.DispatchPredictivePrefetch(2, routingLogits, 2)
	if err != nil {
		t.Fatalf("prefetch on last layer failed: %v", err)
	}
	if len(dispatched) != 0 {
		t.Fatalf("expected 0 dispatches on last layer, got %d", len(dispatched))
	}
}

func TestMoEExpertP2PStreamer_ZeroCopyInvariant(t *testing.T) {
	hal := newTestHAL(t)
	cfg := MoEExpertP2PStreamerConfig{
		NodeID:             0,
		NumLayers:          1,
		NumExpertsPerLayer: 16,
		ExpertSizeBytes:    64 * 1024,
		SimulatedIOLatency: 5 * time.Microsecond,
	}

	streamer, err := NewMoEExpertP2PStreamer(hal, cfg)
	if err != nil {
		t.Fatalf("NewMoEExpertP2PStreamer failed: %v", err)
	}

	if streamer.StagingCopyCount() != 0 {
		t.Fatalf("initial StagingCopyCount=%d != 0", streamer.StagingCopyCount())
	}

	// Stream 4 experts directly
	for e := 0; e < 4; e++ {
		key := MoEExpertKey{LayerIndex: 0, ExpertID: e}
		addr, err := streamer.LoadExpert(key)
		if err != nil {
			t.Fatalf("LoadExpert %s failed: %v", key, err)
		}
		if addr == 0 {
			t.Fatalf("LoadExpert returned 0 VRAM address")
		}
	}

	cmds := streamer.CompletedCommands()
	if len(cmds) == 0 {
		t.Fatalf("expected completed commands, got 0")
	}
	for _, c := range cmds {
		if c.StagingCopyCount() != 0 {
			t.Errorf("cmd %d has staging copy count %d != 0", c.CommandID, c.StagingCopyCount())
		}
	}
}
