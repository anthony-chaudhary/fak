package strix

import (
	"errors"
	"sync"
	"testing"
)

// TestMALLTiler executes the comprehensive test suite for AMD Strix Halo
// 32MB MALL Infinity Cache attention working set tiling and RDNA 3.5 cache bypass hints.
func TestMALLTiler(t *testing.T) {
	t.Run("Criterion1_GeometryAndTilingCalculation", testMALLTiler_GeometryAndTilingCalculation)
	t.Run("Criterion2_AssemblyModifierBitmasks", testMALLTiler_AssemblyModifierBitmasks)
	t.Run("Criterion3_TileTokenMapping", testMALLTiler_TileTokenMapping)
	t.Run("Criterion4_MultiAgentMALLResidencyAndBandwidthSavings", testMALLTiler_MultiAgentMALLResidencyAndBandwidthSavings)
	t.Run("Criterion5_ConcurrencyAndThreadSafety", testMALLTiler_ConcurrencyAndThreadSafety)
	t.Run("Criterion6_QuarantinedFallbackAndLifecycle", testMALLTiler_QuarantinedFallbackAndLifecycle)
}

// testMALLTiler_GeometryAndTilingCalculation verifies that standard GQA 8 KV heads,
// dim 128, FP16 maps 8,192 tokens exactly to 32MB MALL capacity with zero boundary overflow.
func testMALLTiler_GeometryAndTilingCalculation(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	bytesPerToken := gqaGeo.BytesPerTokenPerLayer()
	if bytesPerToken != 4096 {
		t.Fatalf("expected 4,096 bytes per token per layer, got %d", bytesPerToken)
	}

	// 8,192 tokens @ 4,096 bytes/token = 33,554,432 bytes (exactly 32MB)
	cfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: true,
	}

	tiler, err := NewAttentionMALLTiler(cfg)
	if err != nil {
		t.Fatalf("failed to initialize MALL tiler: %v", err)
	}
	defer tiler.Close()

	plan := tiler.Plan()
	if plan == nil {
		t.Fatal("expected non-nil tiling plan")
	}

	if plan.TotalMALLCapacityBytes != MALLCapacityBytes {
		t.Errorf("expected TotalMALLCapacityBytes %d, got %d", MALLCapacityBytes, plan.TotalMALLCapacityBytes)
	}

	if plan.RootAllocatedBytes != MALLCapacityBytes {
		t.Errorf("expected RootAllocatedBytes %d, got %d", MALLCapacityBytes, plan.RootAllocatedBytes)
	}

	if plan.RemainingMALLBytes != 0 {
		t.Errorf("expected RemainingMALLBytes 0 (exact boundary fit), got %d", plan.RemainingMALLBytes)
	}

	if plan.CapacityUtilizationPct != 100.0 {
		t.Errorf("expected 100%% capacity utilization, got %.2f%%", plan.CapacityUtilizationPct)
	}

	// Verify strict overflow rejection when requested tokens exceed 32MB MALL capacity
	overflowCfg := MALLTilerConfig{
		Geometry:       gqaGeo,
		RootTokenCount: 8193, // 8193 * 4096 = 33,558,528 > 33,554,432
	}
	_, err = NewAttentionMALLTiler(overflowCfg)
	if err == nil {
		t.Fatal("expected ErrTileOverflow for 8,193 tokens, got nil")
	}
	if !errors.Is(err, ErrTileOverflow) {
		t.Errorf("expected ErrTileOverflow, got %v", err)
	}
}

// testMALLTiler_AssemblyModifierBitmasks validates that RDNA 3.5 assembly modifier bitmasks
// are synthesized correctly: SLC=0, GLC=0 for root KV, SLC=1, GLC=0 for streaming weights.
func testMALLTiler_AssemblyModifierBitmasks(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	cfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: true,
		AblationArm:           Arm1ExplicitBitmasks,
	}

	tiler, err := NewAttentionMALLTiler(cfg)
	if err != nil {
		t.Fatalf("failed to initialize MALL tiler: %v", err)
	}
	defer tiler.Close()

	// 1. Root KV Flags & Bitmasks (SLC=0, GLC=0)
	rootFlags := tiler.RootKVModifierFlags()
	if rootFlags.SLC != 0 || rootFlags.GLC != 0 {
		t.Errorf("expected root KV SLC=0, GLC=0, got SLC=%d, GLC=%d", rootFlags.SLC, rootFlags.GLC)
	}
	if !rootFlags.Temporal {
		t.Error("expected root KV to have Temporal=true")
	}
	if rootFlags.Bypass {
		t.Error("expected root KV to have Bypass=false")
	}

	rootTile := tiler.Plan().RootTile
	if rootTile.Bitmask.InstructionDword1&MUBUFSLCBit != 0 {
		t.Errorf("expected root KV MUBUFSLCBit cleared in DWord1, got 0x%08X", rootTile.Bitmask.InstructionDword1)
	}
	if rootTile.Bitmask.InstructionDword0&MUBUFGLCBit != 0 {
		t.Errorf("expected root KV MUBUFGLCBit cleared in DWord0, got 0x%08X", rootTile.Bitmask.InstructionDword0)
	}
	if rootTile.Bitmask.DescriptorWord3Mask&DescriptorWord3SLCBit != 0 {
		t.Errorf("expected root KV DescriptorWord3SLCBit cleared, got 0x%08X", rootTile.Bitmask.DescriptorWord3Mask)
	}

	// 2. Weight Streaming Flags & Bitmasks (SLC=1, GLC=0)
	weightFlags := tiler.WeightModifierFlags()
	if weightFlags.SLC != 1 || weightFlags.GLC != 0 {
		t.Errorf("expected weight streaming SLC=1, GLC=0, got SLC=%d, GLC=%d", weightFlags.SLC, weightFlags.GLC)
	}
	if !weightFlags.NonTemporal {
		t.Error("expected weight streaming to have NonTemporal=true")
	}
	if !weightFlags.Bypass {
		t.Error("expected weight streaming to have Bypass=true")
	}

	weightTile := tiler.Plan().WeightTilePolicy
	if weightTile.Bitmask.InstructionDword1&MUBUFSLCBit == 0 {
		t.Errorf("expected weight MUBUFSLCBit set in DWord1, got 0x%08X", weightTile.Bitmask.InstructionDword1)
	}
	if weightTile.Bitmask.DescriptorWord3Mask&DescriptorWord3SLCBit == 0 {
		t.Errorf("expected weight DescriptorWord3SLCBit set, got 0x%08X", weightTile.Bitmask.DescriptorWord3Mask)
	}

	// 3. Divergent KV Flags (SLC=1, GLC=0)
	divFlags := tiler.DivergentKVModifierFlags()
	if divFlags.SLC != 1 || divFlags.GLC != 0 {
		t.Errorf("expected divergent KV SLC=1, GLC=0, got SLC=%d, GLC=%d", divFlags.SLC, divFlags.GLC)
	}
}

// testMALLTiler_TileTokenMapping verifies mapping between token indices and MALL tiles.
func testMALLTiler_TileTokenMapping(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	cfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: true,
	}

	tiler, err := NewAttentionMALLTiler(cfg)
	if err != nil {
		t.Fatalf("failed to initialize MALL tiler: %v", err)
	}
	defer tiler.Close()

	// Token 0 (first root token) -> RootTile
	tile0, err := tiler.TileForToken(0)
	if err != nil {
		t.Fatalf("unexpected error mapping token 0: %v", err)
	}
	if !tile0.IsRootPinned || tile0.Hint != HintTemporalRootKV {
		t.Errorf("expected root tile for token 0, got is_pinned=%v, hint=%v", tile0.IsRootPinned, tile0.Hint)
	}

	// Token 8191 (last root token) -> RootTile
	tile8191, err := tiler.TileForToken(8191)
	if err != nil {
		t.Fatalf("unexpected error mapping token 8191: %v", err)
	}
	if !tile8191.IsRootPinned || tile8191.Hint != HintTemporalRootKV {
		t.Errorf("expected root tile for token 8191, got is_pinned=%v, hint=%v", tile8191.IsRootPinned, tile8191.Hint)
	}

	// Token 8192 (first divergent continuation token) -> DivergentTile
	tile8192, err := tiler.TileForToken(8192)
	if err != nil {
		t.Fatalf("unexpected error mapping token 8192: %v", err)
	}
	if tile8192.IsRootPinned || tile8192.Hint != HintDivergentKV {
		t.Errorf("expected divergent tile for token 8192, got is_pinned=%v, hint=%v", tile8192.IsRootPinned, tile8192.Hint)
	}
	if tile8192.ModifierFlags.SLC != 1 {
		t.Errorf("expected divergent tile to have SLC=1, got %d", tile8192.ModifierFlags.SLC)
	}

	// Negative token index -> error
	_, err = tiler.TileForToken(-1)
	if !errors.Is(err, ErrInvalidTokenIndex) {
		t.Errorf("expected ErrInvalidTokenIndex for token -1, got %v", err)
	}
}

// testMALLTiler_MultiAgentMALLResidencyAndBandwidthSavings simulates multi-agent concurrent
// decode and validates >= 95% root MALL residency and >= 32 GB/s DRAM bandwidth savings.
func testMALLTiler_MultiAgentMALLResidencyAndBandwidthSavings(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	cfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: true,
	}

	tiler, err := NewAttentionMALLTiler(cfg)
	if err != nil {
		t.Fatalf("failed to initialize MALL tiler: %v", err)
	}
	defer tiler.Close()

	// Initial residency of pinned root KV must be 100%
	initialResidency := tiler.PinnedResidencyPercent()
	if initialResidency < 99.0 {
		t.Fatalf("expected initial pinned KV residency >= 99%%, got %.2f%%", initialResidency)
	}

	// Simulate 8-agent concurrent decode over 10 steps with streaming weights and divergent tokens
	telemetry := tiler.SimulateMultiAgentDecode(8, 10, 16)

	if telemetry.WeightBypasses <= 0 {
		t.Error("expected non-zero weight bypasses")
	}

	if telemetry.RootHitRate < DefaultMALLHitRateTarget {
		t.Errorf("expected root hit rate >= %.2f, got %.4f", DefaultMALLHitRateTarget, telemetry.RootHitRate)
	}

	postResidency := tiler.PinnedResidencyPercent()
	if postResidency < MinRootMALLResidencyPct {
		t.Errorf("expected post-decode pinned KV residency >= %.1f%%, got %.2f%%", MinRootMALLResidencyPct, postResidency)
	}

	if telemetry.EffectiveDRAMSavingsGBps < DefaultDRAMBandwidthSavingsFloorGBs {
		t.Errorf("expected EffectiveDRAMSavingsGBps >= %.1f GB/s, got %.2f GB/s",
			DefaultDRAMBandwidthSavingsFloorGBs, telemetry.EffectiveDRAMSavingsGBps)
	}

	if telemetry.DRAMBytesSaved <= 0 {
		t.Error("expected positive DRAM bytes saved")
	}

	// Check stringer
	str := telemetry.String()
	if len(str) == 0 {
		t.Error("expected non-empty telemetry string")
	}
}

// testMALLTiler_ConcurrencyAndThreadSafety tests race-free concurrent execution across multiple goroutines.
func testMALLTiler_ConcurrencyAndThreadSafety(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	cfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: true,
	}

	tiler, err := NewAttentionMALLTiler(cfg)
	if err != nil {
		t.Fatalf("failed to initialize MALL tiler: %v", err)
	}
	defer tiler.Close()

	const numWorkers = 16
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tokenIdx := (workerID*iterations + i) % 10000
				_, _ = tiler.TileForToken(tokenIdx)
				_, _ = tiler.AccessKV(tokenIdx % 8192)
				_, _ = tiler.AccessWeight(uint64(0x200000000000+tokenIdx*4096), 4096)
				_ = tiler.Telemetry()
			}
		}(w)
	}

	wg.Wait()

	telem := tiler.Telemetry()
	if telem.TotalAccesses < numWorkers*iterations*2 {
		t.Errorf("expected total accesses >= %d, got %d", numWorkers*iterations*2, telem.TotalAccesses)
	}
}

// testMALLTiler_QuarantinedFallbackAndLifecycle verifies graceful fallback and closed state handling.
func testMALLTiler_QuarantinedFallbackAndLifecycle(t *testing.T) {
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	// 1. Quarantined Fallback: Assembly tagging disabled
	fallbackCfg := MALLTilerConfig{
		Geometry:              gqaGeo,
		RootTokenCount:        8192,
		EnableAssemblyTagging: false,
	}

	fallbackTiler, err := NewAttentionMALLTiler(fallbackCfg)
	if err != nil {
		t.Fatalf("failed to initialize fallback tiler: %v", err)
	}
	defer fallbackTiler.Close()

	// Untagged loads should have empty assembly suffix but valid plan
	if fallbackTiler.Plan() == nil {
		t.Fatal("expected non-nil plan under fallback mode")
	}

	// 2. Lifecycle: operations on closed tiler
	tiler, err := NewAttentionMALLTiler(MALLTilerConfig{
		Geometry:       gqaGeo,
		RootTokenCount: 8192,
	})
	if err != nil {
		t.Fatalf("failed to initialize tiler: %v", err)
	}

	if err := tiler.Close(); err != nil {
		t.Fatalf("failed to close tiler: %v", err)
	}

	_, err = tiler.TileForToken(0)
	if !errors.Is(err, ErrTilerClosed) {
		t.Errorf("expected ErrTilerClosed for TileForToken on closed tiler, got %v", err)
	}

	_, err = tiler.AccessKV(0)
	if !errors.Is(err, ErrTilerClosed) {
		t.Errorf("expected ErrTilerClosed for AccessKV on closed tiler, got %v", err)
	}

	_, err = tiler.AccessWeight(0x1000, 4096)
	if !errors.Is(err, ErrTilerClosed) {
		t.Errorf("expected ErrTilerClosed for AccessWeight on closed tiler, got %v", err)
	}
}
