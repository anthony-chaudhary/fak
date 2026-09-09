package strix

import (
	"errors"
	"sync"
	"testing"

)

// TestMALLPartitioner executes the complete verification suite for attention working set
// geometry calculation, strict 32MB MALL capacity partitioning, ablation arms, and quarantined fallback.
func TestMALLPartitioner(t *testing.T) {
	t.Run("GeometryCalculation", testMALL_GeometryCalculation)
	t.Run("Static8kPinnedRoot", testMALL_Static8kPinnedRoot)
	t.Run("Dynamic4kRootDraft", testMALL_Dynamic4kRootDraft)
	t.Run("UnpartitionedBaseline", testMALL_UnpartitionedBaseline)
	t.Run("StrictBoundaryOverflowRejection", testMALL_StrictBoundaryOverflowRejection)
	t.Run("QuarantinedFallback", testMALL_QuarantinedFallback)
	t.Run("ConcurrentSafety", testMALL_ConcurrentSafety)
	t.Run("TelemetryAndMetrics", testMALL_TelemetryAndMetrics)
}

// testMALL_GeometryCalculation verifies mathematical correctness of KV byte stride computations
// across standard GQA, MHA, FP8, and multi-layer attention topologies.
func testMALL_GeometryCalculation(t *testing.T) {
	// Standard GQA on AMD Strix Halo (8 KV heads, dim 128, FP16 = 2 bytes)
	gqaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	if err := gqaGeo.Validate(); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}

	// Stride per head: 128 * 2 = 256 bytes
	if stride := gqaGeo.HeadStrideBytes(); stride != 256 {
		t.Errorf("expected HeadStrideBytes 256, got %d", stride)
	}

	// KV stride: 8 * 128 * 2 = 2,048 bytes
	if kvStride := gqaGeo.KVStrideBytes(); kvStride != 2048 {
		t.Errorf("expected KVStrideBytes 2048, got %d", kvStride)
	}

	// Single-layer token byte footprint: 2 (K+V) * 8 * 128 * 2 = 4,096 bytes/token
	expectedLayerBytes := int64(4096)
	if layerBytes := gqaGeo.BytesPerTokenPerLayer(); layerBytes != expectedLayerBytes {
		t.Fatalf("expected BytesPerTokenPerLayer %d, got %d", expectedLayerBytes, layerBytes)
	}

	// Multi-layer (32 layers) token byte footprint: 32 * 4,096 = 131,072 bytes/token
	expectedTotalBytes := int64(32 * 4096)
	if totalBytes := gqaGeo.TotalBytesPerToken(); totalBytes != expectedTotalBytes {
		t.Fatalf("expected TotalBytesPerToken %d, got %d", expectedTotalBytes, totalBytes)
	}

	// Verify MHA topology (32 heads, dim 128, FP16 = 2 bytes): 2 * 32 * 128 * 2 = 16,384 bytes/token
	mhaGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         32,
		HeadDim:         128,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}
	if layerBytes := mhaGeo.BytesPerTokenPerLayer(); layerBytes != 16384 {
		t.Errorf("expected MHA BytesPerTokenPerLayer 16384, got %d", layerBytes)
	}

	// Verify FP8 precision (8 heads, dim 128, FP8 = 1 byte): 2 * 8 * 128 * 1 = 2,048 bytes/token
	fp8Geo := AttentionGeometry{
		Layers:          32,
		KVHeads:         8,
		HeadDim:         128,
		BytesPerElement: 1,
		Precision:       AttentionPrecisionFP8,
	}
	if layerBytes := fp8Geo.BytesPerTokenPerLayer(); layerBytes != 2048 {
		t.Errorf("expected FP8 BytesPerTokenPerLayer 2048, got %d", layerBytes)
	}

	// Invalid geometry parameters must fail validation
	invalidGeos := []AttentionGeometry{
		{Layers: 0, KVHeads: 8, HeadDim: 128, BytesPerElement: 2},
		{Layers: 32, KVHeads: 0, HeadDim: 128, BytesPerElement: 2},
		{Layers: 32, KVHeads: 8, HeadDim: 0, BytesPerElement: 2},
		{Layers: 32, KVHeads: 8, HeadDim: 128, BytesPerElement: 0},
	}
	for i, bad := range invalidGeos {
		if err := bad.Validate(); err == nil {
			t.Errorf("case %d: expected validation error for invalid geometry %+v", i, bad)
		}
	}
}

// testMALL_Static8kPinnedRoot verifies Ablation Arm 1 (Static 8k Pinned Root):
// Allocates full 32MB to 8,192-token shared system prefix (8 KV heads, dim 128, FP16 = 4 KB/token across layer tiles).
func testMALL_Static8kPinnedRoot(t *testing.T) {
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes, // 33,554,432 bytes
		LayerTiled:             true,
		Arm:                    ArmStatic8kPinnedRoot,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err != nil {
		t.Fatalf("failed to initialize Arm 1 partitioner: %v", err)
	}

	plan := partitioner.Plan()
	if plan == nil {
		t.Fatalf("partition plan is nil")
	}

	if !plan.IsPartitioned {
		t.Fatalf("expected plan.IsPartitioned == true")
	}
	if plan.FallbackToDRAM {
		t.Fatalf("expected FallbackToDRAM == false for valid 8k root partition")
	}

	// 8,192 tokens * 4,096 bytes/token = exactly 33,554,432 bytes (32MB)
	expectedRootBytes := int64(8192 * 4096)
	if plan.RootPrefix.SizeBytes != expectedRootBytes {
		t.Errorf("expected RootPrefix size %d bytes, got %d", expectedRootBytes, plan.RootPrefix.SizeBytes)
	}
	if plan.RootPrefix.Tokens != 8192 {
		t.Errorf("expected RootPrefix tokens 8192, got %d", plan.RootPrefix.Tokens)
	}
	if plan.ActiveDraft.SizeBytes != 0 {
		t.Errorf("expected Draft size 0 in static root arm, got %d", plan.ActiveDraft.SizeBytes)
	}
	if plan.TotalAllocatedBytes != MALLSizeBytes {
		t.Errorf("expected TotalAllocatedBytes %d, got %d", MALLSizeBytes, plan.TotalAllocatedBytes)
	}
	if plan.RemainingHeadroomBytes != 0 {
		t.Errorf("expected RemainingHeadroomBytes 0 for 8k pinned root, got %d", plan.RemainingHeadroomBytes)
	}
	if plan.RootPrefix.CachePolicy != "TEMPORAL_PINNED" {
		t.Errorf("expected RootPrefix CachePolicy TEMPORAL_PINNED, got %s", plan.RootPrefix.CachePolicy)
	}
	if plan.PolicyDirectives.SLC != 0 || plan.PolicyDirectives.GLC != 0 {
		t.Errorf("expected SLC=0, GLC=0 for pinned root, got %+v", plan.PolicyDirectives)
	}
}

// testMALL_Dynamic4kRootDraft verifies Ablation Arm 2 (Dynamic 4k Root + Draft):
// Partitions 16MB to 4,096-token root prefix, 14MB to active speculative draft trees, and 2MB to safety reserve.
func testMALL_Dynamic4kRootDraft(t *testing.T) {
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             true,
		Arm:                    ArmDynamic4kRootDraft,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err != nil {
		t.Fatalf("failed to initialize Arm 2 partitioner: %v", err)
	}

	plan := partitioner.Plan()
	if plan == nil {
		t.Fatalf("partition plan is nil")
	}

	if !plan.IsPartitioned {
		t.Fatalf("expected plan.IsPartitioned == true")
	}

	// 16MB root = 16 * 1024 * 1024 = 16,777,216 bytes (4,096 tokens at 4KB)
	expectedRootBytes := int64(16 * 1024 * 1024)
	if plan.RootPrefix.SizeBytes != expectedRootBytes {
		t.Errorf("expected RootPrefix size %d bytes (16MB), got %d", expectedRootBytes, plan.RootPrefix.SizeBytes)
	}
	if plan.RootPrefix.Tokens != 4096 {
		t.Errorf("expected RootPrefix tokens 4096, got %d", plan.RootPrefix.Tokens)
	}

	// 14MB draft = 14 * 1024 * 1024 = 14,680,064 bytes (3,584 tokens at 4KB)
	expectedDraftBytes := int64(14 * 1024 * 1024)
	if plan.ActiveDraft.SizeBytes != expectedDraftBytes {
		t.Errorf("expected ActiveDraft size %d bytes (14MB), got %d", expectedDraftBytes, plan.ActiveDraft.SizeBytes)
	}
	if plan.ActiveDraft.Tokens != 3584 {
		t.Errorf("expected ActiveDraft tokens 3584, got %d", plan.ActiveDraft.Tokens)
	}

	// 2MB reserve = 2 * 1024 * 1024 = 2,097,152 bytes
	expectedReserveBytes := int64(2 * 1024 * 1024)
	if plan.Reserve.SizeBytes != expectedReserveBytes {
		t.Errorf("expected Reserve size %d bytes (2MB), got %d", expectedReserveBytes, plan.Reserve.SizeBytes)
	}

	// Sum: 16MB + 14MB + 2MB = 32MB exactly
	expectedTotal := expectedRootBytes + expectedDraftBytes + expectedReserveBytes
	if plan.TotalAllocatedBytes != expectedTotal {
		t.Fatalf("expected total allocated %d, got %d", expectedTotal, plan.TotalAllocatedBytes)
	}
	if plan.TotalAllocatedBytes != MALLSizeBytes {
		t.Fatalf("expected exact 32MB allocation, got %d", plan.TotalAllocatedBytes)
	}
	if plan.RemainingHeadroomBytes != 0 {
		t.Errorf("expected 0 headroom on exact 32MB budget, got %d", plan.RemainingHeadroomBytes)
	}

	// Verify contiguous offsets
	if plan.RootPrefix.OffsetBytes != 0 {
		t.Errorf("expected root offset 0, got %d", plan.RootPrefix.OffsetBytes)
	}
	if plan.ActiveDraft.OffsetBytes != expectedRootBytes {
		t.Errorf("expected draft offset %d, got %d", expectedRootBytes, plan.ActiveDraft.OffsetBytes)
	}
	if plan.Reserve.OffsetBytes != expectedRootBytes+expectedDraftBytes {
		t.Errorf("expected reserve offset %d, got %d", expectedRootBytes+expectedDraftBytes, plan.Reserve.OffsetBytes)
	}
}

// testMALL_UnpartitionedBaseline verifies Ablation Arm 3 (Unpartitioned Baseline):
// Unmanaged ad-hoc allocation directing runtime to stream from DRAM unpinned.
func testMALL_UnpartitionedBaseline(t *testing.T) {
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             true,
		Arm:                    ArmUnpartitionedBaseline,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err != nil {
		t.Fatalf("failed to initialize Arm 3 partitioner: %v", err)
	}

	plan := partitioner.Plan()
	if plan == nil {
		t.Fatalf("partition plan is nil")
	}

	if plan.IsPartitioned {
		t.Fatalf("expected plan.IsPartitioned == false for unpartitioned baseline")
	}
	if !plan.FallbackToDRAM {
		t.Fatalf("expected FallbackToDRAM == true for unpartitioned baseline")
	}
	if plan.PolicyDirectives.SLC != 1 {
		t.Errorf("expected SLC=1 (streaming bypass) for unpartitioned baseline, got %d", plan.PolicyDirectives.SLC)
	}
	if !partitioner.IsFallbackActive() {
		t.Errorf("expected IsFallbackActive() == true")
	}
}

// testMALL_StrictBoundaryOverflowRejection verifies that any configuration requesting allocations
// in excess of the physical 32MB MALL capacity (33,554,432 bytes) is strictly rejected with
// ErrMALLPartitionOverflow and triggers quarantined fallback.
func testMALL_StrictBoundaryOverflowRejection(t *testing.T) {
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             true,
		RootPrefixTokens:       8192, // 8,192 * 4,096 = 33,554,432 bytes (100% MALL)
		DraftTokens:            1,    // 1 token = 4,096 bytes (over-budget by 4KB!)
		ReserveBytes:           0,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err == nil {
		t.Fatalf("expected ErrMALLPartitionOverflow, got nil error")
	}
	if !errors.Is(err, ErrMALLPartitionOverflow) {
		t.Fatalf("expected ErrMALLPartitionOverflow, got %v", err)
	}

	// Partitioner should return a quarantined fallback plan
	plan := partitioner.Plan()
	if plan == nil {
		t.Fatalf("expected non-nil fallback plan")
	}
	if plan.IsPartitioned {
		t.Errorf("expected IsPartitioned == false on overflow")
	}
	if !plan.FallbackToDRAM {
		t.Errorf("expected FallbackToDRAM == true on overflow")
	}
	if plan.PolicyDirectives.SLC != 1 {
		t.Errorf("expected SLC=1 on overflow fallback, got %d", plan.PolicyDirectives.SLC)
	}

	// Test dynamic re-partitioning over-allocation
	overflowCfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             true,
		RootPrefixTokens:       9000, // 9,000 * 4KB = 36MB > 32MB
	}
	rePlan, reErr := partitioner.Partition(overflowCfg)
	if reErr == nil || !errors.Is(reErr, ErrMALLPartitionOverflow) {
		t.Fatalf("expected ErrMALLPartitionOverflow on dynamic partition, got %v", reErr)
	}
	if !rePlan.FallbackToDRAM {
		t.Errorf("expected FallbackToDRAM on dynamic overflow")
	}

	// PartitionSafe should handle overflow smoothly without returning error
	safePlan := partitioner.PartitionSafe(overflowCfg)
	if safePlan == nil {
		t.Fatalf("expected non-nil plan from PartitionSafe")
	}
	if !safePlan.FallbackToDRAM {
		t.Errorf("expected FallbackToDRAM in PartitionSafe on overflow")
	}
}

// testMALL_QuarantinedFallback validates the quarantined fallback mechanism:
// If model geometry exceeds 32MB MALL capacity, partitioner safely returns an unpartitioned fallback flag,
// instructing the runtime to stream KV blocks from DRAM without asserting cache pin hints.
func testMALL_QuarantinedFallback(t *testing.T) {
	// Massive model geometry where a single layer exceeds 32MB:
	// 128 KV heads, dim 256, FP16 = 2 * 128 * 256 * 2 = 131,072 bytes per token.
	// For full 32 layers: 4,194,304 bytes per token.
	// 10 tokens = 41.9 MB > 32MB!
	hugeGeo := AttentionGeometry{
		Layers:          32,
		KVHeads:         128,
		HeadDim:         256,
		BytesPerElement: 2,
		Precision:       AttentionPrecisionFP16,
	}

	cfg := MALLPartitionConfig{
		Geometry:               hugeGeo,
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             false, // across all layers
		RootPrefixTokens:       10,    // 10 * 4.19MB = ~41.9MB > 32MB
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err == nil {
		t.Fatalf("expected overflow error for huge geometry, got nil")
	}

	plan := partitioner.Plan()
	if !plan.FallbackToDRAM {
		t.Fatalf("expected FallbackToDRAM == true for huge geometry")
	}
	if plan.IsPartitioned {
		t.Fatalf("expected IsPartitioned == false")
	}
	if plan.PolicyDirectives.SLC != 1 {
		t.Errorf("expected SLC=1 (bypass), got %d", plan.PolicyDirectives.SLC)
	}

	// Fallback counter must have recorded at least 1 fallback
	if partitioner.FallbackCount() == 0 {
		t.Errorf("expected non-zero fallback counter, got %d", partitioner.FallbackCount())
	}
}

// testMALL_ConcurrentSafety validates thread safety and data-race freedom under parallel access.
func testMALL_ConcurrentSafety(t *testing.T) {
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes,
		LayerTiled:             true,
		RootPrefixTokens:       4096,
		DraftTokens:            2048,
		ReserveBytes:           1024 * 1024,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err != nil {
		t.Fatalf("failed to initialize partitioner: %v", err)
	}

	var wg sync.WaitGroup
	workers := 16
	iterations := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				switch (workerID + j) % 5 {
				case 0:
					_ = partitioner.Plan()
				case 1:
					_ = partitioner.Metrics()
				case 2:
					_ = partitioner.IsFallbackActive()
				case 3:
					_, _ = partitioner.ComputeTotalWorkingSet(cfg.Geometry, 1024, true)
				case 4:
					if j%10 == 0 {
						arm := ArmDynamic4kRootDraft
						if j%20 == 0 {
							arm = ArmStatic8kPinnedRoot
						}
						_, _ = partitioner.ConfigureArm(arm)
					}
				}
			}
		}(i)
	}

	wg.Wait()
}

// testMALL_TelemetryAndMetrics verifies real-time partition utilization and capacity margin metrics.
func testMALL_TelemetryAndMetrics(t *testing.T) {
	// 4,096 tokens root (16MB) + 2,048 tokens draft (8MB) + 2MB reserve = 26MB total (6MB headroom)
	cfg := MALLPartitionConfig{
		Geometry:               DefaultGQAAttentionGeometry(),
		TotalMALLCapacityBytes: MALLSizeBytes, // 32MB
		LayerTiled:             true,
		RootPrefixTokens:       4096,
		DraftTokens:            2048,
		ReserveBytes:           2 * 1024 * 1024,
	}

	partitioner, err := NewMALLPartitioner(cfg)
	if err != nil {
		t.Fatalf("failed to initialize partitioner: %v", err)
	}

	metrics := partitioner.Metrics()
	expectedAllocated := int64(4096*4096 + 2048*4096 + 2*1024*1024) // 16MB + 8MB + 2MB = 26MB = 27,262,976 bytes
	if metrics.TotalAllocatedBytes != expectedAllocated {
		t.Errorf("expected TotalAllocatedBytes %d, got %d", expectedAllocated, metrics.TotalAllocatedBytes)
	}

	expectedHeadroom := MALLSizeBytes - expectedAllocated // 6MB = 6,291,456 bytes
	if metrics.RemainingHeadroomBytes != expectedHeadroom {
		t.Errorf("expected RemainingHeadroomBytes %d, got %d", expectedHeadroom, metrics.RemainingHeadroomBytes)
	}

	expectedUtilRatio := float64(expectedAllocated) / float64(MALLSizeBytes) // ~0.8125 (81.25%)
	if metrics.MALLUtilizationRatio < expectedUtilRatio-0.001 || metrics.MALLUtilizationRatio > expectedUtilRatio+0.001 {
		t.Errorf("expected utilization ratio ~%.4f, got %.4f", expectedUtilRatio, metrics.MALLUtilizationRatio)
	}

	if metrics.BytesPerToken != 4096 {
		t.Errorf("expected BytesPerToken 4096, got %d", metrics.BytesPerToken)
	}

	if metrics.MaxMALLTokens != 8192 {
		t.Errorf("expected MaxMALLTokens 8192, got %d", metrics.MaxMALLTokens)
	}
}
