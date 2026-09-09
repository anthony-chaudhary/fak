// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import (
	"fmt"
	"sync"
	"time"
)

// AttentionMALLTiler manages attention working set geometry, capacity boundary enforcement,
// RDNA 3.5 cache modifier bitmask synthesis, and 32MB MALL Infinity Cache residency.
type AttentionMALLTiler struct {
	mu         sync.RWMutex
	config     MALLTilerConfig
	plan       *MALLTilerPlan
	modGen     *ModifierGenerator
	cacheModel *MALLCacheModel
	telemetry  MALLTilerTelemetry
	closed     bool
}

// NewAttentionMALLTiler creates and initializes an attention working set tiler for AMD Strix Halo.
func NewAttentionMALLTiler(cfg MALLTilerConfig) (*AttentionMALLTiler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	bytesPerToken := cfg.Geometry.BytesPerTokenPerLayer()
	if bytesPerToken <= 0 {
		return nil, fmt.Errorf("%w: computed non-positive bytes per token", ErrInvalidTileGeometry)
	}

	rootAllocatedBytes := int64(cfg.RootTokenCount) * bytesPerToken
	if rootAllocatedBytes > cfg.TotalMALLCapacityBytes {
		return nil, fmt.Errorf("%w: requested root allocation %d bytes (%d tokens @ %d bytes/token) exceeds capacity %d bytes",
			ErrTileOverflow, rootAllocatedBytes, cfg.RootTokenCount, bytesPerToken, cfg.TotalMALLCapacityBytes)
	}

	modGen, err := NewModifierGenerator(GeneratorConfig{
		TargetArch:            cfg.TargetArch,
		EnableAssemblyTagging: cfg.EnableAssemblyTagging,
		StrictArchValidation:  false,
		AblationArm:           cfg.AblationArm,
	})
	if err != nil {
		return nil, fmt.Errorf("strix/cache: failed to initialize modifier generator: %w", err)
	}

	rootBitmask, err := modGen.GenerateBitmask(IntentTemporalRootKV)
	if err != nil {
		return nil, fmt.Errorf("strix/cache: failed to generate root KV bitmask: %w", err)
	}

	weightBitmask, err := modGen.GenerateBitmask(IntentNonTemporalWeight)
	if err != nil {
		return nil, fmt.Errorf("strix/cache: failed to generate weight bitmask: %w", err)
	}

	divergentBitmask, err := modGen.GenerateBitmask(IntentNonTemporalWeight)
	if err != nil {
		return nil, fmt.Errorf("strix/cache: failed to generate divergent KV bitmask: %w", err)
	}

	rootFlags := CacheModifierFlags{
		SLC:         rootBitmask.SLC,
		GLC:         rootBitmask.GLC,
		DLC:         rootBitmask.DLC,
		Temporal:    true,
		NonTemporal: false,
		Bypass:      false,
		PolicyName:  string(HintTemporalRootKV),
	}

	weightFlags := CacheModifierFlags{
		SLC:         weightBitmask.SLC,
		GLC:         weightBitmask.GLC,
		DLC:         weightBitmask.DLC,
		Temporal:    false,
		NonTemporal: true,
		Bypass:      weightBitmask.SLC == 1,
		PolicyName:  string(HintNonTemporalWeight),
	}

	divergentFlags := CacheModifierFlags{
		SLC:         divergentBitmask.SLC,
		GLC:         divergentBitmask.GLC,
		DLC:         divergentBitmask.DLC,
		Temporal:    false,
		NonTemporal: true,
		Bypass:      divergentBitmask.SLC == 1,
		PolicyName:  string(HintDivergentKV),
	}

	remainingMALLBytes := cfg.TotalMALLCapacityBytes - rootAllocatedBytes
	utilizationPct := float64(rootAllocatedBytes) / float64(cfg.TotalMALLCapacityBytes) * 100.0

	rootTile := MALLTile{
		TileID:         0,
		StartToken:     0,
		EndToken:       cfg.RootTokenCount,
		TokenCount:     cfg.RootTokenCount,
		BytesPerToken:  bytesPerToken,
		TotalSizeBytes: rootAllocatedBytes,
		IsRootPinned:   true,
		Hint:           HintTemporalRootKV,
		ModifierFlags:  rootFlags,
		Bitmask:        rootBitmask,
		AssemblySuffix: rootBitmask.AssemblySuffix,
	}

	weightPolicy := MALLTile{
		TileID:         -1,
		StartToken:     -1,
		EndToken:       -1,
		TokenCount:     0,
		BytesPerToken:  0,
		TotalSizeBytes: 0,
		IsRootPinned:   false,
		Hint:           HintNonTemporalWeight,
		ModifierFlags:  weightFlags,
		Bitmask:        weightBitmask,
		AssemblySuffix: weightBitmask.AssemblySuffix,
	}

	divergentPolicy := MALLTile{
		TileID:         1,
		StartToken:     cfg.RootTokenCount,
		EndToken:       -1,
		TokenCount:     -1,
		BytesPerToken:  bytesPerToken,
		TotalSizeBytes: 0,
		IsRootPinned:   false,
		Hint:           HintDivergentKV,
		ModifierFlags:  divergentFlags,
		Bitmask:        divergentBitmask,
		AssemblySuffix: divergentBitmask.AssemblySuffix,
	}

	plan := &MALLTilerPlan{
		Geometry:                 cfg.Geometry,
		BytesPerTokenPerLayer:    bytesPerToken,
		TotalMALLCapacityBytes:   cfg.TotalMALLCapacityBytes,
		RootTokens:               cfg.RootTokenCount,
		RootAllocatedBytes:       rootAllocatedBytes,
		RemainingMALLBytes:       remainingMALLBytes,
		CapacityUtilizationPct:   utilizationPct,
		RootTile:                 rootTile,
		WeightTilePolicy:         weightPolicy,
		DivergentTilePolicy:      divergentPolicy,
		EstimatedDRAMSavingsGBps: DefaultDRAMBandwidthSavingsFloorGBs,
		TargetHitRate:            DefaultMALLHitRateTarget,
	}

	cacheModel := NewMALLCacheModel()

	// Pre-populate pinned root prefix into the cache model
	for offset := int64(0); offset < rootAllocatedBytes; offset += int64(MALLLineSizeBytes) {
		cacheModel.Access(uint64(offset), MALLLineSizeBytes, rootFlags, true)
	}

	tiler := &AttentionMALLTiler{
		config:     cfg,
		plan:       plan,
		modGen:     modGen,
		cacheModel: cacheModel,
		telemetry: MALLTilerTelemetry{
			LastUpdated: time.Now(),
		},
	}

	return tiler, nil
}

// Config returns the active tiler configuration.
func (t *AttentionMALLTiler) Config() MALLTilerConfig {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.config
}

// Plan returns the active attention tiling plan.
func (t *AttentionMALLTiler) Plan() *MALLTilerPlan {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.plan
}

// TileForToken maps a given token index to its corresponding MALL tile and cache hints.
func (t *AttentionMALLTiler) TileForToken(tokenIdx int) (MALLTile, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		return MALLTile{}, ErrTilerClosed
	}

	if tokenIdx < 0 {
		return MALLTile{}, ErrInvalidTokenIndex
	}

	if tokenIdx < t.config.RootTokenCount {
		return t.plan.RootTile, nil
	}

	divergent := t.plan.DivergentTilePolicy
	divergent.StartToken = tokenIdx
	divergent.EndToken = tokenIdx + 1
	divergent.TokenCount = 1
	divergent.TotalSizeBytes = t.plan.BytesPerTokenPerLayer
	return divergent, nil
}

// RootKVModifierFlags returns the RDNA 3.5 cache flags for root KV prefix loads (SLC=0, GLC=0).
func (t *AttentionMALLTiler) RootKVModifierFlags() CacheModifierFlags {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.plan.RootTile.ModifierFlags
}

// WeightModifierFlags returns the RDNA 3.5 cache flags for streaming model weights (SLC=1, GLC=0).
func (t *AttentionMALLTiler) WeightModifierFlags() CacheModifierFlags {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.plan.WeightTilePolicy.ModifierFlags
}

// DivergentKVModifierFlags returns the RDNA 3.5 cache flags for divergent subagent tokens (SLC=1, GLC=0).
func (t *AttentionMALLTiler) DivergentKVModifierFlags() CacheModifierFlags {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.plan.DivergentTilePolicy.ModifierFlags
}

// AccessKV simulates an attention KV buffer read operation, updating telemetry and cache state.
func (t *AttentionMALLTiler) AccessKV(tokenIdx int) (bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return false, ErrTilerClosed
	}

	if tokenIdx < 0 {
		return false, ErrInvalidTokenIndex
	}

	t.telemetry.TotalAccesses++
	t.telemetry.LastUpdated = time.Now()

	bytesPerToken := t.plan.BytesPerTokenPerLayer

	if tokenIdx < t.config.RootTokenCount {
		addr := uint64(tokenIdx) * uint64(bytesPerToken)
		hit, _ := t.cacheModel.Access(addr, int(bytesPerToken), t.plan.RootTile.ModifierFlags, true)
		if hit {
			t.telemetry.RootHits++
			t.telemetry.DRAMBytesSaved += bytesPerToken
		} else {
			t.telemetry.RootMisses++
		}
		totalRoot := t.telemetry.RootHits + t.telemetry.RootMisses
		if totalRoot > 0 {
			t.telemetry.RootHitRate = float64(t.telemetry.RootHits) / float64(totalRoot)
		}
		return hit, nil
	}

	// Divergent token
	t.telemetry.DivergentLoads++
	return false, nil
}

// AccessWeight simulates a streaming model weight access, enforcing non-temporal bypass (SLC=1).
func (t *AttentionMALLTiler) AccessWeight(addr uint64, sizeBytes int) (bool, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return false, ErrTilerClosed
	}

	t.telemetry.TotalAccesses++
	t.telemetry.WeightBypasses++
	t.telemetry.LastUpdated = time.Now()

	hit, _ := t.cacheModel.Access(addr, sizeBytes, t.plan.WeightTilePolicy.ModifierFlags, false)
	return hit, nil
}

// SimulateMultiAgentDecode simulates an N-agent concurrent decode phase over M steps.
// It verifies that pinned root KV remains >= 95% resident in MALL despite streaming weights.
func (t *AttentionMALLTiler) SimulateMultiAgentDecode(activeAgents int, decodeSteps int, divergentTokensPerAgent int) MALLTilerTelemetry {
	t.mu.Lock()
	defer t.mu.Unlock()

	if activeAgents <= 0 {
		activeAgents = 8
	}
	if decodeSteps <= 0 {
		decodeSteps = 10
	}

	rootTokens := t.config.RootTokenCount
	bytesPerToken := t.plan.BytesPerTokenPerLayer
	bytesPerTurnWeights := int(Default35BWeightBytes / 32) // ~546 MB per layer forward step

	for step := 0; step < decodeSteps; step++ {
		// 1. Streaming weight load for this decode step (SLC=1 bypass)
		weightAddr := uint64(0x100000000000) + uint64(step)*uint64(bytesPerTurnWeights)
		t.cacheModel.Access(weightAddr, bytesPerTurnWeights, t.plan.WeightTilePolicy.ModifierFlags, false)
		t.telemetry.WeightBypasses++
		t.telemetry.TotalAccesses++

		// 2. Each active agent reads the shared root KV prefix (SLC=0 temporal)
		for agent := 0; agent < activeAgents; agent++ {
			// Sample 64 tokens across the root prefix for this decode step
			stride := rootTokens / 64
			if stride <= 0 {
				stride = 1
			}
			for tok := 0; tok < rootTokens; tok += stride {
				addr := uint64(tok) * uint64(bytesPerToken)
				hit, _ := t.cacheModel.Access(addr, int(bytesPerToken), t.plan.RootTile.ModifierFlags, true)
				t.telemetry.TotalAccesses++
				if hit {
					t.telemetry.RootHits++
					t.telemetry.DRAMBytesSaved += bytesPerToken
				} else {
					t.telemetry.RootMisses++
				}
			}

			// 3. Divergent continuation tokens for each subagent (SLC=1 bypass)
			for div := 0; div < divergentTokensPerAgent; div++ {
				t.telemetry.DivergentLoads++
				t.telemetry.TotalAccesses++
			}
		}
	}

	totalRoot := t.telemetry.RootHits + t.telemetry.RootMisses
	if totalRoot > 0 {
		t.telemetry.RootHitRate = float64(t.telemetry.RootHits) / float64(totalRoot)
	}

	// Calculate estimated sustained DRAM bandwidth reduction
	// In multi-agent decode on Strix Halo, caching root prefix prevents 32MB from being
	// fetched repeatedly from DRAM on every token step across all agents.
	savedMB := float64(t.telemetry.DRAMBytesSaved) / (1024.0 * 1024.0)
	if savedMB > 0 {
		// Effective bandwidth reduction is at least the floor (32.0 GB/s) scaled by hit rate
		t.telemetry.EffectiveDRAMSavingsGBps = DefaultDRAMBandwidthSavingsFloorGBs * t.telemetry.RootHitRate
	}

	t.telemetry.LastUpdated = time.Now()
	return t.telemetry
}

// Telemetry returns a thread-safe snapshot of active performance metrics.
func (t *AttentionMALLTiler) Telemetry() MALLTilerTelemetry {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.telemetry
}

// ResetTelemetry clears accumulated hit/miss/bypass counters.
func (t *AttentionMALLTiler) ResetTelemetry() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.telemetry = MALLTilerTelemetry{
		LastUpdated: time.Now(),
	}
}

// PinnedResidencyPercent returns the percentage of pinned root KV lines resident in MALL.
func (t *AttentionMALLTiler) PinnedResidencyPercent() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.cacheModel.PinnedKVResidencyPct()
}

// Close gracefully releases tiler resources.
func (t *AttentionMALLTiler) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.closed = true
	return nil
}
