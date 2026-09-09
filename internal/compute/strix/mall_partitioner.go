// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// MALLPartitioner calculates attention head working set geometry and enforces strict
// byte-level capacity partitioning across the 32MB MALL Infinity Cache.
type MALLPartitioner struct {
	mu        sync.RWMutex
	config    MALLPartitionConfig
	plan      *MALLPartitionPlan
	fallbacks atomic.Int64
}

// NewMALLPartitioner initializes an attention geometry calculator and 32MB MALL capacity partitioner.
// If TotalMALLCapacityBytes is not specified, it defaults to the physical 32MB MALL size (33,554,432 bytes).
func NewMALLPartitioner(cfg MALLPartitionConfig) (*MALLPartitioner, error) {
	if cfg.TotalMALLCapacityBytes <= 0 {
		cfg.TotalMALLCapacityBytes = MALLSizeBytes
	}

	if cfg.Geometry.BytesPerElement <= 0 {
		cfg.Geometry.BytesPerElement = FP16BytesPerElement
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	p := &MALLPartitioner{
		config: cfg,
	}

	var plan *MALLPartitionPlan
	var err error

	switch cfg.Arm {
	case ArmStatic8kPinnedRoot:
		plan, err = p.configureArmStatic8kPinned(cfg)
	case ArmDynamic4kRootDraft:
		plan, err = p.configureArmDynamic4kRootDraft(cfg)
	case ArmUnpartitionedBaseline:
		plan, err = p.configureArmUnpartitionedBaseline(cfg)
	default:
		plan, err = p.generatePartitionPlan(cfg)
	}

	p.plan = plan
	return p, err
}

// ComputeGeometry calculates and verifies the byte stride per token for a given attention geometry.
// Returns the single-layer KV footprint in bytes (e.g. 4,096 bytes for standard GQA 8 KV heads, dim 128, FP16).
func (p *MALLPartitioner) ComputeGeometry(geo AttentionGeometry) (int64, error) {
	if err := geo.Validate(); err != nil {
		return 0, err
	}
	return geo.BytesPerTokenPerLayer(), nil
}

// ComputeTotalWorkingSet calculates the full KV cache memory footprint in bytes for a sequence of tokens.
func (p *MALLPartitioner) ComputeTotalWorkingSet(geo AttentionGeometry, tokens int, layerTiled bool) (int64, error) {
	if err := geo.Validate(); err != nil {
		return 0, err
	}
	if tokens < 0 {
		return 0, ErrInvalidPartitionConfig
	}

	var bytesPerToken int64
	if layerTiled {
		bytesPerToken = geo.BytesPerTokenPerLayer()
	} else {
		bytesPerToken = geo.TotalBytesPerToken()
	}

	return int64(tokens) * bytesPerToken, nil
}

// Partition evaluates the requested partition configuration, enforces the strict <= 33,554,432 bytes
// capacity invariant, and returns a deterministic partition plan.
// If the requested allocation exceeds the 32MB physical MALL limit, it returns the fallback plan
// (with FallbackToDRAM=true and IsPartitioned=false) alongside ErrMALLPartitionOverflow.
func (p *MALLPartitioner) Partition(cfg MALLPartitionConfig) (*MALLPartitionPlan, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if cfg.TotalMALLCapacityBytes <= 0 {
		cfg.TotalMALLCapacityBytes = MALLSizeBytes
	}
	if cfg.Geometry.BytesPerElement <= 0 {
		cfg.Geometry.BytesPerElement = FP16BytesPerElement
	}

	if err := cfg.Validate(); err != nil {
		p.fallbacks.Add(1)
		fallbackPlan := p.createFallbackPlan(cfg, fmt.Sprintf("invalid geometry or config: %v", err))
		p.plan = fallbackPlan
		return fallbackPlan, err
	}

	p.config = cfg
	plan, err := p.generatePartitionPlan(cfg)
	p.plan = plan
	return plan, err
}

// PartitionSafe performs capacity partitioning and returns a valid plan, gracefully triggering the
// quarantined fallback mechanism on overflow or invalid parameters without returning an error.
func (p *MALLPartitioner) PartitionSafe(cfg MALLPartitionConfig) *MALLPartitionPlan {
	plan, _ := p.Partition(cfg)
	return plan
}

// ConfigureArm switches the active partitioner to a designated ablation arm and re-partitions.
func (p *MALLPartitioner) ConfigureArm(arm PartitionArm) (*MALLPartitionPlan, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	cfg := p.config
	cfg.Arm = arm

	var plan *MALLPartitionPlan
	var err error

	switch arm {
	case ArmStatic8kPinnedRoot:
		plan, err = p.configureArmStatic8kPinned(cfg)
	case ArmDynamic4kRootDraft:
		plan, err = p.configureArmDynamic4kRootDraft(cfg)
	case ArmUnpartitionedBaseline:
		plan, err = p.configureArmUnpartitionedBaseline(cfg)
	default:
		plan, err = p.generatePartitionPlan(cfg)
	}

	p.config = cfg
	p.plan = plan
	return plan, err
}

// configureArmStatic8kPinned configures Arm 1 (Static 8k Pinned Root):
// Allocates full 32MB (or 32MB minus reserve) to 8,192-token shared root prompt prefix.
func (p *MALLPartitioner) configureArmStatic8kPinned(cfg MALLPartitionConfig) (*MALLPartitionPlan, error) {
	bytesPerToken := cfg.EffectiveBytesPerToken()
	maxTokens := int(cfg.TotalMALLCapacityBytes / bytesPerToken)

	// Target 8,192 tokens or maximum capacity tokens
	rootTokens := 8192
	if rootTokens > maxTokens {
		rootTokens = maxTokens
	}

	cfg.RootPrefixTokens = rootTokens
	cfg.DraftTokens = 0
	if cfg.ReserveBytes < 0 {
		cfg.ReserveBytes = 0
	}

	// If reserve was specified, adjust root tokens to fit strictly within 32MB
	availableForRoot := cfg.TotalMALLCapacityBytes - cfg.ReserveBytes
	if availableForRoot > 0 {
		fittedTokens := int(availableForRoot / bytesPerToken)
		if fittedTokens < cfg.RootPrefixTokens {
			cfg.RootPrefixTokens = fittedTokens
		}
	}

	return p.generatePartitionPlan(cfg)
}

// configureArmDynamic4kRootDraft configures Arm 2 (Dynamic 4k Root + Draft):
// Allocates 16MB to 4,096-token root prefix, 14MB to active speculative draft trees, and 2MB to safety reserve.
func (p *MALLPartitioner) configureArmDynamic4kRootDraft(cfg MALLPartitionConfig) (*MALLPartitionPlan, error) {
	bytesPerToken := cfg.EffectiveBytesPerToken()

	// 16MB root = 16 * 1024 * 1024 = 16,777,216 bytes (4,096 tokens at 4KB/token)
	targetRootBytes := int64(16 * 1024 * 1024)
	cfg.RootPrefixTokens = int(targetRootBytes / bytesPerToken)

	// 14MB draft = 14 * 1024 * 1024 = 14,680,064 bytes (3,584 tokens at 4KB/token)
	targetDraftBytes := int64(14 * 1024 * 1024)
	cfg.DraftTokens = int(targetDraftBytes / bytesPerToken)

	// 2MB reserve = 2 * 1024 * 1024 = 2,097,152 bytes
	cfg.ReserveBytes = int64(2 * 1024 * 1024)

	return p.generatePartitionPlan(cfg)
}

// configureArmUnpartitionedBaseline configures Arm 3 (Unpartitioned Baseline):
// Ad-hoc unmanaged allocation directing the runtime to stream KV blocks from DRAM without asserting pin hints.
func (p *MALLPartitioner) configureArmUnpartitionedBaseline(cfg MALLPartitionConfig) (*MALLPartitionPlan, error) {
	p.fallbacks.Add(1)
	plan := p.createFallbackPlan(cfg, "Ablation Arm 3: unpartitioned baseline active, unpinned DRAM streaming")
	return plan, nil
}

// generatePartitionPlan executes the strict capacity calculation and produces the MALLPartitionPlan.
func (p *MALLPartitioner) generatePartitionPlan(cfg MALLPartitionConfig) (*MALLPartitionPlan, error) {
	bytesPerToken := cfg.EffectiveBytesPerToken()
	if bytesPerToken <= 0 {
		p.fallbacks.Add(1)
		plan := p.createFallbackPlan(cfg, "invalid geometry: zero bytes per token")
		return plan, ErrInvalidGeometry
	}

	rootBytes := int64(cfg.RootPrefixTokens) * bytesPerToken
	draftBytes := int64(cfg.DraftTokens) * bytesPerToken
	reserveBytes := cfg.ReserveBytes

	totalAllocated := rootBytes + draftBytes + reserveBytes
	capacity := cfg.TotalMALLCapacityBytes

	// Enforce strict <= 33,554,432 bytes invariant across Root Prefix, Active Draft, and Reserve segments.
	if totalAllocated > capacity {
		p.fallbacks.Add(1)
		statusMsg := fmt.Sprintf("MALL partition overflow: requested %d bytes (Root=%d, Draft=%d, Reserve=%d) exceeds capacity %d bytes by %d bytes",
			totalAllocated, rootBytes, draftBytes, reserveBytes, capacity, totalAllocated-capacity)

		fallbackPlan := p.createFallbackPlan(cfg, statusMsg)
		fallbackPlan.TotalAllocatedBytes = totalAllocated
		fallbackPlan.RemainingHeadroomBytes = capacity - totalAllocated // negative headroom
		return fallbackPlan, ErrMALLPartitionOverflow
	}

	headroomBytes := capacity - totalAllocated

	// Compute segment byte offsets
	rootOffset := int64(0)
	draftOffset := rootOffset + rootBytes
	reserveOffset := draftOffset + draftBytes
	headroomOffset := reserveOffset + reserveBytes

	rootSeg := PartitionSegment{
		Name:        "RootPrefix",
		SizeBytes:   rootBytes,
		OffsetBytes: rootOffset,
		Tokens:      cfg.RootPrefixTokens,
		CachePolicy: "TEMPORAL_PINNED",
	}

	draftSeg := PartitionSegment{
		Name:        "ActiveDraft",
		SizeBytes:   draftBytes,
		OffsetBytes: draftOffset,
		Tokens:      cfg.DraftTokens,
		CachePolicy: "TRANSIENT_PINNED",
	}

	reserveSeg := PartitionSegment{
		Name:        "Reserve",
		SizeBytes:   reserveBytes,
		OffsetBytes: reserveOffset,
		Tokens:      0,
		CachePolicy: "RESERVE",
	}

	headroomSeg := PartitionSegment{
		Name:        "Headroom",
		SizeBytes:   headroomBytes,
		OffsetBytes: headroomOffset,
		Tokens:      int(headroomBytes / bytesPerToken),
		CachePolicy: "UNALLOCATED",
	}

	metrics := PartitionMetrics{
		TotalCapacityBytes:     capacity,
		TotalAllocatedBytes:    totalAllocated,
		RemainingHeadroomBytes: headroomBytes,
		MALLUtilizationRatio:   float64(totalAllocated) / float64(capacity),
		RootPrefixUtilization:  float64(rootBytes) / float64(capacity),
		DraftUtilization:       float64(draftBytes) / float64(capacity),
		ReserveRatio:           float64(reserveBytes) / float64(capacity),
		HeadroomRatio:          float64(headroomBytes) / float64(capacity),
		BytesPerToken:          bytesPerToken,
		MaxMALLTokens:          int(capacity / bytesPerToken),
		Fallbacks:              p.fallbacks.Load(),
	}

	plan := &MALLPartitionPlan{
		Config:                 cfg,
		RootPrefix:             rootSeg,
		ActiveDraft:            draftSeg,
		Reserve:                reserveSeg,
		Headroom:               headroomSeg,
		TotalAllocatedBytes:    totalAllocated,
		RemainingHeadroomBytes: headroomBytes,
		IsPartitioned:          true,
		FallbackToDRAM:         false,
		StatusMessage:          fmt.Sprintf("MALL partitioned successfully: %d bytes allocated (%.2f%% utilized), %d bytes headroom", totalAllocated, metrics.MALLUtilizationRatio*100.0, headroomBytes),
		Metrics:                metrics,
		PolicyDirectives:       CacheHintTemporalPinned,
	}

	return plan, nil
}

// createFallbackPlan constructs a quarantined unpartitioned fallback plan instructing the runtime
// to stream KV blocks from DRAM without asserting cache pin hints.
func (p *MALLPartitioner) createFallbackPlan(cfg MALLPartitionConfig, reason string) *MALLPartitionPlan {
	bytesPerToken := cfg.EffectiveBytesPerToken()
	capacity := cfg.TotalMALLCapacityBytes
	if capacity <= 0 {
		capacity = MALLSizeBytes
	}

	var maxTokens int
	if bytesPerToken > 0 {
		maxTokens = int(capacity / bytesPerToken)
	}

	return &MALLPartitionPlan{
		Config: cfg,
		RootPrefix: PartitionSegment{
			Name:        "RootPrefix",
			SizeBytes:   0,
			OffsetBytes: 0,
			Tokens:      0,
			CachePolicy: "UNPINNED_DRAM",
		},
		ActiveDraft: PartitionSegment{
			Name:        "ActiveDraft",
			SizeBytes:   0,
			OffsetBytes: 0,
			Tokens:      0,
			CachePolicy: "UNPINNED_DRAM",
		},
		Reserve: PartitionSegment{
			Name:        "Reserve",
			SizeBytes:   0,
			OffsetBytes: 0,
			Tokens:      0,
			CachePolicy: "RESERVE",
		},
		Headroom: PartitionSegment{
			Name:        "Headroom",
			SizeBytes:   capacity,
			OffsetBytes: 0,
			Tokens:      maxTokens,
			CachePolicy: "UNPINNED_DRAM",
		},
		TotalAllocatedBytes:    0,
		RemainingHeadroomBytes: capacity,
		IsPartitioned:          false,
		FallbackToDRAM:         true,
		StatusMessage:          fmt.Sprintf("Quarantined fallback active: %s", reason),
		Metrics: PartitionMetrics{
			TotalCapacityBytes:     capacity,
			TotalAllocatedBytes:    0,
			RemainingHeadroomBytes: capacity,
			MALLUtilizationRatio:   0.0,
			RootPrefixUtilization:  0.0,
			DraftUtilization:       0.0,
			ReserveRatio:           0.0,
			HeadroomRatio:          1.0,
			BytesPerToken:          bytesPerToken,
			MaxMALLTokens:          maxTokens,
			Fallbacks:              p.fallbacks.Load(),
		},
		PolicyDirectives: CacheHintStreamingBypass,
	}
}

// Plan returns the current active partition plan.
func (p *MALLPartitioner) Plan() *MALLPartitionPlan {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.plan
}

// Metrics returns the latest partition telemetry metrics.
func (p *MALLPartitioner) Metrics() PartitionMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.plan != nil {
		m := p.plan.Metrics
		m.Fallbacks = p.fallbacks.Load()
		return m
	}
	return PartitionMetrics{
		TotalCapacityBytes: p.config.TotalMALLCapacityBytes,
		Fallbacks:          p.fallbacks.Load(),
	}
}

// IsFallbackActive returns true if the partitioner is currently directing traffic to DRAM streaming fallback.
func (p *MALLPartitioner) IsFallbackActive() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.plan == nil || p.plan.FallbackToDRAM
}

// Config returns the current partition configuration.
func (p *MALLPartitioner) Config() MALLPartitionConfig {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.config
}

// FallbackCount returns the total number of fallback transitions recorded.
func (p *MALLPartitioner) FallbackCount() int64 {
	return p.fallbacks.Load()
}
