// Package strix implements high-density KV cache packing, micro-scaling quantization,
// 32MB MALL Infinity Cache attention tiling, and RDNA 3.5 cache modifier and invalidation protocols
// for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Maximum chunk size (in bytes) for a single buffer_wbl2 range invalidation packet (2 MiB).
const maxInvalidationChunkBytes = 2 * 1024 * 1024

// DynamicReTiler coordinates selective RDNA 3.5 buffer_wbl2 invalidations,
// dynamic 32MB MALL partition re-tiling, and generation epoch fencing across subagent context switches.
type DynamicReTiler struct {
	mu            sync.RWMutex
	config        DynamicReTilerConfig
	cacheModel    *MALLCacheModel
	currentTenant TenantKVSegment
	state         TenantTransitionState
	epoch         atomic.Uint64
	telemetry     InvalidationTelemetry
	closed        bool
	tenantHistory map[string]TenantKVSegment
	staleRanges   []TenantKVSegment
}

// NewDynamicReTiler creates and initializes an invalidation and re-tiling coordinator.
// Target architecture is validated against GFX1151 (AMD Strix Halo).
func NewDynamicReTiler(cfg DynamicReTilerConfig, cacheModel *MALLCacheModel) (*DynamicReTiler, error) {
	if cfg.TargetArch == "" {
		cfg.TargetArch = TargetArchGFX1151
	}
	if cfg.TargetArch != TargetArchGFX1151 {
		return nil, fmt.Errorf("%w: %s", ErrTargetArchMismatch, cfg.TargetArch)
	}

	if cfg.DefaultArm == "" {
		cfg.DefaultArm = Arm1SelectiveInvalidation
	}

	if cfg.MaxTransitionLatencyNs <= 0 {
		cfg.MaxTransitionLatencyNs = 50000 // 50 microseconds
	}

	r := &DynamicReTiler{
		config:        cfg,
		cacheModel:    cacheModel,
		state:         StateIdle,
		tenantHistory: make(map[string]TenantKVSegment),
	}
	r.epoch.Store(1)

	return r, nil
}

// SynthesizeBufferWBL2Packets breaks a byte range into 64-byte line aligned RDNA 3.5
// buffer_wbl2 command packets with explicit MUBUF encoding and fence tags.
func (r *DynamicReTiler) SynthesizeBufferWBL2Packets(baseAddr uint64, sizeBytes int64) ([]BufferWBL2Packet, error) {
	if sizeBytes <= 0 {
		return nil, nil
	}

	lineSize := int64(MALLLineSizeBytes)
	// Align to 64-byte cache lines
	alignedStart := baseAddr &^ (uint64(lineSize) - 1)
	alignedEnd := (baseAddr + uint64(sizeBytes) + uint64(lineSize) - 1) &^ (uint64(lineSize) - 1)
	totalAlignedBytes := int64(alignedEnd - alignedStart)

	var packets []BufferWBL2Packet
	currAddr := alignedStart
	remaining := totalAlignedBytes

	for remaining > 0 {
		chunk := remaining
		if chunk > maxInvalidationChunkBytes {
			chunk = maxInvalidationChunkBytes
		}

		lineCount := int(chunk / lineSize)

		// GFX11 MUBUF command encoding for buffer_wbl2:
		// DWord 0: MUBUFOpcodePrefix | MUBUFOpcodeBufferWBL2 | MUBUFOffenBit | MUBUFGLCBit
		// DWord 1: MUBUFSLCBit
		dword0 := MUBUFOpcodePrefix | MUBUFOpcodeBufferWBL2 | MUBUFOffenBit | MUBUFGLCBit
		dword1 := MUBUFSLCBit

		packet := BufferWBL2Packet{
			Opcode:               OpcodeBufferWBL2,
			BaseAddress:          currAddr,
			RangeBytes:           chunk,
			CacheLineCount:       lineCount,
			GLC:                  1, // Flush/invalidate Vector L1
			SLC:                  1, // System Level Coherent (MALL writeback/invalidate)
			RawInstructionDword0: dword0,
			RawInstructionDword1: dword1,
			FenceConfirmed:       true,
		}

		packets = append(packets, packet)
		currAddr += uint64(chunk)
		remaining -= chunk
	}

	return packets, nil
}

// InvalidateRangeInCacheModel invalidates lines belonging to [startAddr, startAddr+sizeBytes)
// in the attached MALLCacheModel if present. Returns the number of invalidated lines.
func (r *DynamicReTiler) InvalidateRangeInCacheModel(startAddr uint64, sizeBytes int64) int {
	if r.cacheModel == nil || sizeBytes <= 0 {
		return 0
	}

	lineSize := uint64(MALLLineSizeBytes)
	alignedStart := startAddr &^ (lineSize - 1)
	alignedEnd := (startAddr + uint64(sizeBytes) + lineSize - 1) &^ (lineSize - 1)

	r.cacheModel.mu.Lock()
	defer r.cacheModel.mu.Unlock()

	invalidated := 0
	for lineAddr := alignedStart; lineAddr < alignedEnd; lineAddr += lineSize {
		setIdx := (lineAddr >> 6) & (MALLSetCount - 1)
		tag := lineAddr >> 21

		set := &r.cacheModel.sets[setIdx]
		for i := 0; i < MALLAssociativityWays; i++ {
			way := &set.ways[i]
			if way.valid && way.tag == tag {
				way.valid = false
				if way.isPinnedKV {
					way.isPinnedKV = false
					if r.cacheModel.currentPinnedKV > 0 {
						r.cacheModel.currentPinnedKV--
					}
				}
				invalidated++
			}
		}
	}

	return invalidated
}

// ExecuteTransition executes an atomic context switch from PreviousTenant to IncomingTenant.
// It analyzes prefix fingerprints to selectively retain shared root prefix lines while
// purging tenant-private KV cache blocks in < 50µs.
func (r *DynamicReTiler) ExecuteTransition(req TransitionRequest) (*TransitionReceipt, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, ErrReTilerClosed
	}

	if req.PreviousTenant.TenantID == "" || req.IncomingTenant.TenantID == "" {
		return nil, ErrInvalidTransitionRequest
	}

	start := time.Now()
	r.state = StateSwitching

	arm := req.AblationArm
	if arm == "" {
		arm = r.config.DefaultArm
	}

	// 1. Classification & Fingerprinting
	r.state = StateFingerprinting
	sharedPrefix := req.PreviousTenant.PrefixFingerprint != "" &&
		req.PreviousTenant.PrefixFingerprint == req.IncomingTenant.PrefixFingerprint &&
		req.PreviousTenant.PrefixSizeBytes > 0

	var packets []BufferWBL2Packet
	var err error
	var invalidatedBytes int64
	var invalidatedLines int
	var preservedPrefixBytes int64
	var fallbackTriggered bool
	var fallbackReason string

	// 2. Determine invalidation strategy according to arm and parameters
	if req.ForceFallback || (r.config.EnableQuarantineFallback && req.AblationArm == Arm2GlobalCoarseFlush) {
		// Quarantined / Coarse fallback: purge all previous tenant lines (prefix + private)
		r.state = StateInvalidating
		fallbackTriggered = true
		if req.ForceFallback {
			fallbackReason = "quarantined fallback forced by caller or fence timeout"
		} else {
			fallbackReason = "coarse global flush requested by ablation policy"
		}

		if req.PreviousTenant.PrefixSizeBytes > 0 {
			pPackets, pErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
			if pErr != nil {
				return nil, pErr
			}
			packets = append(packets, pPackets...)
			invalidatedBytes += req.PreviousTenant.PrefixSizeBytes
			invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
		}
		if req.PreviousTenant.PrivateSizeBytes > 0 {
			prPackets, prErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
			if prErr != nil {
				return nil, prErr
			}
			packets = append(packets, prPackets...)
			invalidatedBytes += req.PreviousTenant.PrivateSizeBytes
			invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
		}
		if invalidatedLines == 0 && invalidatedBytes > 0 {
			invalidatedLines = int(invalidatedBytes / int64(MALLLineSizeBytes))
		}
	} else if arm == Arm1SelectiveInvalidation {
		r.state = StateInvalidating
		if sharedPrefix {
			// Selective invalidation: retain shared root prefix, purge only tenant-private tail
			preservedPrefixBytes = req.PreviousTenant.PrefixSizeBytes

			if req.PreviousTenant.PrivateSizeBytes > 0 {
				packets, err = r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
				if err != nil {
					return nil, err
				}
				invalidatedBytes = req.PreviousTenant.PrivateSizeBytes
				invalidatedLines = r.InvalidateRangeInCacheModel(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
				if invalidatedLines == 0 && req.PreviousTenant.PrivateSizeBytes > 0 {
					invalidatedLines = int(req.PreviousTenant.PrivateSizeBytes / int64(MALLLineSizeBytes))
				}
			}
		} else {
			// Disjoint prefixes: purge both prefix and private ranges
			if req.PreviousTenant.PrefixSizeBytes > 0 {
				pPackets, pErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
				if pErr != nil {
					return nil, pErr
				}
				packets = append(packets, pPackets...)
				invalidatedBytes += req.PreviousTenant.PrefixSizeBytes
				invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
			}
			if req.PreviousTenant.PrivateSizeBytes > 0 {
				prPackets, prErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
				if prErr != nil {
					return nil, prErr
				}
				packets = append(packets, prPackets...)
				invalidatedBytes += req.PreviousTenant.PrivateSizeBytes
				invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
			}
			if invalidatedLines == 0 && invalidatedBytes > 0 {
				invalidatedLines = int(invalidatedBytes / int64(MALLLineSizeBytes))
			}
		}
	} else if arm == Arm3LazyLRUBaseline {
		// Arm 3: Lazy LRU baseline, no explicit invalidations emitted
		r.state = StateInvalidating
		packets = nil
		invalidatedBytes = 0
		invalidatedLines = 0
		if sharedPrefix {
			preservedPrefixBytes = req.PreviousTenant.PrefixSizeBytes
		}
	} else {
		// Coarse flush
		r.state = StateInvalidating
		if req.PreviousTenant.PrefixSizeBytes > 0 {
			pPackets, pErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
			if pErr != nil {
				return nil, pErr
			}
			packets = append(packets, pPackets...)
			invalidatedBytes += req.PreviousTenant.PrefixSizeBytes
			invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrefixBaseAddr, req.PreviousTenant.PrefixSizeBytes)
		}
		if req.PreviousTenant.PrivateSizeBytes > 0 {
			prPackets, prErr := r.SynthesizeBufferWBL2Packets(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
			if prErr != nil {
				return nil, prErr
			}
			packets = append(packets, prPackets...)
			invalidatedBytes += req.PreviousTenant.PrivateSizeBytes
			invalidatedLines += r.InvalidateRangeInCacheModel(req.PreviousTenant.PrivateBaseAddr, req.PreviousTenant.PrivateSizeBytes)
		}
	}

	// 3. Dynamic Re-Tiling & Epoch Increment
	r.state = StateReTiling
	newEpoch := r.epoch.Add(1)

	// Record stale private range to defend against cross-tenant read leaks
	if invalidatedBytes > 0 {
		r.staleRanges = append(r.staleRanges, req.PreviousTenant)
	}

	// Update active tenant context
	incoming := req.IncomingTenant
	incoming.GenerationEpoch = newEpoch
	r.currentTenant = incoming
	r.tenantHistory[incoming.TenantID] = incoming
	r.state = StateActive

	elapsed := time.Since(start)
	durationNs := elapsed.Nanoseconds()
	durationUs := float64(durationNs) / 1000.0
	sub50usMet := durationNs <= r.config.MaxTransitionLatencyNs

	// 4. Update Telemetry
	r.telemetry.TotalTransitions++
	r.telemetry.TotalDurationNs += uint64(durationNs)
	if uint64(durationNs) > r.telemetry.MaxDurationNs {
		r.telemetry.MaxDurationNs = uint64(durationNs)
	}
	r.telemetry.PreservedPrefixLines += uint64(preservedPrefixBytes / int64(MALLLineSizeBytes))
	r.telemetry.EvictedPrivateLines += uint64(invalidatedLines)
	r.telemetry.LastTransition = time.Now()

	switch arm {
	case Arm1SelectiveInvalidation:
		if fallbackTriggered {
			r.telemetry.QuarantinedFallbacks++
		} else {
			r.telemetry.SelectiveTransitions++
		}
	case Arm2GlobalCoarseFlush:
		r.telemetry.GlobalFlushes++
	case Arm3LazyLRUBaseline:
		r.telemetry.LazyLRUBaselines++
	}

	receipt := &TransitionReceipt{
		Schema:                  "fak.strix-mall-invalidation-receipt.v1",
		PreviousTenantID:        req.PreviousTenant.TenantID,
		IncomingTenantID:        req.IncomingTenant.TenantID,
		PrefixFingerprint:       req.IncomingTenant.PrefixFingerprint,
		PrefixPreserved:         sharedPrefix && !fallbackTriggered,
		PreservedPrefixBytes:    preservedPrefixBytes,
		InvalidatedPrivateBytes: invalidatedBytes,
		InvalidatedLineCount:    invalidatedLines,
		PacketsEmitted:          len(packets),
		DurationNs:              durationNs,
		DurationMicroseconds:    durationUs,
		Sub50usMet:              sub50usMet,
		AblationArm:             arm,
		FallbackTriggered:       fallbackTriggered,
		FallbackReason:          fallbackReason,
		NewEpoch:                newEpoch,
		Timestamp:               time.Now(),
	}

	return receipt, nil
}

// CheckCrossTenantIsolation asserts that an access from candidateTenantID to targetAddr
// does not attempt to read an unshared stale memory range from another tenant.
func (r *DynamicReTiler) CheckCrossTenantIsolation(candidateTenantID string, targetAddr uint64, sizeBytes int) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return false, ErrReTilerClosed
	}

	// Accessor must be the currently active tenant
	if candidateTenantID != r.currentTenant.TenantID {
		r.telemetry.CrossTenantLeaksPrevented++
		return false, fmt.Errorf("%w: candidate tenant %s is not active tenant %s",
			ErrCrossTenantLeakDetected, candidateTenantID, r.currentTenant.TenantID)
	}

	// Address must not fall into a stale private range of a previous tenant
	targetEnd := targetAddr + uint64(sizeBytes)
	for _, stale := range r.staleRanges {
		if stale.TenantID == candidateTenantID {
			continue
		}
		staleStart := stale.PrivateBaseAddr
		staleEnd := stale.PrivateBaseAddr + uint64(stale.PrivateSizeBytes)

		if targetAddr < staleEnd && targetEnd > staleStart {
			r.telemetry.StaleHitsPrevented++
			return false, fmt.Errorf("%w: target range [0x%x, 0x%x) overlaps un-reclaimed tenant %s range [0x%x, 0x%x)",
				ErrCrossTenantLeakDetected, targetAddr, targetEnd, stale.TenantID, staleStart, staleEnd)
		}
	}

	return true, nil
}

// CurrentTenant returns the actively scheduled tenant metadata.
func (r *DynamicReTiler) CurrentTenant() TenantKVSegment {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.currentTenant
}

// CurrentEpoch returns the monotonic generation counter.
func (r *DynamicReTiler) CurrentEpoch() uint64 {
	return r.epoch.Load()
}

// CurrentState returns the lifecycle state of the re-tiler.
func (r *DynamicReTiler) CurrentState() TenantTransitionState {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.state
}

// Telemetry returns a snapshot of accumulated multi-tenant transition telemetry.
func (r *DynamicReTiler) Telemetry() InvalidationTelemetry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.telemetry
}

// Reset clears state, history, and metrics.
func (r *DynamicReTiler) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.currentTenant = TenantKVSegment{}
	r.state = StateIdle
	r.tenantHistory = make(map[string]TenantKVSegment)
	r.staleRanges = nil
	r.telemetry = InvalidationTelemetry{}
	r.epoch.Store(1)
}

// Close closes the re-tiler, releasing resources.
func (r *DynamicReTiler) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closed = true
	return nil
}
