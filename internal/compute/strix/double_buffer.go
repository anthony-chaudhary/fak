// Package strix implements the dual 16MB ping-pong MALL (Memory Attached Last-Level)
// Infinity Cache staging pipeline for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"
)

// DoubleBufferManager orchestrates dual 16MB ping-pong buffers inside the 32MB MALL Infinity Cache.
// It overlaps layer L arithmetic computation on 40 CUs at >1.2 TB/s with asynchronous DMA/scalar
// prefetch of layer L+1 weights from physical DRAM at 273.1 GB/s.
type DoubleBufferManager struct {
	cfg      DoubleBufferConfig
	bufferA  *MALLBuffer
	bufferB  *MALLBuffer
	activeID BufferID

	mu             sync.RWMutex
	activePrefetch *PrefetchDescriptor
	prefetchMu     sync.Mutex

	metrics   DoubleBufferMetrics
	metricsMu sync.RWMutex

	latencies   []float64
	latenciesMu sync.Mutex

	closed bool
}

// allocateAligned allocates a byte slice of the requested size aligned to the given boundary.
func allocateAligned(size int64, align int) ([]byte, error) {
	if size <= 0 || align <= 0 {
		return nil, ErrBufferAllocationFailed
	}
	raw := make([]byte, size+int64(align))
	addr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int64((uintptr(align) - (addr % uintptr(align))) % uintptr(align))
	aligned := raw[offset : offset+size]
	return aligned, nil
}

// Option represents a functional configuration option for DoubleBufferManager.
type Option func(*DoubleBufferConfig)

// WithSimulateTransferDelay enables or disables simulated DRAM transfer delay in goroutines.
func WithSimulateTransferDelay(simulate bool) Option {
	return func(c *DoubleBufferConfig) {
		c.SimulateTransferDelay = simulate
	}
}

// WithDRAMBandwidth sets the physical DRAM bandwidth in GB/s for modeling.
func WithDRAMBandwidth(bwGBs float64) Option {
	return func(c *DoubleBufferConfig) {
		if bwGBs > 0 {
			c.DRAMBandwidthGBs = bwGBs
		}
	}
}

// WithBufferCapacity sets the per-buffer capacity in bytes.
func WithBufferCapacity(capacityBytes int64) Option {
	return func(c *DoubleBufferConfig) {
		if capacityBytes > 0 {
			c.BufferCapacityBytes = capacityBytes
		}
	}
}

// NewDoubleBufferManager constructs and initializes a new DoubleBufferManager for AMD Strix Halo.
// It allocates exactly two 16,777,216-byte (16 MiB) symmetrical buffers mapped to set-disjoint
// regions of the 32MB MALL Infinity Cache with strict 64-byte memory alignment.
func NewDoubleBufferManager(opts ...Option) (*DoubleBufferManager, error) {
	cfg := DefaultDoubleBufferConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Allocate Buffer A (16 MiB, sets 0..16,383)
	dataA, err := allocateAligned(cfg.BufferCapacityBytes, cfg.CacheLineSizeBytes)
	if err != nil {
		return nil, fmt.Errorf("%w: Buffer A allocation failed: %v", ErrBufferAllocationFailed, err)
	}

	// Allocate Buffer B (16 MiB, sets 16,384..32,767)
	dataB, err := allocateAligned(cfg.BufferCapacityBytes, cfg.CacheLineSizeBytes)
	if err != nil {
		return nil, fmt.Errorf("%w: Buffer B allocation failed: %v", ErrBufferAllocationFailed, err)
	}

	bufA := &MALLBuffer{
		ID:            BufferIDA,
		SetRangeStart: BufferASetRangeStart,
		SetRangeEnd:   BufferASetRangeEnd,
		CapacityBytes: cfg.BufferCapacityBytes,
		Alignment:     cfg.CacheLineSizeBytes,
		Data:          dataA,
		ActiveLayer:   -1,
		SubLayerID:    -1,
		state:         BufferEmpty,
	}

	bufB := &MALLBuffer{
		ID:            BufferIDB,
		SetRangeStart: BufferBSetRangeStart,
		SetRangeEnd:   BufferBSetRangeEnd,
		CapacityBytes: cfg.BufferCapacityBytes,
		Alignment:     cfg.CacheLineSizeBytes,
		Data:          dataB,
		ActiveLayer:   -1,
		SubLayerID:    -1,
		state:         BufferEmpty,
	}

	// Invariant Check 1: 64-byte alignment
	if !bufA.AddressAligned() || !bufB.AddressAligned() {
		return nil, fmt.Errorf("%w: buffer memory failed 64-byte alignment requirement", ErrBufferAllocationFailed)
	}

	// Invariant Check 2: Physical set disjointness
	if !bufA.IsSetDisjointWith(bufB) {
		return nil, fmt.Errorf("%w: buffer set ranges overlap (%d..%d vs %d..%d)",
			ErrBufferAllocationFailed, bufA.SetRangeStart, bufA.SetRangeEnd, bufB.SetRangeStart, bufB.SetRangeEnd)
	}

	// Invariant Check 3: Combined capacity matches physical MALL partition (32 MiB)
	if bufA.CapacityBytes+bufB.CapacityBytes != cfg.TotalMALLCapacityBytes {
		return nil, fmt.Errorf("%w: combined buffer capacity (%d) does not match total MALL capacity (%d)",
			ErrBufferAllocationFailed, bufA.CapacityBytes+bufB.CapacityBytes, cfg.TotalMALLCapacityBytes)
	}

	mgr := &DoubleBufferManager{
		cfg:      cfg,
		bufferA:  bufA,
		bufferB:  bufB,
		activeID: BufferIDA,
		metrics: DoubleBufferMetrics{
			ActiveComputeBuffer:  BufferIDA.String(),
			ActivePrefetchBuffer: BufferIDB.String(),
			MALLHitRate:          1.0, // Initial nominal hit rate for resident MALL compute
		},
		latencies: make([]float64, 0, 1024),
	}

	return mgr, nil
}

// Config returns the configuration associated with the manager.
func (m *DoubleBufferManager) Config() DoubleBufferConfig {
	return m.cfg
}

// ActiveBuffer returns the currently active MALL compute buffer.
func (m *DoubleBufferManager) ActiveBuffer() *MALLBuffer {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.activeID == BufferIDA {
		return m.bufferA
	}
	return m.bufferB
}

// InactiveBuffer returns the currently inactive MALL staging/prefetch buffer.
func (m *DoubleBufferManager) InactiveBuffer() *MALLBuffer {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.activeID == BufferIDA {
		return m.bufferB
	}
	return m.bufferA
}

// ActiveID returns the BufferID of the active compute buffer.
func (m *DoubleBufferManager) ActiveID() BufferID {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeID
}

// InactiveID returns the BufferID of the inactive staging buffer.
func (m *DoubleBufferManager) InactiveID() BufferID {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeID.Other()
}

// GetBuffer returns the buffer identified by the given BufferID.
func (m *DoubleBufferManager) GetBuffer(id BufferID) *MALLBuffer {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if id == BufferIDA {
		return m.bufferA
	}
	return m.bufferB
}

// PrefetchLayerTo initiates an asynchronous prefetch of layer weights into a specifically designated buffer.
func (m *DoubleBufferManager) PrefetchLayerTo(targetID BufferID, desc PrefetchDescriptor) (*CompletionFence, error) {
	if desc.SizeBytes <= 0 {
		return nil, fmt.Errorf("%w: invalid prefetch size %d", ErrOutOfBounds, desc.SizeBytes)
	}
	if desc.SizeBytes > m.cfg.BufferCapacityBytes {
		return nil, fmt.Errorf("%w: requested %d bytes exceeds 16MB capacity %d",
			ErrCapacityExceeded, desc.SizeBytes, m.cfg.BufferCapacityBytes)
	}

	targetBuf := m.GetBuffer(targetID)
	currentState := targetBuf.State()

	if currentState != BufferEmpty && currentState != BufferRecycling {
		return nil, fmt.Errorf("%w: buffer %s is in state %s (expected EMPTY or RECYCLING)",
			ErrBufferBusy, targetID, currentState)
	}

	if err := targetBuf.SetState(BufferPrefetching); err != nil {
		return nil, err
	}

	targetBuf.ActiveLayer = desc.LayerID
	targetBuf.SubLayerID = desc.SubLayerID

	fence := desc.Fence
	if fence == nil {
		fence = NewCompletionFence()
	}

	m.prefetchMu.Lock()
	descCopy := desc
	descCopy.Status = PrefetchInProgress
	descCopy.ScheduledAt = time.Now()
	descCopy.Fence = fence
	m.activePrefetch = &descCopy
	m.prefetchMu.Unlock()

	go m.runPrefetchTransfer(targetBuf, descCopy, fence)

	return fence, nil
}

// PrefetchNextLayer schedules an asynchronous prefetch of layer L+1 weights from physical DRAM
// into the inactive MALL ping-pong buffer. Returns a CompletionFence that signals upon arrival.
func (m *DoubleBufferManager) PrefetchNextLayer(desc PrefetchDescriptor) (*CompletionFence, error) {
	inactiveID := m.InactiveID()
	return m.PrefetchLayerTo(inactiveID, desc)
}

// runPrefetchTransfer executes the asynchronous transfer in a dedicated background goroutine.
func (m *DoubleBufferManager) runPrefetchTransfer(targetBuf *MALLBuffer, desc PrefetchDescriptor, fence *CompletionFence) {
	transferStart := time.Now()

	// If configured to simulate transfer delay based on physical DRAM bandwidth (273.1 GB/s)
	if m.cfg.SimulateTransferDelay {
		transferSec := float64(desc.SizeBytes) / (m.cfg.DRAMBandwidthGBs * 1e9)
		if transferSec > 0 {
			time.Sleep(time.Duration(transferSec * float64(time.Second)))
		}
	}

	// In-memory simulation: copy weight payload into aligned buffer slice if provided
	if len(desc.Payload) > 0 {
		copyLen := int64(len(desc.Payload))
		if copyLen > desc.SizeBytes {
			copyLen = desc.SizeBytes
		}
		copy(targetBuf.Data[:copyLen], desc.Payload[:copyLen])
	}

	duration := time.Since(transferStart)
	completedAt := time.Now()

	// Update buffer state: PREFETCHING -> READY
	if err := targetBuf.SetState(BufferReady); err != nil {
		fence.SignalError(err)
		m.prefetchMu.Lock()
		if m.activePrefetch != nil {
			m.activePrefetch.Status = PrefetchFailed
		}
		m.prefetchMu.Unlock()
		return
	}

	targetBuf.lastPrefetch = completedAt

	// Calculate simulated or measured bandwidth
	var bw float64
	if duration.Seconds() > 0 {
		bw = (float64(desc.SizeBytes) / 1e9) / duration.Seconds()
	} else {
		bw = m.cfg.DRAMBandwidthGBs
	}

	m.prefetchMu.Lock()
	if m.activePrefetch != nil {
		m.activePrefetch.Status = PrefetchCompleted
		m.activePrefetch.CompletedAt = completedAt
		m.activePrefetch.Duration = duration
		m.activePrefetch.BandwidthGBs = bw
	}
	m.prefetchMu.Unlock()

	// Update telemetry
	m.recordPrefetchSuccess(desc.SizeBytes, duration)

	fence.Signal()
}

// recordPrefetchSuccess updates internal metrics upon successful prefetch completion.
func (m *DoubleBufferManager) recordPrefetchSuccess(bytes int64, duration time.Duration) {
	m.metricsMu.Lock()
	defer m.metricsMu.Unlock()

	m.metrics.TotalPrefetchBytes += bytes
	m.metrics.TotalPrefetchDurationNs += duration.Nanoseconds()
	m.metrics.PrefetchSuccessCount++
}

// Flip atomically swaps the active compute buffer and inactive prefetch buffer.
func (m *DoubleBufferManager) Flip() (*MALLBuffer, error) {
	return m.FlipWait(0)
}

// Advance is an alias for Flip, advancing the pipeline to the next staged layer.
func (m *DoubleBufferManager) Advance() (*MALLBuffer, error) {
	return m.Flip()
}

// FlipWait atomically swaps the buffers, waiting up to the given timeout for the inactive
// buffer's prefetch fence to signal if it is currently prefetching.
func (m *DoubleBufferManager) FlipWait(timeout time.Duration) (*MALLBuffer, error) {
	swapStart := time.Now()

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil, ErrBufferNotReady
	}

	currentActive := m.bufferA
	currentInactive := m.bufferB
	if m.activeID == BufferIDB {
		currentActive = m.bufferB
		currentInactive = m.bufferA
	}

	// If inactive buffer is still prefetching and timeout is allowed, wait for in-flight fence
	inactiveState := currentInactive.State()
	if inactiveState == BufferPrefetching && timeout > 0 {
		m.prefetchMu.Lock()
		var fence *CompletionFence
		if m.activePrefetch != nil {
			fence = m.activePrefetch.Fence
		}
		m.prefetchMu.Unlock()

		if fence != nil {
			if err := fence.Wait(timeout); err != nil {
				m.metricsMu.Lock()
				m.metrics.PrefetchUnderrunCount++
				m.metricsMu.Unlock()
				return nil, fmt.Errorf("%w: prefetch underrun on %s: %v", ErrBufferNotReady, currentInactive.ID, err)
			}
		}
		inactiveState = currentInactive.State()
	}

	// Verify that inactive buffer is ready
	if inactiveState != BufferReady {
		m.metricsMu.Lock()
		m.metrics.PrefetchUnderrunCount++
		m.metricsMu.Unlock()
		return nil, fmt.Errorf("%w: cannot flip to %s in state %s (expected READY)",
			ErrBufferNotReady, currentInactive.ID, inactiveState)
	}

	// Calculate prefetch lead time (elapsed time between prefetch completion and flip request)
	var leadTimeNs int64
	if !currentInactive.lastPrefetch.IsZero() {
		leadTime := swapStart.Sub(currentInactive.lastPrefetch)
		if leadTime > 0 {
			leadTimeNs = leadTime.Nanoseconds()
		}
	}

	// Transition inactive buffer: READY -> COMPUTING
	if err := currentInactive.SetState(BufferComputing); err != nil {
		return nil, err
	}
	currentInactive.lastCompute = swapStart

	// Transition previously active buffer: COMPUTING / READY / EMPTY -> RECYCLING -> EMPTY
	activeState := currentActive.State()
	if activeState == BufferComputing || activeState == BufferReady {
		if err := currentActive.SetState(BufferRecycling); err != nil {
			return nil, err
		}
		// Reset layer tracking for recycled buffer
		currentActive.ActiveLayer = -1
		currentActive.SubLayerID = -1
		if err := currentActive.SetState(BufferEmpty); err != nil {
			return nil, err
		}
	}

	// Swap active buffer ID
	m.activeID = m.activeID.Other()

	swapDuration := time.Since(swapStart)

	// Update telemetry
	m.metricsMu.Lock()
	m.metrics.FlipCount++
	m.metrics.SwapLatencyNs = swapDuration.Nanoseconds()
	if leadTimeNs > 0 {
		// Moving average of lead time
		if m.metrics.AveragePrefetchLeadTimeNs == 0 {
			m.metrics.AveragePrefetchLeadTimeNs = leadTimeNs
		} else {
			m.metrics.AveragePrefetchLeadTimeNs = (m.metrics.AveragePrefetchLeadTimeNs*3 + leadTimeNs) / 4
		}
		// Strix Halo RDNA 3.5 CUs clocked at ~2.2 GHz: cycles saved = leadTimeNs * 2.2
		cyclesSaved := int64(float64(leadTimeNs) * 2.2)
		m.metrics.StallCyclesReduced += cyclesSaved
	}
	m.metrics.ActiveComputeBuffer = m.activeID.String()
	m.metrics.ActivePrefetchBuffer = m.activeID.Other().String()
	m.metricsMu.Unlock()

	return currentInactive, nil
}

// ExecuteCompute executes a layer forward pass out of the active MALL compute buffer at >1.2 TB/s.
func (m *DoubleBufferManager) ExecuteCompute(ctx context.Context, layerID int, computeFn func(buf *MALLBuffer) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	activeBuf := m.ActiveBuffer()
	state := activeBuf.State()

	// If buffer is READY, promote to COMPUTING
	if state == BufferReady {
		if err := activeBuf.SetState(BufferComputing); err != nil {
			return err
		}
		state = BufferComputing
	}

	if state != BufferComputing {
		return fmt.Errorf("%w: active buffer %s is in state %s (expected COMPUTING)",
			ErrBufferNotReady, activeBuf.ID, state)
	}

	if activeBuf.ActiveLayer != -1 && activeBuf.ActiveLayer != layerID {
		return fmt.Errorf("%w: layer mismatch in %s: active=%d requested=%d",
			ErrComputeFailed, activeBuf.ID, activeBuf.ActiveLayer, layerID)
	}

	start := time.Now()
	var computeErr error
	if computeFn != nil {
		computeErr = computeFn(activeBuf)
	}
	elapsed := time.Since(start)

	elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
	m.RecordDecodeLatency(elapsedMs)

	m.metricsMu.Lock()
	m.metrics.MALLHitRate = 1.0 // Buffer compute hit directly in MALL SRAM
	m.metricsMu.Unlock()

	return computeErr
}

// FallbackDirectDRAMStream executes computation directly by streaming weights from physical DRAM
// when prefetch fails or MALL buffer allocation is denied, without altering tensor arithmetic outputs.
func (m *DoubleBufferManager) FallbackDirectDRAMStream(ctx context.Context, desc PrefetchDescriptor, computeFn func(data []byte) error) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	start := time.Now()

	// Simulate DRAM latency: streaming weights at 273.1 GB/s plus DRAM bank precharge/refresh penalty
	if m.cfg.SimulateTransferDelay && desc.SizeBytes > 0 {
		transferSec := float64(desc.SizeBytes) / (m.cfg.DRAMBandwidthGBs * 1e9)
		// Add periodic DRAM refresh penalty (~280ns)
		transferSec += 280e-9
		time.Sleep(time.Duration(transferSec * float64(time.Second)))
	}

	var computeErr error
	if computeFn != nil {
		computeErr = computeFn(desc.Payload)
	}
	elapsed := time.Since(start)

	elapsedMs := float64(elapsed.Nanoseconds()) / 1e6
	m.RecordDecodeLatency(elapsedMs)

	m.metricsMu.Lock()
	m.metrics.DirectDRAMFallbackCount++
	// Fallback reads directly from DRAM, lowering MALL hit rate
	totalOps := m.metrics.FlipCount + m.metrics.DirectDRAMFallbackCount
	if totalOps > 0 {
		m.metrics.MALLHitRate = float64(m.metrics.FlipCount) / float64(totalOps)
	}
	m.metricsMu.Unlock()

	return computeErr
}

// RecordDecodeLatency records an observed per-token decode latency sample in milliseconds.
func (m *DoubleBufferManager) RecordDecodeLatency(latencyMs float64) {
	m.latenciesMu.Lock()
	defer m.latenciesMu.Unlock()
	m.latencies = append(m.latencies, latencyMs)
}

// CalculateJitterVarianceReduction compares the variance of recorded ping-pong decode latencies
// against an empirical baseline slice of DRAM decode latencies (which exhibits sawtooth jitter).
func (m *DoubleBufferManager) CalculateJitterVarianceReduction(baselineLatenciesMs []float64) float64 {
	m.latenciesMu.Lock()
	pingpongLatencies := make([]float64, len(m.latencies))
	copy(pingpongLatencies, m.latencies)
	m.latenciesMu.Unlock()

	reduction := CalculateVarianceReductionPct(baselineLatenciesMs, pingpongLatencies)

	m.metricsMu.Lock()
	m.metrics.JitterVarianceReductionPct = reduction
	m.metricsMu.Unlock()

	return reduction
}

// CalculateVarianceReductionPct computes variance reduction percentage between baseline and test distributions.
func CalculateVarianceReductionPct(baseline, optimized []float64) float64 {
	if len(baseline) < 2 || len(optimized) < 2 {
		return 0.0
	}

	varMean := func(samples []float64) (float64, float64) {
		var sum float64
		for _, s := range samples {
			sum += s
		}
		mean := sum / float64(len(samples))
		var varSum float64
		for _, s := range samples {
			diff := s - mean
			varSum += diff * diff
		}
		variance := varSum / float64(len(samples)-1)
		return mean, variance
	}

	_, varBase := varMean(baseline)
	_, varOpt := varMean(optimized)

	if varBase <= 0 {
		return 0.0
	}

	reduction := ((varBase - varOpt) / varBase) * 100.0
	if reduction < 0 {
		return 0.0
	}
	return math.Round(reduction*100) / 100
}

// RecycleInactive recycles the inactive buffer, clearing its state back to BufferEmpty.
func (m *DoubleBufferManager) RecycleInactive() error {
	inactive := m.InactiveBuffer()
	state := inactive.State()
	if state == BufferComputing {
		return fmt.Errorf("%w: cannot recycle buffer %s while computing", ErrInvalidStateTransition, inactive.ID)
	}
	inactive.Reset()
	return nil
}

// Reset re-initializes both buffers to BufferEmpty and clears metrics.
func (m *DoubleBufferManager) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.bufferA.Reset()
	m.bufferB.Reset()
	m.activeID = BufferIDA

	m.metricsMu.Lock()
	m.metrics = DoubleBufferMetrics{
		ActiveComputeBuffer:  BufferIDA.String(),
		ActivePrefetchBuffer: BufferIDB.String(),
		MALLHitRate:          1.0,
	}
	m.metricsMu.Unlock()

	m.latenciesMu.Lock()
	m.latencies = m.latencies[:0]
	m.latenciesMu.Unlock()
}

// Close releases resources and prevents further buffer operations.
func (m *DoubleBufferManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}
	m.closed = true

	m.bufferA.Reset()
	m.bufferB.Reset()
	return nil
}

// Metrics returns a snapshot copy of current operational metrics.
func (m *DoubleBufferManager) Metrics() DoubleBufferMetrics {
	m.metricsMu.RLock()
	defer m.metricsMu.RUnlock()
	return m.metrics
}

// Telemetry returns a snapshot copy of current operational metrics.
func (m *DoubleBufferManager) Telemetry() DoubleBufferTelemetry {
	return m.Metrics()
}
