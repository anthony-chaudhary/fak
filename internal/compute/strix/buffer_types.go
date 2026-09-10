// Package strix implements the dual 16MB ping-pong MALL (Memory Attached Last-Level)
// Infinity Cache staging pipeline for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
//
// Hardware Physics and Architectural Invariants:
//   - Physical MALL Capacity: 33,554,432 bytes (32 MiB on-die Infinity Cache).
//   - Cache Geometry: 32,768 sets, 16 ways per set, 64-byte line size (32,768 * 16 * 64 = 33,554,432 B).
//   - Symmetrical Dual Buffers: Buffer A (16 MiB) and Buffer B (16 MiB).
//   - Set-Disjoint Mapping:
//   - Buffer A: sets 0 .. 16,383
//   - Buffer B: sets 16,384 .. 32,767
//     Guarantees physical set-disjointness in cache geometry to eliminate way-conflicts
//     between concurrent scalar/DMA prefetch writes and WMMA compute reads.
//   - Physical DRAM Bandwidth: 273.1 GB/s (16-channel LPDDR5X-8533 256-bit bus).
//   - Prefetch Transfer Time: 58.58 µs for 16MB (16 MiB / 273.1 GB/s).
//   - Internal MALL Read Bandwidth: > 1.2 TB/s.
//   - Memory Alignment: 64-byte physical cache line alignment.
package strix

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// Hardware constants for AMD Strix Halo 32MB MALL on-die Infinity Cache.
const (
	// MALLTotalCapacityBytes is the exact 32 MiB physical MALL Infinity Cache capacity.
	MALLTotalCapacityBytes int64 = 32 * 1024 * 1024 // 33,554,432 bytes

	// MALLTotalSets is the total number of cache sets in the Strix Halo MALL architecture.
	MALLTotalSets = 32768

	// MALLWaysPerSet is the set-associativity way depth (16-way associative).
	MALLWaysPerSet = 16

	// MALLCacheLineSizeBytes is the physical cache line granularity (64 bytes).
	MALLCacheLineSizeBytes = 64

	// MALLBufferCapacityBytes is the symmetrical partition size for each ping-pong buffer (16 MiB).
	MALLBufferCapacityBytes int64 = 16 * 1024 * 1024 // 16,777,216 bytes

	// MALLBufferSets is the number of sets allocated to each ping-pong buffer (32,768 / 2 = 16,384).
	MALLBufferSets = 16384

	// MALLAlignmentBytes is the required physical memory alignment boundary (64 bytes).
	MALLAlignmentBytes = 64

	// TheoreticalPrefetchDurationSec is the ideal transfer time for a 16MB layer at 273.1 GB/s (~58.58 µs).
	TheoreticalPrefetchDurationSec = float64(MALLBufferCapacityBytes) / (PhysicalDRAMBandwidthGBs * 1e9)

	// BufferASetRangeStart is the starting set index for Buffer A.
	BufferASetRangeStart = 0
	// BufferASetRangeEnd is the ending set index for Buffer A (inclusive).
	BufferASetRangeEnd = 16383

	// BufferBSetRangeStart is the starting set index for Buffer B.
	BufferBSetRangeStart = 16384
	// BufferBSetRangeEnd is the ending set index for Buffer B (inclusive).
	BufferBSetRangeEnd = 32767
)

// Sentinel errors for Strix Halo MALL double-buffering.
var (
	// ErrBufferNotReady is returned when a compute operation attempts to read a buffer not in BufferReady or BufferComputing state.
	ErrBufferNotReady = errors.New("strix/cache: buffer is not ready for compute")

	// ErrInvalidStateTransition is returned when an illegal buffer state transition is requested.
	ErrInvalidStateTransition = errors.New("strix/cache: invalid buffer state transition")

	// ErrBufferAllocationFailed is returned when 16MB aligned buffer allocation fails.
	ErrBufferAllocationFailed = errors.New("strix/cache: buffer allocation failed")

	// ErrFenceTimeout is returned when waiting for an asynchronous prefetch fence exceeds the deadline.
	ErrFenceTimeout = errors.New("strix/cache: completion fence timed out")

	// ErrBufferCapacityExceeded is returned when a requested prefetch size exceeds the 16MB buffer capacity.
	ErrBufferCapacityExceeded = errors.New("strix/cache: prefetch size exceeds 16MB buffer capacity")

	// ErrBufferBusy is returned when prefetch is scheduled onto a buffer currently being computed or preloaded.
	ErrBufferBusy = errors.New("strix/cache: target buffer is busy")

	// ErrComputeFailed is returned when layer compute execution encounters an unrecoverable failure.
	ErrComputeFailed = errors.New("strix/cache: layer compute execution failed")

	// ErrNilDescriptor is returned when a nil prefetch descriptor is passed.
	ErrNilDescriptor = errors.New("strix/cache: prefetch descriptor is nil")
)

// BufferID identifies one of the dual ping-pong buffers (Buffer A or Buffer B).
type BufferID int

const (
	// BufferIDA identifies Buffer A (sets 0..16,383).
	BufferIDA BufferID = 0
	// BufferIDB identifies Buffer B (sets 16,384..32,767).
	BufferIDB BufferID = 1
)

// String returns the string representation of the BufferID.
func (b BufferID) String() string {
	switch b {
	case BufferIDA:
		return "BUFFER_A"
	case BufferIDB:
		return "BUFFER_B"
	default:
		return fmt.Sprintf("BUFFER_%d", b)
	}
}

// Other returns the alternate buffer in the ping-pong pair.
func (b BufferID) Other() BufferID {
	if b == BufferIDA {
		return BufferIDB
	}
	return BufferIDA
}

// BufferState represents the lifecycle state of a MALL ping-pong buffer.
type BufferState string

const (
	// BufferEmpty indicates the buffer holds no valid layer weights and is idle.
	BufferEmpty BufferState = "EMPTY"
	// BufferPrefetching indicates an asynchronous DMA/scalar prefetch from DRAM is actively filling this buffer.
	BufferPrefetching BufferState = "PREFETCHING"
	// BufferReady indicates prefetch has completed and verified weights are staged and ready for compute.
	BufferReady BufferState = "READY"
	// BufferComputing indicates 40-CU RDNA 3.5 wavefronts are actively reading weights out of this MALL buffer.
	BufferComputing BufferState = "COMPUTING"
	// BufferRecycling indicates compute for this layer is complete and buffer is being scrubbed/re-armed for the next prefetch.
	BufferRecycling BufferState = "RECYCLING"
)

// String returns the string representation of the BufferState.
func (s BufferState) String() string {
	return string(s)
}

// IsValidTransition checks whether moving from state 'from' to state 'to' is permitted.
func IsValidTransition(from, to BufferState) bool {
	if from == to {
		return true
	}
	switch from {
	case BufferEmpty:
		return to == BufferPrefetching || to == BufferReady
	case BufferPrefetching:
		return to == BufferReady || to == BufferEmpty
	case BufferReady:
		return to == BufferComputing || to == BufferRecycling || to == BufferEmpty
	case BufferComputing:
		return to == BufferRecycling || to == BufferEmpty
	case BufferRecycling:
		return to == BufferEmpty || to == BufferPrefetching || to == BufferReady
	default:
		return false
	}
}

// MALLBuffer represents one 16MB symmetrical partition of the 32MB MALL Infinity Cache.
type MALLBuffer struct {
	ID            BufferID
	SetRangeStart int
	SetRangeEnd   int
	CapacityBytes int64
	Alignment     int
	Data          []byte
	ActiveLayer   int
	SubLayerID    int

	state        BufferState
	stateMu      sync.RWMutex
	lastPrefetch time.Time
	lastCompute  time.Time
}

// State returns the current lifecycle state of the MALL buffer in a thread-safe manner.
func (b *MALLBuffer) State() BufferState {
	b.stateMu.RLock()
	defer b.stateMu.RUnlock()
	return b.state
}

// SetState attempts an atomic state transition, validating against the transition rules.
func (b *MALLBuffer) SetState(next BufferState) error {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()

	if !IsValidTransition(b.state, next) {
		return fmt.Errorf("%w: cannot transition from %s to %s for %s",
			ErrInvalidStateTransition, b.state, next, b.ID)
	}
	b.state = next
	return nil
}

// Reset clears the buffer data, resets layer metadata, and sets state to BufferEmpty.
func (b *MALLBuffer) Reset() {
	b.stateMu.Lock()
	defer b.stateMu.Unlock()

	b.state = BufferEmpty
	b.ActiveLayer = -1
	b.SubLayerID = -1
	b.lastPrefetch = time.Time{}
	b.lastCompute = time.Time{}
	if len(b.Data) > 0 {
		clear(b.Data)
	}
}

// AddressAligned verifies that the underlying slice begins on the required 64-byte alignment boundary.
func (b *MALLBuffer) AddressAligned() bool {
	if len(b.Data) == 0 {
		return false
	}
	addr := uintptr(unsafe.Pointer(&b.Data[0]))
	return addr%uintptr(b.Alignment) == 0
}

// ContainsSet returns true if the specified cache set index falls within this buffer's set-disjoint range.
func (b *MALLBuffer) ContainsSet(setIdx int) bool {
	return setIdx >= b.SetRangeStart && setIdx <= b.SetRangeEnd
}

// IsSetDisjointWith verifies that this buffer's cache set range does not overlap with another buffer's set range.
func (b *MALLBuffer) IsSetDisjointWith(other *MALLBuffer) bool {
	if other == nil {
		return true
	}
	return b.SetRangeEnd < other.SetRangeStart || b.SetRangeStart > other.SetRangeEnd
}

// Slice returns a sub-slice of the buffer data with bounds checking.
func (b *MALLBuffer) Slice(offset, length int64) ([]byte, error) {
	if offset < 0 || length < 0 || offset+length > b.CapacityBytes {
		return nil, fmt.Errorf("%w: offset=%d length=%d capacity=%d",
			ErrOutOfBounds, offset, length, b.CapacityBytes)
	}
	return b.Data[offset : offset+length], nil
}

// PrefetchStatus represents the state of an asynchronous DRAM-to-MALL prefetch operation.
type PrefetchStatus string

const (
	// PrefetchPending indicates the prefetch descriptor is queued but has not yet initiated transfer.
	PrefetchPending PrefetchStatus = "PENDING"
	// PrefetchInProgress indicates DMA/scalar prefetch is actively in-flight from DRAM to MALL.
	PrefetchInProgress PrefetchStatus = "IN_PROGRESS"
	// PrefetchCompleted indicates all layer weight bytes have arrived and fence has signaled.
	PrefetchCompleted PrefetchStatus = "COMPLETED"
	// PrefetchFailed indicates transfer failed or timed out.
	PrefetchFailed PrefetchStatus = "FAILED"
)

// CompletionFence provides a thread-safe, deadlock-free synchronization primitive
// between the asynchronous DRAM prefetch pipeline and WMMA compute wavefronts.
type CompletionFence struct {
	done       chan struct{}
	mu         sync.Mutex
	signaled   bool
	signaledAt time.Time
	err        error
}

// NewCompletionFence constructs an unsignaled completion fence.
func NewCompletionFence() *CompletionFence {
	return &CompletionFence{
		done: make(chan struct{}),
	}
}

// Signal marks the fence as completed, unblocking all current and future waiters.
func (f *CompletionFence) Signal() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.signaled {
		f.signaled = true
		f.signaledAt = time.Now()
		close(f.done)
	}
}

// SignalError marks the fence as completed with an error, unblocking waiters.
func (f *CompletionFence) SignalError(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.signaled {
		f.signaled = true
		f.signaledAt = time.Now()
		f.err = err
		close(f.done)
	}
}

// Wait blocks until the fence is signaled or the given timeout elapses.
func (f *CompletionFence) Wait(timeout time.Duration) error {
	if timeout <= 0 {
		select {
		case <-f.done:
			f.mu.Lock()
			defer f.mu.Unlock()
			return f.err
		default:
			return ErrFenceTimeout
		}
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-f.done:
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.err
	case <-timer.C:
		return ErrFenceTimeout
	}
}

// WaitContext blocks until the fence is signaled, the context is cancelled, or the timeout elapses.
func (f *CompletionFence) WaitContext(ctx context.Context, timeout time.Duration) error {
	if timeout <= 0 {
		return f.Wait(timeout)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-f.done:
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.err
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return ErrFenceTimeout
	}
}

// IsSignaled returns true if the fence has already been signaled.
func (f *CompletionFence) IsSignaled() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.signaled
}

// SignaledAt returns the time when the fence was signaled, or zero if unsignaled.
func (f *CompletionFence) SignaledAt() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.signaledAt
}

// DurationSinceSignal returns the elapsed duration since the fence was signaled.
func (f *CompletionFence) DurationSinceSignal() time.Duration {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.signaled {
		return 0
	}
	return time.Since(f.signaledAt)
}

// Err returns the error associated with the fence completion, if any.
func (f *CompletionFence) Err() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.err
}

// Reset resets the fence to an unsignaled state for reuse.
func (f *CompletionFence) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.done = make(chan struct{})
	f.signaled = false
	f.signaledAt = time.Time{}
	f.err = nil
}

// PrefetchDescriptor specifies an asynchronous DMA/scalar prefetch request
// for a transformer layer's weights from DRAM into a target MALL ping-pong buffer.
type PrefetchDescriptor struct {
	LayerID      int
	SubLayerID   int
	DRAMAddress  uintptr
	SizeBytes    int64
	Payload      []byte // in-memory payload for simulation and verification
	Fence        *CompletionFence
	Status       PrefetchStatus
	ScheduledAt  time.Time
	CompletedAt  time.Time
	Duration     time.Duration
	BandwidthGBs float64
}

// DoubleBufferConfig defines sizing, geometry, and bandwidth parameters for the ping-pong manager.
type DoubleBufferConfig struct {
	TotalMALLCapacityBytes int64
	BufferCapacityBytes    int64
	CacheLineSizeBytes     int
	DRAMBandwidthGBs       float64
	MALLBandwidthGBs       float64
	SimulateTransferDelay  bool
	PrefetchLeadTimeTarget time.Duration
}

// DefaultDoubleBufferConfig returns the canonical configuration grounded in AMD Strix Halo silicon specs.
func DefaultDoubleBufferConfig() DoubleBufferConfig {
	return DoubleBufferConfig{
		TotalMALLCapacityBytes: MALLTotalCapacityBytes,
		BufferCapacityBytes:    MALLBufferCapacityBytes,
		CacheLineSizeBytes:     MALLCacheLineSizeBytes,
		DRAMBandwidthGBs:       PhysicalDRAMBandwidthGBs,
		MALLBandwidthGBs:       PeakMALLBandwidthGBs,
		SimulateTransferDelay:  false,
		PrefetchLeadTimeTarget: 58580 * time.Nanosecond, // 58.58 µs theoretical prefetch window
	}
}

// DoubleBufferMetrics captures operational telemetry, latency, and stall reductions for the double buffer manager.
type DoubleBufferMetrics struct {
	FlipCount                  int64   `json:"flip_count"`
	TotalPrefetchBytes         int64   `json:"total_prefetch_bytes"`
	TotalPrefetchDurationNs    int64   `json:"total_prefetch_duration_ns"`
	AveragePrefetchLeadTimeNs  int64   `json:"average_prefetch_lead_time_ns"`
	StallCyclesReduced         int64   `json:"stall_cycles_reduced"`
	JitterVarianceReductionPct float64 `json:"jitter_variance_reduction_pct"`
	MALLHitRate                float64 `json:"mall_hit_rate"`
	SwapLatencyNs              int64   `json:"swap_latency_ns"`
	DirectDRAMFallbackCount    int64   `json:"direct_dram_fallback_count"`
	PrefetchSuccessCount       int64   `json:"prefetch_success_count"`
	PrefetchUnderrunCount      int64   `json:"prefetch_underrun_count"`
	ActiveComputeBuffer        string  `json:"active_compute_buffer"`
	ActivePrefetchBuffer       string  `json:"active_prefetch_buffer"`
}

// DoubleBufferTelemetry is a type alias for DoubleBufferMetrics for operations and telemetry reporting.
type DoubleBufferTelemetry = DoubleBufferMetrics
