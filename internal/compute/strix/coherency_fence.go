// Package strix implements hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

const (
	// CacheLineSize is the standard 64-byte x86-64 / Zen 5 cache line size.
	CacheLineSize = 64

	// MaxAcceptableSFenceLatencyNs defines the upper bound for sfence() latency (<15ns).
	MaxAcceptableSFenceLatencyNs = 15.0
)

// Common errors for store buffer coherency protocol.
var (
	ErrCoherencyFenceRequired = errors.New("strix: store buffer coherency fence required prior to AQL doorbell ring")
	ErrInvalidFenceToken      = errors.New("strix: invalid or expired store buffer fence token")
	ErrBufferOutOfBounds      = errors.New("strix: memory access exceeds USWC buffer bounds")
	ErrUnfencedRejected       = errors.New("strix: unfenced store buffer access rejected by strict coherency policy")
)

// StoreBufferProtocol identifies the coherency enforcement mechanism.
type StoreBufferProtocol string

const (
	// StoreBufferProtocolSFENCE is Arm 1 (Target): issues native SFENCE instruction to drain
	// CPU write-combining store buffers (WCBs) to the shared memory crossbar in <10ns.
	StoreBufferProtocolSFENCE StoreBufferProtocol = "sfence"

	// StoreBufferProtocolUnfenced is Arm 2 (Baseline): omits memory barrier, provoking
	// stale data read hazards when GPU reads uncommitted partial cache-lines.
	StoreBufferProtocolUnfenced StoreBufferProtocol = "unfenced"

	// StoreBufferProtocolCLFlushOpt is Arm 3: iterates cache line by line using CLFLUSHOPT,
	// providing line invalidation at higher instruction overhead (~50-100x higher than SFENCE).
	StoreBufferProtocolCLFlushOpt StoreBufferProtocol = "clflushopt"
)

// StoreBufferFence represents an immutable proof token that CPU write-combining
// store buffers have been drained and serialized prior to GPU dispatch.
type StoreBufferFence struct {
	TokenID      uint64              `json:"token_id"`
	Timestamp    time.Time           `json:"timestamp"`
	Protocol     StoreBufferProtocol `json:"protocol"`
	BytesFlushed int64               `json:"bytes_flushed"`
	LatencyNs    int64               `json:"latency_ns"`
	Valid        bool                `json:"valid"`
}

var (
	globalStoreBufferFenceCounter uint64
)

// SFence executes an SFENCE instruction to drain CPU write-combining store buffers.
func SFence() {
	sfence()
}

// MFence executes an MFENCE instruction to serialize all loads and stores.
func MFence() {
	mfence()
}

// CLFlushOpt flushes the 64-byte cache line containing addr using CLFLUSHOPT.
func CLFlushOpt(addr uintptr) {
	clflushopt(addr)
}

// FlushRangeCLFlushOpt flushes each 64-byte cache line spanning [base, base+size)
// and finishes with an SFENCE to serialize the line invalidations.
func FlushRangeCLFlushOpt(base uintptr, size uintptr) {
	if size == 0 {
		return
	}
	end := base + size
	startLine := base &^ uintptr(CacheLineSize-1)
	for addr := startLine; addr < end; addr += CacheLineSize {
		clflushopt(addr)
	}
	sfence()
}

// FlushStoreBuffer issues a native SFENCE barrier and returns a validated fence token.
func FlushStoreBuffer(bytesFlushed int64) StoreBufferFence {
	t0 := time.Now()
	sfence()
	lat := time.Since(t0).Nanoseconds()

	token := atomic.AddUint64(&globalStoreBufferFenceCounter, 1)
	return StoreBufferFence{
		TokenID:      token,
		Timestamp:    time.Now(),
		Protocol:     StoreBufferProtocolSFENCE,
		BytesFlushed: bytesFlushed,
		LatencyNs:    lat,
		Valid:        true,
	}
}

// FlushStoreBufferWithProtocol executes the requested coherency protocol and returns a fence token.
func FlushStoreBufferWithProtocol(proto StoreBufferProtocol, base uintptr, bytesFlushed int64) (StoreBufferFence, error) {
	token := atomic.AddUint64(&globalStoreBufferFenceCounter, 1)
	now := time.Now()

	switch proto {
	case StoreBufferProtocolSFENCE:
		t0 := time.Now()
		sfence()
		lat := time.Since(t0).Nanoseconds()
		return StoreBufferFence{
			TokenID:      token,
			Timestamp:    now,
			Protocol:     StoreBufferProtocolSFENCE,
			BytesFlushed: bytesFlushed,
			LatencyNs:    lat,
			Valid:        true,
		}, nil

	case StoreBufferProtocolUnfenced:
		// Arm 2: Omits memory barrier; store buffers linger in Zen 5 CPU WCBs.
		return StoreBufferFence{
			TokenID:      token,
			Timestamp:    now,
			Protocol:     StoreBufferProtocolUnfenced,
			BytesFlushed: 0,
			LatencyNs:    0,
			Valid:        false,
		}, nil

	case StoreBufferProtocolCLFlushOpt:
		t0 := time.Now()
		FlushRangeCLFlushOpt(base, uintptr(bytesFlushed))
		lat := time.Since(t0).Nanoseconds()
		return StoreBufferFence{
			TokenID:      token,
			Timestamp:    now,
			Protocol:     StoreBufferProtocolCLFlushOpt,
			BytesFlushed: bytesFlushed,
			LatencyNs:    lat,
			Valid:        true,
		}, nil

	default:
		return StoreBufferFence{}, fmt.Errorf("strix: unsupported store buffer protocol %q", proto)
	}
}

// CoherencyAQLPacket models a packet submitted to an AMD RDNA 3.5 hardware queue.
type CoherencyAQLPacket struct {
	PacketID      uint64 `json:"packet_id"`
	Payload       []byte `json:"payload"`
	DestinationVA uint64 `json:"destination_va"`
	Flags         uint32 `json:"flags"`
}

// AQLDoorbellQueue models an AQL hardware queue protected by the coherency gate.
type AQLDoorbellQueue struct {
	mu             sync.Mutex
	StrictFence    bool
	SubmittedCount uint64
	Submissions    []CoherencyAQLPacket
}

// NewAQLDoorbellQueue creates a new AQL hardware queue.
// When strict is true, submitting without a valid SFENCE/CLFLUSHOPT token is prohibited.
func NewAQLDoorbellQueue(strict bool) *AQLDoorbellQueue {
	return &AQLDoorbellQueue{
		StrictFence: strict,
		Submissions: make([]CoherencyAQLPacket, 0),
	}
}

// Submit enqueues an AQL packet only if preceded by a valid store buffer coherency fence.
func (q *AQLDoorbellQueue) Submit(pkt CoherencyAQLPacket, fence StoreBufferFence) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.StrictFence {
		if !fence.Valid {
			return ErrCoherencyFenceRequired
		}
		if fence.Protocol == StoreBufferProtocolUnfenced {
			return ErrUnfencedRejected
		}
		if fence.TokenID == 0 {
			return ErrInvalidFenceToken
		}
		// Invalidate fences older than 5 seconds to prevent replay
		if time.Since(fence.Timestamp) > 5*time.Second {
			return ErrInvalidFenceToken
		}
	}

	q.SubmittedCount++
	q.Submissions = append(q.Submissions, pkt)
	return nil
}

// USWCMemoryBuffer models an APU unified memory allocation configured with
// AMDGPU_GEM_CREATE_CPU_GTT_USWC (Uncached Speculative Write Combining).
type USWCMemoryBuffer struct {
	mu           sync.RWMutex
	data         []byte
	baseVA       uintptr
	size         int
	pendingDirty int64
	dirtyLines   map[int]bool
	lastFence    StoreBufferFence
}

// NewUSWCMemoryBuffer allocates a modeled USWC unified memory buffer.
func NewUSWCMemoryBuffer(size int, baseVA uintptr) *USWCMemoryBuffer {
	if baseVA == 0 {
		baseVA = SimulatedUnifiedVABase
	}
	return &USWCMemoryBuffer{
		data:       make([]byte, size),
		baseVA:     baseVA,
		size:       size,
		dirtyLines: make(map[int]bool),
	}
}

// Write writes payload into the USWC buffer. CPU writes accumulate in WCBs.
func (b *USWCMemoryBuffer) Write(offset int, payload []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if offset < 0 || offset+len(payload) > b.size {
		return 0, ErrBufferOutOfBounds
	}

	copy(b.data[offset:offset+len(payload)], payload)
	b.pendingDirty += int64(len(payload))

	startLine := offset / CacheLineSize
	endLine := (offset + len(payload) - 1) / CacheLineSize
	for l := startLine; l <= endLine; l++ {
		b.dirtyLines[l] = true
	}
	return len(payload), nil
}

// Flush executes the chosen coherency protocol across all pending writes.
func (b *USWCMemoryBuffer) Flush(proto StoreBufferProtocol) (StoreBufferFence, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	var mappedBase uintptr
	if len(b.data) > 0 {
		mappedBase = uintptr(unsafe.Pointer(&b.data[0]))
	}

	fence, err := FlushStoreBufferWithProtocol(proto, mappedBase, b.pendingDirty)
	if err != nil {
		return fence, err
	}

	if fence.Valid {
		b.pendingDirty = 0
		b.dirtyLines = make(map[int]bool)
		b.lastFence = fence
	}
	return fence, nil
}

// ReadGPU simulates a GPU shader read from the physical memory crossbar.
// If simRace is true and there are unflushed pending writes (or unfenced protocol),
// the GPU observes stale data with a controlled race hazard.
func (b *USWCMemoryBuffer) ReadGPU(offset, length int, simRace bool) ([]byte, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if offset < 0 || offset+length > b.size {
		return nil, false
	}

	out := make([]byte, length)
	copy(out, b.data[offset:offset+length])

	if simRace && len(b.dirtyLines) > 0 {
		startLine := offset / CacheLineSize
		endLine := (offset + length - 1) / CacheLineSize
		for l := startLine; l <= endLine; l++ {
			if b.dirtyLines[l] {
				// Simulating uncommitted data in DRAM crossbar for unfenced writes
				return out, false
			}
		}
	}

	return out, true
}

// BaseVA returns the virtual address base of the buffer.
func (b *USWCMemoryBuffer) BaseVA() uintptr {
	return b.baseVA
}

// Size returns the buffer size in bytes.
func (b *USWCMemoryBuffer) Size() int {
	return b.size
}

// AblationArmMetrics captures empirical benchmark results for one ablation arm.
type AblationArmMetrics struct {
	Protocol       StoreBufferProtocol `json:"protocol"`
	Iterations     int                 `json:"iterations"`
	TotalLatencyNs int64               `json:"total_latency_ns"`
	AvgLatencyNs   float64             `json:"avg_latency_ns"`
	StaleReadCount int                 `json:"stale_read_count"`
	ConsistencyPct float64             `json:"consistency_pct"`
}

// RunAblationEvaluation runs a comparative evaluation between Arm 1 (SFENCE),
// Arm 2 (Unfenced), and Arm 3 (CLFlushOpt).
func RunAblationEvaluation(iterations int, bufferSize int) map[StoreBufferProtocol]AblationArmMetrics {
	if iterations <= 0 {
		iterations = 100
	}
	if bufferSize <= 0 {
		bufferSize = 4096
	}

	protocols := []StoreBufferProtocol{
		StoreBufferProtocolSFENCE,
		StoreBufferProtocolUnfenced,
		StoreBufferProtocolCLFlushOpt,
	}

	results := make(map[StoreBufferProtocol]AblationArmMetrics)

	for _, proto := range protocols {
		buf := NewUSWCMemoryBuffer(bufferSize, SimulatedUnifiedVABase)
		var totalLat int64
		staleReads := 0

		testPayload := make([]byte, 128)
		for i := range testPayload {
			testPayload[i] = byte((i + 1) % 255)
		}

		for it := 0; it < iterations; it++ {
			// Step 1: CPU write to USWC buffer
			_, _ = buf.Write(0, testPayload)

			// Step 2: Flush according to protocol
			fence, _ := buf.Flush(proto)
			totalLat += fence.LatencyNs

			// Step 3: GPU reads data with simulated crossbar access
			readBack, clean := buf.ReadGPU(0, len(testPayload), true)
			if !clean {
				staleReads++
			} else {
				// Verify content matches
				for i := range testPayload {
					if readBack[i] != testPayload[i] {
						staleReads++
						break
					}
				}
			}
		}

		consistency := float64(iterations-staleReads) / float64(iterations) * 100.0
		avgLat := float64(totalLat) / float64(iterations)

		results[proto] = AblationArmMetrics{
			Protocol:       proto,
			Iterations:     iterations,
			TotalLatencyNs: totalLat,
			AvgLatencyNs:   avgLat,
			StaleReadCount: staleReads,
			ConsistencyPct: consistency,
		}
	}

	return results
}
