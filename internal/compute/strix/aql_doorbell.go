// Package strix implements direct userspace AQL queue management,
// cacheline-aligned 64-byte packet formatting, and low-overhead MMIO hardware
// doorbell signaling for sub-50µs tool-yield preemption and resumption on AMD
// Strix Halo (Ryzen AI Max+ 395 / GFX1151).
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
	// DefaultAQLQueueSize is the standard circular ring buffer capacity (1024 packets).
	DefaultAQLQueueSize uint32 = 1024

	// MinAQLQueueSize is the minimum queue capacity (must be a non-zero power of two).
	MinAQLQueueSize uint32 = 4

	// MaxAQLQueueSize is the maximum user-mode queue depth supported on Strix Halo.
	MaxAQLQueueSize uint32 = 65536

	// TargetDoorbellLatencyP99Ns is the P99 latency SLA (50 microseconds) for lockless resumption.
	TargetDoorbellLatencyP99Ns int64 = 50 * 1000

	// TargetDoorbellLatencyP50Ns is the P50 latency SLA (22 microseconds) for lockless resumption.
	TargetDoorbellLatencyP50Ns int64 = 22 * 1000

	// MockMMIOWriteLatencyCeilingNs is the benchmark ceiling (< 5µs) for mock MMIO writes.
	MockMMIOWriteLatencyCeilingNs int64 = 5 * 1000
)

// DoorbellWidth defines the register width for the MMIO doorbell aperture.
type DoorbellWidth int

const (
	// DoorbellWidth32 specifies a 32-bit MMIO doorbell write (standard on KFD/GFX).
	DoorbellWidth32 DoorbellWidth = 32

	// DoorbellWidth64 specifies a 64-bit MMIO doorbell write.
	DoorbellWidth64 DoorbellWidth = 64
)

// String returns the string representation of DoorbellWidth.
func (w DoorbellWidth) String() string {
	switch w {
	case DoorbellWidth32:
		return "32-bit"
	case DoorbellWidth64:
		return "64-bit"
	default:
		return fmt.Sprintf("unknown(%d)", int(w))
	}
}

// QueueState represents the execution lifecycle of an AQL queue during tool yield and resumption.
type QueueState int32

const (
	// QueueStateActive indicates normal GPU execution and decode.
	QueueStateActive QueueState = 0

	// QueueStateYielded indicates the CPU agent has yielded execution to run external tools (YIELDED_IO).
	QueueStateYielded QueueState = 1

	// QueueStateResuming indicates preemption resumption is in flight via doorbell signaling.
	QueueStateResuming QueueState = 2
)

// String returns the string representation of QueueState.
func (s QueueState) String() string {
	switch s {
	case QueueStateActive:
		return "ACTIVE"
	case QueueStateYielded:
		return "YIELDED"
	case QueueStateResuming:
		return "RESUMING"
	default:
		return fmt.Sprintf("UNKNOWN_STATE(%d)", int32(s))
	}
}

// Doorbell queue errors.
var (
	// ErrQueueFull indicates the AQL user-mode queue ring buffer is full.
	ErrQueueFull = errors.New("strix/doorbell: AQL queue ring buffer is full")

	// ErrInvalidQueueSize indicates queue size is not a valid non-zero power of two.
	ErrInvalidQueueSize = errors.New("strix/doorbell: queue size must be a non-zero power of two between 4 and 65536")

	// ErrQueueClosed indicates operations on a closed AQL queue.
	ErrQueueClosed = errors.New("strix/doorbell: AQL queue is closed")

	// ErrNilPacket indicates submission of a nil packet.
	ErrNilPacket = errors.New("strix/doorbell: nil packet submitted")

	// ErrInvalidPacketSize indicates packet bytes length does not equal 64 bytes.
	ErrInvalidPacketSize = errors.New("strix/doorbell: packet must be strictly 64 bytes")
)

// AQLQueue represents an architected userspace AQL ring buffer queue wrapping
// 64-byte AQL packets with direct MMIO hardware doorbell dispatch.
type AQLQueue struct {
	// Ring buffer memory & sizing
	rawMemory  []byte     // underlying backing memory
	ringBuffer [][64]byte // 64-byte aligned packet slots
	queueSize  uint32     // power-of-two capacity
	queueMask  uint32     // queueSize - 1 for circular masking

	// Queue identity
	queueID   uint32
	sessionID string

	// Pointers (accessed atomically)
	writePtr uint64 // monotonic write pointer index
	readPtr  uint64 // monotonic read pointer index

	// MMIO Doorbell mapping & simulation
	mmioPtr         unsafe.Pointer // mapped MMIO aperture address (nil = fallback simulation)
	doorbellWidth   int32          // atomic DoorbellWidth (32 or 64 bit)
	simulatedMMIO32 uint32         // in-memory fallback register for 32-bit MMIO
	simulatedMMIO64 uint64         // in-memory fallback register for 64-bit MMIO
	fallbackCount   uint64         // count of doorbell rings using in-memory simulation

	// Telemetry & lifecycle
	lastRingNs   int64  // latency of the most recent doorbell ring (ns)
	ringCount    uint64 // total doorbell rings
	barrierCount uint64 // total barrier packets submitted
	state        int32  // atomic QueueState
	lastExitCode int32  // tool child process exit code
	closed       int32  // atomic closed flag

	// Mutex for synchronizing concurrent packet writes into the ring buffer
	mu sync.RWMutex
}

// QueueOption configures optional parameters during AQLQueue initialization.
type QueueOption func(*AQLQueue)

// WithDoorbellWidth sets the register width for the MMIO doorbell aperture (32-bit or 64-bit).
func WithDoorbellWidth(width DoorbellWidth) QueueOption {
	return func(q *AQLQueue) {
		atomic.StoreInt32(&q.doorbellWidth, int32(width))
	}
}

// WithMMIOPointer sets an initial mapped MMIO doorbell aperture pointer.
func WithMMIOPointer(ptr unsafe.Pointer) QueueOption {
	return func(q *AQLQueue) {
		atomic.StorePointer(&q.mmioPtr, ptr)
	}
}

// WithQueueID configures the numeric queue ID.
func WithQueueID(id uint32) QueueOption {
	return func(q *AQLQueue) {
		q.queueID = id
	}
}

// NewAQLQueue allocates and initializes a user-mode AQL queue ring buffer wrapping
// 64-byte packets. Backing memory is strictly aligned to a 64-byte cache line boundary.
func NewAQLQueue(sessionID string, queueSize uint32, opts ...QueueOption) (*AQLQueue, error) {
	// Verify non-zero power of two within supported bounds
	if queueSize == 0 || (queueSize&(queueSize-1)) != 0 || queueSize < MinAQLQueueSize || queueSize > MaxAQLQueueSize {
		return nil, fmt.Errorf("%w: requested %d", ErrInvalidQueueSize, queueSize)
	}

	// Allocate backing memory with extra 64 bytes to guarantee 64-byte alignment
	totalBytes := int(queueSize) * AQLPacketSizeBytes
	rawMemory := make([]byte, totalBytes+64)

	baseAddr := uintptr(unsafe.Pointer(&rawMemory[0]))
	alignedOffset := (64 - (baseAddr % 64)) % 64
	alignedPtr := unsafe.Pointer(&rawMemory[alignedOffset])
	ringBuffer := unsafe.Slice((*[64]byte)(alignedPtr), queueSize)

	q := &AQLQueue{
		rawMemory:     rawMemory,
		ringBuffer:    ringBuffer,
		queueSize:     queueSize,
		queueMask:     queueSize - 1,
		queueID:       1,
		sessionID:     sessionID,
		doorbellWidth: int32(DoorbellWidth32),
		state:         int32(QueueStateActive),
	}

	for _, opt := range opts {
		opt(q)
	}

	return q, nil
}

// NewDefaultAQLQueue creates an AQLQueue with the default capacity (1024 packets).
func NewDefaultAQLQueue(sessionID string, opts ...QueueOption) (*AQLQueue, error) {
	return NewAQLQueue(sessionID, DefaultAQLQueueSize, opts...)
}

// QueueSize returns the capacity of the ring buffer.
func (q *AQLQueue) QueueSize() uint32 {
	return q.queueSize
}

// QueueMask returns the bitmask used for circular wrapping (queueSize - 1).
func (q *AQLQueue) QueueMask() uint32 {
	return q.queueMask
}

// SessionID returns the mapped agent session ID.
func (q *AQLQueue) SessionID() string {
	return q.sessionID
}

// QueueID returns the numeric queue ID.
func (q *AQLQueue) QueueID() uint32 {
	return q.queueID
}

// RingBufferBase returns the base memory address of the ring buffer.
func (q *AQLQueue) RingBufferBase() uintptr {
	if len(q.ringBuffer) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&q.ringBuffer[0]))
}

// Is64ByteAligned confirms that the ring buffer base pointer is aligned to 64 bytes.
func (q *AQLQueue) Is64ByteAligned() bool {
	base := q.RingBufferBase()
	return base != 0 && (base%64) == 0
}

// LoadWritePtr atomically reads the current monotonic write pointer index.
func (q *AQLQueue) LoadWritePtr() uint64 {
	return atomic.LoadUint64(&q.writePtr)
}

// LoadReadPtr atomically reads the current monotonic read pointer index.
func (q *AQLQueue) LoadReadPtr() uint64 {
	return atomic.LoadUint64(&q.readPtr)
}

// SetReadPtr atomically updates the monotonic read pointer index (simulating GPU progress).
func (q *AQLQueue) SetReadPtr(val uint64) {
	atomic.StoreUint64(&q.readPtr, val)
}

// AdvanceReadPtr atomically increments the read pointer by delta and returns the new value.
func (q *AQLQueue) AdvanceReadPtr(delta uint64) uint64 {
	return atomic.AddUint64(&q.readPtr, delta)
}

// SlotIndex returns the circular wrapped slot index for a given pointer index.
func (q *AQLQueue) SlotIndex(ptr uint64) uint64 {
	return ptr & uint64(q.queueMask)
}

// PendingCount returns the number of unconsumed packets in the ring buffer.
func (q *AQLQueue) PendingCount() uint64 {
	w := q.LoadWritePtr()
	r := q.LoadReadPtr()
	if w >= r {
		return w - r
	}
	return 0
}

// IsFull returns true if the queue has reached its power-of-two capacity.
func (q *AQLQueue) IsFull() bool {
	return q.PendingCount() >= uint64(q.queueSize)
}

// State returns the current execution state of the queue.
func (q *AQLQueue) State() QueueState {
	return QueueState(atomic.LoadInt32(&q.state))
}

// PreemptForTool marks the queue as yielded for tool execution (QueueStateYielded).
func (q *AQLQueue) PreemptForTool() error {
	atomic.StoreInt32(&q.state, int32(QueueStateYielded))
	return nil
}

// IsYielded reports whether the queue is currently in the yielded state.
func (q *AQLQueue) IsYielded() bool {
	return q.State() == QueueStateYielded
}

// LastExitCode returns the tool exit code recorded during the last resumption.
func (q *AQLQueue) LastExitCode() int {
	return int(atomic.LoadInt32(&q.lastExitCode))
}

// SetMMIOPointer updates the mapped MMIO doorbell register address.
func (q *AQLQueue) SetMMIOPointer(ptr unsafe.Pointer) {
	atomic.StorePointer(&q.mmioPtr, ptr)
}

// LoadMMIOPointer returns the currently mapped MMIO doorbell pointer.
func (q *AQLQueue) LoadMMIOPointer() unsafe.Pointer {
	return atomic.LoadPointer(&q.mmioPtr)
}

// SetDoorbellWidth updates the MMIO doorbell register width (32-bit or 64-bit).
func (q *AQLQueue) SetDoorbellWidth(w DoorbellWidth) {
	atomic.StoreInt32(&q.doorbellWidth, int32(w))
}

// DoorbellWidth returns the current MMIO doorbell register width.
func (q *AQLQueue) DoorbellWidth() DoorbellWidth {
	return DoorbellWidth(atomic.LoadInt32(&q.doorbellWidth))
}

// LoadSimulatedDoorbell32 returns the in-memory fallback 32-bit doorbell register.
func (q *AQLQueue) LoadSimulatedDoorbell32() uint32 {
	return atomic.LoadUint32(&q.simulatedMMIO32)
}

// LoadSimulatedDoorbell64 returns the in-memory fallback 64-bit doorbell register.
func (q *AQLQueue) LoadSimulatedDoorbell64() uint64 {
	return atomic.LoadUint64(&q.simulatedMMIO64)
}

// FallbackCount returns the number of times doorbell writes used in-memory simulation.
func (q *AQLQueue) FallbackCount() uint64 {
	return atomic.LoadUint64(&q.fallbackCount)
}

// LastRingNs returns the latency of the most recent doorbell ring in nanoseconds.
func (q *AQLQueue) LastRingNs() int64 {
	return atomic.LoadInt64(&q.lastRingNs)
}

// RingCount returns the cumulative number of doorbell rings executed.
func (q *AQLQueue) RingCount() uint64 {
	return atomic.LoadUint64(&q.ringCount)
}

// BarrierCount returns the total number of barrier packets submitted.
func (q *AQLQueue) BarrierCount() uint64 {
	return atomic.LoadUint64(&q.barrierCount)
}

// isClosed checks if the queue has been closed.
func (q *AQLQueue) isClosed() bool {
	return atomic.LoadInt32(&q.closed) != 0
}

// Close closes the queue and prevents further packet submissions.
func (q *AQLQueue) Close() error {
	atomic.StoreInt32(&q.closed, 1)
	return nil
}

// RingDoorbell executes an atomic 32-bit or 64-bit store to the MMIO doorbell
// register address. If the MMIO pointer is nil or unmapped, it automatically
// falls back to an in-memory atomic simulation without failing.
// Returns the dispatch latency in nanoseconds.
func (q *AQLQueue) RingDoorbell(writePtr uint64) (latencyNs int64, err error) {
	start := time.Now()

	mmio := atomic.LoadPointer(&q.mmioPtr)
	width := DoorbellWidth(atomic.LoadInt32(&q.doorbellWidth))

	if mmio != nil {
		if width == DoorbellWidth32 {
			atomic.StoreUint32((*uint32)(mmio), uint32(writePtr))
		} else {
			atomic.StoreUint64((*uint64)(mmio), writePtr)
		}
		// Mirror value into simulated registers for inspection
		atomic.StoreUint32(&q.simulatedMMIO32, uint32(writePtr))
		atomic.StoreUint64(&q.simulatedMMIO64, writePtr)
	} else {
		// Fallback mechanism: in-memory atomic simulation
		atomic.StoreUint32(&q.simulatedMMIO32, uint32(writePtr))
		atomic.StoreUint64(&q.simulatedMMIO64, writePtr)
		atomic.AddUint64(&q.fallbackCount, 1)
	}

	elapsed := time.Since(start).Nanoseconds()
	if elapsed <= 0 {
		elapsed = 1
	}

	atomic.StoreInt64(&q.lastRingNs, elapsed)
	atomic.AddUint64(&q.ringCount, 1)
	return elapsed, nil
}

// extractPacketBytes decodes any valid AQL packet representation into a raw 64-byte array
// and extracted 16-bit header.
func extractPacketBytes(packet any) ([64]byte, AQLPacketHeader, error) {
	if packet == nil {
		return [64]byte{}, 0, ErrNilPacket
	}

	switch p := packet.(type) {
	case AQLKernelDispatchPacket:
		return p.Bytes(), p.Header, nil
	case *AQLKernelDispatchPacket:
		if p == nil {
			return [64]byte{}, 0, ErrNilPacket
		}
		return p.Bytes(), p.Header, nil
	case AQLBarrierPacket:
		return p.Bytes(), p.Header, nil
	case *AQLBarrierPacket:
		if p == nil {
			return [64]byte{}, 0, ErrNilPacket
		}
		return p.Bytes(), p.Header, nil
	case AQLPacket:
		return p.Bytes(), p.PacketHeader(), nil
	case [64]byte:
		hdr := AQLPacketHeader(uint16(p[0]) | (uint16(p[1]) << 8))
		return p, hdr, nil
	case *[64]byte:
		if p == nil {
			return [64]byte{}, 0, ErrNilPacket
		}
		hdr := AQLPacketHeader(uint16((*p)[0]) | (uint16((*p)[1]) << 8))
		return *p, hdr, nil
	case []byte:
		if len(p) != AQLPacketSizeBytes {
			return [64]byte{}, 0, fmt.Errorf("%w: got %d bytes", ErrInvalidPacketSize, len(p))
		}
		var raw [64]byte
		copy(raw[:], p)
		hdr := AQLPacketHeader(uint16(raw[0]) | (uint16(raw[1]) << 8))
		return raw, hdr, nil
	default:
		return [64]byte{}, 0, fmt.Errorf("strix/doorbell: unsupported packet type %T", packet)
	}
}

// SubmitPacket writes a 64-byte AQL packet into the circular ring buffer,
// advances the monotonic write pointer, and rings the MMIO hardware doorbell.
// Returns the updated write pointer, doorbell latency in nanoseconds, and error.
func (q *AQLQueue) SubmitPacket(packet any) (uint64, int64, error) {
	raw, header, err := extractPacketBytes(packet)
	if err != nil {
		return 0, 0, err
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	if q.isClosed() {
		return 0, 0, ErrQueueClosed
	}

	w := atomic.LoadUint64(&q.writePtr)
	r := atomic.LoadUint64(&q.readPtr)

	// Check if queue is full
	if w-r >= uint64(q.queueSize) {
		return w, 0, ErrQueueFull
	}

	// Track barrier packet submissions
	if header.Barrier() || header.Type() == AQLPacketTypeBarrierAnd || header.Type() == AQLPacketTypeBarrierOr {
		atomic.AddUint64(&q.barrierCount, 1)
	}

	// Write 64 bytes to the circular slot index
	slotIdx := w & uint64(q.queueMask)
	q.ringBuffer[slotIdx] = raw

	// Advance write pointer
	nextPtr := w + 1
	atomic.StoreUint64(&q.writePtr, nextPtr)

	// Ring doorbell
	lat, ringErr := q.RingDoorbell(nextPtr)
	return nextPtr, lat, ringErr
}

// SubmitKernelDispatch is a typed convenience method for submitting an AQLKernelDispatchPacket.
func (q *AQLQueue) SubmitKernelDispatch(packet AQLKernelDispatchPacket) (uint64, int64, error) {
	return q.SubmitPacket(packet)
}

// SubmitBarrier is a typed convenience method for submitting an AQLBarrierPacket.
func (q *AQLQueue) SubmitBarrier(packet AQLBarrierPacket) (uint64, int64, error) {
	return q.SubmitPacket(packet)
}

// ResumeToolYield coordinates lockless tool resumption:
//  1. Records the child process / tool exit code.
//  2. Transitions queue state from QueueStateYielded back to QueueStateActive.
//  3. If an optional resumption packet is provided, submits it to the ring buffer.
//     Otherwise, advances the write pointer to signal the RDNA 3.5 command processor.
//  4. Rings the hardware MMIO doorbell register (or in-memory fallback simulation).
//
// Returns the dispatch latency in nanoseconds (< 50µs SLA).
func (q *AQLQueue) ResumeToolYield(exitCode int, resumptionPacket ...any) (latencyNs int64, err error) {
	atomic.StoreInt32(&q.lastExitCode, int32(exitCode))
	atomic.StoreInt32(&q.state, int32(QueueStateActive))

	if len(resumptionPacket) > 0 && resumptionPacket[0] != nil {
		_, lat, err := q.SubmitPacket(resumptionPacket[0])
		return lat, err
	}

	// Advance write pointer and ring doorbell
	nextPtr := atomic.AddUint64(&q.writePtr, 1)
	lat, err := q.RingDoorbell(nextPtr)
	return lat, err
}

// GetPacket reads the raw 64-byte packet currently stored at the wrapped slot index.
func (q *AQLQueue) GetPacket(slotIdx uint64) [64]byte {
	q.mu.RLock()
	defer q.mu.RUnlock()
	idx := slotIdx & uint64(q.queueMask)
	return q.ringBuffer[idx]
}

// GetKernelDispatchPacket decodes the packet at the wrapped slot index into an AQLKernelDispatchPacket.
func (q *AQLQueue) GetKernelDispatchPacket(slotIdx uint64) (AQLKernelDispatchPacket, error) {
	raw := q.GetPacket(slotIdx)
	pkt := KernelDispatchFromBytes(raw)
	if pkt.Header.Type() != AQLPacketTypeKernelDispatch {
		return pkt, fmt.Errorf("strix/doorbell: slot %d contains packet type %s, not KERNEL_DISPATCH",
			slotIdx, pkt.Header.Type())
	}
	return pkt, nil
}

// GetBarrierPacket decodes the packet at the wrapped slot index into an AQLBarrierPacket.
func (q *AQLQueue) GetBarrierPacket(slotIdx uint64) (AQLBarrierPacket, error) {
	raw := q.GetPacket(slotIdx)
	pkt := BarrierFromBytes(raw)
	t := pkt.Header.Type()
	if t != AQLPacketTypeBarrierAnd && t != AQLPacketTypeBarrierOr {
		return pkt, fmt.Errorf("strix/doorbell: slot %d contains packet type %s, not BARRIER", slotIdx, t)
	}
	return pkt, nil
}

// AQLQueueRegistry provides thread-safe mapping and coordination of AQLQueues
// across concurrent agent sessions.
type AQLQueueRegistry struct {
	mu     sync.RWMutex
	queues map[string]*AQLQueue
}

// NewAQLQueueRegistry instantiates an empty AQLQueueRegistry.
func NewAQLQueueRegistry() *AQLQueueRegistry {
	return &AQLQueueRegistry{
		queues: make(map[string]*AQLQueue),
	}
}

// Register allocates and registers a new AQLQueue for sessionID.
func (r *AQLQueueRegistry) Register(sessionID string, queueSize uint32, opts ...QueueOption) (*AQLQueue, error) {
	if sessionID == "" {
		return nil, errors.New("strix/doorbell: empty session ID")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.queues[sessionID]; exists {
		return nil, fmt.Errorf("strix/doorbell: queue already registered for session %s", sessionID)
	}

	q, err := NewAQLQueue(sessionID, queueSize, opts...)
	if err != nil {
		return nil, err
	}

	r.queues[sessionID] = q
	return q, nil
}

// Get retrieves an existing AQLQueue for sessionID.
func (r *AQLQueueRegistry) Get(sessionID string) (*AQLQueue, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	q, ok := r.queues[sessionID]
	return q, ok
}

// GetOrCreate retrieves an existing queue or creates one with default parameters.
func (r *AQLQueueRegistry) GetOrCreate(sessionID string, queueSize uint32, opts ...QueueOption) (*AQLQueue, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if q, ok := r.queues[sessionID]; ok {
		return q, nil
	}

	q, err := NewAQLQueue(sessionID, queueSize, opts...)
	if err != nil {
		return nil, err
	}
	r.queues[sessionID] = q
	return q, nil
}

// Unregister closes and removes the queue for sessionID.
func (r *AQLQueueRegistry) Unregister(sessionID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	q, exists := r.queues[sessionID]
	if !exists {
		return fmt.Errorf("strix/doorbell: queue not found for session %s", sessionID)
	}

	_ = q.Close()
	delete(r.queues, sessionID)
	return nil
}

// Count returns the number of active registered queues.
func (r *AQLQueueRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.queues)
}
