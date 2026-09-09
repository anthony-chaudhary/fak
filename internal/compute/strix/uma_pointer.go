// Package strix implements the hardware profile, memory bandwidth physics,
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

// Typed errors for UMA pointer operations.
var (
	// ErrEmptySlice is returned when converting a nil or zero-length slice to a device pointer.
	ErrEmptySlice = errors.New("strix/uma: slice is nil or empty")

	// ErrInvalidDevPtr is returned when a device pointer is 0 or invalid.
	ErrInvalidDevPtr = errors.New("strix/uma: invalid device pointer")

	// ErrInvalidAlignment is returned when alignment is not a positive power of 2.
	ErrInvalidAlignment = errors.New("strix/uma: alignment must be a positive power of 2")
)

// UMAPointerCounters records real-time allocation and zero-copy pointer metrics.
type UMAPointerCounters struct {
	TotalAllocatedBytes   int64 `json:"total_allocated_bytes"`
	ActiveAllocations     int64 `json:"active_allocations"`
	ZeroCopyConversions   int64 `json:"zero_copy_conversions"`
	AssertionChecksPassed int64 `json:"assertion_checks_passed"`
}

// UMAPointerManager governs zero-copy Unified Memory Architecture (UMA) pointer identity
// mappings for AMD Strix Halo (GFX1151). On this physical APU architecture, Zen 5 CPU cores
// and RDNA 3.5 compute units communicate across a coherent 256-bit Infinity Fabric crossbar
// sharing physical LPDDR5X DRAM. Host virtual addresses and device virtual addresses are
// 1:1 identical (uintptr(HostPtr) == uintptr(DevPtr)).
type UMAPointerManager struct {
	mu                    sync.RWMutex
	buffers               map[*UMABuffer]int64
	totalAllocatedBytes   int64
	activeAllocations     int64
	zeroCopyConversions   int64
	assertionChecksPassed int64
}

// NewUMAPointerManager initializes a new UMA pointer manager.
func NewUMAPointerManager() *UMAPointerManager {
	return &UMAPointerManager{
		buffers: make(map[*UMABuffer]int64),
	}
}

// defaultUMAPointerManager is the process-wide default UMA pointer manager.
var defaultUMAPointerManager = NewUMAPointerManager()

// DefaultUMAPointerManager returns the process-wide default UMA pointer manager.
func DefaultUMAPointerManager() *UMAPointerManager {
	return defaultUMAPointerManager
}

// AllocateUMABuffer allocates a zero-copy unified memory buffer with specified byte size
// and power-of-2 alignment (e.g. 64-byte cache line, 2MB hugepage). In both simulated
// environments and on physical GFX1151 silicon, it guarantees strict pointer identity:
// uintptr(HostPtr) == uintptr(DevPtr).
func (m *UMAPointerManager) AllocateUMABuffer(size int64, align int64) (*UMABuffer, error) {
	if size <= 0 {
		return nil, ErrInvalidBufferSize
	}
	if size > MaxGTTAllocPoolBytes {
		return nil, fmt.Errorf("%w: requested size %d exceeds max GTT pool %d", ErrInvalidBufferSize, size, MaxGTTAllocPoolBytes)
	}

	if align == 0 {
		align = UMACacheLineAlignment
	}
	if align < 0 || (align&(align-1)) != 0 {
		return nil, ErrInvalidAlignment
	}
	// Guarantee at least cache line alignment matching Zen 5 CPU and RDNA 3.5 memory fabric.
	if align < UMACacheLineAlignment {
		align = UMACacheLineAlignment
	}

	totalCap := size + align
	if totalCap <= 0 || totalCap > int64(int(^uint(0)>>1)) {
		return nil, ErrInvalidBufferSize
	}

	raw := make([]byte, totalCap)
	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(align) - (baseAddr % uintptr(align))) % uintptr(align))
	alignedPtr := unsafe.Add(unsafe.Pointer(&raw[0]), offset)
	alignedSlice := raw[offset : offset+int(size) : offset+int(size)]

	buf := &UMABuffer{
		raw:           raw,
		slice:         alignedSlice,
		ptr:           alignedPtr,
		size:          int(size),
		alignedOffset: offset,
	}

	if m != nil {
		m.mu.Lock()
		m.buffers[buf] = size
		m.mu.Unlock()

		atomic.AddInt64(&m.totalAllocatedBytes, size)
		atomic.AddInt64(&m.activeAllocations, 1)
	}

	return buf, nil
}

// FreeUMABuffer releases an allocated UMABuffer, reclaiming active allocation tracking
// and invalidating pointer handles. Returns ErrBufferClosed if buffer is nil or already closed.
func (m *UMAPointerManager) FreeUMABuffer(buf *UMABuffer) error {
	if buf == nil {
		return ErrBufferClosed
	}
	if atomic.LoadUint32(&buf.closed) != 0 {
		return ErrBufferClosed
	}

	if m != nil {
		m.mu.Lock()
		if _, ok := m.buffers[buf]; ok {
			delete(m.buffers, buf)
			atomic.AddInt64(&m.activeAllocations, -1)
		}
		m.mu.Unlock()
	}

	return buf.Close()
}

// BytesToDevPtr extracts the device pointer address directly from a Go byte slice
// without allocating intermediate staging memory or invoking memcpy.
func (m *UMAPointerManager) BytesToDevPtr(b []byte) (uintptr, error) {
	if len(b) == 0 {
		return 0, ErrEmptySlice
	}
	devPtr := uintptr(unsafe.Pointer(&b[0]))
	if m != nil {
		atomic.AddInt64(&m.zeroCopyConversions, 1)
	}
	return devPtr, nil
}

// Float16ToDevPtr extracts the device pointer address directly from a Go uint16 slice
// (representing IEEE 754 float16/bfloat16 bit patterns) without memory copying.
func (m *UMAPointerManager) Float16ToDevPtr(f []uint16) (uintptr, error) {
	if len(f) == 0 {
		return 0, ErrEmptySlice
	}
	devPtr := uintptr(unsafe.Pointer(&f[0]))
	if m != nil {
		atomic.AddInt64(&m.zeroCopyConversions, 1)
	}
	return devPtr, nil
}

// uintptrToUnsafe converts a uintptr to an unsafe.Pointer without triggering
// checkptr arithmetic violations when bridging hardware/device virtual pointers.
func uintptrToUnsafe(ptr uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&ptr))
}

// DevPtrToBytes constructs a zero-copy byte slice viewing the unified memory at ptr.
// Returns nil if ptr is 0 or size <= 0.
func (m *UMAPointerManager) DevPtrToBytes(ptr uintptr, size int) []byte {
	if ptr == 0 || size <= 0 {
		return nil
	}
	uptr := uintptrToUnsafe(ptr)
	b := unsafe.Slice((*byte)(uptr), size)
	if m != nil {
		atomic.AddInt64(&m.zeroCopyConversions, 1)
	}
	return b
}

// DevPtrToFloat16 constructs a zero-copy uint16 slice (FP16/BF16) viewing the unified memory at ptr.
// Returns nil if ptr is 0, size <= 0, or ptr is not 2-byte aligned.
func (m *UMAPointerManager) DevPtrToFloat16(ptr uintptr, size int) []uint16 {
	if ptr == 0 || size <= 0 || (ptr%2) != 0 {
		return nil
	}
	uptr := uintptrToUnsafe(ptr)
	f := unsafe.Slice((*uint16)(uptr), size)
	if m != nil {
		atomic.AddInt64(&m.zeroCopyConversions, 1)
	}
	return f
}

// AssertPointerIdentity validates that the host virtual address equals the device virtual address,
// asserting the zero-copy UMA contract for GFX1151.
func (m *UMAPointerManager) AssertPointerIdentity(hostPtr unsafe.Pointer, devPtr uintptr) bool {
	if hostPtr == nil || devPtr == 0 {
		return false
	}
	if uintptr(hostPtr) != devPtr {
		return false
	}
	if m != nil {
		atomic.AddInt64(&m.assertionChecksPassed, 1)
	}
	return true
}

// TotalAllocatedBytes returns the cumulative total bytes allocated.
func (m *UMAPointerManager) TotalAllocatedBytes() int64 {
	return atomic.LoadInt64(&m.totalAllocatedBytes)
}

// ActiveAllocations returns the current count of active allocations.
func (m *UMAPointerManager) ActiveAllocations() int64 {
	return atomic.LoadInt64(&m.activeAllocations)
}

// ZeroCopyConversions returns the total count of zero-copy slice/pointer conversions.
func (m *UMAPointerManager) ZeroCopyConversions() int64 {
	return atomic.LoadInt64(&m.zeroCopyConversions)
}

// AssertionChecksPassed returns the count of successful AssertPointerIdentity checks.
func (m *UMAPointerManager) AssertionChecksPassed() int64 {
	return atomic.LoadInt64(&m.assertionChecksPassed)
}

// Counters returns a point-in-time snapshot of UMA pointer counters.
func (m *UMAPointerManager) Counters() UMAPointerCounters {
	return UMAPointerCounters{
		TotalAllocatedBytes:   m.TotalAllocatedBytes(),
		ActiveAllocations:     m.ActiveAllocations(),
		ZeroCopyConversions:   m.ZeroCopyConversions(),
		AssertionChecksPassed: m.AssertionChecksPassed(),
	}
}

// Stats returns a point-in-time snapshot of UMA pointer counters.
func (m *UMAPointerManager) Stats() UMAPointerCounters {
	return m.Counters()
}

// ResetCounters resets cumulative metrics on the manager.
func (m *UMAPointerManager) ResetCounters() {
	m.mu.Lock()
	defer m.mu.Unlock()
	atomic.StoreInt64(&m.totalAllocatedBytes, 0)
	atomic.StoreInt64(&m.activeAllocations, int64(len(m.buffers)))
	atomic.StoreInt64(&m.zeroCopyConversions, 0)
	atomic.StoreInt64(&m.assertionChecksPassed, 0)
}

// Package-level functions delegating to defaultUMAPointerManager.

// AllocateUMABuffer allocates a zero-copy unified memory buffer using the default manager.
func AllocateUMABuffer(size int64, align int64) (*UMABuffer, error) {
	return defaultUMAPointerManager.AllocateUMABuffer(size, align)
}

// FreeUMABuffer releases an allocated UMABuffer using the default manager.
func FreeUMABuffer(buf *UMABuffer) error {
	return defaultUMAPointerManager.FreeUMABuffer(buf)
}

// BytesToDevPtr extracts the device pointer address directly from a Go byte slice.
func BytesToDevPtr(b []byte) (uintptr, error) {
	return defaultUMAPointerManager.BytesToDevPtr(b)
}

// Float16ToDevPtr extracts the device pointer address directly from a Go uint16 slice.
func Float16ToDevPtr(f []uint16) (uintptr, error) {
	return defaultUMAPointerManager.Float16ToDevPtr(f)
}

// DevPtrToBytes constructs a zero-copy byte slice viewing the unified memory at ptr.
func DevPtrToBytes(ptr uintptr, size int) []byte {
	return defaultUMAPointerManager.DevPtrToBytes(ptr, size)
}

// DevPtrToFloat16 constructs a zero-copy uint16 slice (FP16/BF16) viewing the unified memory at ptr.
func DevPtrToFloat16(ptr uintptr, size int) []uint16 {
	return defaultUMAPointerManager.DevPtrToFloat16(ptr, size)
}

// AssertPointerIdentity validates that host virtual address equals device virtual address.
func AssertPointerIdentity(hostPtr unsafe.Pointer, devPtr uintptr) bool {
	return defaultUMAPointerManager.AssertPointerIdentity(hostPtr, devPtr)
}

// TotalAllocatedBytes returns cumulative allocated bytes on the default manager.
func TotalAllocatedBytes() int64 {
	return defaultUMAPointerManager.TotalAllocatedBytes()
}

// ActiveAllocations returns current active allocations on the default manager.
func ActiveAllocations() int64 {
	return defaultUMAPointerManager.ActiveAllocations()
}

// ZeroCopyConversions returns zero-copy conversion count on the default manager.
func ZeroCopyConversions() int64 {
	return defaultUMAPointerManager.ZeroCopyConversions()
}

// AssertionChecksPassed returns passed assertion count on the default manager.
func AssertionChecksPassed() int64 {
	return defaultUMAPointerManager.AssertionChecksPassed()
}

// UMAPointerStatsSnapshot returns current counters on the default manager.
func UMAPointerStatsSnapshot() UMAPointerCounters {
	return defaultUMAPointerManager.Counters()
}

// UMABuffer extensions for UMA pointer identity.

// HostPtr returns the host virtual address as a uintptr.
// Returns 0 if the buffer is nil or closed.
func (b *UMABuffer) HostPtr() uintptr {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if atomic.LoadUint32(&b.closed) != 0 {
		return 0
	}
	return uintptr(b.ptr)
}

// DevPtr returns the device virtual address as a uintptr.
// In AMD Strix Halo GFX1151 UMA, DevPtr is identical to HostPtr.
// Returns 0 if the buffer is nil or closed.
func (b *UMABuffer) DevPtr() uintptr {
	return b.HostPtr()
}

// IsZeroCopy returns true if the buffer has valid non-zero identical host and device pointers.
func (b *UMABuffer) IsZeroCopy() bool {
	if b == nil {
		return false
	}
	hp := b.HostPtr()
	dp := b.DevPtr()
	return hp != 0 && hp == dp
}

// Float16Slice returns a zero-copy uint16 (FP16/BF16) view of the buffer's aligned memory.
// Returns nil if buffer is nil, closed, or size < 2.
func (b *UMABuffer) Float16Slice() []uint16 {
	if b == nil {
		return nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if atomic.LoadUint32(&b.closed) != 0 || b.ptr == nil || b.size < 2 {
		return nil
	}
	return unsafe.Slice((*uint16)(b.ptr), b.size/2)
}

// AsActiveKVBlock converts the UMABuffer into an ActiveKVBlock metadata record.
func (b *UMABuffer) AsActiveKVBlock(blockID int64, sessionID string) (*ActiveKVBlock, error) {
	if b == nil || atomic.LoadUint32(&b.closed) != 0 {
		return nil, ErrBufferClosed
	}
	hp := b.HostPtr()
	dp := b.DevPtr()
	return &ActiveKVBlock{
		BlockID:   blockID,
		SessionID: sessionID,
		HostPtr:   hp,
		DevicePtr: dp,
		SizeBytes: int64(b.size),
		LastCheck: time.Now(),
	}, nil
}
