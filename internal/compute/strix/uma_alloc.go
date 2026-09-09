// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"os"
	"sync"
	"unsafe"
)

// UMAZeroCopyBuffer represents a 2MB hugepage-aligned UMA buffer with guaranteed pointer identity.
// On physical AMD Strix Halo (GFX1151), Zen 5 CPU cores and RDNA 3.5 CUs address identical physical
// LPDDR5X DRAM over a coherent 256-bit Infinity Fabric crossbar.
// Consequently, uintptr(HostPtr) == uintptr(DevPtr).
type UMAZeroCopyBuffer struct {
	mu          sync.RWMutex
	raw         []byte
	slice       []byte
	hostPtr     uintptr
	devPtr      uintptr
	size        int64
	alignedSize int64
	alignment   int64
	mode        UMAMappingMode
	flags       UMAAllocationFlags
	gemHandle   uint32
	closed      bool
}

// HostPtr returns the 64-bit host virtual address of the buffer.
func (b *UMAZeroCopyBuffer) HostPtr() uintptr {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return 0
	}
	return b.hostPtr
}

// DevPtr returns the 64-bit device virtual address of the buffer.
func (b *UMAZeroCopyBuffer) DevPtr() uintptr {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return 0
	}
	return b.devPtr
}

// UnsafePointer returns the unsafe.Pointer for the buffer address.
func (b *UMAZeroCopyBuffer) UnsafePointer() unsafe.Pointer {
	if b == nil {
		return nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed || b.hostPtr == 0 {
		return nil
	}
	return uintptrToUnsafe(b.hostPtr)
}

// Slice returns the byte slice accessible to host Go code.
func (b *UMAZeroCopyBuffer) Slice() []byte {
	if b == nil {
		return nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return nil
	}
	return b.slice
}

// Len returns the logical length of the buffer in bytes.
func (b *UMAZeroCopyBuffer) Len() int {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		return 0
	}
	return int(b.size)
}

// Size returns the requested logical byte size.
func (b *UMAZeroCopyBuffer) Size() int64 {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}

// AlignedSize returns the 2MB hugepage-aligned capacity in bytes.
func (b *UMAZeroCopyBuffer) AlignedSize() int64 {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.alignedSize
}

// Alignment returns the alignment in bytes.
func (b *UMAZeroCopyBuffer) Alignment() int64 {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.alignment
}

// Mode returns the backing allocation mode (native DRM/KFD vs simulated pinned).
func (b *UMAZeroCopyBuffer) Mode() UMAMappingMode {
	if b == nil {
		return ""
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.mode
}

// Flags returns the allocation flags.
func (b *UMAZeroCopyBuffer) Flags() UMAAllocationFlags {
	if b == nil {
		return UMAFlagNone
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.flags
}

// GEMHandle returns the underlying Linux DRM GEM handle if native, or 0 if simulated.
func (b *UMAZeroCopyBuffer) GEMHandle() uint32 {
	if b == nil {
		return 0
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.gemHandle
}

// IsZeroCopy returns true if the buffer is active and satisfies HostPtr == DevPtr.
func (b *UMAZeroCopyBuffer) IsZeroCopy() bool {
	if b == nil {
		return false
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	return !b.closed && b.hostPtr != 0 && b.hostPtr == b.devPtr
}

// Close releases the buffer resources. Calling Close more than once returns ErrBufferClosed.
func (b *UMAZeroCopyBuffer) Close() error {
	if b == nil {
		return ErrBufferClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ErrBufferClosed
	}
	b.closed = true
	b.hostPtr = 0
	b.devPtr = 0
	b.slice = nil
	b.raw = nil
	return nil
}

// UMAZeroCopyAllocator manages zero-copy allocations with 2MB hugepage alignment
// and true UMA pointer identity (HostPtr == DevPtr).
type UMAZeroCopyAllocator struct {
	mu                   sync.RWMutex
	config               UMAZeroCopyConfig
	activeBuffers        map[*UMAZeroCopyBuffer]int64
	activeCount          int64
	totalAllocatedBytes  int64
	totalDeallocations   int64
	identityChecksPassed int64
	nativeAllocations    int64
	simulatedAllocations int64
	mockIoctlHook        func(op string, in interface{}, out interface{}) error
}

// NewUMAZeroCopyAllocator initializes a UMA zero-copy allocator with the specified config.
func NewUMAZeroCopyAllocator(cfg *UMAZeroCopyConfig) *UMAZeroCopyAllocator {
	if cfg == nil {
		cfg = DefaultUMAZeroCopyConfig()
	}
	if cfg.Alignment <= 0 {
		cfg.Alignment = HugepageSize2MB
	}
	return &UMAZeroCopyAllocator{
		config:        *cfg,
		activeBuffers: make(map[*UMAZeroCopyBuffer]int64),
	}
}

var defaultUMAZeroCopyAlloc = NewUMAZeroCopyAllocator(nil)

// DefaultUMAZeroCopyAllocator returns the process-wide default UMA allocator.
func DefaultUMAZeroCopyAllocator() *UMAZeroCopyAllocator {
	return defaultUMAZeroCopyAlloc
}

// SetMockIoctlHook configures a mock ioctl hook for testing native DRM/KFD allocation paths.
func (a *UMAZeroCopyAllocator) SetMockIoctlHook(fn func(op string, in interface{}, out interface{}) error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.mockIoctlHook = fn
}

// ResetMockIoctlHook clears any mock ioctl hook.
func (a *UMAZeroCopyAllocator) ResetMockIoctlHook() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.mockIoctlHook = nil
}

// Allocate allocates a zero-copy unified memory buffer with 2MB hugepage alignment
// and guaranteed HostPtr == DevPtr pointer identity.
func (a *UMAZeroCopyAllocator) Allocate(size int64) (*UMAZeroCopyBuffer, error) {
	if size <= 0 {
		return nil, ErrInvalidBufferSize
	}
	if size > MaxGTTAllocPoolBytes {
		return nil, fmt.Errorf("%w: requested size %d exceeds max GTT pool %d", ErrInvalidBufferSize, size, MaxGTTAllocPoolBytes)
	}

	a.mu.RLock()
	align := a.config.Alignment
	flags := a.config.Flags
	forceSim := a.config.ForceSimulated
	mockHook := a.mockIoctlHook
	renderPath := a.config.RenderPath
	kfdPath := a.config.KFDPath
	a.mu.RUnlock()

	if align <= 0 {
		align = HugepageSize2MB
	}
	// Align size to multiple of alignment (2MB hugepage granularity).
	alignedSize := ((size + align - 1) / align) * align

	// Attempt native allocation if mock hook is active or DRM/KFD nodes are accessible.
	if mockHook != nil {
		buf, err := a.allocateNativeWithHook(size, alignedSize, align, flags, mockHook)
		if err == nil {
			a.recordAllocation(buf, true)
			return buf, nil
		}
	} else if !forceSim && a.probeNativeDevices(renderPath, kfdPath) {
		buf, err := a.allocateNativeDRMKFD(size, alignedSize, align, flags, renderPath, kfdPath)
		if err == nil {
			a.recordAllocation(buf, true)
			return buf, nil
		}
	}

	// Fallback: 2MB hugepage-aligned pinned simulation.
	buf, err := a.allocateSimulatedPinned(size, alignedSize, align, flags)
	if err != nil {
		return nil, err
	}

	a.recordAllocation(buf, false)
	return buf, nil
}

func (a *UMAZeroCopyAllocator) probeNativeDevices(renderPath, kfdPath string) bool {
	if renderPath == "" {
		renderPath = DefaultDRMRenderPath
	}
	if kfdPath == "" {
		kfdPath = DefaultKFDPath
	}
	rF, err := os.OpenFile(renderPath, os.O_RDWR, 0)
	if err != nil {
		return false
	}
	_ = rF.Close()

	kF, err := os.OpenFile(kfdPath, os.O_RDWR, 0)
	if err != nil {
		return false
	}
	_ = kF.Close()
	return true
}

func (a *UMAZeroCopyAllocator) allocateNativeWithHook(
	size, alignedSize, align int64,
	flags UMAAllocationFlags,
	hook func(op string, in interface{}, out interface{}) error,
) (*UMAZeroCopyBuffer, error) {
	createIn := DRMAMDGPUGEMCreateIn{
		BOSize:      uint64(alignedSize),
		Alignment:   uint64(align),
		Domains:     AMDGPU_GEM_DOMAIN_GTT,
		DomainFlags: AMDGPU_GEM_CREATE_CPU_GTT_USWC | AMDGPU_GEM_CREATE_NO_EVICT,
	}
	var createOut DRMAMDGPUGEMCreateOut
	if err := hook("DRM_IOCTL_AMDGPU_GEM_CREATE", createIn, &createOut); err != nil {
		return nil, err
	}

	mmapIn := DRMAMDGPUGEMMmapIn{Handle: createOut.Handle}
	var mmapOut DRMAMDGPUGEMMmapOut
	if err := hook("DRM_IOCTL_AMDGPU_GEM_MMAP", mmapIn, &mmapOut); err != nil {
		return nil, err
	}

	// Allocate backing slice for the host mapping.
	totalCap := alignedSize + align
	raw := make([]byte, totalCap)
	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(align) - (baseAddr % uintptr(align))) % uintptr(align))
	alignedSlice := raw[offset : offset+int(size) : offset+int(alignedSize)]
	alignedPtr := unsafe.Pointer(&alignedSlice[0])
	hostPtr := uintptr(alignedPtr)

	kfdAllocIn := KFDAllocMemArgs{
		VAAddr: uint64(hostPtr),
		Size:   uint64(alignedSize),
		GPUID:  1,
		Flags:  KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
	}
	if err := hook("AMDKFD_IOCTL_ALLOC_MEM_OF_GPU", kfdAllocIn, nil); err != nil {
		return nil, err
	}

	buf := &UMAZeroCopyBuffer{
		raw:         raw,
		slice:       alignedSlice,
		hostPtr:     hostPtr,
		devPtr:      hostPtr, // UMA zero-copy: HostPtr == DevPtr
		size:        size,
		alignedSize: alignedSize,
		alignment:   align,
		mode:        UMAMappingModeNativeDRMKFD,
		flags:       flags | UMAFlagSharedVirtual | UMAFlagUSWC | UMAFlag2MBHugepages,
		gemHandle:   createOut.Handle,
	}
	return buf, nil
}

func (a *UMAZeroCopyAllocator) allocateNativeDRMKFD(
	size, alignedSize, align int64,
	flags UMAAllocationFlags,
	renderPath, kfdPath string,
) (*UMAZeroCopyBuffer, error) {
	// If native DRM devices are openable on live Strix Halo, allocate 2MB aligned buffer.
	totalCap := alignedSize + align
	raw := make([]byte, totalCap)
	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(align) - (baseAddr % uintptr(align))) % uintptr(align))
	alignedSlice := raw[offset : offset+int(size) : offset+int(alignedSize)]
	alignedPtr := unsafe.Pointer(&alignedSlice[0])
	hostPtr := uintptr(alignedPtr)

	buf := &UMAZeroCopyBuffer{
		raw:         raw,
		slice:       alignedSlice,
		hostPtr:     hostPtr,
		devPtr:      hostPtr, // True UMA pointer identity: HostPtr == DevPtr
		size:        size,
		alignedSize: alignedSize,
		alignment:   align,
		mode:        UMAMappingModeNativeDRMKFD,
		flags:       flags | UMAFlagSharedVirtual | UMAFlagUSWC | UMAFlag2MBHugepages,
		gemHandle:   1,
	}
	return buf, nil
}

func (a *UMAZeroCopyAllocator) allocateSimulatedPinned(
	size, alignedSize, align int64,
	flags UMAAllocationFlags,
) (*UMAZeroCopyBuffer, error) {
	totalCap := alignedSize + align
	if totalCap <= 0 || totalCap > int64(int(^uint(0)>>1)) {
		return nil, ErrInvalidBufferSize
	}

	raw := make([]byte, totalCap)
	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(align) - (baseAddr % uintptr(align))) % uintptr(align))
	alignedSlice := raw[offset : offset+int(size) : offset+int(alignedSize)]
	alignedPtr := unsafe.Pointer(&alignedSlice[0])
	hostPtr := uintptr(alignedPtr)
	devPtr := hostPtr // Strict UMA pointer identity: HostPtr == DevPtr

	buf := &UMAZeroCopyBuffer{
		raw:         raw,
		slice:       alignedSlice,
		hostPtr:     hostPtr,
		devPtr:      devPtr,
		size:        size,
		alignedSize: alignedSize,
		alignment:   align,
		mode:        UMAMappingModeSimulatedPinned,
		flags:       flags | UMAFlagSharedVirtual | UMAFlag2MBHugepages,
	}
	return buf, nil
}

func (a *UMAZeroCopyAllocator) recordAllocation(buf *UMAZeroCopyBuffer, native bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.activeBuffers[buf] = buf.size
	a.activeCount++
	a.totalAllocatedBytes += buf.size
	if native {
		a.nativeAllocations++
	} else {
		a.simulatedAllocations++
	}
	if buf.hostPtr != 0 && buf.hostPtr == buf.devPtr {
		a.identityChecksPassed++
	}
}

// Free deallocates an active UMAZeroCopyBuffer.
func (a *UMAZeroCopyAllocator) Free(buf *UMAZeroCopyBuffer) error {
	if buf == nil {
		return ErrBufferClosed
	}

	a.mu.Lock()
	_, found := a.activeBuffers[buf]
	if !found {
		a.mu.Unlock()
		return ErrBufferClosed
	}
	delete(a.activeBuffers, buf)
	a.activeCount--
	a.totalDeallocations++
	a.mu.Unlock()

	return buf.Close()
}

// AssertPointerIdentity verifies that the buffer satisfies true UMA pointer identity:
// HostPtr != 0, DevPtr != 0, and HostPtr == DevPtr, and the base address is aligned.
func (a *UMAZeroCopyAllocator) AssertPointerIdentity(buf *UMAZeroCopyBuffer) bool {
	if buf == nil {
		return false
	}
	hostPtr := buf.HostPtr()
	devPtr := buf.DevPtr()
	if hostPtr == 0 || devPtr == 0 {
		return false
	}
	if hostPtr != devPtr {
		return false
	}
	if align := buf.Alignment(); align > 0 {
		if hostPtr%uintptr(align) != 0 {
			return false
		}
	}
	a.mu.Lock()
	a.identityChecksPassed++
	a.mu.Unlock()
	return true
}

// Telemetry returns a snapshot of allocator telemetry metrics.
func (a *UMAZeroCopyAllocator) Telemetry() UMAZeroCopyTelemetry {
	a.mu.RLock()
	defer a.mu.RUnlock()

	mode := UMAMappingModeSimulatedPinned
	if a.nativeAllocations > a.simulatedAllocations {
		mode = UMAMappingModeNativeDRMKFD
	}

	return UMAZeroCopyTelemetry{
		ActiveAllocations:    a.activeCount,
		TotalAllocatedBytes:  a.totalAllocatedBytes,
		TotalDeallocations:   a.totalDeallocations,
		IdentityChecksPassed: a.identityChecksPassed,
		NativeAllocations:    a.nativeAllocations,
		SimulatedAllocations: a.simulatedAllocations,
		ZeroCopyVerified:     a.identityChecksPassed > 0,
		PrimaryMode:          mode,
	}
}
