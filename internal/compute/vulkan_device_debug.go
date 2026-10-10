//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo !v41_indexer_witness LDFLAGS: -L${SRCDIR} -lfakvulkan
#cgo v41_indexer_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_indexer_witness
#include <stdlib.h>
#include "vulkan_backend.h"

// Issue-local restore transaction ABI. The stable public header remains owned by
// the arena/recovery leaves; this adapter is intentionally private to the Go shim.
void *fvk_malloc_weight(size_t bytes, uint64_t max_arena_bytes);
int fvk_restore_begin(size_t max_bytes, size_t max_entries);
int fvk_restore_add(void *dst, size_t dst_offset, const void *src, size_t bytes);
int fvk_restore_submit(void);
void fvk_restore_finish(void);
void fvk_restore_abort(void);
int fvk_restore_active(void);
void fvk_debug_restore_fail_after_submits(int successful_submits);
int fvk_phase_performance_query_available(void);
int fvk_phase_counter_count(void);
int fvk_phase_counter_describe(int index, char* name, size_t name_len, char* unit, size_t unit_len, int* scope);
// Pure test adapter for the production compatible-memory-type selector.
int fvk_debug_memory_type_selection(uint32_t type_bits, uint32_t want,
    uint32_t type_count, const uint32_t* flags, const int* results,
    const uint32_t* heap_indices, uint32_t heap_count, const uint64_t* heap_sizes,
    uint64_t allocation_size, int try_all, uint32_t* attempts, uint32_t* attempt_count, uint32_t* selected_type,
    int* published_handle);
int fvk_debug_select_q8_gateup_coop(int enabled);
int fvk_debug_select_q2k_matvec(int enabled);
int fvk_debug_select_iq_matvec(int fmt, int enabled);
*/
import "C"
import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"unsafe"
)

// VulkanRestoreLimits bounds both axes that can grow a recovery submit. Bytes
// bound the one reusable host-visible staging allocation; entries bound command
// recording even when a group contains many tiny immutable objects.
type VulkanRestoreLimits struct {
	MaxBatchBytes   int
	MaxBatchEntries int
}

// VulkanImmutableResidencySource is one source-bound immutable restore object.
// Binding is the stable identity supplied by the model/residency owner (for
// example, a checkpoint tensor path plus source digest); an empty or duplicate
// binding is rejected before any Vulkan allocation.
type VulkanImmutableResidencySource struct {
	Binding string
	Bytes   []byte
}

// VulkanRestoreReceipt describes only work observed by this restore call.
// Published is the ownership boundary: false means the caller received no
// destination handles, even when earlier bounded submits completed physically.
type VulkanRestoreReceipt struct {
	RequestedObjects int
	RequestedBytes   uint64
	SubmittedBytes   uint64
	Submits          uint64
	PeakStagingBytes uint64
	Status           int
	Published        bool
}

// VulkanRestoreImmutableResidencyGroup recreates one immutable residency group
// into fresh transaction-owned buffers. Sources are consumed in slice order and
// must remain immutable for this synchronous call. A source larger than the byte
// cap is split deterministically; each submit is also capped by entry count.
//
// This is the issue-local adapter between #11288 and #12217: #11288 calls it only
// after creating a fresh Vulkan context, and every destination is allocated from
// #12217's immutable-weight arena. On any interruption, all destinations are
// retired and nil is returned, so partially restored bytes can never become
// visible through this API.
func (v *vulkanBackend) VulkanRestoreImmutableResidencyGroup(ctx context.Context, sources []VulkanImmutableResidencySource, limits VulkanRestoreLimits) (buffers []*vulkanBuf, receipt VulkanRestoreReceipt, err error) {
	receipt.RequestedObjects = len(sources)
	if ctx == nil {
		return nil, receipt, fmt.Errorf("compute: Vulkan restore requires a context")
	}
	if len(sources) == 0 {
		return nil, receipt, fmt.Errorf("compute: Vulkan restore requires at least one immutable object")
	}
	if limits.MaxBatchBytes < 4 || limits.MaxBatchBytes%4 != 0 {
		return nil, receipt, fmt.Errorf("compute: Vulkan restore byte cap must be a positive multiple of 4")
	}
	if limits.MaxBatchEntries <= 0 {
		return nil, receipt, fmt.Errorf("compute: Vulkan restore entry cap must be positive")
	}
	bindings := make(map[string]struct{}, len(sources))
	for i, source := range sources {
		if strings.TrimSpace(source.Binding) == "" {
			return nil, receipt, fmt.Errorf("compute: Vulkan immutable object %d has no source binding", i)
		}
		if _, duplicate := bindings[source.Binding]; duplicate {
			return nil, receipt, fmt.Errorf("compute: Vulkan immutable source binding %q is duplicated", source.Binding)
		}
		bindings[source.Binding] = struct{}{}
		src := source.Bytes
		if len(src) == 0 || len(src)%4 != 0 {
			return nil, receipt, fmt.Errorf("compute: Vulkan immutable object %d size must be a positive multiple of 4", i)
		}
		if ^uint64(0)-receipt.RequestedBytes < uint64(len(src)) {
			return nil, receipt, fmt.Errorf("compute: Vulkan restore byte count overflows receipt")
		}
		receipt.RequestedBytes += uint64(len(src))
	}
	if err := ctx.Err(); err != nil {
		return nil, receipt, fmt.Errorf("compute: Vulkan restore cancelled before allocation: %w", err)
	}

	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	owned := make([]*vulkanBuf, 0, len(sources))
	cleanup := true
	defer func() {
		if cleanup {
			for _, b := range owned {
				if b != nil && b.ptr != nil {
					C.fvk_free(b.ptr)
					b.ptr = nil
				}
			}
		}
	}()
	// Abort/discard any recorded command before retiring its destination
	// buffers. Defer order is load-bearing here: this runs before cleanup above.
	defer C.fvk_restore_abort()
	arenaLimit := v.totalMem
	if v.budgetBytes > 0 {
		arenaLimit = v.budgetBytes
	}
	var maxArenaBytes C.uint64_t
	if arenaLimit > 0 {
		maxArenaBytes = C.uint64_t(arenaLimit)
	}
	for i, source := range sources {
		src := source.Bytes
		if err := ctx.Err(); err != nil {
			return nil, receipt, fmt.Errorf("compute: Vulkan restore cancelled allocating object %d: %w", i, err)
		}
		p := C.fvk_malloc_weight(C.size_t(len(src)), maxArenaBytes)
		b := &vulkanBuf{ptr: unsafe.Pointer(p), n: len(src), class: MemoryWeights}
		if p == nil || !v.debugBufferDeviceLocal(b) {
			if p != nil {
				C.fvk_free(p)
			}
			return nil, receipt, fmt.Errorf("compute: Vulkan restore could not allocate device-local object %d (%d bytes)", i, len(src))
		}
		owned = append(owned, b)
	}
	if status := int(C.fvk_restore_begin(C.size_t(limits.MaxBatchBytes), C.size_t(limits.MaxBatchEntries))); status != 0 {
		receipt.Status = status
		return nil, receipt, fmt.Errorf("compute: Vulkan restore transaction begin failed with status %d", status)
	}
	// The transaction allocates the full declared staging capacity up front, so
	// report that memory peak rather than only the payload high-water mark.
	receipt.PeakStagingBytes = uint64(limits.MaxBatchBytes)

	batchUsed, batchPayload, batchEntries := 0, 0, 0
	submit := func() error {
		if batchEntries == 0 {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("compute: Vulkan restore cancelled before submit %d: %w", receipt.Submits+1, err)
		}
		status := int(C.fvk_restore_submit())
		if status != 0 {
			receipt.Status = status
			return fmt.Errorf("compute: Vulkan restore submit %d failed with status %d", receipt.Submits+1, status)
		}
		receipt.Submits++
		receipt.SubmittedBytes += uint64(batchPayload)
		batchUsed, batchPayload, batchEntries = 0, 0, 0
		return nil
	}
	for objectIndex, source := range sources {
		src := source.Bytes
		for sourceOffset := 0; sourceOffset < len(src); {
			if err := ctx.Err(); err != nil {
				return nil, receipt, fmt.Errorf("compute: Vulkan restore cancelled at object %d offset %d: %w", objectIndex, sourceOffset, err)
			}
			aligned := (batchUsed + 3) &^ 3
			if batchEntries == limits.MaxBatchEntries || aligned == limits.MaxBatchBytes {
				if err := submit(); err != nil {
					return nil, receipt, err
				}
				aligned = 0
			}
			room := limits.MaxBatchBytes - aligned
			chunk := len(src) - sourceOffset
			if chunk > room {
				chunk = room
			}
			chunk &^= 3
			if chunk == 0 {
				if err := submit(); err != nil {
					return nil, receipt, err
				}
				continue
			}
			status := int(C.fvk_restore_add(
				owned[objectIndex].ptr,
				C.size_t(sourceOffset),
				unsafe.Pointer(&src[sourceOffset]),
				C.size_t(chunk),
			))
			if status != 0 {
				receipt.Status = status
				return nil, receipt, fmt.Errorf("compute: Vulkan restore add object %d offset %d failed with status %d", objectIndex, sourceOffset, status)
			}
			batchUsed = aligned + chunk
			batchPayload += chunk
			batchEntries++
			sourceOffset += chunk
		}
	}
	if err := submit(); err != nil {
		return nil, receipt, err
	}
	C.fvk_restore_finish()
	receipt.Published = true
	cleanup = false
	buffers = owned
	return buffers, receipt, nil
}

func (v *vulkanBackend) VulkanDebugReadRestoreBuffer(b *vulkanBuf) []byte {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if b == nil || b.ptr == nil || b.n <= 0 {
		return nil
	}
	out := make([]byte, b.n)
	C.fvk_d2h(unsafe.Pointer(&out[0]), b.ptr, C.size_t(len(out)))
	return out
}

func (v *vulkanBackend) VulkanDebugFreeRestoreBuffers(buffers []*vulkanBuf) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	for _, b := range buffers {
		if b != nil && b.ptr != nil {
			C.fvk_free(b.ptr)
			b.ptr = nil
		}
	}
}

func (v *vulkanBackend) VulkanDebugSetRestoreFailureAfterSubmits(successfulSubmits int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_debug_restore_fail_after_submits(C.int(successfulSubmits))
}

// VulkanDebugSetD2HStagingFailureOnce makes the next device-to-host copy return
// the shim's unattributed staging-allocation failure. It is a one-shot test seam;
// passing false disarms a pending injection during test cleanup.
func (v *vulkanBackend) VulkanDebugSetD2HStagingFailureOnce(enabled bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var value C.int
	if enabled {
		value = 1
	}
	C.fvk_debug_d2h_staging_failure_once(value)
}

func (v *vulkanBackend) VulkanDebugRestoreActive() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_restore_active() != 0
}

// VulkanDRMRenderNode reports the DRM render-node identity of the initialized,
// selected physical device. Unsupported platforms, headers, drivers, or devices
// return unavailable. The pair identifies a node, not a dedicated memory pool;
// callers must independently prove any capacity and host-memory relationship.
func (v *vulkanBackend) VulkanDRMRenderNode() (major, minor uint64, available bool) {
	if v == nil {
		return 0, 0, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var drmMajor, drmMinor C.uint64_t
	if C.fvk_device_drm_render_node(&drmMajor, &drmMinor) == 0 || drmMajor == 0 {
		return 0, 0, false
	}
	return uint64(drmMajor), uint64(drmMinor), true
}

// BackendExecutionSnapshot reports identity and cumulative counters from the
// selected Vulkan backend. Callers must bracket one execution and subtract via
// BackendExecutionDelta; this process-global snapshot is never a receipt.
func (v *vulkanBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.backendExecutionSnapshotLocked()
}

func (v *vulkanBackend) backendExecutionSnapshotLocked() (BackendExecutionSnapshot, error) {
	var name [256]C.char
	var vendorID, deviceID, driverVersion, apiVersion C.uint32_t
	if C.fvk_device_identity(&name[0], 256, &vendorID, &deviceID, &driverVersion, &apiVersion) == 0 {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan physical-device identity is unavailable")
	}
	device := strings.TrimSpace(C.GoString(&name[0]))
	if device == "" {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan physical-device name is unavailable")
	}
	api := uint32(apiVersion)
	runtimeIdentity := fmt.Sprintf("vulkan-%d.%d.%d", api>>22, (api>>12)&0x3ff, api&0xfff)
	driverIdentity := fmt.Sprintf("vendor=0x%04x device=0x%04x driver=0x%08x", uint32(vendorID), uint32(deviceID), uint32(driverVersion))

	dispatch := vulkanDebugDispatchProfileSnapshotLocked()
	if dispatch.Q4KMatmulDispatches > dispatch.ComputeDispatches {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan Q4_K dispatch count exceeds compute total")
	}
	var h2dCount, h2dBytes, d2hCount, d2hBytes, d2dCount, d2dBytes C.uint64_t
	if C.fvk_transfer_counters(&h2dCount, &h2dBytes, &d2hCount, &d2hBytes, &d2dCount, &d2dBytes) == 0 {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan transfer counters are unavailable")
	}
	var allocationLive C.uint64_t
	if C.fvk_device_allocation_snapshot(&allocationLive) == 0 {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan device-allocation accounting is unavailable")
	}
	stageCalls, stageBytes, fallbacks := v.q4kStagedCalls, v.q4kStagedBytes, v.q4kStageFallbacks
	hits, admissions, bypasses := v.homeHits, v.homeMisses, v.homeBypasses
	entries, residentBytes, copiedBytes := len(v.homes), v.homeBytes, v.homeCopied
	if entries < 0 {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan tensor-home entry count is negative")
	}
	for name, value := range map[string]int64{
		"stage calls": stageCalls, "stage bytes": stageBytes, "fallbacks": fallbacks,
		"tensor-home hits": hits, "tensor-home admissions": admissions, "tensor-home bypasses": bypasses,
		"tensor-home resident bytes": residentBytes, "tensor-home copied bytes": copiedBytes,
	} {
		if value < 0 {
			return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan %s counter is negative", name)
		}
	}
	total, free, memoryObserved := v.totalMem, int64(FreeUnknown), v.totalMem > 0
	if memoryObserved && v.haveMemoryBudget {
		var budget, usage, freeBytes C.uint64_t
		if C.fvk_device_local_memory_budget(&budget, &usage, &freeBytes) != 0 {
			free = vulkanCapInt64(freeBytes)
		}
	}
	if memoryObserved && (total <= 0 || free < 0 || free > total) {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan device-memory observation is invalid")
	}

	return BackendExecutionSnapshot{
		Identity: BackendRuntimeIdentity{
			Backend: v.Name(), Device: device, Driver: driverIdentity, Runtime: runtimeIdentity,
		},
		Counters: BackendCounterSnapshot{
			ComputeDispatches: dispatch.ComputeDispatches, Q4KMatmulDispatches: dispatch.Q4KMatmulDispatches,
			OtherDispatches: dispatch.ComputeDispatches - dispatch.Q4KMatmulDispatches,
			DispatchSubmits: dispatch.BatchSubmits + dispatch.OneShotSubmits,
			H2DBytes:        uint64(h2dBytes), H2DCount: uint64(h2dCount),
			D2HBytes: uint64(d2hBytes), D2HCount: uint64(d2hCount),
			D2DCopies: uint64(d2dCount), D2DBytes: uint64(d2dBytes),
			Q4KStageCalls: uint64(stageCalls), Q4KStageBytes: uint64(stageBytes), Fallbacks: uint64(fallbacks),
			TensorHomeHits: uint64(hits), TensorHomeAdmissions: uint64(admissions),
			TensorHomeBypasses: uint64(bypasses), TensorHomeCopiedBytes: uint64(copiedBytes),
		},
		TensorHomeEntries: uint64(entries), TensorHomeResidentBytes: uint64(residentBytes),
		DeviceMemoryTotalBytes: uint64(total), DeviceMemoryFreeBytes: uint64(free), DeviceMemoryObserved: memoryObserved,
		TransferCountersObserved: true, DeviceAllocationLiveBytes: uint64(allocationLive), DeviceAllocationObserved: true,
	}, nil
}

type vulkanExecutionWindow struct {
	backend *vulkanBackend
	before  BackendExecutionSnapshot
	token   uint64
	mu      sync.Mutex
	ended   bool
}

func (v *vulkanBackend) BeginBackendExecutionWindow() (BackendExecutionWindow, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	before, err := v.backendExecutionSnapshotLocked()
	if err != nil {
		return nil, err
	}
	var token C.uint64_t
	if C.fvk_device_allocation_window_begin(&token) == 0 || token == 0 {
		return nil, fmt.Errorf("compute: Vulkan execution observation window is unavailable or already active")
	}
	return &vulkanExecutionWindow{backend: v, before: before, token: uint64(token)}, nil
}

func (w *vulkanExecutionWindow) End() (BackendExecutionObservation, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.ended {
		return BackendExecutionObservation{}, fmt.Errorf("compute: Vulkan execution observation window is stale")
	}
	w.ended = true

	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	after, snapshotErr := w.backend.backendExecutionSnapshotLocked()
	var live, peak C.uint64_t
	ended := C.fvk_device_allocation_window_end(C.uint64_t(w.token), &live, &peak) != 0
	if snapshotErr != nil {
		return BackendExecutionObservation{}, snapshotErr
	}
	if !ended {
		return BackendExecutionObservation{}, fmt.Errorf("compute: Vulkan execution observation window could not be completed")
	}
	return BackendExecutionWindowDelta(w.before, after, uint64(live), uint64(peak))
}

func (v *vulkanBackend) debugBufferHostVisible(b *vulkanBuf) bool {
	return b != nil && b.ptr != nil && C.fvk_debug_buffer_is_host_visible(b.ptr) != 0
}

func (v *vulkanBackend) debugBufferDeviceLocal(b *vulkanBuf) bool {
	return b != nil && b.ptr != nil && C.fvk_debug_buffer_is_device_local(b.ptr) != 0
}

// VulkanBufferBacking describes one live buffer's allocation selection. Part is
// "data" or "scales"; Chunk is -1 for an unchunked tensor. Memory type and heap
// indices are local to the selected device's current lifetime. Flags are raw
// VkMemoryPropertyFlags and VkMemoryHeapFlags, not physical-pool guarantees.
type VulkanBufferBacking struct {
	Part                   string
	Chunk                  int
	RequestedPropertyFlags uint32
	MemoryTypeIndex        uint32
	PropertyFlags          uint32
	HeapIndex              uint32
	HeapFlags              uint32
	HostVisibleFallback    bool
	WeightArenaBound       bool
}

// VulkanTensorBufferBackings observes every data/scale buffer of a live tensor
// owned by v, in data-then-scales order for each chunk. A missing buffer or shim
// observation returns nil, false; partial evidence is never returned. The query
// performs no Vulkan allocation, transfer, or synchronization of device work.
//
// HostVisibleFallback denotes only the shim's successful device-local retry in
// host-visible storage. A false value does not exclude Go-side host-visible
// placement or recovery; inspect RequestedPropertyFlags and PropertyFlags too.
// Arena-bound buffers can share one allocation. These observations cannot
// establish host/device disjointness, reserve future memory, or authorize a
// larger capacity limit. Call outside hot token loops.
func (v *vulkanBackend) VulkanTensorBufferBackings(t Tensor) ([]VulkanBufferBacking, bool) {
	if v == nil || t.be != v {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := t.buf.(*vulkanBuf)
	if !ok || b == nil {
		return nil, false
	}
	var backings []VulkanBufferBacking
	appendBacking := func(ptr unsafe.Pointer, part string, chunk int) bool {
		backing, ok := vulkanBufferBackingLocked(ptr, part, chunk)
		if !ok {
			return false
		}
		backings = append(backings, backing)
		return true
	}
	if len(b.q8Chunks) != 0 {
		if b.ptr != nil || b.scalePtr != nil || t.Dtype != Q8_0 {
			return nil, false
		}
		for i, chunk := range b.q8Chunks {
			if !appendBacking(chunk.ptr, "data", i) || !appendBacking(chunk.scalePtr, "scales", i) {
				return nil, false
			}
		}
	} else {
		if !appendBacking(b.ptr, "data", -1) {
			return nil, false
		}
		if b.scalePtr != nil || t.Dtype == Q8_0 {
			if !appendBacking(b.scalePtr, "scales", -1) {
				return nil, false
			}
		}
	}
	return backings, true
}

// VulkanBufferReservation reports a buffer binding's nominal native length and
// its complete VkDeviceMemory allocation request. ReservationBytes repeats the
// whole allocation size for each binding, including shared weight-arena bindings;
// it is not a per-tensor charge, driver overhead or physical residency measure.
// BufferBytes can exceed a borrowed tensor view's logical extent. AllocationID
// names the memory allocation, not the tensor or buffer binding. It is opaque and
// valid only within the current initialized-device lifetime. Backing identifies
// the part/chunk and selected memory type/heap without changing the older record.
type VulkanBufferReservation struct {
	BufferBytes      uint64
	ReservationBytes uint64
	AllocationID     uint64
	BindingOffset    uint64
	Backing          VulkanBufferBacking
}

// VulkanTensorBufferReservations observes all data/scale bindings in data-then-
// scales order per chunk under vulkanMu. Missing metadata returns nil, false,
// never partial evidence. The caller must retain a live tensor and, for borrowed
// views, its live backing owner; stale aliases cannot be validated from t.be or
// a lock. The current initialized-device lifetime is a precondition, not a
// generation check. Explicit shim reinitialization invalidates this contract.
//
// Nonzero IDs distinguish shim allocations in that lifetime; arena bindings share
// a block ID, pool retention preserves an ID, and new allocations do not reuse
// retired IDs. This supplies no whole-backend inventory or general deduplication
// guarantee. Separate calls are not an atomic snapshot. Do not infer disjoint
// physical pools, sum capacities, authorize headroom or predict future placement.
// Other owner inventories (homes, stages, scratch, pools) are not included here.
//
// The query reads host metadata without allocating/freeing Vulkan storage,
// transferring, flushing batches, fencing device work or changing owners.
// The result can allocate Go memory. Call outside hot token loops. A matching
// rebuilt libfakvulkan is required for the additive reservation symbol.
func (v *vulkanBackend) VulkanTensorBufferReservations(t Tensor) ([]VulkanBufferReservation, bool) {
	if v == nil || t.be != v {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := t.buf.(*vulkanBuf)
	if !ok || b == nil {
		return nil, false
	}
	var backings []VulkanBufferReservation
	appendBacking := func(ptr unsafe.Pointer, part string, chunk int) bool {
		backing, ok := vulkanBufferReservationLocked(ptr, part, chunk)
		if !ok {
			return false
		}
		backings = append(backings, backing)
		return true
	}
	if len(b.q8Chunks) != 0 {
		if b.ptr != nil || b.scalePtr != nil || t.Dtype != Q8_0 {
			return nil, false
		}
		for i, chunk := range b.q8Chunks {
			if !appendBacking(chunk.ptr, "data", i) || !appendBacking(chunk.scalePtr, "scales", i) {
				return nil, false
			}
		}
	} else {
		if !appendBacking(b.ptr, "data", -1) {
			return nil, false
		}
		if b.scalePtr != nil || t.Dtype == Q8_0 {
			if !appendBacking(b.scalePtr, "scales", -1) {
				return nil, false
			}
		}
	}
	return backings, true
}

// The caller holds vulkanMu and proves live current-lifetime ownership. No raw
// handle or borrowed native pointer escapes into the returned value record.
func vulkanBufferReservationLocked(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferReservation, bool) {
	var info C.fvk_buffer_reservation_info
	if ptr == nil || C.fvk_buffer_reservation(ptr, &info) == 0 {
		return VulkanBufferReservation{}, false
	}
	backing, ok := vulkanBufferBackingLocked(ptr, part, chunk)
	if !ok {
		return VulkanBufferReservation{}, false
	}
	return VulkanBufferReservation{
		BufferBytes:      uint64(info.buffer_bytes),
		ReservationBytes: uint64(info.reservation_bytes),
		AllocationID:     uint64(info.allocation_id),
		BindingOffset:    uint64(info.binding_offset),
		Backing:          backing,
	}, true
}

// VulkanQ4KStageBufferBacking describes the backend-owned shared Q4_K dispatch
// stage. BufferBytes is its retained nominal buffer length, not the most recent
// copy length or reserved VkDeviceMemory bytes. Backing uses Part "data", Chunk -1.
// A record supplies no persistent identity and cannot be used to deduplicate memory.
type VulkanQ4KStageBufferBacking struct {
	BufferBytes uint64
	Backing     VulkanBufferBacking
}

// VulkanQ4KStageBufferBackings observes the zero or one retained shared Q4_K
// stage under the same lock as its allocation, growth and retirement. A missing
// stage with zero length is known-empty; inconsistent ownership or unavailable
// backing returns nil, false. A retained stage is reported even when Q4_K staging
// is disabled. Its presence does not establish current dispatch use or contents.
// Known-empty describes this owner, not device health or driver memory release.
// Observations require the owner's current initialized device lifetime; they do
// not validate ownership across shim reinitialization.
//
// This inventory excludes tensors, Q4_K homes, ordinary H2D/D2H and restore
// staging, KV, GDN and other scratch, and pool storage. Separate calls do not
// form an atomic whole-backend snapshot. Buffer lengths and memory type/heap
// flags do not establish reserved sizes, disjoint physical pools or headroom;
// arena/shared-heap deduplication and future placement remain unknown.
// The query performs no Vulkan allocation, transfer, device-work fence or owner
// mutation. Its result may allocate Go memory. Call outside hot token loops.
func (v *vulkanBackend) VulkanQ4KStageBufferBackings() ([]VulkanQ4KStageBufferBacking, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanQ4KStageBufferBackingsLocked(v.q4kStagePtr, v.q4kStageBytes, vulkanBufferBackingLocked)
}

// The caller holds vulkanMu through the backing read so stage growth, reset and
// teardown cannot retire the owned handle in flight. The local reader lets tests
// exercise ownership checks without sending synthetic handles to the C ABI.
func vulkanQ4KStageBufferBackingsLocked(ptr unsafe.Pointer, bufferBytes int64, readBacking func(unsafe.Pointer, string, int) (VulkanBufferBacking, bool)) ([]VulkanQ4KStageBufferBacking, bool) {
	if ptr == nil && bufferBytes == 0 {
		return []VulkanQ4KStageBufferBacking{}, true
	}
	if ptr == nil || bufferBytes <= 0 {
		return nil, false
	}
	backing, ok := readBacking(ptr, "data", -1)
	if !ok {
		return nil, false
	}
	return []VulkanQ4KStageBufferBacking{{BufferBytes: uint64(bufferBytes), Backing: backing}}, true
}

// VulkanQ4KStageBufferReservations observes the zero or one retained backend-owned
// shared Q4_K stage under the same lock as its growth and retirement. A nil handle
// with zero retained bytes is known-empty; inconsistent ownership, a native length
// mismatch or unavailable reservation metadata returns nil, false. A retained
// stage is reported even when staging is disabled. Backing uses Part "data", Chunk -1.
//
// The owner must belong to the current initialized-device lifetime. The native
// reader checks current device availability, but no generation validates an owner
// across shim reinitialization. Known-empty describes only this Go owner; neither
// it nor a retained record proves device health, dispatch use or physical residency.
// ReservationBytes is the complete allocation request, repeated for shared IDs,
// not a per-binding charge. IDs and memory indices are limited to that lifetime.
//
// This excludes tensors, Q4_K homes, ordinary H2D/D2H and restore stages, scratch
// and pools. Separate observations are not an atomic whole-backend inventory or
// proof of disjoint physical pools, capacity headroom or future placement.
// It reads host metadata without Vulkan allocation/free, transfers, batch flushes,
// device-work fences or owner/counter changes. Results can allocate Go memory;
// call outside hot token loops with a matching rebuilt reservation-capable shim.
func (v *vulkanBackend) VulkanQ4KStageBufferReservations() ([]VulkanBufferReservation, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanQ4KStageBufferReservationsLocked(v.q4kStagePtr, v.q4kStageBytes, vulkanBufferReservationLocked)
}

// The caller holds vulkanMu through the read. The injected reader is a complete
// reservation observer, allowing owner controls to be tested without C handles.
func vulkanQ4KStageBufferReservationsLocked(ptr unsafe.Pointer, bufferBytes int64, readReservation func(unsafe.Pointer, string, int) (VulkanBufferReservation, bool)) ([]VulkanBufferReservation, bool) {
	if ptr == nil && bufferBytes == 0 {
		return []VulkanBufferReservation{}, true
	}
	if ptr == nil || bufferBytes <= 0 {
		return nil, false
	}
	reservation, ok := readReservation(ptr, "data", -1)
	if !ok || reservation.BufferBytes != uint64(bufferBytes) {
		return nil, false
	}
	return []VulkanBufferReservation{reservation}, true
}

// VulkanTransferStageBufferBacking describes the shim-global ordinary H2D/D2H
// stage shared by both transfer directions. BufferBytes is its retained nominal
// buffer length, not the last transfer or VkDeviceMemory reservation. Backing has
// Part "data" and Chunk -1. The record provides no persistent allocation identity.
type VulkanTransferStageBufferBacking struct {
	BufferBytes uint64
	Backing     VulkanBufferBacking
}

// VulkanTransferStageBufferBackings observes zero or one retained ordinary
// transfer stage under vulkanMu, the same lock as stage reuse and growth. This
// owner belongs to the process-global shim's selected device, not to the receiver
// or either transfer direction separately. Do not count it once per receiver.
// A coherent absent stage in an initialized shim is known-empty; an unavailable
// shim, inconsistent owner or missing backing returns nil, false.
//
// The observation requires the current initialized device lifetime. The shim has
// no generation token to validate ownership across reinitialization. Presence
// does not prove recent use, contents or device health, and pool trim does not
// retire this stage. It excludes tensors, Q4_K homes/staging, restore staging,
// KV, GDN/other scratch and reusable pools. Separate queries are not an atomic
// whole-backend inventory. Nominal lengths and type/heap flags cannot establish
// physical-pool disjointness, reserved bytes, headroom or future placement.
//
// The query performs no Vulkan allocation, transfer, device-work fence or owner
// mutation. It does not flush pending batches or establish device quiescence;
// its result can allocate Go memory. Call outside hot token loops.
// The additive C accessor requires libfakvulkan rebuilt from matching sources.
func (v *vulkanBackend) VulkanTransferStageBufferBackings() ([]VulkanTransferStageBufferBacking, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var bufferBytes C.uint64_t
	var info C.fvk_buffer_backing_info
	if C.fvk_transfer_stage_backing(&bufferBytes, &info) == 0 {
		return nil, false
	}
	if bufferBytes == 0 {
		return []VulkanTransferStageBufferBacking{}, true
	}
	return []VulkanTransferStageBufferBacking{{
		BufferBytes: uint64(bufferBytes),
		Backing:     vulkanBufferBackingFromInfo(info, "data", -1),
	}}, true
}

// VulkanTransferStageBufferReservations observes zero or one retained ordinary
// H2D/D2H stage under vulkanMu. This is one shim-global owner shared by both
// transfer directions and every nonnil receiver; do not count it per receiver.
// A coherent absent stage in an initialized shim returns a nonnil empty slice,
// true. Unavailable or inconsistent owner/reservation metadata returns nil, false.
// A retained record uses Backing.Part "data" and Backing.Chunk -1.
//
// The current initialized-device lifetime is a precondition; no generation
// validates ownership across reinitialization. Retained metadata remains visible
// under sticky submission failure and proves neither health, quiescence, recent
// use nor contents. Pool trim does not retire this stage. ReservationBytes is the
// whole VkDeviceMemory allocation request, not physical residency, driver overhead
// or a per-binding charge. IDs and memory indices are local to that lifetime.
//
// This excludes tensors, Q4_K homes/staging, restore staging, scratch and pools.
// Separate queries are not an atomic whole-backend inventory or evidence of
// disjoint physical pools, capacity headroom or future placement. The query reads
// host metadata without Vulkan allocation/free, transfers, batch flushes, device
// fences or owner/counter changes. Results may allocate Go memory; call outside
// hot token loops with a matching rebuilt reservation-capable shim.
func (v *vulkanBackend) VulkanTransferStageBufferReservations() ([]VulkanBufferReservation, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanTransferStageBufferReservationsLocked(func() (VulkanBufferReservation, bool) {
		var info C.fvk_buffer_reservation_info
		var backing C.fvk_buffer_backing_info
		if C.fvk_transfer_stage_reservation(&info, &backing) == 0 {
			return VulkanBufferReservation{}, false
		}
		return VulkanBufferReservation{
			BufferBytes:      uint64(info.buffer_bytes),
			ReservationBytes: uint64(info.reservation_bytes),
			AllocationID:     uint64(info.allocation_id),
			BindingOffset:    uint64(info.binding_offset),
			Backing:          vulkanBufferBackingFromInfo(backing, "", 0),
		}, true
	})
}

// The caller holds vulkanMu across the complete native read and value conversion.
// The reader returns unlabelled value metadata, including two zero native records
// for known-empty. Injection tests result shaping without exposing a stage handle.
func vulkanTransferStageBufferReservationsLocked(readReservation func() (VulkanBufferReservation, bool)) ([]VulkanBufferReservation, bool) {
	reservation, ok := readReservation()
	if !ok {
		return nil, false
	}
	if reservation.BufferBytes == 0 {
		if reservation != (VulkanBufferReservation{}) {
			return nil, false
		}
		return []VulkanBufferReservation{}, true
	}
	reservation.Backing.Part, reservation.Backing.Chunk = "data", -1
	return []VulkanBufferReservation{reservation}, true
}

// vulkanBufferBackingLocked requires vulkanMu and a live, owned handle. The shim
// cannot validate an arbitrary or retired pointer; ownership supplies that proof.
func vulkanBufferBackingLocked(ptr unsafe.Pointer, part string, chunk int) (VulkanBufferBacking, bool) {
	var info C.fvk_buffer_backing_info
	if ptr == nil || C.fvk_buffer_backing(ptr, &info) == 0 {
		return VulkanBufferBacking{}, false
	}
	return vulkanBufferBackingFromInfo(info, part, chunk), true
}

func vulkanBufferBackingFromInfo(info C.fvk_buffer_backing_info, part string, chunk int) VulkanBufferBacking {
	return VulkanBufferBacking{
		Part: part, Chunk: chunk,
		RequestedPropertyFlags: uint32(info.requested_property_flags),
		MemoryTypeIndex:        uint32(info.memory_type_index),
		PropertyFlags:          uint32(info.property_flags),
		HeapIndex:              uint32(info.heap_index),
		HeapFlags:              uint32(info.heap_flags),
		HostVisibleFallback:    info.host_visible_fallback != 0,
		WeightArenaBound:       info.weight_arena_bound != 0,
	}
}

// VulkanQ4KHomeBufferBacking describes one backend-owned Q4_K dispatch home copy.
// BufferBytes is its nominal buffer/copy length, not reserved VkDeviceMemory bytes.
// Backing describes the home allocation itself, with Part "data" and Chunk -1.
// Records have no persistent identity and must not be used to deduplicate memory.
type VulkanQ4KHomeBufferBacking struct {
	BufferBytes uint64
	Backing     VulkanBufferBacking
}

// VulkanQ4KHomeBufferBackings observes all currently retained Q4_K home copies
// under the same lock as their admission and retirement. Order is unspecified.
// An empty cache returns an empty slice and true; an invalid owner record or
// unavailable backing returns nil, false, never partial evidence.
//
// This is a backend-cache inventory, not a tensor-liveness or dispatch-use claim:
// homes can outlive their source tensors, and the source keys are not dereferenced
// or exported. It excludes Tensor.buf, shared staging, KV, scratch and pool storage.
// Buffer lengths and type/heap flags do not prove allocation sizes, disjoint physical
// pools, or capacity headroom. Arena and shared-heap deduplication remain unknown.
// The query makes no Vulkan allocation, transfer, device-work fence or cache change;
// constructing the result can allocate Go memory. Call outside hot token loops.
func (v *vulkanBackend) VulkanQ4KHomeBufferBackings() ([]VulkanQ4KHomeBufferBacking, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanQ4KHomeBufferBackingsLocked(v.homes, v.homeBytes, vulkanBufferBackingLocked)
}

// The caller holds vulkanMu through every backing read so freeHomesLocked cannot
// retire a handle in flight. The reader argument keeps enumeration testable without
// passing synthetic handles to the C ABI or changing a process-global query hook.
func vulkanQ4KHomeBufferBackingsLocked(homes map[vulkanQ4KHomeKey]vulkanQ4KHome, residentBytes int64, readBacking func(unsafe.Pointer, string, int) (VulkanBufferBacking, bool)) ([]VulkanQ4KHomeBufferBacking, bool) {
	if residentBytes < 0 {
		return nil, false
	}
	backings := make([]VulkanQ4KHomeBufferBacking, 0, len(homes))
	seen := make(map[unsafe.Pointer]struct{}, len(homes))
	var total int64
	for key, home := range homes {
		if key.src == nil || key.bytes <= 0 || home.ptr == nil || home.bytes != int64(key.bytes) || home.bytes > residentBytes-total {
			return nil, false
		}
		if _, duplicate := seen[home.ptr]; duplicate {
			return nil, false
		}
		seen[home.ptr] = struct{}{}
		backing, ok := readBacking(home.ptr, "data", -1)
		if !ok {
			return nil, false
		}
		backings = append(backings, VulkanQ4KHomeBufferBacking{BufferBytes: uint64(home.bytes), Backing: backing})
		total += home.bytes
	}
	if total != residentBytes {
		return nil, false
	}
	return backings, true
}

// VulkanQ4KHomeBufferReservations observes all retained backend-owned Q4_K homes
// under the same lock as admission and retirement. Order is unspecified. An empty
// cache with zero owned bytes is known-empty; invalid ownership, a native length
// mismatch or unavailable reservation metadata returns nil, false, never a partial
// inventory. Backing uses Part "data", Chunk -1. Source keys are neither read as
// native handles nor exported: homes can outlive their source tensors.
//
// Owners must belong to the current initialized-device lifetime. The native reader
// checks current device availability, but cannot validate stale owners after shim
// reinitialization. Known-empty describes the Go cache, not device health or release
// of driver memory. ReservationBytes repeats the complete allocation request for
// shared AllocationID values; equal IDs do not make distinct owned bindings invalid.
// IDs and memory indices are meaningful only in that initialized-device lifetime.
//
// This excludes tensors, shared Q4_K/ordinary H2D/D2H/restore stages, scratch and
// pools. Separate observations are not an atomic whole-backend inventory and prove
// neither physical residency, disjoint physical pools, capacity headroom nor future
// placement. This reads host metadata without Vulkan allocation/free, transfers,
// batch flushes, device-work fences or owner/counter changes. Results can allocate
// Go memory; call outside hot token loops with a matching reservation-capable shim.
func (v *vulkanBackend) VulkanQ4KHomeBufferReservations() ([]VulkanBufferReservation, bool) {
	if v == nil {
		return nil, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanQ4KHomeBufferReservationsLocked(v.homes, v.homeBytes, vulkanBufferReservationLocked)
}

// The caller holds vulkanMu through every read so retirement cannot invalidate an
// owned handle. Only home.ptr reaches the complete reservation reader, never key.src.
func vulkanQ4KHomeBufferReservationsLocked(homes map[vulkanQ4KHomeKey]vulkanQ4KHome, residentBytes int64, readReservation func(unsafe.Pointer, string, int) (VulkanBufferReservation, bool)) ([]VulkanBufferReservation, bool) {
	if residentBytes < 0 {
		return nil, false
	}
	reservations := make([]VulkanBufferReservation, 0, len(homes))
	seen := make(map[unsafe.Pointer]struct{}, len(homes))
	var total int64
	for key, home := range homes {
		if key.src == nil || key.bytes <= 0 || home.ptr == nil || home.bytes != int64(key.bytes) || home.bytes > residentBytes-total {
			return nil, false
		}
		if _, duplicate := seen[home.ptr]; duplicate {
			return nil, false
		}
		seen[home.ptr] = struct{}{}
		reservation, ok := readReservation(home.ptr, "data", -1)
		if !ok || reservation.BufferBytes != uint64(home.bytes) {
			return nil, false
		}
		reservations = append(reservations, reservation)
		total += home.bytes
	}
	if total != residentBytes {
		return nil, false
	}
	return reservations, true
}

func (v *vulkanBackend) VulkanDebugResidencyBudget() (budgetBytes, dlUsed int64, hostvisN int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.budgetBytes, v.dlUsed, v.hostvisN
}

func (v *vulkanBackend) VulkanDebugSetResidencyBudget(budgetBytes int64) (oldBudgetBytes, oldDLUsed int64, oldHostvisN int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	oldBudgetBytes, oldDLUsed, oldHostvisN = v.budgetBytes, v.dlUsed, v.hostvisN
	v.budgetBytes, v.dlUsed, v.hostvisN = budgetBytes, 0, 0
	return oldBudgetBytes, oldDLUsed, oldHostvisN
}

func (v *vulkanBackend) VulkanDebugRestoreResidencyBudget(budgetBytes, dlUsed int64, hostvisN int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.budgetBytes, v.dlUsed, v.hostvisN = budgetBytes, dlUsed, hostvisN
}

func (v *vulkanBackend) VulkanDebugResourceCaps() (maxBufferBytes, maxStorageBufferRange, maxMemoryAllocationSize int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize
}

func (v *vulkanBackend) VulkanDebugMemoryBudgetAvailable() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.haveMemoryBudget
}

// VulkanDebugTransferBytes returns the process-global payload bytes copied across
// the Vulkan host/device boundary. It is a cumulative observability counter: take
// snapshots around a serialized operation to prove that operation stayed resident.
func (v *vulkanBackend) VulkanDebugTransferBytes() (h2d, d2h uint64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return uint64(C.fvk_h2d_bytes()), uint64(C.fvk_d2h_bytes())
}

func (v *vulkanBackend) VulkanDebugQ4KProfileSnapshot() (enabled bool, deviceCalls, devicePackedBytes, hostVisibleCalls, hostVisiblePackedBytes int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.q4kProfile, v.q4kDeviceCalls, v.q4kDevicePackedBytes, v.q4kHostVisibleCalls, v.q4kHostVisiblePackedBytes
}

type VulkanDispatchProfile struct {
	ComputeDispatches, Q4KMatmulDispatches, Q2KMatmulDispatches, OtherComputeDispatches uint64
	ComputeBarriers, D2DCopies, BatchSubmits, BatchFlushes, OneShotSubmits              uint64
	OtherMatmulDispatches, OtherNormDispatches, OtherRoPEDispatches                     uint64
	OtherSwiGLUDispatches, OtherAddDispatches, OtherAttentionDispatches                 uint64
	OtherArgmaxDispatches, OtherGDNDispatches, OtherUnclassifiedDispatches              uint64
	OneShotComputeSubmits, OneShotH2DSubmits                                            uint64
	OneShotD2HSubmits, OneShotD2DSubmits                                                uint64
}

func (v *vulkanBackend) VulkanDebugDispatchProfileSnapshot() VulkanDispatchProfile {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return vulkanDebugDispatchProfileSnapshotLocked()
}

func vulkanDebugDispatchProfileSnapshotLocked() VulkanDispatchProfile {
	var p C.fvk_dispatch_profile
	C.fvk_dispatch_profile_snapshot(&p)
	return VulkanDispatchProfile{
		ComputeDispatches:           uint64(p.compute_dispatches),
		Q4KMatmulDispatches:         uint64(p.q4k_matmul_dispatches),
		Q2KMatmulDispatches:         uint64(p.q2k_matmul_dispatches),
		OtherComputeDispatches:      uint64(p.other_compute_dispatches),
		ComputeBarriers:             uint64(p.compute_barriers),
		D2DCopies:                   uint64(p.d2d_copies),
		BatchSubmits:                uint64(p.batch_submits),
		BatchFlushes:                uint64(p.batch_flushes),
		OneShotSubmits:              uint64(p.one_shot_submits),
		OtherMatmulDispatches:       uint64(p.other_matmul_dispatches),
		OtherNormDispatches:         uint64(p.other_norm_dispatches),
		OtherRoPEDispatches:         uint64(p.other_rope_dispatches),
		OtherSwiGLUDispatches:       uint64(p.other_swiglu_dispatches),
		OtherAddDispatches:          uint64(p.other_add_dispatches),
		OtherAttentionDispatches:    uint64(p.other_attention_dispatches),
		OtherArgmaxDispatches:       uint64(p.other_argmax_dispatches),
		OtherGDNDispatches:          uint64(p.other_gdn_dispatches),
		OtherUnclassifiedDispatches: uint64(p.other_unclassified_dispatches),
		OneShotComputeSubmits:       uint64(p.one_shot_compute_submits),
		OneShotH2DSubmits:           uint64(p.one_shot_h2d_submits),
		OneShotD2HSubmits:           uint64(p.one_shot_d2h_submits),
		OneShotD2DSubmits:           uint64(p.one_shot_d2d_submits),
	}
}
func (v *vulkanBackend) VulkanDebugResetDispatchProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_dispatch_profile_reset()
}

func (v *vulkanBackend) VulkanDebugResetQ4KProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.q4kDeviceCalls = 0
	v.q4kDevicePackedBytes = 0
	v.q4kHostVisibleCalls = 0
	v.q4kHostVisiblePackedBytes = 0
}

func (v *vulkanBackend) SetDisableVectorGDN(disable bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.disableVectorGDN = disable
}

func (v *vulkanBackend) IsVectorGDNDisabled() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.isVectorGDNDisabledLocked()
}

func (v *vulkanBackend) isVectorGDNDisabledLocked() bool {
	if v != nil && v.disableVectorGDN {
		return true
	}
	if env := os.Getenv("FAK_DISABLE_VECTOR_GDN"); env == "1" || strings.EqualFold(env, "true") || strings.EqualFold(env, "yes") || strings.EqualFold(env, "on") {
		return true
	}
	if env := os.Getenv("FAK_VECTORIZED_DELTANET"); env == "0" || strings.EqualFold(env, "false") || strings.EqualFold(env, "no") || strings.EqualFold(env, "off") {
		return true
	}
	if env := os.Getenv("FAK_VECTOR_GDN"); env == "0" || strings.EqualFold(env, "false") || strings.EqualFold(env, "no") || strings.EqualFold(env, "off") {
		return true
	}
	return false
}

func (v *vulkanBackend) VulkanDebugGDNProfileSnapshot() (vectorCalls, scalarCalls int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.vectorGDNCalls, v.scalarGDNCalls
}

func (v *vulkanBackend) VulkanDebugGDNProjectionProfileSnapshot() (fusedCalls, composedCalls int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.q8GDNFusedInProjCalls, v.q8GDNComposedInProjCalls
}

func (v *vulkanBackend) VulkanDebugQ8GDNInProjAvailable() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.haveQ8GDNInProj
}

func (v *vulkanBackend) VulkanDebugResetGDNProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.vectorGDNCalls = 0
	v.scalarGDNCalls = 0
	v.q8GDNFusedInProjCalls = 0
	v.q8GDNComposedInProjCalls = 0
}

// VulkanDebugInitShim attempts initialization against an explicit SPIR-V directory
// and returns the raw exit code from the underlying shim (0 = success, non-zero = failure).
func VulkanDebugInitShim(spirvDir string) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	cdir := C.CString(spirvDir)
	defer C.free(unsafe.Pointer(cdir))
	var name [256]C.char
	var discrete C.int
	return int(C.fvk_init(&name[0], 256, &discrete, cdir))
}

// PhasePerformanceQuerySupported reports whether the live RADV/Vulkan device
// exposes the KHR performance-query surface. A nil/absent extension is a typed
// false, never a fabricated true.
func (v *vulkanBackend) PhasePerformanceQuerySupported() (bool, string) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if C.fvk_phase_performance_query_available() == 0 {
		return false, "device does not expose VK_KHR_performance_query"
	}
	if C.fvk_phase_counter_count() <= 0 {
		return false, "VK_KHR_performance_query is present but this queue family declares no enumerable counters"
	}
	return true, ""
}

// PhasePerformanceCounterDescriptors enumerates the device-declared counters
// once. observed=false means the device did not enumerate a live counter set,
// so the caller must publish a typed-unavailable observation.
func (v *vulkanBackend) PhasePerformanceCounterDescriptors() ([]PhasePerformanceCounterDescriptor, bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	count := int(C.fvk_phase_counter_count())
	if count <= 0 {
		return nil, false
	}
	descriptors := make([]PhasePerformanceCounterDescriptor, 0, count)
	for i := 0; i < count; i++ {
		var name [256]C.char
		var unit [64]C.char
		var scope C.int
		if C.fvk_phase_counter_describe(C.int(i), &name[0], 256, &unit[0], 64, &scope) == 0 {
			return nil, false
		}
		descriptors = append(descriptors, PhasePerformanceCounterDescriptor{
			Index: i,
			Name:  strings.TrimSpace(C.GoString(&name[0])),
			Unit:  PhasePerformanceCounterUnit(strings.TrimSpace(C.GoString(&unit[0]))),
			Scope: PhasePerformanceCounterScope(int(scope)),
		})
	}
	return descriptors, true
}

// vulkanDebugSelectQ8GateUpCoop selects the cooperative Q8 gate/up kernel (enabled and
// built) or the one-thread-per-output kernel, returning the previous selection. Test-only:
// it exists so the bit-parity witness can run both kernels in one process.
func vulkanDebugSelectQ8GateUpCoop(enabled bool) bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_debug_select_q8_gateup_coop(boolToCInt(enabled)) != 0
}

// vulkanDebugSelectQ2KMatvec selects the single-token Q2_K matvec kernel (enabled and
// built) or the original Q2_K decode kernel, returning the previous selection. Test-only.
func vulkanDebugSelectQ2KMatvec(enabled bool) bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_debug_select_q2k_matvec(boolToCInt(enabled)) != 0
}

// vulkanDebugSelectIQMatvec enables (when built) or disables the native matvec kernel for the
// raw i-quant dtype dt, returning the previous selection. Disabled, a later Upload of that
// dtype takes the Q8_0 expansion path exactly as under its FAK_VULKAN_<FMT>=0 kill switch;
// already-uploaded native weights still require the kernel. Test-only.
func vulkanDebugSelectIQMatvec(dt Dtype, enabled bool) bool {
	f, ok := rawIQFormats[dt]
	if !ok {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_debug_select_iq_matvec(C.int(f.vulkanID), boolToCInt(enabled)) != 0
}

func boolToCInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// vulkanDebugMemoryTypeSelection runs only the shared pure native selector with
// synthetic results. It performs no Vulkan calls and changes no backend state.
func vulkanDebugMemoryTypeSelection(typeBits, want uint32, flags []uint32, results []int32, heapIndices []uint32, heapSizes []uint64, allocationSize uint64, tryAll bool) (status int32, attempts []uint32, selected uint32, published bool) {
	if len(flags) != len(results) || len(flags) != len(heapIndices) {
		panic("compute: synthetic Vulkan memory type fixtures differ in length")
	}
	cflags := make([]C.uint32_t, len(flags))
	cresults := make([]C.int, len(results))
	cindices := make([]C.uint32_t, len(heapIndices))
	csizes := make([]C.uint64_t, len(heapSizes))
	for i := range heapSizes {
		csizes[i] = C.uint64_t(heapSizes[i])
	}
	for i := range flags {
		cflags[i] = C.uint32_t(flags[i])
		cresults[i] = C.int(results[i])
		cindices[i] = C.uint32_t(heapIndices[i])
	}
	var fp *C.uint32_t
	var rp *C.int
	var ip *C.uint32_t
	var sp *C.uint64_t
	if len(flags) > 0 {
		fp, rp, ip = &cflags[0], &cresults[0], &cindices[0]
	}
	if len(heapSizes) > 0 {
		sp = &csizes[0]
	}
	var tried [32]C.uint32_t
	var count, chosen C.uint32_t
	var handle C.int
	all := C.int(0)
	if tryAll {
		all = 1
	}
	status = int32(C.fvk_debug_memory_type_selection(C.uint32_t(typeBits), C.uint32_t(want), C.uint32_t(len(flags)), fp, rp, ip, C.uint32_t(len(heapSizes)), sp, C.uint64_t(allocationSize), all, &tried[0], &count, &chosen, &handle))
	for i := 0; i < int(count); i++ {
		attempts = append(attempts, uint32(tried[i]))
	}
	return status, attempts, uint32(chosen), handle != 0
}
