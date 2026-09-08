//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
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
*/
import "C"
import (
	"context"
	"fmt"
	"os"
	"strings"
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

func (v *vulkanBackend) VulkanDebugRestoreActive() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_restore_active() != 0
}

// BackendExecutionSnapshot reports identity and cumulative counters from the
// selected Vulkan backend. Callers must bracket one execution and subtract via
// BackendExecutionDelta; this process-global snapshot is never a receipt.
func (v *vulkanBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
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

	dispatch := v.VulkanDebugDispatchProfileSnapshot()
	if dispatch.Q4KMatmulDispatches > dispatch.ComputeDispatches {
		return BackendExecutionSnapshot{}, fmt.Errorf("compute: Vulkan Q4_K dispatch count exceeds compute total")
	}
	h2d, d2h := v.VulkanDebugTransferBytes()
	_, _, stageCalls, stageBytes, fallbacks := v.VulkanDebugQ4KStageSnapshot()
	hits, admissions, bypasses, entries, residentBytes, copiedBytes := v.VulkanDebugQ4KTensorHomeSnapshot()
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
	total, free, memoryObserved := DeviceMemoryInfo(v)
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
			H2DBytes:        h2d, D2HBytes: d2h, D2DCopies: dispatch.D2DCopies,
			Q4KStageCalls: uint64(stageCalls), Q4KStageBytes: uint64(stageBytes), Fallbacks: uint64(fallbacks),
			TensorHomeHits: uint64(hits), TensorHomeAdmissions: uint64(admissions),
			TensorHomeBypasses: uint64(bypasses), TensorHomeCopiedBytes: uint64(copiedBytes),
		},
		TensorHomeEntries: uint64(entries), TensorHomeResidentBytes: uint64(residentBytes),
		DeviceMemoryTotalBytes: uint64(total), DeviceMemoryFreeBytes: uint64(free), DeviceMemoryObserved: memoryObserved,
	}, nil
}

func (v *vulkanBackend) debugBufferHostVisible(b *vulkanBuf) bool {
	return b != nil && b.ptr != nil && C.fvk_debug_buffer_is_host_visible(b.ptr) != 0
}

func (v *vulkanBackend) debugBufferDeviceLocal(b *vulkanBuf) bool {
	return b != nil && b.ptr != nil && C.fvk_debug_buffer_is_device_local(b.ptr) != 0
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

func (v *vulkanBackend) VulkanDebugResetGDNProfile() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.vectorGDNCalls = 0
	v.scalarGDNCalls = 0
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
