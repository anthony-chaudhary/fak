//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"
import (
	"os"
	"strings"
	"unsafe"
)

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
