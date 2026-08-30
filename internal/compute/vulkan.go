//go:build vulkan && windows

// vulkan.go registers a Windows Vulkan compute backend behind the compute.Backend seam.
// It mirrors cuda.go closely: default builds exclude it, it is Approx rather than
// Reference, and device buffers are opaque handles that the Go forward loop never
// dereferences. The C++ shim is built offline by build_vulkan.ps1 into libfakvulkan.a.

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

var vulkanMu sync.Mutex

type vulkanQ4KHomeKey struct {
	src   unsafe.Pointer
	bytes int
}

type vulkanQ4KHome struct {
	ptr   unsafe.Pointer
	bytes int64
}

func (v *vulkanBackend) homeCapLocked() int64 {
	if v.budgetBytes <= 0 {
		return 0
	}
	capBytes := v.budgetBytes / 4
	const maxHomeCap = int64(512 << 20)
	if capBytes > maxHomeCap {
		capBytes = maxHomeCap
	}
	return capBytes
}

func (v *vulkanBackend) freeHomesLocked() {
	if len(v.homes) == 0 {
		v.homes = nil
		v.homeBytes = 0
		return
	}
	C.fvk_batch_flush()
	for _, home := range v.homes {
		if home.ptr != nil {
			C.fvk_free(home.ptr)
		}
	}
	v.dlUsed -= v.homeBytes
	if v.dlUsed < 0 {
		v.dlUsed = 0
	}
	v.homes = nil
	v.homeBytes = 0
}

func (v *vulkanBackend) q4kHomeLocked(wb *vulkanBuf) (unsafe.Pointer, bool) {
	if wb == nil || wb.ptr == nil || wb.n <= 0 {
		v.homeBypasses++
		return nil, false
	}
	key := vulkanQ4KHomeKey{src: wb.ptr, bytes: wb.n}
	if home, ok := v.homes[key]; ok && home.ptr != nil {
		v.homeHits++
		return home.ptr, true
	}
	capBytes := v.homeCapLocked()
	remaining := capBytes - v.homeBytes
	if capBytes == 0 || int64(wb.n) > remaining || v.dlUsed+int64(wb.n)+v.q4kStageBytes > v.budgetBytes {
		v.homeBypasses++
		return nil, false
	}
	wasBatch := C.fvk_batch_active()
	if wasBatch {
		C.fvk_batch_flush()
	}
	ptr := C.fvk_malloc(C.size_t(wb.n))
	if ptr != nil && !v.debugBufferDeviceLocal(&vulkanBuf{ptr: unsafe.Pointer(ptr), n: wb.n}) {
		C.fvk_free(ptr)
		ptr = nil
	}
	if ptr == nil {
		if wasBatch {
			C.fvk_batch_begin()
		}
		v.homeBypasses++
		return nil, false
	}
	C.fvk_d2d(ptr, wb.ptr, C.size_t(wb.n))
	C.fvk_batch_flush()
	if wasBatch {
		C.fvk_batch_begin()
	}
	if v.homes == nil {
		v.homes = make(map[vulkanQ4KHomeKey]vulkanQ4KHome)
	}
	home := vulkanQ4KHome{ptr: ptr, bytes: int64(wb.n)}
	v.homes[key] = home
	v.dlUsed += home.bytes
	v.homeBytes += home.bytes
	v.homeCopied += home.bytes
	v.homeMisses++
	return ptr, true
}
func init() {
	spirv := os.Getenv("FAK_VULKAN_SPIRV")
	if spirv == "" {
		return
	}
	cdir := C.CString(spirv)
	defer C.free(unsafe.Pointer(cdir))

	var name [256]C.char
	var discrete C.int
	if C.fvk_init(&name[0], 256, &discrete, cdir) != 0 {
		return
	}
	tier := "integrated"
	if discrete != 0 {
		tier = "discrete"
	}
	totalDeviceLocal := vulkanCapInt64(C.fvk_total_device_local_memory())
	vulkanDev = &vulkanBackend{
		name:                    "vulkan",
		tier:                    tier + ":" + C.GoString(&name[0]),
		haveQ8:                  C.fvk_have_q8() != 0,
		haveMemoryBudget:        C.fvk_have_memory_budget() != 0,
		totalMem:                totalDeviceLocal,
		budgetBytes:             vulkanBudgetBytes(totalDeviceLocal),
		maxBufferBytes:          vulkanCapInt64(C.fvk_max_buffer_bytes()),
		maxStorageBufferRange:   vulkanCapInt64(C.fvk_max_storage_buffer_range()),
		maxMemoryAllocationSize: vulkanCapInt64(C.fvk_max_memory_allocation_size()),
	}
	Register(vulkanDev)
}

func (v *vulkanBackend) configureVulkanQ4K(profile, stage bool) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.q4kProfile = profile
	v.q4kStage = stage
}

// vulkanBudgetBytes resolves FAK_GPU_BUDGET_MB — the device-local weight budget in MiB — against
// this device's total device-local memory. 0 / unset / invalid = unbounded (place every weight
// device-local, the prior behavior); a positive value caps device-local weight residency; "auto"
// derives the cap from totalDeviceLocal (see resolveGPUBudgetBytes), failing open to unbounded when
// capacity is unknown. Weights past the cap go host-visible in upload order.
func vulkanBudgetBytes(totalDeviceLocal int64) int64 {
	return resolveGPUBudgetBytes(os.Getenv("FAK_GPU_BUDGET_MB"), totalDeviceLocal, totalDeviceLocal > 0)
}

func vulkanCapInt64(v C.uint64_t) int64 {
	u := uint64(v)
	const maxInt64 = uint64(1<<63 - 1)
	if u > maxInt64 {
		return 0
	}
	return int64(u)
}

var vulkanDev *vulkanBackend

type vulkanBuf struct {
	ptr                 unsafe.Pointer
	n                   int
	class               MemoryClass
	scalePtr            unsafe.Pointer
	scaleN              int
	scaleBudgetedBytes  int64
	scaleHostVisible    bool
	q8Chunks            []vulkanQ8Chunk
	budgetedWeightBytes int64
	hostVisibleWeight   bool
}

type vulkanQ8Chunk struct {
	rowStart            int
	rows                int
	ptr                 unsafe.Pointer
	n                   int
	scalePtr            unsafe.Pointer
	scaleN              int
	budgetedWeightBytes int64
	hostVisibleWeight   bool
}

// Ready always reports true: Vulkan dispatches are submitted synchronously, so a
// vulkanBuf handle is materialized as soon as it exists.
func (b *vulkanBuf) Ready() bool { return true }

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

func (v *vulkanBackend) VulkanDebugQ4KProfileSnapshot() (enabled bool, deviceCalls, devicePackedBytes, hostVisibleCalls, hostVisiblePackedBytes int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.q4kProfile, v.q4kDeviceCalls, v.q4kDevicePackedBytes, v.q4kHostVisibleCalls, v.q4kHostVisiblePackedBytes
}

type VulkanDispatchProfile struct {
	ComputeDispatches, Q4KMatmulDispatches, OtherComputeDispatches         uint64
	ComputeBarriers, D2DCopies, BatchSubmits, BatchFlushes, OneShotSubmits uint64
	OtherMatmulDispatches, OtherNormDispatches, OtherRoPEDispatches        uint64
	OtherSwiGLUDispatches, OtherAddDispatches, OtherAttentionDispatches    uint64
	OtherArgmaxDispatches, OtherGDNDispatches, OtherUnclassifiedDispatches uint64
	OneShotComputeSubmits, OneShotH2DSubmits                               uint64
	OneShotD2HSubmits, OneShotD2DSubmits                                   uint64
}

func (v *vulkanBackend) VulkanDebugDispatchProfileSnapshot() VulkanDispatchProfile {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var p C.fvk_dispatch_profile
	C.fvk_dispatch_profile_snapshot(&p)
	return VulkanDispatchProfile{
		ComputeDispatches:           uint64(p.compute_dispatches),
		Q4KMatmulDispatches:         uint64(p.q4k_matmul_dispatches),
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

type vulkanBackend struct {
	name          string
	tier          string
	haveQ8        bool
	transient     []*vulkanBuf
	freeTransient map[int][]*vulkanBuf
	// Device-local residency budget (Stage-1 offload). budgetBytes is the cap on device-local
	// memory fak will request for weights; 0 = unbounded (the prior behavior). dlUsed tracks
	// bytes placed device-local so far. When the next weight would exceed the budget it is
	// placed host-visible deliberately (in upload order — early layers stay device-local), so
	// the cold tail spills by CHOICE instead of by losing the allocation race. Set via
	// FAK_GPU_BUDGET_MB. Guarded by vulkanMu (mutated only inside locked upload paths).
	budgetBytes int64
	dlUsed      int64
	hostvisN    int // count of weights placed host-visible (for the bench report)
	// Single-resource caps queried from the Vulkan physical device. maxBufferBytes is the
	// effective STORAGE buffer ceiling: min(maxStorageBufferRange, maxMemoryAllocationSize)
	// when both are known. It does not solve chunking, but it turns a raw driver allocation
	// failure into a deterministic refusal that names the over-cap buffer (#362).
	haveMemoryBudget          bool
	totalMem                  int64
	maxBufferBytes            int64
	maxStorageBufferRange     int64
	maxMemoryAllocationSize   int64
	q4kProfile                bool
	q4kDeviceCalls            int64
	q4kDevicePackedBytes      int64
	q4kHostVisibleCalls       int64
	q4kHostVisiblePackedBytes int64
	q4kStage                  bool
	q4kStagePtr               unsafe.Pointer
	q4kStageBytes             int64
	q4kStagedCalls            int64
	q4kStagedBytes            int64
	q4kStageFallbacks         int64
	homes                     map[vulkanQ4KHomeKey]vulkanQ4KHome
	homeHits                  int64
	homeMisses                int64
	homeBypasses              int64
	homeBytes                 int64
	homeCopied                int64
}

var _ TensorCloner = (*vulkanBackend)(nil)

const vulkanGoPoolBucketCap = 64

// RetireRequestResources fences any recorded work before returning request-owned
// transient buffers to reusable capacity. Cancellation and request-end paths call
// this explicitly; token-boundary Recycle uses the same ordering.
func (v *vulkanBackend) RetireRequestResources() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	// The Go transient pool bypasses fvk_free, so it cannot rely on the shim's
	// g_batchFreed parking. Complete the command buffer first: only then may an
	// address be handed to the next request or token.
	C.fvk_retire_request()
	for _, b := range v.transient {
		if b.ptr != nil {
			v.recycleTransientLocked(b)
			b.ptr = nil
		}
	}
	v.transient = v.transient[:0]
	C.fvk_trim_pool_if_over(512)
}

// Recycle returns every transient buffer from the current op cycle to the per-size
// free pool after the completion fence required by their recorded descriptors.
func (v *vulkanBackend) Recycle() {
	v.RetireRequestResources()
}

// Trim frees all pooled transient buffers and asks the C++ shim to release its idle
// device-pool memory, reclaiming VRAM held only for reuse.
func (v *vulkanBackend) Trim() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	v.trimTransientLocked()
	C.fvk_trim_pool()
}

// Name returns the backend's stable registry id ("vulkan").
func (v *vulkanBackend) Name() string            { return v.name }
func (v *vulkanBackend) Tier() string            { return v.tier }
func (v *vulkanBackend) Class() CorrectnessClass { return Approx }
func (v *vulkanBackend) Caps() Caps {
	_, _, hostKnown := hostSystemMemory()
	return Caps{DeviceMemory: true, UploadDtype: v.haveQ8, CapacityProbe: v.totalMem > 0, HostCapacityProbe: hostKnown}
}

// DeviceMemory reports the Vulkan device-local heap total and, when VK_EXT_memory_budget is
// available, the current device-local budget headroom. Drivers without the extension keep
// the prior fail-open behavior: total known, free unknown.
func (v *vulkanBackend) DeviceMemory() (total, free int64, known bool) {
	if v == nil || v.totalMem <= 0 {
		return 0, FreeUnknown, false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if v.haveMemoryBudget {
		var budget C.uint64_t
		var usage C.uint64_t
		var freeBytes C.uint64_t
		if C.fvk_device_local_memory_budget(&budget, &usage, &freeBytes) != 0 {
			if free := vulkanCapInt64(freeBytes); free >= 0 {
				return v.totalMem, free, true
			}
		}
	}
	return v.totalMem, FreeUnknown, true
}

// DeviceWeightBudget reports the explicit device-local weight cap. A positive
// cap means immutable weights above it are deliberately placed in host-visible
// Vulkan storage; callers must plan those excess bytes against host RAM rather
// than rejecting the full checkpoint against VRAM.
// MaxWeightBufferBytes reports the single-resource ceiling used to decide
// whether a table-shaped weight can be addressed directly by a Vulkan kernel.
func (v *vulkanBackend) MaxWeightBufferBytes() int64 {
	if v == nil {
		return 0
	}
	return v.maxBufferBytes
}
func (v *vulkanBackend) DeviceWeightBudget() (int64, bool) {
	if v == nil || v.budgetBytes <= 0 {
		return 0, false
	}
	return v.budgetBytes, true
}
func (v *vulkanBackend) HostMemory() (total, free int64, known bool) {
	return hostSystemMemory()
}

func (v *vulkanBackend) BeginBatch() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_batch_begin()
}

// FlushBatch submits the recorded command batch to the device, ending the batching
// window opened by BeginBatch.
func (v *vulkanBackend) FlushBatch() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_batch_flush()
}

// TeardownResources flushes in-flight work before releasing backend-owned
// reusable resources. Repeated calls are valid; staging buffers can join this
// lifecycle without adding per-operation queue fences.
func (v *vulkanBackend) TeardownResources() error {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_batch_flush()
	v.freeHomesLocked()
	if v.q4kStagePtr != nil {
		C.fvk_free(v.q4kStagePtr)
		v.dlUsed -= v.q4kStageBytes
		v.q4kStagePtr = nil
		v.q4kStageBytes = 0
	}
	C.fvk_trim_pool()
	return nil
}

func (v *vulkanBackend) checkResourceCap(nbytes int, what string) {
	if what == "" {
		what = "storage buffer"
	}
	if singleResourceCapExceeded(nbytes, v.maxBufferBytes) {
		panic(formatVulkanResourceCapError(what, nbytes, v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize))
	}
}

func (v *vulkanBackend) dalloc(nbytes int) *vulkanBuf {
	return v.dallocFor(nbytes, "storage buffer")
}

func (v *vulkanBackend) dallocFor(nbytes int, what string) *vulkanBuf {
	return v.dallocForClass(nbytes, memoryClassForVulkanAlloc(what), what)
}

func (v *vulkanBackend) dallocForClass(nbytes int, class MemoryClass, what string) *vulkanBuf {
	v.checkResourceCap(nbytes, what)
	p := C.fvk_malloc(C.size_t(nbytes))
	if p == nil {
		// Device-local (and the shim's own host-visible storage fallback) is exhausted. Rather
		// than crash the whole run, try a clean host-visible allocation as a last resort — slow
		// but alive. This is what makes a budgeted run degrade gracefully when KV/scratch (which
		// don't go through the weight budget) outgrow the remaining device-local headroom,
		// instead of the old hard panic. A nil here too is a genuine OOM with nowhere left.
		p = C.fvk_malloc_hostvis(C.size_t(nbytes))
		if p == nil {
			panic(&DeviceAllocError{Bytes: nbytes, Site: "vulkan:" + what, Class: class})
		}
	}
	return &vulkanBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class}
}

// dallocHostVis allocates a storage buffer in host-visible memory directly (no device-local
// attempt). Used by the residency-budget path for cold weights. Caller holds vulkanMu.
func (v *vulkanBackend) dallocHostVis(nbytes int) *vulkanBuf {
	return v.dallocHostVisFor(nbytes, "host-visible storage buffer")
}

func (v *vulkanBackend) dallocHostVisFor(nbytes int, what string) *vulkanBuf {
	v.checkResourceCap(nbytes, what)
	p := C.fvk_malloc_hostvis(C.size_t(nbytes))
	if p == nil {
		panic(&DeviceAllocError{Bytes: nbytes, Site: "vulkan:" + what, Class: MemoryOffload})
	}
	return &vulkanBuf{ptr: unsafe.Pointer(p), n: nbytes, class: MemoryOffload}
}

// dallocWeight places a weight buffer device-local while under the residency budget, else
// host-visible (deliberately, in upload order). budgetBytes==0 means unbounded -> always
// device-local. Caller holds vulkanMu.
func (v *vulkanBackend) dallocWeight(nbytes int) *vulkanBuf {
	return v.dallocWeightFor(nbytes, "weight buffer")
}

func (v *vulkanBackend) dallocWeightFor(nbytes int, what string) *vulkanBuf {
	if v.budgetBytes > 0 && v.dlUsed+int64(nbytes) > v.budgetBytes {
		buf := v.dallocHostVisFor(nbytes, what)
		v.accountWeightPlacement(buf, nbytes)
		return buf
	}
	buf := v.dallocForClass(nbytes, MemoryWeights, what)
	v.accountWeightPlacement(buf, nbytes)
	return buf
}

func (v *vulkanBackend) dallocKVFor(nbytes int, what string) *vulkanBuf {
	if what == "" {
		what = "KV cache buffer"
	}
	return v.dallocForClass(nbytes, MemoryKVCache, what)
}

func memoryClassForVulkanAlloc(what string) MemoryClass {
	what = strings.ToLower(what)
	switch {
	case strings.Contains(what, "kv"):
		return MemoryKVCache
	case strings.Contains(what, "transient"):
		return MemoryScratchpad
	case strings.Contains(what, "weight"):
		return MemoryWeights
	case strings.Contains(what, "host-visible"):
		return MemoryOffload
	default:
		return MemoryUnknown
	}
}

func (v *vulkanBackend) accountWeightPlacement(buf *vulkanBuf, nbytes int) {
	if v.budgetBytes == 0 || buf == nil || buf.ptr == nil {
		return
	}
	if v.debugBufferDeviceLocal(buf) {
		v.dlUsed += int64(nbytes)
		buf.budgetedWeightBytes = int64(nbytes)
		return
	}
	v.hostvisN++
	buf.hostVisibleWeight = true
}

func (v *vulkanBackend) dallocTransient(nbytes int) *vulkanBuf {
	if v.freeTransient != nil {
		bucket := v.freeTransient[nbytes]
		if len(bucket) > 0 {
			b := bucket[len(bucket)-1]
			v.freeTransient[nbytes] = bucket[:len(bucket)-1]
			if !v.debugBufferDeviceLocal(b) {
				C.fvk_free(b.ptr)
				return v.dallocFor(nbytes, "transient storage buffer")
			}
			return b
		}
	}
	return v.dallocFor(nbytes, "transient storage buffer")
}

func (v *vulkanBackend) recycleTransientLocked(b *vulkanBuf) {
	if b == nil || b.ptr == nil {
		return
	}
	if v.freeTransient == nil {
		v.freeTransient = make(map[int][]*vulkanBuf)
	}
	if !v.debugBufferDeviceLocal(b) {
		C.fvk_free(b.ptr)
		return
	}
	bucket := v.freeTransient[b.n]
	owner := &vulkanBuf{ptr: b.ptr, n: b.n, class: b.class}
	if len(bucket) < vulkanGoPoolBucketCap {
		v.freeTransient[b.n] = append(bucket, owner)
	} else {
		C.fvk_free(owner.ptr)
	}
}

func (v *vulkanBackend) trimTransientLocked() {
	for _, bucket := range v.freeTransient {
		for _, b := range bucket {
			if b.ptr != nil {
				C.fvk_free(b.ptr)
				b.ptr = nil
			}
		}
	}
	clear(v.freeTransient)
}

func (v *vulkanBackend) dev(shape []int, dt Dtype) (Tensor, *vulkanBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := v.dallocFor(n*dt.Bytes(), dt.String()+" tensor "+shapeText(shape))
	return makeTensor(v, dt, RowMajor, append([]int(nil), shape...), nil, buf), buf
}

func (v *vulkanBackend) devTr(shape []int, dt Dtype) (Tensor, *vulkanBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	b := v.dallocTransient(n * dt.Bytes())
	t := makeTensor(v, dt, RowMajor, append([]int(nil), shape...), nil, b)
	v.transient = append(v.transient, b)
	return t, b
}

// Upload copies host weight data to the device: a Q8_0 tensor (or an F32 one narrowed
// to Q8_0 via as) goes through the int8 code+scale path, otherwise F32 is sent H2D as-is.
func (v *vulkanBackend) Upload(t Tensor, as Dtype) Tensor {
	return v.uploadClass(t, as, MemoryWeights, "F32 weight tensor "+shapeText(t.Shape))
}

func (v *vulkanBackend) UploadClass(t Tensor, as Dtype, class MemoryClass, what string) Tensor {
	if class == "" {
		class = MemoryUnknown
	}
	if what == "" {
		what = "F32 " + string(class) + " tensor " + shapeText(t.Shape)
	}
	return v.uploadClass(t, as, class, what)
}

func (v *vulkanBackend) uploadClass(t Tensor, as Dtype, class MemoryClass, what string) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic("compute: vulkan Upload expects host data")
	}
	if t.Dtype == Q4_K {
		return v.uploadQ4KLocked(t)
	}
	if t.Dtype == Q8_0 {
		if t.Quant == nil {
			panic("compute: vulkan Upload Q8 tensor missing QuantSpec")
		}
		return v.uploadQ8Locked(t.Shape, hb.I8(), t.Quant.Scale, t.Quant.Block)
	}
	if t.Dtype != F32 {
		panic("compute: vulkan Upload supports only F32 today (got " + t.Dtype.String() + ")")
	}
	f := hb.F32()
	if class != MemoryWeights {
		if as != F32 {
			panic("compute: vulkan classed Upload supports only F32 activation/runtime uploads")
		}
		buf := v.dallocForClass(t.Numel()*F32.Bytes(), class, what)
		out := makeF32TensorLike(v, t, buf)
		return finishF32Upload(out, f, func(values []float32) {
			C.fvk_h2d(buf.ptr, unsafe.Pointer(&values[0]), C.size_t(len(values)*4))
		})
	}
	if as == Q8_0 {
		q := QuantizeQ8(Default(), t.Shape, f, 32)
		qh := q.buf.(HostBuffer)
		return v.uploadQ8Locked(q.Shape, qh.I8(), q.Quant.Scale, q.Quant.Block)
	}
	buf := v.dallocWeightFor(t.Numel()*F32.Bytes(), what)
	out := makeF32TensorLike(v, t, buf)
	return finishF32Upload(out, f, func(values []float32) {
		C.fvk_h2d(buf.ptr, unsafe.Pointer(&values[0]), C.size_t(len(values)*4))
	})
}

func (v *vulkanBackend) uploadQ4KLocked(t Tensor) Tensor {
	hb, ok := t.buf.(HostBuffer)
	if !ok || len(t.Shape) != 2 || t.Shape[1]%256 != 0 {
		panic("compute: vulkan Q4_K upload requires host raw bytes and [out,in] with in divisible by 256")
	}
	codes := hb.I8()
	raw := i8AsBytes(codes)
	want := t.Shape[0] * (t.Shape[1] / 256) * 144
	if len(raw) != want {
		panic("compute: vulkan Q4_K raw byte length does not match shape")
	}
	buf := v.dallocWeightFor(len(raw), "Q4_K weight buffer "+shapeText(t.Shape))
	if len(raw) > 0 {
		C.fvk_h2d(buf.ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	}
	return makeTensor(v, Q4_K, RowMajor, append([]int(nil), t.Shape...), nil, buf)
}
func (v *vulkanBackend) uploadQ8Locked(shape []int, codes []int8, scales []float32, block int) Tensor {
	if !v.haveQ8 {
		panic("compute: vulkan Q8 upload requested but device lacks int8/8-bit-storage support")
	}
	if len(shape) != 2 {
		panic("compute: vulkan Q8 upload expects a 2D weight tensor")
	}
	out, in := shape[0], shape[1]
	if block != 32 || in%block != 0 {
		panic("compute: vulkan Q8 upload supports only Q8_0 block=32 with divisible input dim")
	}
	if len(codes) != out*in {
		panic("compute: vulkan Q8 code length does not match shape")
	}
	if len(scales) != out*(in/block) {
		panic("compute: vulkan Q8 scale length does not match shape")
	}
	chunks, chunked, ok := q8RowChunksForCap(out, in, block, v.maxBufferBytes)
	if !ok {
		rowBytes := in
		if scaleRowBytes := (in / block) * F32.Bytes(); scaleRowBytes > rowBytes {
			rowBytes = scaleRowBytes
		}
		panic(formatVulkanResourceCapError("Q8_0 weight row "+shapeText(shape), rowBytes, v.maxBufferBytes, v.maxStorageBufferRange, v.maxMemoryAllocationSize))
	}
	if chunked {
		return v.uploadQ8ChunksLocked(shape, codes, scales, block, chunks)
	}
	// The code buffer is the bulk of the weight (in*out bytes) — it's the budget's subject.
	// The scale buffer is ~1/32 the size; keep it device-local so the hot per-block scales
	// stay fast even when the codes spill host-visible.
	shapeName := shapeText(shape)
	codeBuf := v.dallocWeightFor(len(codes), "Q8_0 weight code buffer "+shapeName)
	scaleBuf := v.dallocWeightFor(len(scales)*F32.Bytes(), "Q8_0 weight scale buffer "+shapeName)
	if len(codes) > 0 {
		C.fvk_h2d(codeBuf.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes)))
	}
	if len(scales) > 0 {
		C.fvk_h2d(scaleBuf.ptr, unsafe.Pointer(&scales[0]), C.size_t(len(scales)*F32.Bytes()))
	}
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	buf := &vulkanBuf{
		ptr:                 codeBuf.ptr,
		n:                   codeBuf.n,
		class:               codeBuf.class,
		scalePtr:            scaleBuf.ptr,
		scaleN:              scaleBuf.n,
		budgetedWeightBytes: codeBuf.budgetedWeightBytes,
		hostVisibleWeight:   codeBuf.hostVisibleWeight,
		scaleBudgetedBytes:  scaleBuf.budgetedWeightBytes,
		scaleHostVisible:    scaleBuf.hostVisibleWeight,
	}
	return makeTensor(v, Q8_0, RowMajor, append([]int(nil), shape...), q, buf)
}

func (v *vulkanBackend) uploadQ8ChunksLocked(shape []int, codes []int8, scales []float32, block int, chunks []q8RowChunk) Tensor {
	out, in := shape[0], shape[1]
	scaleCols := in / block
	shapeName := shapeText(shape)
	buf := &vulkanBuf{q8Chunks: make([]vulkanQ8Chunk, 0, len(chunks))}
	for i, chunk := range chunks {
		codeStart := chunk.start * in
		codeEnd := codeStart + chunk.rows*in
		scaleStart := chunk.start * scaleCols
		scaleEnd := scaleStart + chunk.rows*scaleCols
		codeLabel := "Q8_0 weight code chunk " + strconv.Itoa(i) + " rows " + strconv.Itoa(chunk.start) + ":" + strconv.Itoa(chunk.start+chunk.rows) + " " + shapeName
		scaleLabel := "Q8_0 weight scale chunk " + strconv.Itoa(i) + " rows " + strconv.Itoa(chunk.start) + ":" + strconv.Itoa(chunk.start+chunk.rows) + " " + shapeName
		codeBuf := v.dallocWeightFor(codeEnd-codeStart, codeLabel)
		scaleBuf := v.dallocWeightFor((scaleEnd-scaleStart)*F32.Bytes(), scaleLabel)
		if codeEnd > codeStart {
			C.fvk_h2d(codeBuf.ptr, unsafe.Pointer(&codes[codeStart]), C.size_t(codeEnd-codeStart))
		}
		if scaleEnd > scaleStart {
			C.fvk_h2d(scaleBuf.ptr, unsafe.Pointer(&scales[scaleStart]), C.size_t((scaleEnd-scaleStart)*F32.Bytes()))
		}
		buf.q8Chunks = append(buf.q8Chunks, vulkanQ8Chunk{
			rowStart:            chunk.start,
			rows:                chunk.rows,
			ptr:                 codeBuf.ptr,
			n:                   codeBuf.n,
			scalePtr:            scaleBuf.ptr,
			scaleN:              scaleBuf.n,
			budgetedWeightBytes: codeBuf.budgetedWeightBytes,
			hostVisibleWeight:   codeBuf.hostVisibleWeight,
		})
	}
	if out > 0 && len(buf.q8Chunks) == 0 {
		panic("compute: vulkan Q8 chunk upload produced no chunks")
	}
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	return makeTensor(v, Q8_0, RowMajor, append([]int(nil), shape...), q, buf)
}

// Host returns the host-addressable f32 view only when the tensor is backed by a host
// buffer; a device-resident vulkanBuf is not host-addressable, so it returns (nil, false).
func (v *vulkanBackend) Host(t Tensor) ([]float32, bool) {
	return hostF32(t)
}

// Read returns the tensor as host f32: a host-backed buffer is returned directly, a
// device buffer is copied D2H into a fresh slice (the device-to-host fence).
func (v *vulkanBackend) Read(t Tensor) []float32 {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return readF32Tensor(t, func(buf Buffer, out []float32) {
		db := buf.(*vulkanBuf)
		if len(out) > 0 {
			C.fvk_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
		}
	})
}

// CloneTensor makes an independently owned device-to-device copy for persistent
// backend state. fvk_d2d records into an open Vulkan batch when one exists, so the
// flush is part of the clone contract: after this method returns either owner may be
// freed without invalidating the other's bytes.
func (v *vulkanBackend) CloneTensor(t Tensor) (Tensor, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := t.buf.(*vulkanBuf)
	if !ok || b == nil || b.ptr == nil {
		return Tensor{}, fmt.Errorf("vulkan: CloneTensor requires a live vulkan tensor")
	}
	if b.scalePtr != nil || len(b.q8Chunks) != 0 {
		return Tensor{}, fmt.Errorf("vulkan: CloneTensor does not support tensors with auxiliary buffers")
	}
	if b.n <= 0 {
		return Tensor{}, fmt.Errorf("vulkan: CloneTensor invalid allocation size %d", b.n)
	}
	class := b.class
	if class == "" {
		class = MemoryUnknown
	}
	dup := v.dallocForClass(b.n, class, "tensor clone "+shapeText(t.Shape))
	C.fvk_d2d(dup.ptr, b.ptr, C.size_t(b.n))
	C.fvk_batch_flush()
	out := t
	out.Shape = append([]int(nil), t.Shape...)
	out.buf = dup
	return out, nil
}

// Free releases the tensor's device buffer (and its companion Q8 scale buffer, if any)
// back to the shim and nils the handle; it is a no-op for a non-device tensor.
func (v *vulkanBackend) Free(t Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if db, ok := t.buf.(*vulkanBuf); ok {
		for i := range db.q8Chunks {
			chunk := &db.q8Chunks[i]
			if chunk.scalePtr != nil {
				C.fvk_free(chunk.scalePtr)
				chunk.scalePtr = nil
				chunk.scaleN = 0
			}
			if chunk.ptr != nil {
				C.fvk_free(chunk.ptr)
				chunk.ptr = nil
				chunk.n = 0
			}
			if chunk.budgetedWeightBytes > 0 {
				v.dlUsed -= chunk.budgetedWeightBytes
				if v.dlUsed < 0 {
					v.dlUsed = 0
				}
				chunk.budgetedWeightBytes = 0
			}
			if chunk.hostVisibleWeight {
				if v.hostvisN > 0 {
					v.hostvisN--
				}
				chunk.hostVisibleWeight = false
			}
		}
		db.q8Chunks = nil
		if db.ptr == nil {
			return
		}
		if db.scalePtr != nil {
			C.fvk_free(db.scalePtr)
			if db.scaleBudgetedBytes > 0 {
				v.dlUsed -= db.scaleBudgetedBytes
				db.scaleBudgetedBytes = 0
			}
			if db.scaleHostVisible {
				if v.hostvisN > 0 {
					v.hostvisN--
				}
				db.scaleHostVisible = false
			}
			db.scalePtr = nil
			db.scaleN = 0
		}
		C.fvk_free(db.ptr)
		db.ptr = nil
		if db.budgetedWeightBytes > 0 {
			v.dlUsed -= db.budgetedWeightBytes
			if v.dlUsed < 0 {
				v.dlUsed = 0
			}
			db.budgetedWeightBytes = 0
		}
		if db.hostVisibleWeight {
			if v.hostvisN > 0 {
				v.hostvisN--
			}
			db.hostVisibleWeight = false
		}
	}
}

func (v *vulkanBackend) vp(t Tensor) unsafe.Pointer { return t.buf.(*vulkanBuf).ptr }

func vulkanQ4KProfileHostVisible(hostVisibleWeight, deviceLocal bool) bool {
	return hostVisibleWeight || !deviceLocal
}

func (v *vulkanBackend) profileQ4KMatMulLocked(packedBytes int, hostVisibleWeight, deviceLocal bool) {
	if !v.q4kProfile {
		return
	}
	if vulkanQ4KProfileHostVisible(hostVisibleWeight, deviceLocal) {
		v.q4kHostVisibleCalls++
		v.q4kHostVisiblePackedBytes += int64(packedBytes)
	} else {
		v.q4kDeviceCalls++
		v.q4kDevicePackedBytes += int64(packedBytes)
	}
	if (v.q4kDeviceCalls+v.q4kHostVisibleCalls)%256 == 0 {
		log.Printf("compute: vulkan Q4_K profile device_calls=%d device_packed_bytes=%d host_visible_calls=%d host_visible_packed_bytes=%d",
			v.q4kDeviceCalls, v.q4kDevicePackedBytes, v.q4kHostVisibleCalls, v.q4kHostVisiblePackedBytes)
	}
}

func (v *vulkanBackend) VulkanDebugQ4KStageSnapshot() (enabled bool, capacity, calls, bytes, fallbacks int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.q4kStage, v.q4kStageBytes, v.q4kStagedCalls, v.q4kStagedBytes, v.q4kStageFallbacks
}

func (v *vulkanBackend) VulkanDebugBatchActive() bool {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return bool(C.fvk_batch_active())
}

func (v *vulkanBackend) VulkanDebugResetQ4KStage() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_batch_flush()
	v.freeHomesLocked()
	if v.q4kStagePtr != nil {
		C.fvk_free(v.q4kStagePtr)
		v.dlUsed -= v.q4kStageBytes
		v.q4kStagePtr = nil
		v.q4kStageBytes = 0
	}
	v.q4kStagedCalls = 0
	v.q4kStagedBytes = 0
	v.q4kStageFallbacks = 0
	v.homeHits = 0
	v.homeMisses = 0
	v.homeBypasses = 0
	v.homeCopied = 0
}

func (v *vulkanBackend) VulkanDebugQ4KTensorHomeSnapshot() (hits, misses, bypasses int64, entries int, residentBytes, copiedBytes int64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return v.homeHits, v.homeMisses, v.homeBypasses, len(v.homes), v.homeBytes, v.homeCopied
}

func (v *vulkanBackend) ensureQ4KStageLocked(bytes int) unsafe.Pointer {
	need := int64(bytes)
	if v.q4kStagePtr != nil && v.q4kStageBytes >= need {
		return v.q4kStagePtr
	}
	old := v.q4kStageBytes
	if v.budgetBytes > 0 && v.dlUsed-old+need > v.budgetBytes {
		v.q4kStageFallbacks++
		return nil
	}
	resumeBatch := C.fvk_batch_active()
	if resumeBatch {
		defer C.fvk_batch_begin()
	}
	// Growth is rare and must not release storage referenced by pending commands.
	C.fvk_batch_flush()
	if v.q4kStagePtr != nil {
		C.fvk_free(v.q4kStagePtr)
		v.dlUsed -= old
		v.q4kStagePtr = nil
		v.q4kStageBytes = 0
	}
	v.checkResourceCap(bytes, "Q4_K staging buffer")
	p := C.fvk_malloc(C.size_t(bytes))
	if p == nil {
		v.q4kStageFallbacks++
		return nil
	}
	stage := &vulkanBuf{ptr: unsafe.Pointer(p), n: bytes}
	if !v.debugBufferDeviceLocal(stage) {
		C.fvk_free(p)
		v.q4kStageFallbacks++
		return nil
	}
	v.q4kStagePtr = unsafe.Pointer(p)
	v.q4kStageBytes = need
	v.dlUsed += need
	return v.q4kStagePtr
}

func (v *vulkanBackend) q4kMatMulLocked(w, x, y Tensor, out, in, P int) {
	wb := w.buf.(*vulkanBuf)
	if v.q4kProfile {
		v.profileQ4KMatMulLocked(wb.n, wb.hostVisibleWeight, v.debugBufferDeviceLocal(wb))
	}
	weight := wb.ptr
	if v.q4kStage && wb.hostVisibleWeight {
		if home, ok := v.q4kHomeLocked(wb); ok {
			weight = home
		} else if stage := v.ensureQ4KStageLocked(wb.n); stage != nil {
			C.fvk_d2d(stage, wb.ptr, C.size_t(wb.n))
			weight = stage
			v.q4kStagedCalls++
			v.q4kStagedBytes += int64(wb.n)
		}
	}
	C.fvk_q4k_matmul_f32(weight, v.vp(x), v.vp(y), C.int(out), C.int(in), C.int(P))
}

func (v *vulkanBackend) MatMul(w, x Tensor) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	y, _ := v.devTr([]int{out}, F32)
	switch w.Dtype {
	case F32:
		C.fvk_matmul_f32(v.vp(w), v.vp(x), v.vp(y), C.int(out), C.int(in), 1)
	case Q8_0:
		v.q8MatMulLocked(w, x, y, out, in, 1)
	case Q4_K:
		v.q4kMatMulLocked(w, x, y, out, in, 1)
	default:
		panic("compute: vulkan MatMul unsupported weight dtype " + w.Dtype.String())
	}
	return y
}

func (v *vulkanBackend) q8MatMulLocked(w, x, y Tensor, out, in, P int) {
	wb := v.q8WeightBufLocked(w, in, "Q8 MatMul")
	if len(wb.q8Chunks) > 0 {
		v.q8MatMulChunksLocked(wb, x, y, out, in, P)
		return
	}
	C.fvk_q8_matmul_f32(wb.ptr, wb.scalePtr, v.vp(x), v.vp(y),
		C.int(out), C.int(in), C.int(P))
}

func (v *vulkanBackend) q8MatMulChunksLocked(wb *vulkanBuf, x, y Tensor, out, in, P int) {
	for _, chunk := range wb.q8Chunks {
		tmpShape := []int{P, chunk.rows}
		if P == 1 {
			tmpShape = []int{chunk.rows}
		}
		_, tmpBuf := v.devTr(tmpShape, F32)
		C.fvk_q8_matmul_f32(chunk.ptr, chunk.scalePtr, v.vp(x), tmpBuf.ptr,
			C.int(chunk.rows), C.int(in), C.int(P))
		v.copyQ8ChunkOutputLocked(y.buf.(*vulkanBuf), tmpBuf, out, chunk.rowStart, chunk.rows, P)
	}
}

func (v *vulkanBackend) copyQ8ChunkOutputLocked(dst, src *vulkanBuf, out, rowStart, rows, P int) {
	bytes := rows * F32.Bytes()
	for p := 0; p < P; p++ {
		dstOff := (p*out + rowStart) * F32.Bytes()
		srcOff := p * rows * F32.Bytes()
		C.fvk_d2d_range(dst.ptr, C.size_t(dstOff), src.ptr, C.size_t(srcOff), C.size_t(bytes))
	}
}

func (v *vulkanBackend) q8WeightBufLocked(w Tensor, in int, op string) *vulkanBuf {
	if !v.haveQ8 {
		panic("compute: vulkan " + op + " requested but device lacks int8/8-bit-storage support")
	}
	if w.Dtype != Q8_0 || w.Quant == nil || w.Quant.Block != 32 || in%32 != 0 {
		panic("compute: vulkan " + op + " supports only Q8_0 block=32 with divisible input dim")
	}
	// The q8_matmul shader tiles the input in windows of SHARED_CAP floats, so any
	// 32-divisible input dim is supported (e.g. a 1.5B FFN down_proj with in=8960).
	wb := w.buf.(*vulkanBuf)
	if len(wb.q8Chunks) > 0 {
		for _, chunk := range wb.q8Chunks {
			if chunk.ptr == nil || chunk.scalePtr == nil {
				panic("compute: vulkan " + op + " missing Q8 chunk device buffers")
			}
		}
		return wb
	}
	if wb.ptr == nil || wb.scalePtr == nil {
		panic("compute: vulkan " + op + " missing device scale buffer")
	}
	return wb
}

// MatMulArgmax fuses the final F32 projection and the argmax reduction in one shader,
// returning the index of the largest logit without copying the logits host-ward.
func (v *vulkanBackend) MatMulArgmax(w, x Tensor) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: vulkan MatMulArgmax supports only F32 weights today (got " + w.Dtype.String() + ")")
	}
	if in == 0 || x.Numel() != in {
		panic("compute: vulkan MatMulArgmax expects one input row matching the weight input dim")
	}
	return int(C.fvk_matmul_argmax_f32(v.vp(w), v.vp(x), C.int(out), C.int(in)))
}

// RMSNormMatMulArgmax fuses RMSNorm of x, the final F32 projection, and the argmax into
// one shader, returning the top logit's index for greedy decode.
func (v *vulkanBackend) RMSNormMatMulArgmax(w, x, normWeight Tensor, eps float32) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 || normWeight.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMulArgmax supports only F32 weights today")
	}
	if normWeight.Numel() != in {
		panic("compute: vulkan RMSNormMatMulArgmax norm weight shape does not match projection input dim")
	}
	if in == 0 || x.Numel() != in {
		panic("compute: vulkan RMSNormMatMulArgmax expects one input row matching the weight input dim")
	}
	return int(C.fvk_rmsnorm_matmul_argmax_f32(v.vp(w), v.vp(x), v.vp(normWeight),
		C.int(out), C.int(in), C.float(eps)))
}

// BatchedMatMul computes the prefill GEMM Y = X @ Wᵀ over P input rows, dispatching the
// F32 or Q8_0 shader by the weight's dtype.
func (v *vulkanBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	y, _ := v.devTr([]int{P, out}, F32)
	switch w.Dtype {
	case F32:
		C.fvk_matmul_f32(v.vp(w), v.vp(X), v.vp(y), C.int(out), C.int(in), C.int(P))
	case Q8_0:
		v.q8MatMulLocked(w, X, y, out, in, P)
	default:
		panic("compute: vulkan BatchedMatMul unsupported weight dtype " + w.Dtype.String())
	}
	return y
}

// EmbeddingRow returns one row of a 2D F32 embedding table as a new device tensor,
// copied device-to-device so the lookup never round-trips through the host.
func (v *vulkanBackend) EmbeddingRow(table Tensor, row int) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if table.Dtype != F32 {
		panic("compute: vulkan EmbeddingRow supports only F32 tables today (got " + table.Dtype.String() + ")")
	}
	if len(table.Shape) != 2 {
		panic("compute: vulkan EmbeddingRow expects a 2D table")
	}
	rows, width := table.Shape[0], table.Shape[1]
	if row < 0 || row >= rows {
		panic("compute: vulkan EmbeddingRow row out of range")
	}
	y, _ := v.devTr([]int{width}, F32)
	bytes := width * F32.Bytes()
	srcOff := row * bytes
	C.fvk_d2d_range(v.vp(y), C.size_t(0), v.vp(table), C.size_t(srcOff), C.size_t(bytes))
	return y
}

// MatMulAddInPlace accumulates the F32 projection x @ Wᵀ into dst (dst += x @ Wᵀ),
// the residual-add fused into the matmul for any P input rows.
func (v *vulkanBackend) MatMulAddInPlace(dst, w, x Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: vulkan MatMulAddInPlace supports only F32 weights today (got " + w.Dtype.String() + ")")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan MatMulAddInPlace input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if dst.Numel() != P*out {
		panic("compute: vulkan MatMulAddInPlace dst shape does not match projection output")
	}
	C.fvk_matmul_add_f32(v.vp(w), v.vp(x), v.vp(dst), C.int(out), C.int(in), C.int(P))
}

// MatMul2 applies two projections sharing input x in one decode-only dispatch (all-F32
// or all-Q8_0), returning both outputs — the fused gate/up FFN projection.
func (v *vulkanBackend) MatMul2(w0, w1, x Tensor) (Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out0, in := w0.Shape[0], w0.Shape[1]
	out1, in1 := w1.Shape[0], w1.Shape[1]
	if in1 != in {
		panic("compute: vulkan MatMul2 weight input dims differ")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan MatMul2 input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if P != 1 {
		panic("compute: vulkan MatMul2 is decode-only today")
	}
	y0, _ := v.devTr([]int{out0}, F32)
	y1, _ := v.devTr([]int{out1}, F32)

	if w0.Dtype == Q8_0 || w1.Dtype == Q8_0 {
		if w0.Dtype != Q8_0 || w1.Dtype != Q8_0 {
			panic("compute: vulkan MatMul2 requires either all F32 or all Q8_0 weights")
		}
		wb0 := v.q8WeightBufLocked(w0, in, "Q8 MatMul2")
		wb1 := v.q8WeightBufLocked(w1, in, "Q8 MatMul2")
		if len(wb0.q8Chunks) > 0 || len(wb1.q8Chunks) > 0 {
			v.q8MatMulLocked(w0, x, y0, out0, in, P)
			v.q8MatMulLocked(w1, x, y1, out1, in, P)
			return y0, y1
		}
		C.fvk_q8_matmul2_f32(wb0.ptr, wb0.scalePtr, wb1.ptr, wb1.scalePtr,
			v.vp(x), v.vp(y0), v.vp(y1),
			C.int(out0), C.int(out1), C.int(in), C.int(P))
		return y0, y1
	}
	if w0.Dtype != F32 || w1.Dtype != F32 {
		panic("compute: vulkan MatMul2 supports only F32 or all-Q8_0 weights")
	}
	C.fvk_matmul2_f32(v.vp(w0), v.vp(w1), v.vp(x), v.vp(y0), v.vp(y1),
		C.int(out0), C.int(out1), C.int(in), C.int(P))
	return y0, y1
}

// MatMul3 applies the Q, K, and V projections sharing input x in one decode-only
// dispatch (all-F32 or all-Q8_0), returning the three attention projections.
func (v *vulkanBackend) MatMul3(wq, wk, wv, x Tensor) (Tensor, Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	qOut, in := wq.Shape[0], wq.Shape[1]
	kOut, kIn := wk.Shape[0], wk.Shape[1]
	vOut, vIn := wv.Shape[0], wv.Shape[1]
	if kIn != in || vIn != in {
		panic("compute: vulkan MatMul3 weight input dims differ")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan MatMul3 input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if P != 1 {
		panic("compute: vulkan MatMul3 is decode-only today")
	}
	q, _ := v.devTr([]int{qOut}, F32)
	k, _ := v.devTr([]int{kOut}, F32)
	val, _ := v.devTr([]int{vOut}, F32)
	if wq.Dtype == Q8_0 || wk.Dtype == Q8_0 || wv.Dtype == Q8_0 {
		if wq.Dtype != Q8_0 || wk.Dtype != Q8_0 || wv.Dtype != Q8_0 {
			panic("compute: vulkan MatMul3 requires either all F32 or all Q8_0 weights")
		}
		wbq := v.q8WeightBufLocked(wq, in, "Q8 MatMul3")
		wbk := v.q8WeightBufLocked(wk, in, "Q8 MatMul3")
		wbv := v.q8WeightBufLocked(wv, in, "Q8 MatMul3")
		if len(wbq.q8Chunks) > 0 || len(wbk.q8Chunks) > 0 || len(wbv.q8Chunks) > 0 {
			v.q8MatMulLocked(wq, x, q, qOut, in, P)
			v.q8MatMulLocked(wk, x, k, kOut, in, P)
			v.q8MatMulLocked(wv, x, val, vOut, in, P)
			return q, k, val
		}
		C.fvk_q8_matmul3_f32(wbq.ptr, wbq.scalePtr, wbk.ptr, wbk.scalePtr, wbv.ptr, wbv.scalePtr,
			v.vp(x), v.vp(q), v.vp(k), v.vp(val),
			C.int(qOut), C.int(kOut), C.int(vOut), C.int(in), C.int(P))
		return q, k, val
	}
	if wq.Dtype != F32 || wk.Dtype != F32 || wv.Dtype != F32 {
		panic("compute: vulkan MatMul3 supports only F32 or all-Q8_0 weights")
	}
	C.fvk_matmul3_f32(v.vp(wq), v.vp(wk), v.vp(wv), v.vp(x), v.vp(q), v.vp(k), v.vp(val),
		C.int(qOut), C.int(kOut), C.int(vOut), C.int(in), C.int(P))
	return q, k, val
}

// RMSNormMatMul2 fuses RMSNorm of x with two projections sharing that normalized input
// in one decode-only dispatch (all-F32 or all-Q8_0), returning both outputs.
func (v *vulkanBackend) RMSNormMatMul2(w0, w1, x, normWeight Tensor, eps float32) (Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out0, in := w0.Shape[0], w0.Shape[1]
	out1, in1 := w1.Shape[0], w1.Shape[1]
	if normWeight.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMul2 norm weight must be F32")
	}
	if in1 != in {
		panic("compute: vulkan RMSNormMatMul2 weight input dims differ")
	}
	if normWeight.Numel() != in {
		panic("compute: vulkan RMSNormMatMul2 norm weight shape does not match projection input dim")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan RMSNormMatMul2 input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if P != 1 {
		panic("compute: vulkan RMSNormMatMul2 is decode-only today")
	}
	y0, _ := v.devTr([]int{out0}, F32)
	y1, _ := v.devTr([]int{out1}, F32)
	if w0.Dtype == Q4_K || w1.Dtype == Q4_K {
		if w0.Dtype != Q4_K || w1.Dtype != Q4_K {
			panic("compute: vulkan RMSNormMatMul2 requires either all F32, all Q8_0, or all Q4_K weights")
		}
		xn, _ := v.devTr([]int{in}, F32)
		C.fvk_rmsnorm_f32(v.vp(x), v.vp(normWeight), v.vp(xn), C.int(P), C.int(in), C.float(eps))
		v.q4kMatMulLocked(w0, xn, y0, out0, in, P)
		v.q4kMatMulLocked(w1, xn, y1, out1, in, P)
		return y0, y1
	}
	if w0.Dtype == Q8_0 || w1.Dtype == Q8_0 {
		if w0.Dtype != Q8_0 || w1.Dtype != Q8_0 {
			panic("compute: vulkan RMSNormMatMul2 requires either all F32 or all Q8_0 weights")
		}
		wb0 := v.q8WeightBufLocked(w0, in, "Q8 RMSNormMatMul2")
		wb1 := v.q8WeightBufLocked(w1, in, "Q8 RMSNormMatMul2")
		if len(wb0.q8Chunks) > 0 || len(wb1.q8Chunks) > 0 {
			xn, _ := v.devTr([]int{in}, F32)
			C.fvk_rmsnorm_f32(v.vp(x), v.vp(normWeight), v.vp(xn), C.int(P), C.int(in), C.float(eps))
			v.q8MatMulLocked(w0, xn, y0, out0, in, P)
			v.q8MatMulLocked(w1, xn, y1, out1, in, P)
			return y0, y1
		}
		C.fvk_rmsnorm_q8_matmul2_f32(wb0.ptr, wb0.scalePtr, wb1.ptr, wb1.scalePtr,
			v.vp(x), v.vp(normWeight), v.vp(y0), v.vp(y1),
			C.int(out0), C.int(out1), C.int(in), C.int(P), C.float(eps))
		return y0, y1
	}
	if w0.Dtype != F32 || w1.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMul2 supports only F32 or all-Q8_0 weights")
	}
	C.fvk_rmsnorm_matmul2_f32(v.vp(w0), v.vp(w1), v.vp(x), v.vp(normWeight), v.vp(y0), v.vp(y1),
		C.int(out0), C.int(out1), C.int(in), C.int(P), C.float(eps))
	return y0, y1
}

// RMSNormMatMul3 fuses RMSNorm of x with the Q, K, and V projections in one decode-only
// dispatch (all-F32 or all-Q8_0), returning the three normalized-then-projected outputs.
func (v *vulkanBackend) RMSNormMatMul3(wq, wk, wv, x, normWeight Tensor, eps float32) (Tensor, Tensor, Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	qOut, in := wq.Shape[0], wq.Shape[1]
	kOut, kIn := wk.Shape[0], wk.Shape[1]
	vOut, vIn := wv.Shape[0], wv.Shape[1]
	if normWeight.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMul3 norm weight must be F32")
	}
	if kIn != in || vIn != in {
		panic("compute: vulkan RMSNormMatMul3 weight input dims differ")
	}
	if normWeight.Numel() != in {
		panic("compute: vulkan RMSNormMatMul3 norm weight shape does not match projection input dim")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan RMSNormMatMul3 input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if P != 1 {
		panic("compute: vulkan RMSNormMatMul3 is decode-only today")
	}
	q, _ := v.devTr([]int{qOut}, F32)
	k, _ := v.devTr([]int{kOut}, F32)
	val, _ := v.devTr([]int{vOut}, F32)
	if wq.Dtype == Q8_0 || wk.Dtype == Q8_0 || wv.Dtype == Q8_0 {
		if wq.Dtype != Q8_0 || wk.Dtype != Q8_0 || wv.Dtype != Q8_0 {
			panic("compute: vulkan RMSNormMatMul3 requires either all F32 or all Q8_0 weights")
		}
		wbq := v.q8WeightBufLocked(wq, in, "Q8 RMSNormMatMul3")
		wbk := v.q8WeightBufLocked(wk, in, "Q8 RMSNormMatMul3")
		wbv := v.q8WeightBufLocked(wv, in, "Q8 RMSNormMatMul3")
		if len(wbq.q8Chunks) > 0 || len(wbk.q8Chunks) > 0 || len(wbv.q8Chunks) > 0 {
			xn, _ := v.devTr([]int{in}, F32)
			C.fvk_rmsnorm_f32(v.vp(x), v.vp(normWeight), v.vp(xn), C.int(P), C.int(in), C.float(eps))
			v.q8MatMulLocked(wq, xn, q, qOut, in, P)
			v.q8MatMulLocked(wk, xn, k, kOut, in, P)
			v.q8MatMulLocked(wv, xn, val, vOut, in, P)
			return q, k, val
		}
		C.fvk_rmsnorm_q8_matmul3_f32(wbq.ptr, wbq.scalePtr, wbk.ptr, wbk.scalePtr, wbv.ptr, wbv.scalePtr,
			v.vp(x), v.vp(normWeight), v.vp(q), v.vp(k), v.vp(val),
			C.int(qOut), C.int(kOut), C.int(vOut), C.int(in), C.int(P), C.float(eps))
		return q, k, val
	}
	if wq.Dtype != F32 || wk.Dtype != F32 || wv.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMul3 supports only F32 or all-Q8_0 weights")
	}
	C.fvk_rmsnorm_matmul3_f32(v.vp(wq), v.vp(wk), v.vp(wv), v.vp(x), v.vp(normWeight),
		v.vp(q), v.vp(k), v.vp(val),
		C.int(qOut), C.int(kOut), C.int(vOut), C.int(in), C.int(P), C.float(eps))
	return q, k, val
}

// RMSNormMatMul fuses RMSNorm of x and a single F32 projection in one decode-only
// dispatch, returning the normalized-then-projected output.
func (v *vulkanBackend) RMSNormMatMul(w, x, normWeight Tensor, eps float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 || normWeight.Dtype != F32 {
		panic("compute: vulkan RMSNormMatMul supports only F32 weights today")
	}
	if normWeight.Numel() != in {
		panic("compute: vulkan RMSNormMatMul norm weight shape does not match projection input dim")
	}
	if in == 0 || x.Numel()%in != 0 {
		panic("compute: vulkan RMSNormMatMul input shape is not divisible by weight input dim")
	}
	P := x.Numel() / in
	if P != 1 {
		panic("compute: vulkan RMSNormMatMul is decode-only today")
	}
	y, _ := v.devTr([]int{out}, F32)
	C.fvk_rmsnorm_matmul_f32(v.vp(w), v.vp(x), v.vp(normWeight), v.vp(y),
		C.int(out), C.int(in), C.int(P), C.float(eps))
	return y
}

// SwiGLUMatMulAddInPlace computes silu(gate)*up, projects it through the F32 or Q8_0
// down weight, and accumulates the result into dst — the fused FFN down step.
func (v *vulkanBackend) SwiGLUMatMulAddInPlace(dst, w, gate, up Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if gate.Numel() != up.Numel() {
		panic("compute: vulkan SwiGLUMatMulAddInPlace gate/up shapes differ")
	}
	if in == 0 || gate.Numel()%in != 0 {
		panic("compute: vulkan SwiGLUMatMulAddInPlace gate shape is not divisible by weight input dim")
	}
	P := gate.Numel() / in
	if dst.Numel() != P*out {
		panic("compute: vulkan SwiGLUMatMulAddInPlace dst shape does not match projection output")
	}
	switch w.Dtype {
	case F32:
		C.fvk_swiglu_matmul_add_f32(v.vp(w), v.vp(gate), v.vp(up), v.vp(dst), C.int(out), C.int(in), C.int(P))
	case Q4_K:
		sw, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
		C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(sw), C.int(gate.Numel()))
		projShape := []int{P, out}
		if P == 1 {
			projShape = []int{out}
		}
		proj, _ := v.devTr(projShape, F32)
		v.q4kMatMulLocked(w, sw, proj, out, in, P)
		C.fvk_add_f32(v.vp(dst), v.vp(proj), C.int(dst.Numel()))
	case Q8_0:
		wb := v.q8WeightBufLocked(w, in, "Q8 SwiGLUMatMulAddInPlace")
		if len(wb.q8Chunks) > 0 {
			sw, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
			C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(sw), C.int(gate.Numel()))
			projShape := []int{P, out}
			if P == 1 {
				projShape = []int{out}
			}
			proj, _ := v.devTr(projShape, F32)
			v.q8MatMulLocked(w, sw, proj, out, in, P)
			C.fvk_add_f32(v.vp(dst), v.vp(proj), C.int(dst.Numel()))
			return
		}
		C.fvk_swiglu_q8_matmul_add_f32(wb.ptr, wb.scalePtr, v.vp(gate), v.vp(up), v.vp(dst), C.int(out), C.int(in), C.int(P))
	default:
		panic("compute: vulkan SwiGLUMatMulAddInPlace unsupported weight dtype " + w.Dtype.String())
	}
}

// RMSNorm applies row-wise RMS normalization scaled by weight (eps in the denominator)
// to each row of x, returning a new device tensor of the same shape.
func (v *vulkanBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := weight.Numel()
	rows := x.Numel() / n
	y, _ := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_rmsnorm_f32(v.vp(x), v.vp(weight), v.vp(y), C.int(rows), C.int(n), C.float(eps))
	return y
}

// RoPE applies rotary position embedding at position pos to each head of x, returning a
// new device tensor (x is copied D2D first so the input is left unmodified).
func (v *vulkanBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	y, ybuf := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_d2d(ybuf.ptr, x.buf.(*vulkanBuf).ptr, C.size_t(x.Numel()*4))
	C.fvk_rope_f32(v.vp(y), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return y
}

// RoPEInPlace applies rotary position embedding at position pos to x's buffer directly,
// returning the same tensor (no copy) for the case where x may be overwritten.
func (v *vulkanBackend) RoPEInPlace(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_rope_f32(v.vp(x), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return x
}

// SwiGLU computes the elementwise silu(gate)*up activation, returning a new device
// tensor shaped like gate.
func (v *vulkanBackend) SwiGLU(gate, up Tensor) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := gate.Numel()
	y, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
	C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(y), C.int(n))
	return y
}

// AddInPlace adds src into dst elementwise (dst += src) on the device — the residual add.
func (v *vulkanBackend) AddInPlace(dst, src Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_add_f32(v.vp(dst), v.vp(src), C.int(dst.Numel()))
}

// AddBias adds the width-length bias vector to every row of dst (broadcast over rows).
func (v *vulkanBackend) AddBias(dst, bias Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	width := bias.Numel()
	rows := dst.Numel() / width
	C.fvk_add_bias_f32(v.vp(dst), v.vp(bias), C.int(rows), C.int(width))
}

// Attention runs the fused scaled-dot-product attention for one layer over the cached
// keys/values (grp query heads per KV head, scale applied to the scores), returning the
// per-head context vectors as one device tensor.
func (v *vulkanBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	vk := kv.(*vulkanKV)
	hd, nKV := vk.cfg.HeadDim, vk.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := vk.K[layer].len / w
	out, _ := v.devTr([]int{nH * hd}, F32)
	C.fvk_attention_f32(v.vp(q), vk.K[layer].ptr, vk.V[layer].ptr, v.vp(out),
		C.int(nPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// Argmax returns the index of the largest element of the device logits tensor via the
// scalar-reduction shader, so greedy decode never copies the full vector host-ward.
func (v *vulkanBackend) Argmax(logits Tensor) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_argmax_f32(v.vp(logits), C.int(logits.Numel())))
}
