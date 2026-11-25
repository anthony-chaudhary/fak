//go:build cuda && cgo

// Device residency: tagged allocation, upload and free paths for the CUDA backend.

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakcuda -lcudart -lcublas -lcublasLt -lstdc++ -lm
#include <stdlib.h>
#include "cuda_backend.h"
#include "cuda_kv_precision.h"
#include <stdint.h>
// Always-on device-work counters (cuda/kernel_profile.cuh; fold in cuda_device_work.go).
extern void fcuda_dw_counters(uint64_t *out, int n);
extern int fcuda_dw_site_rows(uint32_t *site, uint32_t *stage, uint64_t *syncs, uint64_t *wait_ns, uint64_t *d2h, int cap);
extern const char *fcuda_dw_stage_name(uint32_t id);
// Checked retained-IQ2 sequence ABI. These declarations and calls are enrolled
// by tools/cuda_abi_parity.py; definitions live in cuda/iq_gemv.cuh.
extern int fcuda_qwen35_iq2_sequence_checked_f32(const uint8_t *w, const float *x,
    float *y, int out, int in, int tokens, int xs);
extern int fcuda_qwen35_iq2_sequence_fence(void);
extern void fcuda_qwen35_iq2_sequence_retirement_failed(int status);
*/
import "C"

import (
	"fmt"
	"slices"
	"sync/atomic"
	"unsafe"
)

// ---- residency ------------------------------------------------------------------

func (c *cudaBackend) dallocClass(nbytes int, class MemoryClass, site string) *cudaBuf {
	if c.promptPanel && (class == MemoryScratchpad || class == MemoryActivation) {
		return c.dallocPromptPanel(nbytes, class, site)
	}
	cls, siteID := cudaAllocTag(class, site, false)
	p := C.fcuda_malloc_tagged(C.size_t(nbytes), cls, siteID)
	if p == nil {
		p = C.fcuda_malloc_managed_tagged(C.size_t(nbytes), cls, siteID)
		if p == nil {
			// fcuda_malloc and fcuda_malloc_managed print the real CUDA errors (OOM vs a context
			// poisoned by a prior async kernel fault) to stderr before returning nil; carry the
			// requested size on a typed DeviceAllocError so the in-kernel decode boundary can
			// recover it into an actionable error instead of crashing the serving goroutine.
			panic(&DeviceAllocError{Bytes: nbytes, Site: site, Class: class})
		}
		return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class, managed: true}
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class}
}

// dallocDeviceClass reserves device-local storage without changing the requested
// memory class after a miss. Callers whose state is consumed by every following
// kernel use this path so a capacity refusal is returned before any work is
// submitted instead of silently opting into UVM migration.
// Caller holds cudaMu.
func (c *cudaBackend) dallocDeviceClass(nbytes int, class MemoryClass, site string) (*cudaBuf, error) {
	cls, siteID := cudaAllocTag(class, site, false)
	p := C.fcuda_malloc_tagged(C.size_t(nbytes), cls, siteID)
	if p == nil {
		return nil, &DeviceAllocError{Bytes: nbytes, Site: site, Class: class}
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class}, nil
}

// dallocDeviceOnlyClass is the strict allocator used by the whole-operation
// GDN path. General CUDA operations may fall back to managed memory after a
// device allocation miss; this path must instead return a typed refusal because
// UVM migration would invalidate its device-residency/zero-transfer contract.
// Caller holds cudaMu.
func (c *cudaBackend) dallocDeviceOnlyClass(nbytes int, class MemoryClass, site string) (*cudaBuf, error) {
	if err := qwen35GDNInjectedAllocationFailure(site, nbytes); err != nil {
		return nil, err
	}
	buf, err := c.dallocDeviceClass(nbytes, class, site)
	if err != nil {
		return nil, &Qwen35GDNAllocationError{Operand: site, Bytes: nbytes}
	}
	return buf, nil
}

// dallocManagedClass allocates directly from cudaMallocManaged. Used by the residency-budget
// path for cold weights. Caller holds cudaMu.
func (c *cudaBackend) dallocManagedClass(nbytes int, class MemoryClass, site string) *cudaBuf {
	cls, siteID := cudaAllocTag(class, site, false)
	p := C.fcuda_malloc_managed_tagged(C.size_t(nbytes), cls, siteID)
	if p == nil {
		panic(&DeviceAllocError{Bytes: nbytes, Site: site, Class: class})
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class, managed: true}
}

// Upload copies host data resident -> VRAM, optionally narrowing the weight dtype at H2D
// (Caps.UploadDtype). The narrowing the `as` request selects:
//   - `as == F16`: the fp16 compute path (#484). The f32 is staged on device, narrowed to __half
//     (and, for a ColMajor source, transpose-repacked — the `Layout` repack at H2D); resident F16.
//   - `as == Q8_0` (#485): the f32 weight is quantized to Q8_0 (per-block(32) int8 codes + f32
//     scales, the cpuref QuantizeQ8 scheme) and BOTH narrow operands go resident — codes in the
//     buffer's ptr, scales in its scale side-channel. No f32 weight ever stays resident, so the
//     VRAM footprint is ≈ int8 size, the whole point of the native quantized GEMM.
//   - any other `as`: full-precision F32 bytes resident (the SGEMM path).
//
// A weight whose HOST dtype is already Q4_K (raw GGUF super-block bytes — the resident-Q4_K loader
// in internal/ggufload produces these) is copied resident verbatim (#485): the bytes are already
// narrow, so there is nothing to quantize; the dequant is fused into the GEMM tile on device.
// Every other host dtype must be F32.
func (c *cudaBackend) Upload(t Tensor, as Dtype) Tensor {
	return c.uploadClass(t, as, MemoryWeights, "upload-weight")
}

func (c *cudaBackend) UploadClass(t Tensor, as Dtype, class MemoryClass, site string) Tensor {
	class, site = normalizeUploadClass(class, site, "upload-")
	return c.uploadClass(t, as, class, site)
}

func (c *cudaBackend) uploadClass(t Tensor, as Dtype, class MemoryClass, site string) Tensor {
	requireCUDAUploadLayout("cuda Upload", t, as, class, site)
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.uploadClassLocked(t, as, class, site)
}

// Caller holds cudaMu and has validated the upload layout.
func (c *cudaBackend) uploadClassLocked(t Tensor, as Dtype, class MemoryClass, site string) Tensor {
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  site,
			Msg:   "cuda Upload expects host data",
			Class: class,
		})
	}
	if t.Dtype == NVFP4 {
		if as != NVFP4 || class != MemoryWeights {
			panic(&CUDAOpError{
				Op:    "Upload",
				Site:  site,
				Msg:   "cuda packed NVFP4 upload requires matching resident dtype and weight memory class",
				Class: class,
			})
		}
		return c.uploadNVFP4(t, hb)
	}
	if cudaRawUploadIsVerbatim(t.Dtype) {
		if as != t.Dtype {
			panic(&CUDAOpError{
				Op:    "Upload",
				Site:  site,
				Msg:   "cuda raw k-quant upload requires matching resident dtype",
				Class: class,
			})
		}
		return c.uploadRawKQuantClass(t, hb, class, site)
	}
	if t.Dtype == Q2_0 {
		// An ALREADY-packed ternary Q2_0 host weight (2-bit codes + per-block f32 scales), copied
		// resident with no re-quantization — the packed-ternary counterpart of uploadQ8Resident.
		return c.uploadQ2Resident(t, hb)
	}
	if t.Dtype == Q8_0 {
		// An ALREADY-quantized Q8_0 host weight (codes + per-block scales), copied resident with
		// NO re-quantization. This is the memory-lean load path: the model dropped the f32 weight
		// at load and carries only the Q8 codes+scales, so the HAL hands a Q8_0 host tensor here
		// (hal.go weightHALQ8) — distinct from the witness path that hands F32 and narrows on-device
		// (uploadQ8 below). Without this branch the lean Q8 decode on the cuda backend panics at H2D.
		return c.uploadQ8Resident(t, hb)
	}
	if t.Dtype != F32 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  site,
			Msg:   "cuda Upload supports F32 host data (optionally narrowing to F16/Q8_0), prequantized Q8_0 codes, or raw Q4_K/Q5_K/Q6_K/Q2_K/IQ2_XXS/IQ2_XS/IQ1_M/IQ4_NL/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS bytes today (got " + t.Dtype.String() + ")",
			Class: class,
		})
	}
	if class != MemoryWeights {
		if as != F32 {
			panic(&CUDAOpError{
				Op:    "Upload",
				Site:  site,
				Msg:   "cuda classed Upload supports only F32 activation/runtime uploads",
				Class: class,
			})
		}
		f := hb.F32()
		buf := c.dallocClass(t.Numel()*F32.Bytes(), class, site)
		out := makeF32TensorLike(c, t, buf)
		return finishF32Upload(out, f, func(values []float32) {
			C.fcuda_h2d(buf.ptr, unsafe.Pointer(&values[0]), C.size_t(len(values)*4))
		})
	}
	var store Dtype // the resident dtype the `as` request narrows f32 host weights to
	switch as {
	case F16:
		store = F16
	case Q8_0:
		store = Q8_0
	default:
		// Upload preserves F32 for requests without a narrowing route.
		store = F32
	}
	f := hb.F32()
	// Q8 narrowing and both native matmul routes require complete row blocks.
	// Validate before cache lookup: equal-Numel reshapes must not bypass this.
	// Keep the existing empty-host upload policy and separate rank validation.
	if store == Q8_0 && len(f) > 0 && len(t.Shape) == 2 && t.Shape[1]%q8DeviceBlock != 0 {
		panic(&CUDAOpError{
			Op: "Upload", Site: "upload-q8", Class: MemoryWeights,
			Msg: "cuda Upload(_, Q8_0) needs in divisible by " + itoaC(q8DeviceBlock),
		})
	}
	var hp uintptr
	if len(f) > 0 && len(t.Shape) >= 2 {
		hp = uintptr(unsafe.Pointer(&f[0]))
		// Guard against host-address reuse: a uintptr key does NOT keep the host slice alive, so
		// after GC a freed slice's address can be reused by a DIFFERENT tensor, false-hitting a
		// stale entry. Require the resident element count to match before sharing; a size mismatch
		// (e.g. a freed 1-D activation address reused for a 2-D weight) falls through to a fresh
		// upload, which overwrites the stale entry at this key below. Without this, a witness that
		// uploads both weights and 1-D activation vectors through this cache panicked in MatMul with
		// "index out of range [1] with length 1" when a 2-D weight aliased a stale 1-D entry.
		// Equal element counts do not identify the geometry used for F16 repacking.
		if cached, ok := uploadCache[ucKey{hp, store, t.Layout}]; ok && cached.Numel() == t.Numel() && slices.Equal(cached.Shape, t.Shape) {
			return cached // same host buffer already resident at this dtype/layout; share it
		}
	}
	switch store {
	case F16:
		return c.uploadF16(t, hb, f, hp)
	case Q8_0:
		return c.uploadQ8(t, hb, f, hp)
	default:
		out, buf := c.dev(t.Shape, F32)
		if len(f) > 0 {
			C.fcuda_h2d(buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
			c.accountImmutableWeightUpload(len(f)*F32.Bytes(), buf)
			if hp != 0 {
				buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, F32, t.Layout
				uploadCache[ucKey{hp, F32, t.Layout}] = out
			}
		}
		return out
	}
}

// uploadNVFP4 copies NewNVFP4's owned payload resident without widening it: packed E2M1
// bytes first, followed immediately by the raw E4M3FN block-scale bytes. Keeping the two
// regions in one allocation lets Free/CloneTensor use their ordinary single-buffer paths.
// NVFP4 deliberately does not share uploadCache entries: Tensor has no buffer refcount, so
// two Upload results must own distinct allocations that either caller may safely Free.
func (c *cudaBackend) uploadNVFP4(t Tensor, _ HostBuffer) Tensor {
	packed, scales, _ := nvfp4WeightPayload(t)
	want := len(packed) + len(scales)
	if len(packed) == 0 || len(scales) == 0 {
		panic(fmt.Errorf("%w: invalid packed CUDA upload payload", ErrNVFP4Shape))
	}
	buf := c.dallocWeight(want)
	out := makeTensor(c, NVFP4, RowMajor, append([]int(nil), t.Shape...), cloneQuantSpec(t.Quant), buf)
	// Two copies: a host row view keeps its scale plane apart from the codes.
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&packed[0]), C.size_t(len(packed)))
	C.fcuda_h2d(unsafe.Add(buf.ptr, len(packed)), unsafe.Pointer(&scales[0]), C.size_t(len(scales)))
	c.accountImmutableWeightUpload(want, buf)
	return out
}

// Host returns a host-addressable f32 view only for a host-resident tensor; a device
// (VRAM) tensor is not host-addressable, so it returns (nil, false) — the Caps.DeviceMemory
// contract that forces the loop through Read.
func (c *cudaBackend) Host(t Tensor) ([]float32, bool) {
	return hostF32(t)
}

// cudaReadOperations is scoped to ordinary Read. Read and nonparallel host
// injection tests access it only under cudaMu; no other operation uses this seam.
// The default operations retain the native allocation/conversion/free lifetime.
type cudaReadOperations struct {
	copy            func(device int, dst, src unsafe.Pointer, bytes uintptr) int
	allocConversion func(c *cudaBackend, bytes int) unsafe.Pointer
	convert         func(dst, src unsafe.Pointer, elements int, dtype Dtype)
	freeConversion  func(ptr unsafe.Pointer)
}

var cudaReadNative = cudaReadOperations{
	copy: func(device int, dst, src unsafe.Pointer, bytes uintptr) int {
		return int(C.fcuda_d2h_on_checked(C.int(device), dst, src, C.size_t(bytes)))
	},
	allocConversion: func(c *cudaBackend, bytes int) unsafe.Pointer {
		return c.dallocClass(bytes, MemoryScratchpad, "kv-read-conversion").ptr
	},
	convert: func(dst, src unsafe.Pointer, elements int, dtype Dtype) {
		cudaKVCheck(C.fcuda_kv_load_typed((*C.float)(dst), src, cudaKVElements(elements), C.int(dtype)))
	},
	freeConversion: func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
}

// Read copies device -> host f32. A successful device-0 read is a host fence (#482):
// synchronous D2H drains g_stream and advances the readiness generation. A copy
// on another device cannot certify device-0 work. Failed copies publish neither
// host data nor readiness. Read moves the FULL vector, which Argmax avoids.
func (c *cudaBackend) Read(t Tensor) []float32 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.readLocked(t)
}

// readLocked preserves Read semantics for an operation holding cudaMu.
func (c *cudaBackend) readLocked(t Tensor) []float32 {
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32 // host-resident: no device work or fence
	}
	db := t.buf.(*cudaBuf)
	if err := db.invalidStateError("buffer"); err != nil {
		panic(err)
	}
	// A nil or reconstructed Go latch cannot reopen a terminal native stream.
	// Refuse before compressor fences, conversion allocation or D2H publication.
	cudaV41CheckedSubmit()
	cudaV41CompressorSubmitLocked(db)
	pending := cudaV41CompressorReadBeginLocked(c)
	readOK := false
	defer func() { cudaV41CompressorReadEndLocked(c, pending, readOK) }()
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		if err := AdmitDevice(c, "read-d2h"); err != nil {
			panic(err)
		}
		src := db.ptr
		if t.Dtype == F16 || t.Dtype == BF16 {
			if db.device != 0 {
				panic("cuda: narrowed KV read requires its owning device")
			}
			src = cudaReadNative.allocConversion(c, len(out)*4)
			defer cudaReadNative.freeConversion(src)
			cudaReadNative.convert(src, db.ptr, len(out), t.Dtype)
		}
		status := cudaReadNative.copy(db.device, unsafe.Pointer(&out[0]), src, uintptr(len(out)*4))
		if status != 0 {
			if cudaQ56KMMQQualification && db.device == 0 {
				// Preserve the actual failed completion before deferred Go cleanup;
				// no retry or additional successful-path fence is introduced.
				cudaQ56KMMQRetirementFailedLocked(status)
			}
			// ObserveError does not classify CUDAOpError. Observe directly, then retain
			// the typed operation/site/status even when this backend has no latch.
			c.faultLatch.Observe(DeviceFaultExecution, "read-d2h", status)
			panic(&CUDAOpError{Op: "Read", Site: "read-d2h", Code: status, Msg: "device-to-host copy failed"})
		}
		if db.device == 0 {
			// Only this device-0 fence can certify work tracked by this generation.
			atomic.AddUint64(&c.fenceGen, 1)
		}
	}
	readOK = true
	return out
}

// Free releases a tensor's VRAM — both the codes buffer and any Q8_0 scale side-channel —
// and evicts its (host ptr, dtype, layout) entry from the upload cache so a re-upload re-stages.
// CloneTensor makes an independently owned device-to-device copy. It deliberately
// uses the backend stream rather than Download+Upload: prefix-cache hits must not cross
// the PCIe/host boundary merely to fork recurrent state.
func (c *cudaBackend) CloneTensor(t Tensor) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.view {
		return Tensor{}, fmt.Errorf("cuda: CloneTensor requires a live owning cuda tensor")
	}
	n := b.n
	if n <= 0 {
		return Tensor{}, fmt.Errorf("cuda: CloneTensor invalid allocation size %d", n)
	}
	dup := c.dallocClass(n, b.class, "prefix_snapshot")
	C.fcuda_d2d(dup.ptr, b.ptr, C.size_t(n))
	out := t
	out.Shape = append([]int(nil), t.Shape...)
	out.buf = dup
	return out, nil
}

func (c *cudaBackend) Free(t Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.cudaFreeLocked(t)
}

func (c *cudaBackend) cudaFreeLocked(t Tensor) {
	if db, ok := t.buf.(*cudaBuf); ok && db.view {
		return
	}
	if cudaQ56KMMVQReleaseBlocked() {
		return
	}
	// Cleanup retains handles rather than replacing an already selected error.
	if cudaQ4KAsyncReleaseBlockedLocked() {
		return
	}
	if db, ok := t.buf.(*cudaBuf); ok && db.ptr != nil {
		if db.graphOncePinned {
			return
		}
		c.graphOnceFreeDependencyLocked(db)
		if cudaV41CompressorDeferFreeLocked(c, db) {
			return
		}
	}
	cudaQ4KAsyncRequireDrainLocked(c)
	c.cudaFreeStorageLocked(t)
}

func (c *cudaBackend) cudaFreeStorageLocked(t Tensor) {
	if db, ok := t.buf.(*cudaBuf); ok && db.view {
		return
	}
	if cudaQ56KMMVQReleaseBlocked() {
		return
	}
	if db, ok := t.buf.(*cudaBuf); ok && db.ptr != nil {
		awaitAsyncUploadLocked(db.ptr) // never pool a buffer the copy stream still writes
		if db.host != 0 {
			// A later upload may have replaced this key with another resident.
			// Only the current owner may evict it; freeing an older buffer must
			// not discard the newer resident's zero-copy cache hit.
			key := ucKey{db.host, db.hostDt, db.hostLo}
			if cached, ok := uploadCache[key]; ok && cached.buf == db {
				delete(uploadCache, key)
			}
		}
		if db.scales != nil { // Q8_0 scale side-channel (#485)
			C.fcuda_free(db.scales)
			db.scales = nil
		}
		if db.device != 0 {
			C.fcuda_free_on(C.int(db.device), db.ptr)
		} else {
			C.fcuda_free(db.ptr)
		}
		db.ptr = nil
		db.hostKeep = nil
		if db.budgetedWeightBytes > 0 {
			c.dlUsed -= db.budgetedWeightBytes
			if c.dlUsed < 0 {
				c.dlUsed = 0
			}
			db.budgetedWeightBytes = 0
		}
		if db.managedWeight {
			if c.managedN > 0 {
				c.managedN--
			}
			db.managedWeight = false
		}
	}
}
