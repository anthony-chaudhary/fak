//go:build cuda && cgo

// cuda.go — the cgo wrapper that registers a CUDA device backend into the compute
// registry. It is compiled ONLY under `-tags cuda`; the default `go build ./cmd/fak`
// excludes it entirely, so the shipped artifact stays one pure-Go binary (DIRECTION.md
// rule 1 + reviewer check 3). When linked, it self-registers an *Approx* backend named
// "cuda" that the registry hands out via Pick("cuda") / FAK_BACKEND=cuda; the Reference
// (cpu-ref) stays the Default, so nothing silently runs on the GPU.
//
// Every method delegates to the flat C ABI in cuda_backend.h (implemented by
// cuda_kernels.cu, compiled offline by nvcc into libfakcuda.a). The Go side re-validates
// shapes and owns the Tensor type; the C side carries only device pointers + shapes — a
// seam that carries data, never trust.
//
// Build (WSL, no sudo; see build_cuda.sh):
//   nvcc -O3 -arch=sm_89 -c cuda_kernels.cu -o cuda_kernels.o
//   ar rcs libfakcuda.a cuda_kernels.o
//   CGO_CFLAGS="-I$CUDA_HOME/include" \
//   CGO_LDFLAGS="-L$PWD -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib" \
//   go build -tags cuda ./...

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakcuda -lcudart -lcublas -lcublasLt -lstdc++ -lm
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// cudaMu serializes all device ops: the cuBLAS handle and the single default stream are
// not safe under concurrent use, and this first backend favors obvious correctness over
// intra-backend concurrency (the async/stream seam is a tracked follow-up). Its critical
// sections are also the opt-in kernel profile's timed spans (cuda_profile.go).
var cudaMu cudaDeviceMutex

// graphEnabled gates the reusable CUDA-graph decode path (FAK_CUDA_GRAPH=1, or a host's
// EnableCUDAGraph). The path ships: it instantiates one exec and patches each later token's
// capture in with cudaGraphExecUpdate (7.5 -> 119-120 tok/s on SmolLM2-135M under WSL2,
// GPU.md §3b). It stays OFF by default until graph replay shows bit-exact parity and no
// tok/s regression on CUDA 13.x hardware (P-06, fak#13985): a sm_80 native-Linux run
// measured 56.6 off vs 55.7 on, and #4370 records a CUDA 13.0 graph/eager divergence. When
// on, only a capture-eligible session preallocates its fixed-capacity KV (GraphKVPreallocFor),
// sized to the host's resolved context by SetCUDAGraphKVCapacity.
var graphEnabled bool

// cudaGraphReplays counts graph replays whose fence completed, so a hardware witness can
// prove a graph-enabled run actually replayed instead of comparing eager with eager.
var cudaGraphReplays atomic.Uint64

// tf32Enabled gates the TF32 tensor-core math mode for the f32 SGEMM path (FAK_CUDA_TF32=1,
// Lever 4 of the H100-KERNEL-5X-ROADMAP). OFF by default so the witnessed device-vs-cpuref cosine
// floors hold unchanged on the pedantic FP32-core path; when on, the f32 prefill GEMMs route
// through Hopper/Ampere tensor cores at TF32 input precision (F32 accumulate) for a large
// compute-bound prefill speedup at a small, disclosed mantissa-only precision cost. Read once at
// init() from the env; a host may flip it post-init via EnableCUDATF32() (tf32_cuda.go). Applied
// through applyCUDATF32 so both entry points share the single cuBLAS handle mutation.
var tf32Enabled bool

// applyCUDATF32 pushes tf32Enabled down to the cuBLAS handle (fcuda_set_tf32). No-op-safe before
// the device is registered (fcuda_set_tf32 itself guards a nil handle), so init() and a post-init
// EnableCUDATF32() can both call it unconditionally. The handle mutation is serialized by cudaMu
// — the same mutex every device call already holds — so a concurrent forward never races the mode
// flip.
func applyCUDATF32() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if tf32Enabled {
		C.fcuda_set_tf32(1)
	} else {
		C.fcuda_set_tf32(0)
	}
}

func init() {
	var name [256]C.char
	var sm C.int
	var total C.size_t
	if C.fcuda_init(&name[0], 256, &sm, &total) != 0 {
		return // no reachable CUDA device — leave cpu-ref as the only backend
	}
	graphEnabled = os.Getenv("FAK_CUDA_GRAPH") == "1"
	cudaDev = &cudaBackend{
		name:        "cuda",
		tier:        "sm_" + itoaC(int(sm)),
		totalMem:    int64(total), // KEEP the device VRAM total — it used to be read and discarded
		budgetBytes: cudaBudgetBytes(int64(total)),
		faultLatch:  NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget),
	}
	Register(cudaDev)
	if cudaTimestampProfileEnabled(os.Getenv(cudaTimestampProfileEnv)) {
		installCUDAKernelProfile()
	}
	// Apply the TF32 tensor-core math mode from the env now that fcuda_init has created the
	// cuBLAS handle (Lever 4). Default-off, so a run that does not set FAK_CUDA_TF32 keeps the
	// pedantic FP32-core path and its witnessed cosine floors; a host can still flip it later
	// via EnableCUDATF32().
	tf32Enabled = os.Getenv("FAK_CUDA_TF32") == "1"
	applyCUDATF32()
}

// installCUDAKernelProfile creates the event pool and registers the "cuda" kernel profile
// reader. The reader is registered even when the pool fails, so an opted-in process reads
// the profile as unavailable rather than as off; with the profile off nothing registers.
func installCUDAKernelProfile() {
	status := int(C.fcuda_prof_init(C.int(cudaProfilePairs)))
	if status != 0 {
		fmt.Fprintf(os.Stderr, "fak-cuda: %s=1 but the event pool failed (status %d); kernel profile unavailable\n", cudaTimestampProfileEnv, status)
	}
	cudaMu.profile.Store(newCUDAKernelProfiler(cudaEventClock{}, cudaProfilePairs, status == 0))
	RegisterKernelProfileReader("cuda", cudaMu.readProfile)
}

// cudaEventClock is the profile's cgo clock over the fcuda_prof_* event pool.
type cudaEventClock struct{}

func cudaProfileEvent(m cudaProfileMark) C.int {
	e := m.slot * 2
	if m.end {
		e++
	}
	return C.int(e)
}

func (cudaEventClock) record(m cudaProfileMark) (idle bool, err error) {
	var wasIdle C.int
	if status := int(C.fcuda_prof_record(cudaProfileEvent(m), &wasIdle)); status != 0 {
		return false, fmt.Errorf("fcuda_prof_record: status %d", status)
	}
	return wasIdle != 0, nil
}

func (cudaEventClock) elapsedNS(from, to cudaProfileMark) (float64, error) {
	var ms C.float
	if status := int(C.fcuda_prof_elapsed_ms(cudaProfileEvent(from), cudaProfileEvent(to), &ms)); status != 0 {
		return 0, fmt.Errorf("fcuda_prof_elapsed_ms: status %d", status)
	}
	return float64(ms) * 1e6, nil
}

func (cudaEventClock) capturing() bool { return C.fcuda_prof_capturing() != 0 }

// Recycle returns every transient op-output buffer allocated since the last Recycle to the
// pooled allocator. The HAL calls it at each token boundary (after Read), where all
// intermediates are dead — the KV cache has already copied what it keeps, and weights are
// cached separately via Upload (never transient). cpu-ref has no Recycle, so this is a
// device-only fast path the HAL probes for.
func (c *cudaBackend) Recycle() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	for _, b := range c.transient {
		if b.ptr != nil {
			C.fcuda_free(b.ptr)
			b.ptr = nil
		}
	}
	clear(c.transient) // drop the handles so recycled cudaBufs are not kept reachable
	c.transient = c.transient[:0]
}

// compactTransientLocked drops transients already freed (ptr == nil), so a long prompt's
// per-op handles do not accumulate until the request's Recycle. Caller holds cudaMu.
func (c *cudaBackend) compactTransientLocked() {
	kept := c.transient[:0]
	for _, b := range c.transient {
		if b != nil && b.ptr != nil {
			kept = append(kept, b)
		}
	}
	clear(c.transient[len(kept):])
	c.transient = kept
}

// releaseTransientBuffers immediately returns an operation-owned subset of
// transients to the C-side pool and detaches it from the later Recycle sweep.
// The caller holds cudaMu and has already fenced/drained any kernel that may
// still reference the buffers.
func (c *cudaBackend) releaseTransientBuffers(buffers []*cudaBuf) {
	if len(buffers) == 0 {
		return
	}
	for _, target := range buffers {
		if target != nil && target.ptr != nil {
			C.fcuda_free(target.ptr)
			target.ptr = nil
		}
	}
	kept := c.transient[:0]
	for _, live := range c.transient {
		release := false
		for _, target := range buffers {
			if live == target {
				release = true
				break
			}
		}
		if !release {
			kept = append(kept, live)
		}
	}
	clear(c.transient[len(kept):])
	c.transient = kept
}

// cudaAllocationCountsForTest is the allocation witness for failure-path tests.
// live counts C-side allocations currently checked out of the pool; transient
// counts Go handles awaiting Recycle. Both are read under the allocator mutex so
// one assertion observes a coherent instant.
func (c *cudaBackend) cudaAllocationCountsForTest() (transient, live int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return len(c.transient), int(C.fcuda_live_allocations())
}

// TrimLarge frees cached allocator buckets larger than maxKeepBytes while preserving the
// small-buffer pool that makes steady-state decode cheap. GLM-DSA's sparse-attention gather
// creates large, one-off selK/selV buffers whose exact sizes grow with context; keeping one
// bucket per position can consume multiple GB on the largest sweeps.
func (c *cudaBackend) TrimLarge(maxKeepBytes int) {
	if maxKeepBytes < 0 {
		maxKeepBytes = 0
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_trim_pool_large(C.size_t(maxKeepBytes))
}

// GraphBegin/GraphEndLaunch capture one token's op stream into a CUDA graph and replay it
// as a single launch — the only way past the proven ~12 tok/s op-per-call WSL floor. The
// HAL calls GraphBegin after the (pre-capture) input upload, issues the layer ops (which
// the open capture records on g_stream), then GraphEndLaunch to instantiate+launch+fence
// before reading logits. The caller pins the goroutine to one OS thread for the token.
// Preconditions the HAL guarantees: pool warm (no cudaMalloc during capture) + fixed-
// capacity KV (no realloc during capture).
func (c *cudaBackend) GraphBegin() bool {
	if !graphEnabled {
		return false // per-token capture is a measured no-win; off unless FAK_CUDA_GRAPH=1
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	// #969: deepen every pool bucket BEFORE opening capture. The warm forward pools one set of
	// each transient size, but a single captured decodeChain holds several same-size transients
	// (the per-layer RMSNorm outputs, etc.) live at once, so the pool can drain mid-forward and a
	// same-size devTr then misses -> cudaMalloc, illegal during capture. graphPrewarmDepth spares
	// per bucket bound the peak same-size concurrency for any realistic layer count; this runs
	// outside capture, so the cudaMalloc's here are legal and the captured forward is then served
	// entirely from the free list.
	C.fcuda_graph_prewarm(C.int(graphPrewarmDepth))
	if C.fcuda_graph_begin() == 0 {
		c.capturing = true
		return true
	}
	return false
}

// graphPrewarmDepth is the spare-buffer headroom per pool size class seeded before each capture.
// The peak same-size live transient count in one decode forward scales with layer count (a couple
// per layer); 128 comfortably covers deep models while costing only a few hundred bytes × 128 per
// distinct small size — negligible VRAM, paid once per token boundary and almost entirely reused.
const graphPrewarmDepth = 128

// GraphEndLaunch instantiates, launches, and fences the captured CUDA graph for
// the token — the replay half of GraphBegin.
func (c *cudaBackend) GraphEndLaunch() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.capturing = false
	if code := int(C.fcuda_graph_end_launch()); code != 0 {
		C.fcuda_graph_reset()
		err := graphEndLaunchError(code)
		if code == graphFenceFailed {
			// An asynchronous fault inside the replayed graph (for example a captured GDN
			// decode) surfaces here; latch it so later operations refuse the suspect context.
			c.faultLatch.Observe(DeviceFaultExecution, err.Site, code)
		}
		panic(err)
	}
	cudaGraphReplays.Add(1)
}

// Qwen35GDNCaptureLegal reports that fcuda_qwen35_gdn_decode_f32 skips its in-op stream
// synchronize while a capture is open, so a hybrid decode step can be captured and its
// asynchronous faults surface at the graph fence (GraphEndLaunch).
func (c *cudaBackend) Qwen35GDNCaptureLegal() bool { return true }

// GraphReset drops the kept exec graph so a new session captures fresh (the exec is bound
// to one session's buffer addresses). The HAL calls it at NewBackendSession.
func (c *cudaBackend) GraphReset() {
	if !graphEnabled {
		return
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.capturing = false
	C.fcuda_graph_reset()
}

// GraphAbort ends and DISCARDS an open stream capture without instantiating or launching it.
// The HAL calls it on its panic path (a capture left open because something unwound past
// GraphEndLaunch). It clears the stream's capture state so the next op — and the next request
// — runs normally instead of failing "operation not permitted while capturing" forever.
func (c *cudaBackend) GraphAbort() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.capturing = false
	C.fcuda_graph_abort()
}

// IsCapturing reports whether a CUDA graph capture is currently open on g_stream (#10716).
// Constant and parameter uploads consult this to bypass host-side skip caching.
func (c *cudaBackend) IsCapturing() bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.capturing
}

// UploadConstantParam uploads parameter or constant float32 data into dst. Under
// stream capture (c.capturing is true), the upload is emitted unconditionally so
// that replayed graph executions are self-contained and contain every parameter
// assignment. Outside capture, uploads with matching paramKey may be elided (#10716).
func (c *cudaBackend) UploadConstantParam(dst Tensor, data []float32, paramKey uint64, lastUploaded *uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if !c.capturing && lastUploaded != nil && *lastUploaded == paramKey {
		return
	}
	if len(data) == 0 {
		if lastUploaded != nil {
			*lastUploaded = paramKey
		}
		return
	}
	buf := c.cudaBufForSubmit(dst)
	nbytes := len(data) * F32.Bytes()
	if nbytes > buf.n {
		panic(&CUDAOpError{
			Op:    "UploadConstantParam",
			Site:  "upload-constant-param",
			Msg:   "UploadConstantParam destination buffer too small",
			Class: buf.class,
		})
	}
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&data[0]), C.size_t(nbytes))
	if lastUploaded != nil {
		*lastUploaded = paramKey
	}
}

// DeviceMemory reports CUDA VRAM total plus the current free bytes from cudaMemGetInfo.
// If the runtime cannot provide a fresh snapshot, it preserves the capacity contract by
// returning the init-time total with free=FreeUnknown rather than failing closed.
func (c *cudaBackend) DeviceMemory() (total, free int64, known bool) {
	if c == nil || c.totalMem <= 0 {
		return 0, FreeUnknown, false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var freeMem C.size_t
	var totalMem C.size_t
	if C.fcuda_mem_info(&freeMem, &totalMem) == 0 && totalMem > 0 {
		return uint64ToCapInt64(uint64(totalMem)), uint64ToCapInt64(uint64(freeMem)), true
	}
	return c.totalMem, FreeUnknown, true
}

// ---- residency ------------------------------------------------------------------

func (c *cudaBackend) dallocClass(nbytes int, class MemoryClass, site string) *cudaBuf {
	if c.promptPanel && (class == MemoryScratchpad || class == MemoryActivation) {
		return c.dallocPromptPanel(nbytes, class, site)
	}
	p := C.fcuda_malloc(C.size_t(nbytes))
	if p == nil {
		p = C.fcuda_malloc_managed(C.size_t(nbytes))
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

// dallocDeviceOnlyClass is the strict allocator used by the whole-operation
// GDN path. General CUDA operations may fall back to managed memory after a
// device allocation miss; this path must instead return a typed refusal because
// UVM migration would invalidate its device-residency/zero-transfer contract.
// Caller holds cudaMu.
func (c *cudaBackend) dallocDeviceOnlyClass(nbytes int, class MemoryClass, site string) (*cudaBuf, error) {
	if err := qwen35GDNInjectedAllocationFailure(site, nbytes); err != nil {
		return nil, err
	}
	p := C.fcuda_malloc(C.size_t(nbytes))
	if p == nil {
		return nil, &Qwen35GDNAllocationError{Operand: site, Bytes: nbytes}
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes, class: class}, nil
}

// dallocManagedClass allocates directly from cudaMallocManaged. Used by the residency-budget
// path for cold weights. Caller holds cudaMu.
func (c *cudaBackend) dallocManagedClass(nbytes int, class MemoryClass, site string) *cudaBuf {
	p := C.fcuda_malloc_managed(C.size_t(nbytes))
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
	cudaMu.Lock()
	defer cudaMu.Unlock()
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
	if t.Dtype == Q4_K || t.Dtype == Q5_K || t.Dtype == Q6_K || t.Dtype == Q2_K {
		if as != t.Dtype {
			panic(&CUDAOpError{
				Op:    "Upload",
				Site:  site,
				Msg:   "cuda raw k-quant upload requires matching resident dtype",
				Class: class,
			})
		}
		return c.uploadRawKQuant(t, hb)
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
			Msg:   "cuda Upload supports F32 host data (optionally narrowing to F16/Q8_0), prequantized Q8_0 codes, or raw Q4_K/Q5_K/Q6_K/Q2_K bytes today (got " + t.Dtype.String() + ")",
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
	store := F32 // the resident dtype the `as` request narrows f32 host weights to
	switch as {
	case F16:
		store = F16
	case Q8_0:
		store = Q8_0
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
		if cached, ok := uploadCache[ucKey{hp, store, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached // same host buffer already resident at this dtype/layout; share it
		}
	}
	switch store {
	case F16:
		return c.uploadF16(t, hb, f, hp)
	case Q8_0:
		return c.uploadQ8(t, hb, f, hp)
	}
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

// uploadNVFP4 copies NewNVFP4's owned payload resident without widening it: packed E2M1
// bytes first, followed immediately by the raw E4M3FN block-scale bytes. Keeping the two
// regions in one allocation lets Free/CloneTensor use their ordinary single-buffer paths.
// NVFP4 deliberately does not share uploadCache entries: Tensor has no buffer refcount, so
// two Upload results must own distinct allocations that either caller may safely Free.
func (c *cudaBackend) uploadNVFP4(t Tensor, hb HostBuffer) Tensor {
	packed, scales, _ := nvfp4WeightPayload(t)
	payload := hb.I8()
	want := len(packed) + len(scales)
	if len(payload) != want || want <= 0 {
		panic(fmt.Errorf("%w: invalid packed CUDA upload payload", ErrNVFP4Shape))
	}
	buf := c.dallocWeight(want)
	out := makeTensor(c, NVFP4, RowMajor, append([]int(nil), t.Shape...), cloneQuantSpec(t.Quant), buf)
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&payload[0]), C.size_t(want))
	c.accountImmutableWeightUpload(want, buf)
	return out
}

// uploadQ8 narrows an f32 host weight [out,in] to a resident Q8_0 weight at H2D (#485): the f32 is
// quantized to per-block(32) int8 codes + f32 scales using the EXACT cpuref QuantizeQ8 scheme
// (d=amax/127, q8round), and both narrow operands are uploaded — codes to the buffer's ptr, scales
// to its scale side-channel. The f32 weight never becomes resident, so the VRAM footprint is the
// int8 size (codes) + a thin per-block scale band, not the f32 size. `in` must be divisible by 32.
func cudaUploadQuantPayload(buf *cudaBuf, codes []int8, scales []float32) {
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes)))
	C.fcuda_h2d(buf.scales, unsafe.Pointer(&scales[0]), C.size_t(len(scales)*F32.Bytes()))
}

func (c *cudaBackend) uploadQ8(t Tensor, hb HostBuffer, f []float32, hp uintptr) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8",
			Msg:   "cuda Upload(_, Q8_0) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := q8DeviceBlock
	nblk := in / blk
	if len(f) == 0 { // degenerate empty weight — mirror the F32 path's len==0 tolerance
		res, _ := c.devQ8(t.Shape, blk, out*nblk)
		return res
	}
	codes := make([]int8, out*in)
	scales := make([]float32, out*nblk)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := o*in + b*blk
			var amax float32
			for i := 0; i < blk; i++ {
				if a := absf(f[base+i]); a > amax {
					amax = a
				}
			}
			d := amax / 127
			scales[o*nblk+b] = d
			if d == 0 {
				continue
			}
			inv := 1.0 / d
			for i := 0; i < blk; i++ {
				codes[base+i] = q8round(f[base+i] * inv)
			}
		}
	}
	res, buf := c.devQ8(t.Shape, blk, len(scales))
	cudaUploadQuantPayload(buf, codes, scales)
	c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
	buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, Q8_0, t.Layout
	uploadCache[ucKey{hp, Q8_0, t.Layout}] = res
	return res
}

// uploadQ8Resident copies an ALREADY-quantized Q8_0 host weight resident with NO re-quantization:
// the int8 codes (HostBuffer.I8()) go to the buffer's ptr and the per-block(32) f32 scales
// (QuantSpec.Scale) to its scale side-channel — the SAME resident layout uploadQ8 produces, so
// k_q8_gemm consumes it unchanged. This is the memory-lean decode path (the HAL's weightHALQ8
// hands a NewQ8 host tensor); the f32-narrowing uploadQ8 above is the witness/quant-at-upload path.
func (c *cudaBackend) uploadQ8Resident(t Tensor, hb HostBuffer) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	if t.Quant == nil || t.Quant.Scale == nil {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) requires QuantSpec.Scale (per-block f32 scales)",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := t.Quant.Block
	if blk <= 0 || in%blk != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) needs in divisible by QuantSpec.Block (block=" + itoaC(blk) + ")",
			Class: MemoryWeights,
		})
	}
	codes := hb.I8()
	scales := t.Quant.Scale
	nblk := in / blk
	if len(codes) != 0 && len(codes) != out*in {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) code length " + itoaC(len(codes)) + " != out*in",
			Class: MemoryWeights,
		})
	}
	if len(scales) != out*nblk {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q8-resident",
			Msg:   "cuda Upload(Q8_0 host) scale length " + itoaC(len(scales)) + " != out*(in/block)",
			Class: MemoryWeights,
		})
	}
	var hp uintptr
	if len(codes) > 0 {
		hp = uintptr(unsafe.Pointer(&codes[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, Q8_0, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached
		}
	}
	res, buf := c.devQ8(t.Shape, blk, len(scales))
	if len(codes) > 0 {
		cudaUploadQuantPayload(buf, codes, scales)
		c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, Q8_0, t.Layout
		uploadCache[ucKey{hp, Q8_0, t.Layout}] = res
	}
	return res
}

// uploadQ2Resident copies an ALREADY-packed ternary Q2_0 host weight resident with NO
// re-quantization (#4872): the 2-bit codes (HostBuffer.I8(), out*in/4 bytes) go to the buffer's
// ptr and the per-block(=Block) f32 scales (QuantSpec.Scale) to its scale side-channel — the same
// resident (codes ptr + scales) shape uploadQ8Resident produces, so k_q2_0_gemm consumes it
// unchanged. The weight stays 0.25 byte/elem in VRAM; no dequant-to-f32 round trip.
func (c *cudaBackend) uploadQ2Resident(t Tensor, hb HostBuffer) Tensor {
	if len(t.Shape) != 2 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")",
			Class: MemoryWeights,
		})
	}
	if t.Quant == nil || t.Quant.Scale == nil {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) requires QuantSpec.Scale (per-block f32 scales)",
			Class: MemoryWeights,
		})
	}
	out, in := t.Shape[0], t.Shape[1]
	blk := t.Quant.Block
	if blk <= 0 || in%blk != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) needs in divisible by QuantSpec.Block (block=" + itoaC(blk) + ")",
			Class: MemoryWeights,
		})
	}
	if in%4 != 0 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) needs in divisible by 4 (2-bit codes, 4/byte)",
			Class: MemoryWeights,
		})
	}
	codes := hb.I8()
	scales := t.Quant.Scale
	nblk := in / blk
	if len(codes) != 0 && len(codes) != out*in/4 {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) code length " + itoaC(len(codes)) + " != out*in/4",
			Class: MemoryWeights,
		})
	}
	if len(scales) != out*nblk {
		panic(&CUDAOpError{
			Op:    "Upload",
			Site:  "upload-q2-resident",
			Msg:   "cuda Upload(Q2_0 host) scale length " + itoaC(len(scales)) + " != out*(in/block)",
			Class: MemoryWeights,
		})
	}
	var hp uintptr
	if len(codes) > 0 {
		hp = uintptr(unsafe.Pointer(&codes[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, Q2_0, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached
		}
	}
	res, buf := c.devQ2(t.Shape, blk, len(scales))
	if len(codes) > 0 {
		cudaUploadQuantPayload(buf, codes, scales)
		c.accountImmutableWeightUpload(len(codes)+len(scales)*F32.Bytes(), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, Q2_0, t.Layout
		uploadCache[ucKey{hp, Q2_0, t.Layout}] = res
	}
	return res
}

// devQ2 allocates a resident packed-ternary Q2_0 weight: an out*in/4-byte codes buffer (ptr, 2-bit
// codes, 4/byte) plus an nScales*4-byte f32 scale side-channel (scales). The Tensor carries a
// QuantSpec{Block} so the GEMM reconstructs nblk = in/block. Uses dev-family weight allocs so the
// resident-weight cache is never recycled out from under it.
func (c *cudaBackend) devQ2(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	return c.devQ2WithOperations(shape, block, nScales, c.dallocWeight,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryWeights, "q2-scale") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
}

// devQ2WithOperations runs under the upload owner's cudaMu. It owns both
// allocations until Tensor construction succeeds. Only the code allocation has
// placement charges under the existing CUDA policy; scales remain uncharged.
// This transaction ends before H2D and upload-cache publication.
func (c *cudaBackend) devQ2WithOperations(shape []int, block, nScales int, allocateCodes, allocateScales func(int) *cudaBuf, release func(unsafe.Pointer)) (Tensor, *cudaBuf) {
	out, in := shape[0], shape[1]
	var owners [2]*cudaBuf
	published := false
	defer func() {
		if published {
			return
		}
		for i, owner := range owners {
			if owner == nil {
				continue
			}
			ptr := owner.ptr
			owner.ptr = nil
			if i == 0 {
				c.dlUsed -= owner.budgetedWeightBytes
				owner.budgetedWeightBytes = 0
				if owner.managedWeight {
					c.managedN--
					owner.managedWeight = false
				}
				owner.scales = nil
				owner.scalesN = 0
			}
			if ptr != nil {
				func() {
					// Preserve the original construction panic and attempt the
					// other owned allocation even if cleanup itself panics.
					defer func() { _ = recover() }()
					release(ptr)
				}()
			}
		}
	}()
	owners[0] = allocateCodes(out * in / 4) // packed 2-bit codes, four per byte
	owners[1] = allocateScales(nScales * 4)
	buf, scales := owners[0], owners[1]
	buf.scales = scales.ptr
	buf.scalesN = scales.n
	q := &QuantSpec{Block: block, Axis: 2, Bits: 2, Symmetric: true}
	result := makeTensor(c, Q2_0, RowMajor, append([]int(nil), shape...), q, buf)
	published = true
	return result, buf
}

// uploadQ4K copies raw Q4_K super-block bytes resident (#485). The host tensor carries the bytes
// in its HostBuffer.I8() view (one int8 per byte); they are already narrow (144 bytes / 256 elems),
// so there is no quantize or dtype-narrow step — just an H2D into a uint8 VRAM buffer the
// dequant-fused GEMM tile consumes. Cached on (host ptr, Q4_K, layout) like every other upload.
func (c *cudaBackend) uploadRawKQuant(t Tensor, hb HostBuffer) Tensor {
	raw := hb.I8()
	var hp uintptr
	if len(raw) > 0 {
		hp = uintptr(unsafe.Pointer(&raw[0]))
		// element-count guard against host-address reuse (see uploadClass F32 path).
		if cached, ok := uploadCache[ucKey{hp, t.Dtype, t.Layout}]; ok && cached.Numel() == t.Numel() {
			return cached
		}
	}
	res, buf := c.devRawKQuant(t.Dtype, t.Shape, len(raw))
	if len(raw) > 0 {
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		c.accountImmutableWeightUpload(len(raw), buf)
		buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, t.Dtype, t.Layout
		uploadCache[ucKey{hp, t.Dtype, t.Layout}] = res
	}
	return res
}

// devQ8 allocates a resident Q8_0 weight: an out*in-byte int8 codes buffer (ptr) plus an
// nScales*4-byte f32 scale side-channel (scales). The Tensor carries a QuantSpec{Block} so the
// GEMM kernel reconstructs nblk = in/block. Weights use dev-family allocs, never devTr, so the
// resident-weight cache is never recycled out from under them.
func (c *cudaBackend) devQ8(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	return c.devQ8WithOperations(shape, block, nScales, c.dallocWeight,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryWeights, "q8-scale") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
}

// devQ8WithOperations runs under the upload owner's cudaMu. It owns both
// allocations until Tensor construction succeeds. Only the code allocation has
// placement charges under the existing CUDA policy; scales remain uncharged.
// This transaction ends before H2D and upload-cache publication.
func (c *cudaBackend) devQ8WithOperations(shape []int, block, nScales int, allocateCodes, allocateScales func(int) *cudaBuf, release func(unsafe.Pointer)) (Tensor, *cudaBuf) {
	out, in := shape[0], shape[1]
	var owners [2]*cudaBuf
	published := false
	defer func() {
		if published {
			return
		}
		for i, owner := range owners {
			if owner == nil {
				continue
			}
			ptr := owner.ptr
			owner.ptr = nil
			if i == 0 {
				c.dlUsed -= owner.budgetedWeightBytes
				owner.budgetedWeightBytes = 0
				if owner.managedWeight {
					c.managedN--
					owner.managedWeight = false
				}
				owner.scales = nil
				owner.scalesN = 0
			}
			if ptr != nil {
				func() {
					// Preserve the original construction panic and attempt the
					// other owned allocation even if cleanup itself panics.
					defer func() { _ = recover() }()
					release(ptr)
				}()
			}
		}
	}()
	owners[0] = allocateCodes(out * in) // int8 codes, 1 byte each
	owners[1] = allocateScales(nScales * 4)
	buf, scales := owners[0], owners[1]
	buf.scales = scales.ptr
	buf.scalesN = scales.n
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	result := makeTensor(c, Q8_0, RowMajor, append([]int(nil), shape...), q, buf)
	published = true
	return result, buf
}

// uploadF16 narrows an f32 host weight to a resident F16 weight at H2D (#484). The f32 is staged
// in a transient device buffer, converted to __half by a device kernel — row-major in place, or
// ColMajor transpose-repacked ([out,in] -> col-major) — and the stage is freed. The narrow runs
// on device (one conversion implementation, identical numerics to the GEMM's own half cast),
// never on the host.
func (c *cudaBackend) uploadF16(t Tensor, hb HostBuffer, f []float32, hp uintptr) Tensor {
	out, buf := c.devF16(t.Shape, t.Layout)
	if len(f) == 0 {
		return out
	}
	stage := c.allocateF16UploadStage(buf, len(f)*4,
		func(bytes int) *cudaBuf { return c.dallocClass(bytes, MemoryScratchpad, "f16-stage") },
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
	C.fcuda_h2d(stage.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
	if t.Layout == ColMajor && len(t.Shape) == 2 {
		C.fcuda_f32_to_f16_T(buf.ptr, (*C.float)(stage.ptr), C.int(t.Shape[0]), C.int(t.Shape[1]))
	} else {
		C.fcuda_f32_to_f16(buf.ptr, (*C.float)(stage.ptr), C.int(len(f)))
	}
	C.fcuda_free(stage.ptr)
	c.accountImmutableWeightUpload(len(f)*F32.Bytes(), buf)
	buf.host, buf.hostKeep, buf.hostDt, buf.hostLo = hp, hb, F16, t.Layout
	uploadCache[ucKey{hp, F16, t.Layout}] = out
	return out
}

// allocateF16UploadStage owns the unpublished destination only while the scratch
// allocation is pending. Caller holds cudaMu. A failed allocation must not strand
// the destination or its logical placement charge. H2D, conversion, stage release
// and upload-cache publication occur after this boundary and are unchanged.
func (c *cudaBackend) allocateF16UploadStage(dst *cudaBuf, bytes int, allocate func(int) *cudaBuf, release func(unsafe.Pointer)) *cudaBuf {
	allocated := false
	defer func() {
		if allocated {
			return
		}
		ptr := dst.ptr
		dst.ptr = nil
		c.dlUsed -= dst.budgetedWeightBytes
		dst.budgetedWeightBytes = 0
		if dst.managedWeight {
			c.managedN--
			dst.managedWeight = false
		}
		if ptr != nil {
			func() {
				// Cleanup is best effort and must preserve the allocation panic.
				defer func() { _ = recover() }()
				release(ptr)
			}()
		}
	}()
	stage := allocate(bytes)
	allocated = true
	return stage
}

// Host returns a host-addressable f32 view only for a host-resident tensor; a device
// (VRAM) tensor is not host-addressable, so it returns (nil, false) — the Caps.DeviceMemory
// contract that forces the loop through Read.
func (c *cudaBackend) Host(t Tensor) ([]float32, bool) {
	return hostF32(t)
}

// Read is a host fence (#482): it copies device -> host f32 and, because that synchronous d2h
// drains g_stream, bumps the fence generation so every async buffer enqueued before it flips
// to Ready. It also moves the FULL vector host-ward — the costly path greedy decode avoids by
// using Argmax instead (the witness counts these bytes).
func (c *cudaBackend) Read(t Tensor) []float32 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32 // host-resident: nothing crosses the bus and no device work is fenced
	}
	db := t.buf.(*cudaBuf)
	if err := db.invalidStateError("buffer"); err != nil {
		panic(err)
	}
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		if db.device != 0 {
			C.fcuda_d2h_on(C.int(db.device), unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
		} else {
			C.fcuda_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
		}
		atomic.AddUint64(&c.fenceGen, 1) // stream drained: prior enqueued work is now materialized
	}
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
	if !ok || b == nil || b.ptr == nil {
		return Tensor{}, fmt.Errorf("cuda: CloneTensor requires a live cuda tensor")
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
	if db, ok := t.buf.(*cudaBuf); ok && db.ptr != nil {
		if db.host != 0 {
			// evict the exact (ptr, dtype, layout) entry so a re-upload of the same host buffer re-stages
			delete(uploadCache, ucKey{db.host, db.hostDt, db.hostLo})
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

// ---- primitives -----------------------------------------------------------------

func (c *cudaBackend) cf(t Tensor) *C.float {
	return (*C.float)(c.cudaBufForSubmit(t).ptr)
}

// cptr is the raw device pointer (void*), for dtypes whose element type is not *C.float — the
// F16 weight buffer (__half) the HGEMM path reads.
func (c *cudaBackend) cptr(t Tensor) unsafe.Pointer {
	return c.cudaBufForSubmit(t).ptr
}

// colMajorFlag reports w's HGEMM layout selector: 1 when the weight was transpose-repacked to
// column-major at H2D (op_N), 0 for the row-major SGEMM recipe (op_T).
func colMajorFlag(w Tensor) C.int {
	if w.Layout == ColMajor {
		return 1
	}
	return 0
}

// SupportsDeviceWeightDtype reports the exact dtype set cudaBackend.MatMul has a case
// for, matching that switch: F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, Q2_0, Q2_K and NVFP4. Every other
// dtype falls to the switch's CUDAOpError default, so it is reported false here.
func (c *cudaBackend) SupportsDeviceWeightDtype(dt Dtype) bool {
	switch dt {
	case F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, Q2_0, Q2_K, NVFP4:
		return true
	default:
		return false
	}
}

const cudaNVFP4IntMax = int(^uint32(0) >> 1)

type cudaNVFP4ResidentMeta struct {
	buf              *cudaBuf
	packedBytes      int
	linearScaleBytes int
	globalScale      float32
}

// checkedCUDANVFP4Resident validates only the canonical packed-plus-linear
// resident contract shared by W4A16 and explicit W4A4. Lt tiled geometry is
// intentionally absent here so ordinary uploads retain their original shape.
func (c *cudaBackend) checkedCUDANVFP4Resident(t Tensor) (cudaNVFP4ResidentMeta, error) {
	if t.Backend() != c || t.Dtype != NVFP4 || t.Layout != RowMajor {
		return cudaNVFP4ResidentMeta{}, fmt.Errorf("%w: requires an owned row-major NVFP4 tensor", ErrNVFP4Shape)
	}
	packedBytes, linearBytes, err := nvfp4PayloadShape(t.Shape)
	if err != nil {
		return cudaNVFP4ResidentMeta{}, err
	}
	q := t.Quant
	if q == nil || q.Block != NVFP4BlockSize || q.Axis != 2 || q.Bits != 4 ||
		!q.Symmetric || q.StaticAct || len(q.ZeroPoint) != 0 || len(q.Scale) != 1 {
		return cudaNVFP4ResidentMeta{}, fmt.Errorf("%w: invalid CUDA NVFP4 quantization metadata", ErrNVFP4Scale)
	}
	global := q.Scale[0]
	if !nvfp4Finite(global) || global < 0 {
		return cudaNVFP4ResidentMeta{}, fmt.Errorf("%w: tensor scale must be finite and non-negative", ErrNVFP4Scale)
	}
	b, ok := t.buf.(*cudaBuf)
	if !ok || b == nil || b.ptr == nil || b.n != packedBytes+linearBytes || b.scales != nil {
		return cudaNVFP4ResidentMeta{}, fmt.Errorf("%w: invalid resident CUDA NVFP4 payload", ErrNVFP4Shape)
	}
	return cudaNVFP4ResidentMeta{b, packedBytes, linearBytes, global}, nil
}

func cudaNVFP4SM(tier string) (int, bool) {
	value, ok := strings.CutPrefix(tier, "sm_")
	if !ok {
		return 0, false
	}
	sm, err := strconv.Atoi(value)
	return sm, err == nil
}

func cudaNVFP4LtUnsupportedStatus(status C.int) bool {
	return status == C.FCUDA_NVFP4_LT_CAPTURE ||
		status == C.FCUDA_NVFP4_LT_COMPILE_UNSUPPORTED ||
		status == C.FCUDA_NVFP4_LT_RUNTIME_UNSUPPORTED ||
		status == C.FCUDA_NVFP4_LT_WORKSPACE_UNAVAILABLE ||
		status == C.FCUDA_NVFP4_LT_HEURISTIC_UNAVAILABLE
}

// cudaNVFP4Operands validates every host-visible part of the packed W4A16 contract before
// a pointer crosses the C ABI. Raw E4M3FN bytes were validated by NewNVFP4 and again by
// uploadNVFP4; after upload they are immutable device memory and need no host readback.
func (c *cudaBackend) cudaNVFP4Operands(w, x Tensor, rows int, batched bool) (out, in int, global float32) {
	resident, err := c.checkedCUDANVFP4Resident(w)
	if err != nil {
		panic(err)
	}
	out, in = w.Shape[0], w.Shape[1]
	global = resident.globalScale
	validShape := !batched && rows == 1 && len(x.Shape) == 1 && x.Shape[0] == in
	if batched {
		validShape = rows > 0 && len(x.Shape) == 2 && x.Shape[0] == rows && x.Shape[1] == in
	}
	if !validShape || x.Dtype != F32 || x.Layout != RowMajor || x.Quant != nil {
		panic(fmt.Errorf("%w: CUDA requires matching row-major F32 activations", ErrNVFP4Input))
	}
	xb, ok := x.buf.(*cudaBuf)
	values, valuesOK := fp8Block32Product(rows, in)
	bytes, bytesOK := fp8Block32Product(values, F32.Bytes())
	outputs, outputsOK := fp8Block32Product(rows, out)
	if !ok || xb == nil || xb.ptr == nil || !valuesOK || !bytesOK || !outputsOK || xb.n != bytes ||
		out > cudaNVFP4IntMax || in > cudaNVFP4IntMax || rows > cudaNVFP4IntMax || outputs > cudaNVFP4IntMax {
		panic(fmt.Errorf("%w: invalid or unrepresentable CUDA activation payload", ErrNVFP4Input))
	}
	return out, in, global
}
func (c *cudaBackend) MatMulNVFP4(w, a Tensor) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	wResident, err := c.checkedCUDANVFP4Resident(w)
	if err != nil {
		return Tensor{}, err
	}
	aResident, err := c.checkedCUDANVFP4Resident(a)
	if err != nil {
		return Tensor{}, err
	}
	out, k, rows := w.Shape[0], w.Shape[1], a.Shape[0]
	if a.Shape[1] != k || k%32 != 0 || out%4 != 0 {
		return Tensor{}, fmt.Errorf("%w: Lt W4A4 needs matching K divisible by 32 and output width divisible by 4", ErrNVFP4Shape)
	}
	count, ok := fp8Block32Product(rows, out)
	if !ok || count <= 0 || out > cudaNVFP4IntMax || k > cudaNVFP4IntMax || rows > cudaNVFP4IntMax {
		return Tensor{}, fmt.Errorf("%w: invalid or unrepresentable output shape [%d %d]", ErrNVFP4Shape, rows, out)
	}
	alpha := nvfp4RoundedF32(wResident.globalScale * aResident.globalScale)
	if !nvfp4Finite(alpha) || (wResident.globalScale != 0 && aResident.globalScale != 0 && alpha == 0) {
		return Tensor{}, fmt.Errorf("%w: invalid tensor-global product", ErrNVFP4Scale)
	}
	if c.capturing {
		return Tensor{}, fmt.Errorf("%w: CUDA graph capture is open", ErrNVFP4GEMMUnsupported)
	}
	sm, ok := cudaNVFP4SM(c.tier)
	if !ok {
		return Tensor{}, fmt.Errorf("%w: unknown CUDA architecture %q", ErrNVFP4GEMMUnsupported, c.tier)
	}
	if sm != 100 && sm != 103 && sm != 120 {
		return Tensor{}, fmt.Errorf("%w: sm_%d is not an NVFP4 cuBLASLt target", ErrNVFP4GEMMUnsupported, sm)
	}
	var libraryVersion C.size_t
	status := C.fcuda_nvfp4_w4a4_preflight(&libraryVersion)
	if status != C.FCUDA_NVFP4_LT_OK {
		if cudaNVFP4LtUnsupportedStatus(status) {
			return Tensor{}, fmt.Errorf("%w: cuBLASLt preflight status %d", ErrNVFP4GEMMUnsupported, int(status))
		}
		return Tensor{}, fmt.Errorf("cuda NVFP4 cuBLASLt preflight failed: status %d", int(status))
	}
	eligible, skipAlgo66 := nvfp4CUBLASLtPolicy(sm, int(libraryVersion))
	if !eligible {
		return Tensor{}, fmt.Errorf("%w: sm_%d with cuBLASLt %d is unqualified", ErrNVFP4GEMMUnsupported, sm, uint64(libraryVersion))
	}
	_, wScaleBytes, _, err := nvfp4ScaleLayoutGeometry(out, k/NVFP4BlockSize)
	if err != nil {
		return Tensor{}, err
	}
	_, aScaleBytes, _, err := nvfp4ScaleLayoutGeometry(rows, k/NVFP4BlockSize)
	if err != nil {
		return Tensor{}, err
	}
	nbytes, ok := fp8Block32Product(count, F32.Bytes())
	if !ok {
		return Tensor{}, fmt.Errorf("%w: output byte size overflows", ErrNVFP4Shape)
	}
	allocDevice := func(bytes int, class MemoryClass, site string) (*cudaBuf, error) {
		p := C.fcuda_malloc(C.size_t(bytes))
		if p == nil {
			return nil, &DeviceAllocError{Bytes: bytes, Site: site, Class: class}
		}
		return &cudaBuf{ptr: unsafe.Pointer(p), n: bytes, class: class}, nil
	}
	yBuf, err := allocDevice(nbytes, MemoryActivation, "nvfp4-w4a4-output")
	if err != nil {
		return Tensor{}, err
	}
	wScaleBuf, err := allocDevice(wScaleBytes, MemoryScratchpad, "nvfp4-w4a4-weight-scales")
	if err != nil {
		C.fcuda_free(yBuf.ptr)
		yBuf.ptr = nil
		return Tensor{}, err
	}
	aScaleBuf, err := allocDevice(aScaleBytes, MemoryScratchpad, "nvfp4-w4a4-activation-scales")
	if err != nil {
		C.fcuda_free(wScaleBuf.ptr)
		wScaleBuf.ptr = nil
		C.fcuda_free(yBuf.ptr)
		yBuf.ptr = nil
		return Tensor{}, err
	}
	y := makeTensor(c, F32, RowMajor, []int{rows, out}, nil, yBuf)
	wLinearScale := unsafe.Add(wResident.buf.ptr, wResident.packedBytes)
	aLinearScale := unsafe.Add(aResident.buf.ptr, aResident.packedBytes)
	skip := C.int(0)
	if skipAlgo66 {
		skip = 1
	}
	status = C.fcuda_nvfp4_w4a4_f32(
		(*C.uint8_t)(wResident.buf.ptr), (*C.uint8_t)(wLinearScale), (*C.uint8_t)(wScaleBuf.ptr), C.size_t(wScaleBytes),
		(*C.uint8_t)(aResident.buf.ptr), (*C.uint8_t)(aLinearScale), (*C.uint8_t)(aScaleBuf.ptr), C.size_t(aScaleBytes),
		(*C.float)(yBuf.ptr), C.int(out), C.int(k), C.int(rows), C.float(alpha), skip)
	if status != C.FCUDA_NVFP4_LT_OK {
		// No Lt enqueue succeeded. Any preceding permutation work is ordered on
		// g_stream, so returning these buffers to the same-stream pool is safe.
		C.fcuda_free(aScaleBuf.ptr)
		aScaleBuf.ptr = nil
		C.fcuda_free(wScaleBuf.ptr)
		wScaleBuf.ptr = nil
		C.fcuda_free(yBuf.ptr)
		yBuf.ptr = nil
		if status == C.FCUDA_NVFP4_LT_GEOMETRY {
			return Tensor{}, fmt.Errorf("%w: cuBLASLt rejected validated geometry", ErrNVFP4Shape)
		}
		if cudaNVFP4LtUnsupportedStatus(status) {
			return Tensor{}, fmt.Errorf("%w: cuBLASLt status %d", ErrNVFP4GEMMUnsupported, int(status))
		}
		return Tensor{}, fmt.Errorf("cuda NVFP4 cuBLASLt call failed: status %d", int(status))
	}
	generation := atomic.LoadUint64(&c.fenceGen)
	for _, buffer := range []*cudaBuf{yBuf, wScaleBuf, aScaleBuf} {
		buffer.be = c
		buffer.bornGen = generation
	}
	// Retain both scale images until the normal post-fence Recycle. They cannot
	// re-enter g_pool while the asynchronous Lt enqueue may still read them.
	c.transient = append(c.transient, yBuf, wScaleBuf, aScaleBuf)
	return y, nil
}

// cudaNVFP4LtCalls is an unexported QA witness. Tests snapshot it around
// MatMulNVFP4 and synchronize by Read before asserting a one-call delta.
func (c *cudaBackend) cudaNVFP4LtCallsForTest() uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return uint64(C.fcuda_nvfp4_w4a4_lt_calls())
}

// MatMul computes y = x @ Wᵀ as a decode GEMV (P=1), dispatching on the weight dtype to the
// SGEMM (F32), tensor-core HGEMM (F16), or native Q8_0/Q4_K device GEMV; output is F32-resident.
func (c *cudaBackend) MatMul(w, x Tensor) Tensor {
	cudaMu.Lock()
	trace := computetrace.Enabled()
	started := time.Now()
	if trace {
		C.fcuda_event_elapsed_ms_start()
	}
	defer func() {
		if trace {
			ms := float64(C.fcuda_event_elapsed_ms_end())
			computetrace.Record(computetrace.Event{Operation: "matmul", Phase: "kernel", Backend: c.Name(), Device: "cuda:0", Kernel: w.Dtype.String() + "_matmul", StartedAt: started.UTC(), DurationNS: int64(ms * 1e6), TimerDomain: "cuda_event", Bytes: int64((w.Numel() + x.Numel()) * w.Dtype.Bytes()), Shapes: [][]int{w.Shape, x.Shape}, ProvenanceDigest: computetrace.Digest(c.Name(), w.Dtype.String(), "matmul")})
		}
		cudaMu.Unlock()
	}()
	var out, in int
	var nvfp4Global float32
	if w.Dtype == NVFP4 {
		out, in, nvfp4Global = c.cudaNVFP4Operands(w, x, 1, false)
	} else {
		out, in = w.Shape[0], w.Shape[1]
	}
	var y Tensor
	switch w.Dtype {
	case F32:
		y, _ = c.devTr([]int{out}, F32)
		C.fcuda_matmul_f32(c.cf(w), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
	case F16:
		// tensor-core HGEMM (#484): F16 weight, f32 activation (converted to __half C-side), f32
		// accumulate/output. P=1 (decode GEMV); the activation x stays F32-resident.
		y, _ = c.devTr([]int{out}, F32)
		C.fcuda_matmul_f16(c.cptr(w), c.cf(x), c.cf(y), C.int(out), C.int(in), 1, colMajorFlag(w))
	case Q8_0:
		// native Q8_0 GEMV (#485): int8 codes + per-block f32 scales resident, the activation
		// quantized to int8 ON DEVICE; integer per-block dot scaled by (weight·activation block
		// scales), F32 accumulate. No dequant-to-f32 round trip — the weight stays int8 in VRAM.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		cudaGEMMStatus("MatMul", int(C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(x), c.cf(y),
			C.int(out), C.int(in), 1, C.int(w.Quant.Block))), cudaQ8MatMulScratch(out, in, 1, w.Quant.Block))
	case Q4_K:
		// native Q4_K GEMV (#485): the dequant (w = d·scale·code − dmin·min) is fused into the
		// GEMM tile straight off the resident super-block bytes; the weight stays int4 in VRAM.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		cudaGEMMStatus("MatMul", int(C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)), cudaGEMMScratch{})
	case Q5_K:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		C.fcuda_q5k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
	case Q6_K:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		C.fcuda_q6k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
	case Q2_K:
		// native Q2_K GEMV (#11945): dequant fused into the GEMV tile straight off the
		// resident super-block bytes; the weight stays 2-bit (84 bytes / 256 elements) in VRAM.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		cudaGEMMStatus("MatMul", int(C.fcuda_q2k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)), cudaGEMMScratch{})
	case Q2_0:
		// native packed-ternary Q2_0 GEMV (#4872): 2-bit codes + per-block f32 scales resident, the
		// signed indicator unpacked and multiply-accumulated against the f32 activation on device
		// (no dequant-to-f32, no activation quant); one block scale folded per block, F32 accumulate.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		C.fcuda_q2_0_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(x), c.cf(y),
			C.int(out), C.int(in), 1, C.int(w.Quant.Block))
	case NVFP4:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{out}, F32)
		cudaGEMMStatus("MatMul", int(C.fcuda_nvfp4_w4a16_f32(
			(*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1, C.float(nvfp4Global))), cudaGEMMScratch{})
	default:
		panic(&CUDAOpError{
			Op:    "MatMul",
			Site:  "matmul-weight-dtype",
			Msg:   "cuda MatMul supports F32/F16/Q8_0/Q4_K/Q5_K/Q6_K/Q2_0/Q2_K/NVFP4 weights today (got " + w.Dtype.String() + "); other quantized device GEMM is a tracked follow-up",
			Class: MemoryActivation,
		})
	}
	// Shape post-condition (#972): the GEMV must yield exactly `out` rows. A wrong-shaped result
	// (the sm_80 / CUDA-13.0 witness saw a non-block-aligned out=257 come back as 64) is a launch
	// or binding fault — fail loud at the call site naming the dtype + dims, not silently downstream.
	if n := y.Numel(); n != out {
		panic(&CUDAOpError{
			Op:    "MatMul",
			Site:  "matmul-shape-postcondition",
			Msg:   "cuda MatMul " + w.Dtype.String() + " out=" + itoaC(out) + " in=" + itoaC(in) + " produced " + itoaC(n) + " rows",
			Class: MemoryActivation,
		})
	}
	return y
}

// BatchedMatMul computes the prefill GEMM Y = X @ Wᵀ for P activation rows, dispatching on the
// weight dtype to the SGEMM (F32), tensor-core HGEMM (F16), or native Q8_0/Q4_K device GEMM.
func (c *cudaBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var out, in int
	var nvfp4Global float32
	if w.Dtype == NVFP4 {
		out, in, nvfp4Global = c.cudaNVFP4Operands(w, X, P, true)
	} else {
		out, in = w.Shape[0], w.Shape[1]
	}
	var y Tensor
	switch w.Dtype {
	case F32:
		y, _ = c.devTr([]int{P, out}, F32)
		C.fcuda_matmul_f32(c.cf(w), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
	case F16:
		// tensor-core HGEMM (#484): the prefill GEMM where fp16/tensor-cores pay off most.
		y, _ = c.devTr([]int{P, out}, F32)
		C.fcuda_matmul_f16(c.cptr(w), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P), colMajorFlag(w))
	case Q8_0:
		// native Q8_0 prefill GEMM (#485): each of the P activation rows is quantized to int8 on
		// device, then the per-block integer dot against the resident int8 weight, F32 accumulate.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		cudaGEMMStatus("BatchedMatMul", int(C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(X), c.cf(y),
			C.int(out), C.int(in), C.int(P), C.int(w.Quant.Block))), cudaQ8MatMulScratch(out, in, P, w.Quant.Block))
	case Q4_K:
		// native Q4_K prefill GEMM (#485): dequant fused into the tile off the resident super-block
		// bytes, dotted with each of the P f32 activation rows, F32 accumulate.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		cudaGEMMStatus("BatchedMatMul", int(C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))), cudaKQuantScratch(out, in, P))
	case Q5_K:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		C.fcuda_q5k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
	case Q6_K:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		C.fcuda_q6k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
	case Q2_K:
		// native Q2_K prefill GEMM (#11945): dequant fused into the tile off the resident super-block
		// bytes, dotted with each of the P f32 activation rows, F32 accumulate.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		cudaGEMMStatus("BatchedMatMul", int(C.fcuda_q2k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))), cudaKQuantScratch(out, in, P))
	case Q2_0:
		// native packed-ternary Q2_0 prefill GEMM (#4872): each of the P f32 activation rows dotted
		// against the resident 2-bit ternary weight (unpacked signed indicator, one block scale folded
		// per block), F32 accumulate. The weight stays 0.25 byte/elem in VRAM.
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		C.fcuda_q2_0_matmul_f32((*C.uint8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(X), c.cf(y),
			C.int(out), C.int(in), C.int(P), C.int(w.Quant.Block))
	case NVFP4:
		wb := c.cudaBufForSubmit(w)
		y, _ = c.devTr([]int{P, out}, F32)
		cudaGEMMStatus("BatchedMatMul", int(C.fcuda_nvfp4_w4a16_f32(
			(*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P), C.float(nvfp4Global))), cudaGEMMScratch{})
	default:
		panic(&CUDAOpError{
			Op:    "BatchedMatMul",
			Site:  "batched-matmul-weight-dtype",
			Msg:   "cuda BatchedMatMul supports F32/F16/Q8_0/Q4_K/Q5_K/Q6_K/Q2_0/Q2_K/NVFP4 weights today (got " + w.Dtype.String() + ")",
			Class: MemoryActivation,
		})
	}
	// Shape post-condition (#972): the batched GEMM must yield exactly P*out elements. Catch a
	// short/wrong-shaped device result loud at the call site rather than as silent garbage.
	if n, want := y.Numel(), P*out; n != want {
		panic(&CUDAOpError{
			Op:    "BatchedMatMul",
			Site:  "batched-matmul-shape-postcondition",
			Msg:   "cuda BatchedMatMul " + w.Dtype.String() + " out=" + itoaC(out) + " in=" + itoaC(in) + " P=" + itoaC(P) + " produced " + itoaC(n) + " want " + itoaC(want),
			Class: MemoryActivation,
		})
	}
	return y
}

// RMSNorm runs the RMS-normalization kernel over each row of x (one weight-width row at a time),
// returning a new F32-resident tensor of x's shape.
func (c *cudaBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := weight.Numel()
	rows := x.Numel() / n
	y, _ := c.devTr(append([]int(nil), x.Shape...), F32)
	C.fcuda_rmsnorm_f32(c.cf(x), c.cf(weight), c.cf(y), C.int(rows), C.int(n), C.float(eps))
	return y
}

// RoPE returns a NEW tensor (value semantics, matching cpuref): copy then rotate in place.
func (c *cudaBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	xbuf := c.cudaBufForSubmit(x)
	y, ybuf := c.devTr(append([]int(nil), x.Shape...), F32)
	C.fcuda_d2d(ybuf.ptr, xbuf.ptr, C.size_t(x.Numel()*4))
	C.fcuda_rope_f32(c.cf(y), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return y
}

// PartialRoPEQK applies Qwen3.5/3.6's rotate-half RoPE to rotaryDim values of
// every Q/K head while preserving the unrotated tail. Both results stay resident.
func (c *cudaBackend) PartialRoPEQK(
	q, k Tensor,
	pos, nQHeads, nKHeads, headDim, rotaryDim int,
	theta float64,
) (Tensor, Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	qOut, _ := c.devTr([]int{nQHeads * headDim}, F32)
	kOut, _ := c.devTr([]int{nKHeads * headDim}, F32)
	C.fcuda_partial_rope_qk_f32(
		c.cf(q), c.cf(k), c.cf(qOut), c.cf(kOut),
		C.int(pos), C.int(nQHeads), C.int(nKHeads),
		C.int(headDim), C.int(rotaryDim), C.double(theta),
	)
	return qOut, kOut
}

// SigmoidMulInPlace applies x *= sigmoid(gate) without crossing the host boundary.
func (c *cudaBackend) SigmoidMulInPlace(x, gate Tensor) {
	if x.Numel() != gate.Numel() {
		panic(&CUDAOpError{
			Op:    "SigmoidMulInPlace",
			Site:  "sigmoid-shape-match",
			Msg:   "sigmoid gate shape mismatch",
			Class: MemoryActivation,
		})
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_sigmoid_mul_f32(c.cf(x), c.cf(gate), C.int(x.Numel()))
}

// SplitQwen35QueryGate separates Qwen's per-head [query, gate] projection rows on device.
func (c *cudaBackend) SplitQwen35QueryGate(qg Tensor, nHeads, headDim int) (Tensor, Tensor) {
	if qg.Numel() != 2*nHeads*headDim {
		panic(&CUDAOpError{
			Op:    "SplitQwen35QueryGate",
			Site:  "split-qg-shape-match",
			Msg:   "Qwen query/gate projection shape mismatch",
			Class: MemoryActivation,
		})
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	q, _ := c.devTr([]int{nHeads * headDim}, F32)
	gate, _ := c.devTr([]int{nHeads * headDim}, F32)
	C.fcuda_split_qwen35_qg_f32(c.cf(qg), c.cf(q), c.cf(gate), C.int(nHeads), C.int(headDim))
	return q, gate
}

// SwiGLU computes silu(gate)*up element-wise on device, returning a new F32-resident tensor of
// gate's shape.
func (c *cudaBackend) SwiGLU(gate, up Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := gate.Numel()
	y, _ := c.devTr(append([]int(nil), gate.Shape...), F32)
	C.fcuda_swiglu_f32(c.cf(gate), c.cf(up), c.cf(y), C.int(n))
	return y
}

// AddInPlace adds src into dst element-wise on device (the residual dst += src).
func (c *cudaBackend) AddInPlace(dst, src Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_add_f32(c.cf(dst), c.cf(src), C.int(dst.Numel()))
}

// AddBias adds the width-long bias vector into every row of dst on device (dst += bias broadcast
// across rows).
func (c *cudaBackend) AddBias(dst, bias Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	width := bias.Numel()
	rows := dst.Numel() / width
	C.fcuda_add_bias_f32(c.cf(dst), c.cf(bias), C.int(rows), C.int(width))
}

// Attention lowers the whole fused op to ONE flash/online-softmax kernel (#486, Caps.FusedAttn):
// k_flash_attention streams the KV window with a running (max, sum, accumulator) so no scores[nPos]
// row is ever materialized and no per-call scratch is allocated (the old g_attn_scratch is unused on
// this path). causal/grp/scale arrive as kernel params: grp = nH/nKV selects the KV head; the cache
// holds exactly the attendable keys, so causality is by construction; scale folds into the score.
func (c *cudaBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	ck := kv.(*cudaKV)
	hd, nKV := ck.cfg.HeadDim, ck.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := ck.K[layer].len / w
	out, _ := c.devTr([]int{nH * hd}, F32)
	C.fcuda_flash_attention_f32(c.cf(q),
		(*C.float)(ck.K[layer].ptr), (*C.float)(ck.V[layer].ptr),
		c.cf(out), C.int(nPos), C.int(ck.maxPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// SpecVerifyAttention runs the split-KV multi-query speculative verify attention kernel (#11100).
// q is [qLen, nH, d], k is [kvLen, nHkv, d], v is [kvLen, nHkv, d], out is [qLen, nH, d].
// Inputs and any reused output must be live row-major F32 CUDA tensors.
// Query tiling across BLOCK_M with KV sequence splitting across NUM_SEGMENTS
// thread blocks and online softmax merging.
func (c *cudaBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil || k == nil || v == nil || out == nil {
		return fmt.Errorf("compute: SpecVerifyAttention nil tensor argument")
	}
	if q.buf == nil || k.buf == nil || v.buf == nil {
		return fmt.Errorf("compute: SpecVerifyAttention unallocated input tensor")
	}
	if qLen <= 0 || kvLen <= 0 || kvLen < qLen {
		return fmt.Errorf("compute: SpecVerifyAttention invalid lengths qLen=%d kvLen=%d", qLen, kvLen)
	}
	if nH <= 0 || nHkv <= 0 || (nH%nHkv) != 0 {
		return fmt.Errorf("compute: SpecVerifyAttention invalid heads nH=%d nHkv=%d", nH, nHkv)
	}
	if d <= 0 {
		return fmt.Errorf("compute: SpecVerifyAttention invalid head dim d=%d", d)
	}
	expectedQ := qLen * nH * d
	expectedKV := kvLen * nHkv * d
	if q.Numel() != expectedQ {
		return fmt.Errorf("compute: SpecVerifyAttention q numel %d != expected %d", q.Numel(), expectedQ)
	}
	if k.Numel() != expectedKV {
		return fmt.Errorf("compute: SpecVerifyAttention k numel %d != expected %d", k.Numel(), expectedKV)
	}
	if v.Numel() != expectedKV {
		return fmt.Errorf("compute: SpecVerifyAttention v numel %d != expected %d", v.Numel(), expectedKV)
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()

	if err := c.validateSpecVerifyOperandsLocked(q, k, v, out, expectedQ); err != nil {
		return err
	}

	if out.buf == nil || out.Numel() != expectedQ {
		devOut, _ := c.devTr([]int{qLen, nH, d}, F32)
		*out = devOut
	}

	scale := float32(1.0 / math.Sqrt(float64(d)))
	rc := int(C.fcuda_spec_verify_attention_f32(
		c.cf(*q), c.cf(*k), c.cf(*v), c.cf(*out),
		C.int(qLen), C.int(kvLen), C.int(nH), C.int(nHkv), C.int(d), C.float(scale)))
	if rc != 0 {
		return fmt.Errorf("compute: fcuda_spec_verify_attention_f32 failed rc=%d", rc)
	}
	return nil
}

// TreeVerifyAttention runs one fused K<=32 branch-tree attention pass. The
// uint32 rows are copied into the CUDA launch parameter block; the device does
// not allocate or stream a dense KxK mask.
func (c *cudaBackend) TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error {
	if err := validateTreeVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d); err != nil {
		return err
	}
	if q.buf == nil || k.buf == nil || v.buf == nil {
		return fmt.Errorf("compute: TreeVerifyAttention unallocated input tensor")
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()

	expectedQ := qLen * nH * d
	if out.buf == nil || out.Numel() != expectedQ {
		devOut, _ := c.devTr([]int{qLen, nH, d}, F32)
		*out = devOut
	}
	scale := float32(1 / math.Sqrt(float64(d)))
	rc := int(C.fcuda_tree_verify_attention_f32(
		c.cf(*q), c.cf(*k), c.cf(*v), c.cf(*out),
		(*C.uint32_t)(unsafe.Pointer(&maskRows[0])),
		C.int(qLen), C.int(kvLen), C.int(nH), C.int(nHkv), C.int(d), C.float(scale)))
	if rc != 0 {
		return fmt.Errorf("compute: fcuda_tree_verify_attention_f32 failed rc=%d", rc)
	}
	return nil
}

// PrefillBatch executes batched prompt prefill across a sequence panel (P x D) in 1 pass on CUDA GPU (#11036).
func (c *cudaBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	P, _, err := validatePrefillBatchArgs(&args)
	if err != nil {
		return PrefillBatchResult{}, err
	}

	nH := args.NumHeads
	nKV := args.NumKVHeads
	hd := args.HeadDim
	qOut := nH * hd
	kvOut := nKV * hd
	startPos := args.StartPos

	// Ensure inputs are resident device tensors
	xDev := args.X
	if _, isHost := c.Host(xDev); isHost {
		xDev = c.Upload(xDev, F32)
	}
	wqDev := args.Wq
	if _, isHost := c.Host(wqDev); isHost {
		wqDev = c.Upload(wqDev, F32)
	}
	wkDev := args.Wk
	if _, isHost := c.Host(wkDev); isHost {
		wkDev = c.Upload(wkDev, F32)
	}
	wvDev := args.Wv
	if _, isHost := c.Host(wvDev); isHost {
		wvDev = c.Upload(wvDev, F32)
	}

	// 1. Batched projections on CUDA GPU
	qTen := c.BatchedMatMul(wqDev, xDev, P)
	kTen := c.BatchedMatMul(wkDev, xDev, P)
	vTen := c.BatchedMatMul(wvDev, xDev, P)

	qf := c.Read(qTen)
	kf := c.Read(kTen)
	vf := c.Read(vTen)

	kRawAll := append([]float32(nil), kf...)
	qRoped := make([]float32, len(qf))
	kRoped := make([]float32, len(kf))
	for t := 0; t < P; t++ {
		pos := startPos + t
		qRow := c.Upload(NewF32(Default(), []int{qOut}, qf[t*qOut:(t+1)*qOut]), F32)
		qR := c.RoPE(qRow, pos, nH, hd, args.RopeTheta)
		copy(qRoped[t*qOut:(t+1)*qOut], c.Read(qR))

		kRow := c.Upload(NewF32(Default(), []int{kvOut}, kf[t*kvOut:(t+1)*kvOut]), F32)
		kR := c.RoPE(kRow, pos, nKV, hd, args.RopeTheta)
		copy(kRoped[t*kvOut:(t+1)*kvOut], c.Read(kR))
	}

	if args.KV != nil {
		for t := 0; t < P; t++ {
			pos := startPos + t
			rawRow := kRawAll[t*kvOut : (t+1)*kvOut]
			ropeRowSlice := kRoped[t*kvOut : (t+1)*kvOut]
			vRow := vf[t*kvOut : (t+1)*kvOut]

			rawTen := c.Upload(NewF32(Default(), []int{kvOut}, rawRow), F32)
			ropeTen := c.Upload(NewF32(Default(), []int{kvOut}, ropeRowSlice), F32)
			vRowTen := c.Upload(NewF32(Default(), []int{kvOut}, vRow), F32)

			args.KV.AppendKV(args.Layer, rawTen, ropeTen, vRowTen, pos)
		}
	}

	var allK, allV []float32
	var totalPositions int
	strideKV := kvOut

	if args.KV != nil {
		allK = c.Read(args.KV.KeysView(args.Layer))
		allV = c.Read(args.KV.ValuesView(args.Layer))
		totalPositions = len(allK) / strideKV
	} else {
		allK = kRoped
		allV = vf
		totalPositions = P
		startPos = 0
	}

	context := prefillBatchCausalAttention(qRoped, allK, allV, P, startPos, totalPositions, nH, nKV, hd, args.Scale)
	ctxDev := c.Upload(NewF32(Default(), []int{P, qOut}, context), F32)

	var outTen Tensor
	if args.Wo.buf != nil {
		woDev := args.Wo
		if _, isHost := c.Host(woDev); isHost {
			woDev = c.Upload(woDev, F32)
		}
		outTen = c.BatchedMatMul(woDev, ctxDev, P)
	} else {
		outTen = ctxDev
	}

	return PrefillBatchResult{
		Output:  outTen,
		Context: ctxDev,
		Tokens:  P,
	}, nil
}
