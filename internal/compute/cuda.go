//go:build cuda

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
#cgo LDFLAGS: -L${SRCDIR} -lfakcuda -lcudart -lcublas -lstdc++ -lm
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"os"
	"sync"
	"sync/atomic"
	"unsafe"
)

// cudaMu serializes all device ops: the cuBLAS handle and the single default stream are
// not safe under concurrent use, and this first backend favors obvious correctness over
// intra-backend concurrency (the async/stream seam is a tracked follow-up).
var cudaMu sync.Mutex

// graphEnabled gates the CUDA-graph decode path (FAK_CUDA_GRAPH=1). It is OFF by default
// because PER-TOKEN capture is a measured dead end: re-instantiating a ~600-node graph
// every token costs ~what the 600 launches it replaces cost (7.0 vs 7.5 tok/s on
// SmolLM2-135M — no net win). The real win is instantiate-ONCE + replay-many, which needs a
// length-agnostic graph (device-side pos/nPos + a positioned KV-write kernel so one graph
// serves every position) — a tracked redesign (issue #35/#3). The capture plumbing here is
// kept, gated, as its foundation; when on, it also forces a fixed-capacity KV (no realloc
// during capture). Default-off keeps the lean path (pooled alloc + recycle + async + single
// stream) at its proven 7.5 tok/s without the fixed-KV memory cost.
var graphEnabled bool

// cudaFP16CosineMin is the cuda backend's RECORDED Approx cosine floor for the fp16 (HGEMM /
// tensor-core) compute path (#484) — the device-vs-cpuref-f32 logit/GEMM cosine a witness must
// clear. It is deliberately LOOSER than the Q8 / int8 lane's 0.999 gate, for a recorded reason,
// not an assumed one:
//
//   - Q8_0 keeps a per-block(32) f32 scale beside the 8-bit codes (QuantSpec.Scale), and the
//     activation is dynamically re-quantized per block with its own f32 scale. The dynamic
//     range of every group is therefore carried in FULL f32; only the in-block code rounds, and
//     the dot is integer-exact before the single f32 scale multiply. That structure keeps the
//     int8 lane tight against the f32 reference (0.999).
//   - fp16 (IEEE binary16) rounds BOTH operands to a 10-bit mantissa (~2^-11 relative) with NO
//     per-block f32 scale to preserve magnitude structure, so the per-element rounding enters
//     the product directly and compounds along the contraction. cublasGemmEx accumulates in F32
//     (CUBLAS_COMPUTE_32F), which bounds the SUM error, but the INPUTS are already fp16-rounded
//     before the multiply — a drift source the scaled-int8 path does not have. So the fp16 gate
//     is set below the int8 gate as a conservative floor.
//
// IMPORTANT (honest handoff): this constant RECORDS the threshold; it does not assert the path
// passes it. The realized cosine is measured on a CUDA node by tools/run_484_acceptance_on_gpu.sh
// (the win32 build host has no CUDA toolkit / GPU). Do not read a pass from this value alone.
const cudaFP16CosineMin = 0.997

// cudaQ8CosineMin / cudaQ4KCosineMin are the cuda backend's RECORDED Approx cosine floors for the
// native quantized device GEMMs (#485) — the device-vs-cpuref-f32 GEMM cosine each per-dtype
// witness must clear. They are PER-DTYPE because the two formats sit at different points on the
// precision/footprint curve, and the floor must reflect the floor's own format, not a shared guess:
//
//   - cudaQ8CosineMin (0.999): Q8_0 keeps a per-block(32) f32 scale beside 8-bit codes (256 levels)
//     and the activation is dynamically re-quantized per block with its OWN f32 scale, so every
//     group's dynamic range is carried in full f32 and only the in-block code rounds; the per-block
//     dot is integer-exact before a single f32 scale multiply. That structure keeps the int8 lane
//     tight against the f32 reference — the SAME 0.999 the Q8 lane has always been held to (it is the
//     gate cudaFP16CosineMin's comment calls "the Q8 lane's 0.999").
//   - cudaQ4KCosineMin (0.995): Q4_K is a 4-bit k-quant — codes carry only 16 levels and the
//     per-32-sub-block (scale,min) is itself quantized to 6 bits under ONE f16 super-block (d,dmin)
//     pair shared across 256 elements. So a Q4_K weight reconstructs less of the original f32
//     magnitude structure than Q8_0's 8-bit+f32-scale grouping does; on a real model quantized FROM
//     full-precision f32 the 4-bit reconstruction error is genuinely larger than 8-bit, so the
//     recorded floor is set LOOSER than the Q8 lane. (The -tags cuda Q4_K gate isolates the device
//     dequant-fused tile's arithmetic against an f32 dequant of the SAME super-block bytes — it
//     witnesses the getScaleMinK4 geometry is reproduced bit-for-bit; the full-model true-f32 → Q4_K
//     cosine is measured on a GPU node, the same honest residual as every device number here.)
//
// IMPORTANT (honest handoff, identical to cudaFP16CosineMin): these constants RECORD the thresholds;
// they do NOT assert the paths pass them. The realized cosines are measured on a CUDA node by
// tools/run_485_acceptance_on_gpu.sh (the win32 build host has no CUDA toolkit / GPU). Do not read a
// pass from these values alone.
const (
	cudaQ8CosineMin  = 0.999
	cudaQ4KCosineMin = 0.995
)

// cudaFlashAttnCosineMin is the cuda backend's RECORDED Approx cosine floor for the fused
// flash/online-softmax attention kernel (#486) — the device-vs-cpuref-f32 logit cosine a witness
// must clear. The flash kernel computes the SAME math as the cpuref reference — softmax(scale·q·k)
// then ΣwV — only reordered into the streaming online-softmax form (running max/sum, the
// accumulator rescaled onto each new max instead of a single batched max+exp over a full scores
// row). So the ONLY difference from the reference is f32 reduction order, the same class of drift
// as the SGEMM lane (cuBLAS reorders the contraction); it carries no extra rounding the way the
// quantized/fp16 lanes do (no narrowed operand). The floor is therefore set at the SAME 0.999 the
// full forward-pass gate has always used — a conservative recorded value, not a measured pass: in
// isolation the attention-only cosine should sit far tighter (near the 0.9999 SGEMM op gate), but
// 0.999 is the floor the multi-layer forward witness (TestCUDAForwardMatchesRef, whose Attention is
// now this flash kernel) is held to end-to-end.
//
// IMPORTANT (honest handoff, identical to the constants above): this RECORDS the threshold; it does
// NOT assert the path passes it. The realized cosine + the fused-vs-naive speedup are measured on a
// CUDA node by tools/run_486_acceptance_on_gpu.sh (the win32 build host has no CUDA toolkit / GPU).
// Do not read a pass — or a speedup — from this value alone.
const cudaFlashAttnCosineMin = 0.999

// q8DeviceBlock is the Q8_0 per-block size the device narrow-at-H2D quant uses (llama.cpp block_q8_0
// = 32, the cpuref default). The resident weight carries it in QuantSpec.Block so the GEMM kernel
// reconstructs nblk = in/block; `in` must be divisible by it.
const q8DeviceBlock = 32

func init() {
	var name [256]C.char
	var sm C.int
	var total C.size_t
	if C.fcuda_init(&name[0], 256, &sm, &total) != 0 {
		return // no reachable CUDA device — leave cpu-ref as the only backend
	}
	graphEnabled = os.Getenv("FAK_CUDA_GRAPH") == "1"
	cudaDev = &cudaBackend{
		name: "cuda",
		tier: "sm_" + itoaC(int(sm)),
	}
	Register(cudaDev)
}

var cudaDev *cudaBackend

// cudaBuf is a device-resident Buffer: a VRAM pointer + byte length. Op OUTPUTS (allocated
// via devTr) are ASYNC under #482 — enqueued on g_stream and NOT host-observable until a host
// fence (Read/Argmax) drains the stream — so each records the backend's fence generation at
// enqueue time and Ready() reports whether a later fence has bumped past it. Buffers that are
// synchronous on return (weights, whose Upload H2D is a blocking cudaMemcpy; KV views; the
// argmax scalar) carry be==nil and are always Ready.
type cudaBuf struct {
	ptr     unsafe.Pointer // device pointer (cudaMalloc); int8 codes for Q8_0, raw bytes for Q4_K
	n       int            // bytes at ptr
	host    uintptr        // source host pointer if this came from a cached Upload (0 otherwise)
	hostDt  Dtype          // narrowed dtype this upload was cached under (so Free evicts the right key)
	hostLo  Layout         // layout this upload was cached under (ditto — same host buffer, two layouts)
	be      *cudaBackend   // non-nil => async op output; Ready() tracks be.fenceGen vs bornGen
	bornGen uint64         // fence generation in which this async buffer was enqueued
	// scales is the SECOND VRAM buffer a resident Q8_0 weight carries (#485): the per-block(32)
	// f32 scales living beside the int8 codes in ptr. Q4_K keeps d/dmin/scales/codes packed in the
	// raw super-block bytes at ptr, so it leaves scales==nil. Freed alongside ptr in Free.
	scales  unsafe.Pointer
	scalesN int // bytes at scales (0 when there is no scale side-channel)
}

// residentBytes is the total VRAM the weight occupies — codes (ptr) plus any scale side-channel.
// The #485 VRAM witness reads it to prove a Q8_0/Q4_K weight stays narrow (≈ int8/int4 size, not
// the f32 size a dequant-to-f32 upload would have paid).
func (b *cudaBuf) residentBytes() int { return b.n + b.scalesN }

// Ready reports whether the buffer's producing kernel has been fenced host-ward. An async op
// output is ready once a Read/Argmax has bumped the fence generation past the one it was
// enqueued in: the single g_stream is FIFO and a host fence drains all prior work, so one
// generation bump materializes every buffer enqueued before it. Synchronous buffers (be==nil)
// are ready on return. This is the bit the model loop reads to know the logits are still
// device-resident mid-step (#482) — it never gates device execution, which is stream-ordered.
func (b *cudaBuf) Ready() bool {
	if b == nil {
		return false
	}
	if b.be == nil {
		return true
	}
	return atomic.LoadUint64(&b.be.fenceGen) > b.bornGen
}

// uploadCache shares one VRAM copy per distinct host buffer across all sessions. A model's
// weights are zero-copy views into one blob (m.tensor(name) returns the SAME pointer every
// call), so without this each NewBackendSession re-uploaded the whole model — N sessions ×
// the full weight set, which exhausts VRAM in a multi-session bench. Free evicts (so per-token
// inputs, which have fresh pointers, don't accumulate).
//
// The key is (host pointer, narrowed dtype, layout), NOT the pointer alone: under #484 the SAME
// host weight may be uploaded as F32 and as F16, or as F16 in two layouts (RowMajor vs the
// ColMajor transpose-repack), and those are DISTINCT resident buffers. Keying on the pointer
// alone would alias them and hand back the wrong layout/dtype.
type ucKey struct {
	hp uintptr
	dt Dtype
	lo Layout
}

var uploadCache = map[ucKey]Tensor{}

type cudaBackend struct {
	name string
	tier string
	// fenceGen counts host fences (Read/Argmax — the ONLY two). Each async op output records
	// the generation it was enqueued in (cudaBuf.bornGen); a fence bumps fenceGen, flipping
	// every buffer enqueued before it to Ready (#482). Read/written atomically: producers hold
	// cudaMu but Ready() readers (the model loop / the witness test) do not take the lock.
	fenceGen uint64
	// transient holds per-token op-output buffers (NOT weights or KV). Recycle() returns
	// them all to the C-side pool at a token boundary so steady-state decode stops paying
	// cudaMalloc per op. Guarded by cudaMu (every appender holds it).
	transient []*cudaBuf
}

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
	c.transient = c.transient[:0]
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
	return C.fcuda_graph_begin() == 0
}
func (c *cudaBackend) GraphEndLaunch() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if C.fcuda_graph_end_launch() != 0 {
		panic("compute: cuda graph capture/launch failed")
	}
}

// GraphReset drops the kept exec graph so a new session captures fresh (the exec is bound
// to one session's buffer addresses). The HAL calls it at NewBackendSession.
func (c *cudaBackend) GraphReset() {
	if !graphEnabled {
		return
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_graph_reset()
}

func (c *cudaBackend) Name() string            { return c.name }
func (c *cudaBackend) Tier() string            { return c.tier }
func (c *cudaBackend) Class() CorrectnessClass { return Approx } // device GEMM != fdot order
func (c *cudaBackend) Caps() Caps {
	// Async (#482): ops enqueue on g_stream and return unready Buffers; the SOLE host fences
	// are Read and Argmax. DeviceMemory: resident tensors (incl. the KV cache) are not host-
	// addressable. GraphCompile (#483): the fixed per-token decode op stream is capturable into
	// a cudaGraph_t on g_stream and replayable as ONE cudaGraphLaunch (instead of N kernel
	// launches). It is advertised true exactly when that path is live (graphEnabled /
	// FAK_CUDA_GRAPH=1) so it stays consistent with GraphBegin's consent — a consumer that reads
	// false cleanly falls back to the synchronous per-op core (the cpu-ref/Metal default).
	// UploadDtype (#484/#485): Upload(t, F16) narrows weights to __half at H2D (with a ColMajor
	// transpose-repack) for tensor-core HGEMM; Upload(t, Q8_0) narrows an f32 weight to resident
	// int8 codes + f32 scales, and a Q4_K host weight uploads its raw super-block bytes verbatim —
	// MatMul/BatchedMatMul then run the native quantized device GEMMs (no dequant-to-f32), keeping
	// the weight narrow in VRAM. FusedAttn (#486): Attention lowers to ONE fused flash/online-softmax
	// kernel (k_flash_attention) — tiled over the KV window with a running max/sum so no scores[nPos]
	// row is materialized; the naive kernel is retained only as the microbench baseline.
	return Caps{Async: true, DeviceMemory: true, GraphCompile: graphEnabled, UploadDtype: true, FusedAttn: true}
}

// ---- residency ------------------------------------------------------------------

func (c *cudaBackend) dalloc(nbytes int) *cudaBuf {
	p := C.fcuda_malloc(C.size_t(nbytes))
	if p == nil {
		panic("compute: cuda device allocation failed")
	}
	return &cudaBuf{ptr: unsafe.Pointer(p), n: nbytes}
}

func (c *cudaBackend) dev(shape []int, dt Dtype) (Tensor, *cudaBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := c.dalloc(n * dt.Bytes())
	return makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, buf), buf
}

// devTr is dev() for an op OUTPUT: the buffer is registered as transient so Recycle() can
// return it to the pool at the next token boundary. Weights (Upload) deliberately use dev,
// not devTr, so they are never recycled out from under the resident-weight cache.
func (c *cudaBackend) devTr(shape []int, dt Dtype) (Tensor, *cudaBuf) {
	t, b := c.dev(shape, dt)
	// Mark async: this output is enqueued on g_stream in the current fence generation, so it
	// reports Ready()==false until the next Read/Argmax drains the stream (#482).
	b.be = c
	b.bornGen = atomic.LoadUint64(&c.fenceGen)
	c.transient = append(c.transient, b)
	return t, b
}

// devF16 is dev() for an F16-resident weight: an out*in*2-byte VRAM buffer carrying the
// requested Layout (so MatMul can read w.Layout to pick the HGEMM op). Weights use dev/devF16,
// never devTr, so the resident-weight cache is never recycled out from under them.
func (c *cudaBackend) devF16(shape []int, layout Layout) (Tensor, *cudaBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := c.dalloc(n * F16.Bytes())
	return makeTensor(c, F16, layout, append([]int(nil), shape...), nil, buf), buf
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
	cudaMu.Lock()
	defer cudaMu.Unlock()
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic("compute: cuda Upload expects host data")
	}
	if t.Dtype == Q4_K {
		return c.uploadQ4K(t, hb) // raw super-block bytes, copied resident (already narrow)
	}
	if t.Dtype != F32 {
		panic("compute: cuda Upload supports F32 host data (optionally narrowing to F16/Q8_0) or raw Q4_K bytes today (got " + t.Dtype.String() + ")")
	}
	store := F32 // the resident dtype the `as` request narrows f32 host weights to
	switch as {
	case F16:
		store = F16
	case Q8_0:
		store = Q8_0
	}
	f := hb.F32()
	var hp uintptr
	if len(f) > 0 {
		hp = uintptr(unsafe.Pointer(&f[0]))
		if cached, ok := uploadCache[ucKey{hp, store, t.Layout}]; ok {
			return cached // same host buffer already resident at this dtype/layout; share it
		}
	}
	switch store {
	case F16:
		return c.uploadF16(t, f, hp)
	case Q8_0:
		return c.uploadQ8(t, f, hp)
	}
	out, buf := c.dev(t.Shape, F32)
	if len(f) > 0 {
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
		buf.host, buf.hostDt, buf.hostLo = hp, F32, t.Layout
		uploadCache[ucKey{hp, F32, t.Layout}] = out
	}
	return out
}

// uploadQ8 narrows an f32 host weight [out,in] to a resident Q8_0 weight at H2D (#485): the f32 is
// quantized to per-block(32) int8 codes + f32 scales using the EXACT cpuref QuantizeQ8 scheme
// (d=amax/127, q8round), and both narrow operands are uploaded — codes to the buffer's ptr, scales
// to its scale side-channel. The f32 weight never becomes resident, so the VRAM footprint is the
// int8 size (codes) + a thin per-block scale band, not the f32 size. `in` must be divisible by 32.
func (c *cudaBackend) uploadQ8(t Tensor, f []float32, hp uintptr) Tensor {
	if len(t.Shape) != 2 {
		panic("compute: cuda Upload(_, Q8_0) expects a 2-D [out,in] weight (got rank " + itoaC(len(t.Shape)) + ")")
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
	C.fcuda_h2d(buf.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes)))
	C.fcuda_h2d(buf.scales, unsafe.Pointer(&scales[0]), C.size_t(len(scales)*4))
	buf.host, buf.hostDt, buf.hostLo = hp, Q8_0, t.Layout
	uploadCache[ucKey{hp, Q8_0, t.Layout}] = res
	return res
}

// uploadQ4K copies raw Q4_K super-block bytes resident (#485). The host tensor carries the bytes
// in its HostBuffer.I8() view (one int8 per byte); they are already narrow (144 bytes / 256 elems),
// so there is no quantize or dtype-narrow step — just an H2D into a uint8 VRAM buffer the
// dequant-fused GEMM tile consumes. Cached on (host ptr, Q4_K, layout) like every other upload.
func (c *cudaBackend) uploadQ4K(t Tensor, hb HostBuffer) Tensor {
	raw := hb.I8()
	var hp uintptr
	if len(raw) > 0 {
		hp = uintptr(unsafe.Pointer(&raw[0]))
		if cached, ok := uploadCache[ucKey{hp, Q4_K, t.Layout}]; ok {
			return cached
		}
	}
	res, buf := c.devQ4K(t.Shape, len(raw))
	if len(raw) > 0 {
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		buf.host, buf.hostDt, buf.hostLo = hp, Q4_K, t.Layout
		uploadCache[ucKey{hp, Q4_K, t.Layout}] = res
	}
	return res
}

// devQ8 allocates a resident Q8_0 weight: an out*in-byte int8 codes buffer (ptr) plus an
// nScales*4-byte f32 scale side-channel (scales). The Tensor carries a QuantSpec{Block} so the
// GEMM kernel reconstructs nblk = in/block. Weights use dev-family allocs, never devTr, so the
// resident-weight cache is never recycled out from under them.
func (c *cudaBackend) devQ8(shape []int, block, nScales int) (Tensor, *cudaBuf) {
	out, in := shape[0], shape[1]
	buf := c.dalloc(out * in) // int8 codes, 1 byte each
	buf.scales = unsafe.Pointer(C.fcuda_malloc(C.size_t(nScales * 4)))
	if buf.scales == nil {
		panic("compute: cuda Q8 scale allocation failed")
	}
	buf.scalesN = nScales * 4
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	return makeTensor(c, Q8_0, RowMajor, append([]int(nil), shape...), q, buf), buf
}

// devQ4K allocates a resident Q4_K weight: a single nbytes-long uint8 buffer holding the raw GGUF
// super-block bytes (d/dmin/scales/codes all packed; no scale side-channel). nbytes is the size of
// the host byte slice (= (out*in/256)*144). The QuantSpec records the 256-elem super-block.
func (c *cudaBackend) devQ4K(shape []int, nbytes int) (Tensor, *cudaBuf) {
	buf := c.dalloc(nbytes)
	q := &QuantSpec{Block: 256, Axis: 2, Bits: 4, Symmetric: false}
	return makeTensor(c, Q4_K, RowMajor, append([]int(nil), shape...), q, buf), buf
}

// uploadF16 narrows an f32 host weight to a resident F16 weight at H2D (#484). The f32 is staged
// in a transient device buffer, converted to __half by a device kernel — row-major in place, or
// ColMajor transpose-repacked ([out,in] -> col-major) — and the stage is freed. The narrow runs
// on device (one conversion implementation, identical numerics to the GEMM's own half cast),
// never on the host.
func (c *cudaBackend) uploadF16(t Tensor, f []float32, hp uintptr) Tensor {
	out, buf := c.devF16(t.Shape, t.Layout)
	if len(f) == 0 {
		return out
	}
	stage := c.dalloc(len(f) * 4)
	C.fcuda_h2d(stage.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
	if t.Layout == ColMajor && len(t.Shape) == 2 {
		C.fcuda_f32_to_f16_T(buf.ptr, (*C.float)(stage.ptr), C.int(t.Shape[0]), C.int(t.Shape[1]))
	} else {
		C.fcuda_f32_to_f16(buf.ptr, (*C.float)(stage.ptr), C.int(len(f)))
	}
	C.fcuda_free(stage.ptr)
	buf.host, buf.hostDt, buf.hostLo = hp, F16, t.Layout
	uploadCache[ucKey{hp, F16, t.Layout}] = out
	return out
}

func (c *cudaBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(*hostBuf); ok && hb.f32 != nil {
		return hb.f32, true
	}
	return nil, false // device tensor: not host-addressable
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
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		C.fcuda_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
		atomic.AddUint64(&c.fenceGen, 1) // stream drained: prior enqueued work is now materialized
	}
	return out
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
		C.fcuda_free(db.ptr)
		db.ptr = nil
	}
}

// ---- primitives -----------------------------------------------------------------

func (c *cudaBackend) cf(t Tensor) *C.float { return (*C.float)(t.buf.(*cudaBuf).ptr) }

// cptr is the raw device pointer (void*), for dtypes whose element type is not *C.float — the
// F16 weight buffer (__half) the HGEMM path reads.
func (c *cudaBackend) cptr(t Tensor) unsafe.Pointer { return t.buf.(*cudaBuf).ptr }

// colMajorFlag reports w's HGEMM layout selector: 1 when the weight was transpose-repacked to
// column-major at H2D (op_N), 0 for the row-major SGEMM recipe (op_T).
func colMajorFlag(w Tensor) C.int {
	if w.Layout == ColMajor {
		return 1
	}
	return 0
}

func (c *cudaBackend) MatMul(w, x Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	switch w.Dtype {
	case F32:
		y, _ := c.devTr([]int{out}, F32)
		C.fcuda_matmul_f32(c.cf(w), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
		return y
	case F16:
		// tensor-core HGEMM (#484): F16 weight, f32 activation (converted to __half C-side), f32
		// accumulate/output. P=1 (decode GEMV); the activation x stays F32-resident.
		y, _ := c.devTr([]int{out}, F32)
		C.fcuda_matmul_f16(c.cptr(w), c.cf(x), c.cf(y), C.int(out), C.int(in), 1, colMajorFlag(w))
		return y
	case Q8_0:
		// native Q8_0 GEMV (#485): int8 codes + per-block f32 scales resident, the activation
		// quantized to int8 ON DEVICE; integer per-block dot scaled by (weight·activation block
		// scales), F32 accumulate. No dequant-to-f32 round trip — the weight stays int8 in VRAM.
		wb := w.buf.(*cudaBuf)
		y, _ := c.devTr([]int{out}, F32)
		C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(x), c.cf(y),
			C.int(out), C.int(in), 1, C.int(w.Quant.Block))
		return y
	case Q4_K:
		// native Q4_K GEMV (#485): the dequant (w = d·scale·code − dmin·min) is fused into the
		// GEMM tile straight off the resident super-block bytes; the weight stays int4 in VRAM.
		wb := w.buf.(*cudaBuf)
		y, _ := c.devTr([]int{out}, F32)
		C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
		return y
	default:
		panic("compute: cuda MatMul supports F32/F16/Q8_0/Q4_K weights today (got " + w.Dtype.String() + "); other quantized device GEMM is a tracked follow-up")
	}
}

func (c *cudaBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	switch w.Dtype {
	case F32:
		y, _ := c.devTr([]int{P, out}, F32)
		C.fcuda_matmul_f32(c.cf(w), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
		return y
	case F16:
		// tensor-core HGEMM (#484): the prefill GEMM where fp16/tensor-cores pay off most.
		y, _ := c.devTr([]int{P, out}, F32)
		C.fcuda_matmul_f16(c.cptr(w), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P), colMajorFlag(w))
		return y
	case Q8_0:
		// native Q8_0 prefill GEMM (#485): each of the P activation rows is quantized to int8 on
		// device, then the per-block integer dot against the resident int8 weight, F32 accumulate.
		wb := w.buf.(*cudaBuf)
		y, _ := c.devTr([]int{P, out}, F32)
		C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.cf(X), c.cf(y),
			C.int(out), C.int(in), C.int(P), C.int(w.Quant.Block))
		return y
	case Q4_K:
		// native Q4_K prefill GEMM (#485): dequant fused into the tile off the resident super-block
		// bytes, dotted with each of the P f32 activation rows, F32 accumulate.
		wb := w.buf.(*cudaBuf)
		y, _ := c.devTr([]int{P, out}, F32)
		C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
		return y
	default:
		panic("compute: cuda BatchedMatMul supports F32/F16/Q8_0/Q4_K weights today (got " + w.Dtype.String() + ")")
	}
}

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
	y, ybuf := c.devTr(append([]int(nil), x.Shape...), F32)
	C.fcuda_d2d(ybuf.ptr, x.buf.(*cudaBuf).ptr, C.size_t(x.Numel()*4))
	C.fcuda_rope_f32(c.cf(y), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return y
}

func (c *cudaBackend) SwiGLU(gate, up Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := gate.Numel()
	y, _ := c.devTr(append([]int(nil), gate.Shape...), F32)
	C.fcuda_swiglu_f32(c.cf(gate), c.cf(up), c.cf(y), C.int(n))
	return y
}

func (c *cudaBackend) AddInPlace(dst, src Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_add_f32(c.cf(dst), c.cf(src), C.int(dst.Numel()))
}

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

// attentionNaive runs the same op through the RETAINED naive kernel (full global scores[nH*nPos]
// scratch, four passes). It exists only so the #486 microbench can time fused-vs-naive on identical
// inputs; the live Attention path never calls it. Same arguments, same result up to reduction order.
func (c *cudaBackend) attentionNaive(q Tensor, kv KVStore, layer int, grp int, scale float32) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	ck := kv.(*cudaKV)
	hd, nKV := ck.cfg.HeadDim, ck.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := ck.K[layer].len / w
	out, _ := c.devTr([]int{nH * hd}, F32)
	C.fcuda_attention_f32(c.cf(q),
		(*C.float)(ck.K[layer].ptr), (*C.float)(ck.V[layer].ptr),
		c.cf(out), C.int(nPos), C.int(ck.maxPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// Argmax is the OTHER host fence (#482) and the one greedy decode uses: it runs the reduction
// ON-DEVICE (k_argmax over the resident logits) and copies back only the single winning token
// id — the full logits vector never crosses the bus. Like Read, the int copy drains g_stream,
// so it bumps the fence generation (the logits it reduced are now Ready).
func (c *cudaBackend) Argmax(logits Tensor) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	id := int(C.fcuda_argmax_f32(c.cf(logits), C.int(logits.Numel())))
	atomic.AddUint64(&c.fenceGen, 1)
	return id
}

// ---- async host-transfer witness (#482) -----------------------------------------
//
// HostXferBytes reports the cumulative bytes copied DEVICE->HOST since the last reset. The two
// host fences are the only d2h transfers and both feed this counter: fcuda_d2h (a full Read)
// adds the vector's bytes, while fcuda_argmax_f32 adds only sizeof(int). So over an Argmax-only
// decode step it reads the size of one token id, whereas a full-logits Read reads vocab*4 —
// the seam the witness test reads to prove only the argmax id crosses the bus per token.
// ResetHostXfer zeroes it. These are used only by the -tags cuda witness/benchmarks.
func (c *cudaBackend) HostXferBytes() uint64 { return uint64(C.fcuda_hostxfer_bytes()) }
func (c *cudaBackend) ResetHostXfer()        { C.fcuda_hostxfer_reset() }

// ---- AWQ (Activation-aware Weight Quantization) 4-bit matmul -------------------

// AWQMatMul computes y = W @ x where W is an AWQ 4-bit quantized tensor.
// W is [out, in] stored as 4-bit packed bytes [out, in/2], with per-channel scales [out].
func (c *cudaBackend) AWQMatMul(w, scales, x Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	y, _ := c.devTr([]int{out}, F32)

	// Get device pointers
	wp := w.buf.(*cudaBuf).ptr
	sp := scales.buf.(*cudaBuf).ptr
	xp := x.buf.(*cudaBuf).ptr
	yp := c.cf(y)

	C.fcuda_awq_gemv((*C.uint8_t)(wp), (*C.float)(sp), (*C.float)(xp), yp, C.int(out), C.int(in))
	return y
}

// AWQBatchedMatMul computes Y = X @ W^T where W is an AWQ 4-bit quantized tensor.
// X is [P, in], W is [out, in] stored as 4-bit packed [out, in/2], scales is [out].
// Output Y is [P, out].
func (c *cudaBackend) AWQBatchedMatMul(w, scales, X Tensor, P int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	y, _ := c.devTr([]int{P, out}, F32)

	// Get device pointers
	wp := w.buf.(*cudaBuf).ptr
	sp := scales.buf.(*cudaBuf).ptr
	xp := X.buf.(*cudaBuf).ptr
	yp := c.cf(y)

	C.fcuda_awq_gemm((*C.uint8_t)(wp), (*C.float)(sp), (*C.float)(xp), yp, C.int(out), C.int(in), C.int(P))
	return y
}

// ---- device-resident KV store ---------------------------------------------------

// cudaKVMaxPos is the fixed cache capacity (in positions) each device KV preallocates, so
// AppendKV never reallocs — a hard requirement for CUDA-graph capture (a cudaMalloc during
// capture is illegal). 1024 covers the decode benchmarks; a longer-context session would
// raise this (a future fixed-vs-ring tradeoff, tracked with the device-KV work).
const cudaKVMaxPos = 1024

func (c *cudaBackend) NewKV(cfg KVConfig) KVStore {
	k := &cudaKV{be: c, cfg: cfg}
	k.K = make([]dslice, cfg.NumLayers)
	k.Kraw = make([]dslice, cfg.NumLayers)
	k.V = make([]dslice, cfg.NumLayers)
	if graphEnabled {
		// Graph capture forbids a cudaMalloc mid-token, so preallocate a fixed capacity
		// the cache never has to realloc within. Default (non-graph) path stays growable
		// and lean (no per-session preallocation).
		k.maxPos = cudaKVMaxPos
		capF := k.maxPos * cfg.NumKVHeads * cfg.HeadDim
		for l := 0; l < cfg.NumLayers; l++ {
			k.K[l] = dslice{ptr: unsafe.Pointer(C.fcuda_malloc(C.size_t(capF * 4))), cap: capF}
			k.Kraw[l] = dslice{ptr: unsafe.Pointer(C.fcuda_malloc(C.size_t(capF * 4))), cap: capF}
			k.V[l] = dslice{ptr: unsafe.Pointer(C.fcuda_malloc(C.size_t(capF * 4))), cap: capF}
		}
	}
	return k
}

// dslice is a growable VRAM float buffer (len/cap in floats).
type dslice struct {
	ptr      unsafe.Pointer
	len, cap int
}

func (c *cudaBackend) growAppend(d *dslice, srcPtr unsafe.Pointer, nFloats int) {
	if d.len+nFloats > d.cap {
		ncap := d.cap*2 + nFloats
		np := C.fcuda_malloc(C.size_t(ncap * 4))
		if d.len > 0 {
			C.fcuda_d2d(unsafe.Pointer(np), d.ptr, C.size_t(d.len*4))
		}
		if d.ptr != nil {
			C.fcuda_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(np)
		d.cap = ncap
	}
	// kernel-form append (scalar offset) instead of a cudaMemcpy to a moving pointer, so a
	// captured decode graph stays reusable via cudaGraphExecUpdate as the cache grows.
	C.fcuda_kv_write((*C.float)(d.ptr), (*C.float)(srcPtr), C.int(d.len), C.int(nFloats))
	d.len += nFloats
}

type cudaKV struct {
	be     *cudaBackend
	cfg    KVConfig
	maxPos int // fixed capacity in positions (preallocated so AppendKV never reallocs)
	K      []dslice
	Kraw   []dslice
	V      []dslice
	pos    []int
}

func (k *cudaKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *cudaKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	w := k.stride()
	k.be.growAppend(&k.Kraw[layer], kRaw.buf.(*cudaBuf).ptr, w)
	k.be.growAppend(&k.K[layer], kRoPE.buf.(*cudaBuf).ptr, w)
	k.be.growAppend(&k.V[layer], v.buf.(*cudaBuf).ptr, w)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

func (k *cudaKV) Len() int   { return len(k.pos) }
func (k *cudaKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *cudaKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := k.K[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &cudaBuf{ptr: k.K[layer].ptr, n: k.K[layer].len * 4})
}
func (k *cudaKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := k.V[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &cudaBuf{ptr: k.V[layer].ptr, n: k.V[layer].len * 4})
}

// Evict compacts the cache ON-GPU — no host round-trip (#479). For every layer it shifts
// the survivors of K/Kraw/V down past the [from,from+n) span, then re-derives the post-RoPE
// K of each survivor whose absolute position changed by a SINGLE rotation of its (already
// device-resident) Kraw at the NEW index — the very kernel AppendKV used, so a device evict
// is bit-identical to a device run that never saw the span (the Approx-gate witness). The
// prefix [0,from) is left byte-for-byte untouched; that asymmetry — only the suffix is
// repositioned — is the write-time quarantine witness (MODEL-ARCH-SEAM §3, O1–O3): a span
// evicted before the query attends vanishes, but one evicted after downstream tokens already
// attended cannot be un-seen. The KV never leaves VRAM, so Host() on these tensors stays
// (nil,false). The host round-trip this replaces lived on cpuKV.Evict / earlier cudaKV.
func (k *cudaKV) Evict(from, n int) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if from < 0 || n <= 0 || from >= len(k.pos) {
		return 0
	}
	end := from + n
	if end > len(k.pos) {
		end = len(k.pos)
	}
	w := k.stride()
	hd, nKV := k.cfg.HeadDim, k.cfg.NumKVHeads
	fromF, endF := from*w, end*w
	tailFloats := (len(k.pos) - end) * w // survivors after the span (shared by K/Kraw/V)
	// survivor positions after compaction: prefix keeps its index, suffix shifts down.
	newPos := append(append([]int(nil), k.pos[:from]...), k.pos[end:]...)
	// One reused scratch buffer for the leftward shift: an in-place device-to-device copy of
	// overlapping regions is undefined, so the tail is staged through disjoint VRAM. Stream
	// ordering (everything on g_stream) serializes the per-layer reuse correctly.
	var scratch unsafe.Pointer
	if tailFloats > 0 {
		scratch = unsafe.Pointer(C.fcuda_malloc(C.size_t(tailFloats * 4)))
		if scratch == nil {
			panic("compute: cuda Evict scratch allocation failed")
		}
	}
	for l := 0; l < k.cfg.NumLayers; l++ {
		k.be.compactDS(&k.K[l], fromF, endF, tailFloats, scratch)
		k.be.compactDS(&k.Kraw[l], fromF, endF, tailFloats, scratch)
		k.be.compactDS(&k.V[l], fromF, endF, tailFloats, scratch)
		for i := range newPos {
			if newPos[i] == i {
				continue // prefix survivor: position unchanged, post-RoPE K stays byte-for-byte
			}
			// K[i] <- Kraw[i] (disjoint buffers, no overlap) then one in-place rotation at i.
			kRow := offsetF(k.K[l].ptr, i*w)
			C.fcuda_d2d(kRow, offsetF(k.Kraw[l].ptr, i*w), C.size_t(w*4))
			C.fcuda_rope_f32((*C.float)(kRow), C.int(i), C.int(nKV), C.int(hd), C.double(k.cfg.RopeTheta))
		}
	}
	if scratch != nil {
		C.fcuda_free(scratch)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return end - from
}

// offsetF advances a device pointer by nFloats f32 elements. The KV buffers are C-allocated
// (cudaMalloc), not Go-managed memory, so this is the correct way to address a sub-row and
// is outside the GC's purview (the vet unsafeptr concern is for Go-heap pointers, not these).
func offsetF(p unsafe.Pointer, nFloats int) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(nFloats)*4)
}

// compactDS removes the float span [fromF,endF) from a position-major device buffer in place
// by shifting its tailFloats-long tail down through a caller-supplied disjoint scratch. A
// direct leftward device-to-device copy would overlap (src and dst intersect), which
// cudaMemcpy leaves undefined; staging through scratch is well-defined and never touches the
// host. Both copies ride g_stream, so they stay ordered against each other and the re-RoPE.
func (c *cudaBackend) compactDS(d *dslice, fromF, endF, tailFloats int, scratch unsafe.Pointer) {
	if tailFloats > 0 {
		C.fcuda_d2d(scratch, offsetF(d.ptr, endF), C.size_t(tailFloats*4))
		C.fcuda_d2d(offsetF(d.ptr, fromF), scratch, C.size_t(tailFloats*4))
	}
	d.len -= endF - fromF
}

func (k *cudaKV) Clone() KVStore {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	n := &cudaKV{be: k.be, cfg: k.cfg,
		K: make([]dslice, len(k.K)), Kraw: make([]dslice, len(k.Kraw)), V: make([]dslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	cp := func(dst, src *dslice) {
		if src.len == 0 {
			return
		}
		np := C.fcuda_malloc(C.size_t(src.len * 4))
		C.fcuda_d2d(unsafe.Pointer(np), src.ptr, C.size_t(src.len*4))
		dst.ptr, dst.len, dst.cap = unsafe.Pointer(np), src.len, src.len
	}
	for l := range k.K {
		cp(&n.K[l], &k.K[l])
		cp(&n.Kraw[l], &k.Kraw[l])
		cp(&n.V[l], &k.V[l])
	}
	return n
}

func (k *cudaKV) Free() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	free := func(d *dslice) {
		if d.ptr != nil {
			C.fcuda_free(d.ptr)
			d.ptr = nil
		}
		d.len = 0
		d.cap = 0
	}
	for l := range k.K {
		free(&k.K[l])
		free(&k.Kraw[l])
		free(&k.V[l])
	}
	k.pos = nil
}

// itoaC is a tiny int->string for the tier label (avoids importing strconv into the
// build-tagged file's surface).
func itoaC(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
