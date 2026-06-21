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

// cudaBuf is a device-resident Buffer: a VRAM pointer + byte length. Synchronous backend,
// so Ready() is always true (the async seam would flip this).
type cudaBuf struct {
	ptr  unsafe.Pointer // device pointer (cudaMalloc)
	n    int            // bytes
	host uintptr        // source host pointer if this came from a cached Upload (0 otherwise)
}

func (b *cudaBuf) Ready() bool { return true }

// uploadCache shares one VRAM copy per distinct host buffer across all sessions. A model's
// weights are zero-copy views into one blob (m.tensor(name) returns the SAME pointer every
// call), so without this each NewBackendSession re-uploaded the whole model — N sessions ×
// the full weight set, which exhausts VRAM in a multi-session bench. Keyed by host pointer;
// Free evicts (so per-token inputs, which have fresh pointers, don't accumulate).
var uploadCache = map[uintptr]Tensor{}

type cudaBackend struct {
	name string
	tier string
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
	// Device-resident tensors are not host-addressable. The KV cache lives in VRAM. We do
	// not yet advertise Async/FusedAttn/GraphCompile/UploadDtype — those are tracked seams.
	return Caps{DeviceMemory: true}
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
	c.transient = append(c.transient, b)
	return t, b
}

// Upload copies host data resident -> VRAM. Only F32 is implemented today; quantized
// upload (narrowing weights at H2D) is the UploadDtype seam, deferred.
func (c *cudaBackend) Upload(t Tensor, as Dtype) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic("compute: cuda Upload expects host data")
	}
	if t.Dtype != F32 {
		panic("compute: cuda Upload supports only F32 today (got " + t.Dtype.String() + ")")
	}
	f := hb.F32()
	var hp uintptr
	if len(f) > 0 {
		hp = uintptr(unsafe.Pointer(&f[0]))
		if cached, ok := uploadCache[hp]; ok {
			return cached // same host buffer already resident in VRAM; share it
		}
	}
	out, buf := c.dev(t.Shape, F32)
	if len(f) > 0 {
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
		buf.host = hp
		uploadCache[hp] = out
	}
	return out
}

func (c *cudaBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(*hostBuf); ok && hb.f32 != nil {
		return hb.f32, true
	}
	return nil, false // device tensor: not host-addressable
}

// Read is the host fence: device -> host f32.
func (c *cudaBackend) Read(t Tensor) []float32 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32
	}
	db := t.buf.(*cudaBuf)
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		C.fcuda_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
	}
	return out
}

func (c *cudaBackend) Free(t Tensor) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if db, ok := t.buf.(*cudaBuf); ok && db.ptr != nil {
		if db.host != 0 {
			delete(uploadCache, db.host) // evict so a re-upload of the same host buffer re-stages
		}
		C.fcuda_free(db.ptr)
		db.ptr = nil
	}
}

// ---- primitives -----------------------------------------------------------------

func (c *cudaBackend) cf(t Tensor) *C.float { return (*C.float)(t.buf.(*cudaBuf).ptr) }

func (c *cudaBackend) MatMul(w, x Tensor) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: cuda MatMul supports only F32 weights today (got " + w.Dtype.String() + "); quantized device GEMM is a tracked follow-up")
	}
	y, _ := c.devTr([]int{out}, F32)
	C.fcuda_matmul_f32(c.cf(w), c.cf(x), c.cf(y), C.int(out), C.int(in), 1)
	return y
}

func (c *cudaBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: cuda BatchedMatMul supports only F32 weights today (got " + w.Dtype.String() + ")")
	}
	y, _ := c.devTr([]int{P, out}, F32)
	C.fcuda_matmul_f32(c.cf(w), c.cf(X), c.cf(y), C.int(out), C.int(in), C.int(P))
	return y
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

func (c *cudaBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
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

func (c *cudaBackend) Argmax(logits Tensor) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return int(C.fcuda_argmax_f32(c.cf(logits), C.int(logits.Numel())))
}

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

// Evict: correctness-first host round-trip (read VRAM -> host, compact + single-rotation
// re-RoPE survivors exactly as cpuKV.Evict, write back). The on-GPU Evict that keeps the
// quarantine witness without a host round-trip is a tracked follow-up; this preserves the
// numerics so the contract holds, just not the performance.
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
	for l := 0; l < k.cfg.NumLayers; l++ {
		K := k.readDS(&k.K[l])
		Kraw := k.readDS(&k.Kraw[l])
		V := k.readDS(&k.V[l])
		K = append(K[:from*w], K[end*w:]...)
		Kraw = append(Kraw[:from*w], Kraw[end*w:]...)
		V = append(V[:from*w], V[end*w:]...)
		// re-RoPE survivors whose position changed (mirrors cpuKV.Evict)
		newPos := append(append([]int(nil), k.pos[:from]...), k.pos[end:]...)
		for i := range newPos {
			if newPos[i] != i {
				cos, sin := ropeRow(k.cfg.RopeTheta, hd, i)
				for h := 0; h < nKV; h++ {
					dst := K[i*w+h*hd : i*w+(h+1)*hd]
					copy(dst, Kraw[i*w+h*hd:i*w+(h+1)*hd])
					applyRope(dst, cos, sin)
				}
			}
		}
		k.writeDS(&k.K[l], K)
		k.writeDS(&k.Kraw[l], Kraw)
		k.writeDS(&k.V[l], V)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return end - from
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

func (k *cudaKV) readDS(d *dslice) []float32 {
	out := make([]float32, d.len)
	if d.len > 0 {
		C.fcuda_d2h(unsafe.Pointer(&out[0]), d.ptr, C.size_t(d.len*4))
	}
	return out
}

func (k *cudaKV) writeDS(d *dslice, data []float32) {
	need := len(data)
	if need > d.cap {
		if d.ptr != nil {
			C.fcuda_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(C.fcuda_malloc(C.size_t(need * 4)))
		d.cap = need
	}
	if need > 0 {
		C.fcuda_h2d(d.ptr, unsafe.Pointer(&data[0]), C.size_t(need*4))
	}
	d.len = need
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
