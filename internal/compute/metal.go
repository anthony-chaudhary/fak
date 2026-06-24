//go:build darwin && metal && cgo

// metal.go — the cgo wrapper that registers a Metal device backend (Apple Silicon) into
// the compute registry. It is compiled ONLY under `-tags metal` on darwin; the default
// `go build ./cmd/fak` excludes it entirely, so the shipped artifact stays one pure-Go
// binary (DIRECTION.md rule 1). When linked, it self-registers an *Approx* backend named
// "metal" that the registry hands out via Pick("metal") / FAK_BACKEND=metal; the Reference
// (cpu-ref) stays the Default, so nothing silently runs on the GPU.
//
// Every method delegates to the flat C ABI in metal_backend.h (implemented by
// metal_shim.m, compiled in-process by cgo's clang — no offline kernel build, unlike the
// CUDA/Vulkan backends). The Go side re-validates shapes and owns the Tensor type; the C
// side carries only opaque MTLBuffer handles + shapes — a seam that carries data, never
// trust. This is the first landing of issue #300 (Metal Backend [C-001]): MPS GEMM +
// runtime-compiled MSL compute kernels, full synth-decode parity vs cpuref on this box.
// Quantized device GEMM, async/stream pipelining, and an on-GPU Evict are tracked
// follow-ups (the Go MatMul refuses non-F32 weights with a clear message).

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders
#include <stdlib.h>
#include "metal_backend.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

// metalMu serializes all device ops: the command queue is driven synchronously (encode ->
// commit -> waitUntilCompleted per op) and this first backend favors obvious correctness
// over intra-backend concurrency (the async/stream seam is a tracked follow-up).
var metalMu sync.Mutex

var metalDev *metalBackend

func init() {
	var name [256]C.char
	if C.fmetal_init(&name[0], 256) != 0 {
		return // no reachable Metal device (or a pipeline failed to build) — cpu-ref stays sole backend
	}
	metalDev = &metalBackend{
		name: "metal",
		tier: C.GoString(&name[0]), // e.g. "Apple M3 Pro" (the device's own capability label)
	}
	Register(metalDev)
}

// metalBuf is a device-resident Buffer: an opaque id<MTLBuffer> handle + byte length.
// Synchronous backend, so Ready() is always true (the async seam would flip this).
type metalBuf struct {
	ptr  unsafe.Pointer // id<MTLBuffer> handle (shared storage on unified memory)
	n    int            // bytes
	host uintptr        // source host pointer if this came from a cached Upload (0 otherwise)
}

// Ready reports the buffer is materialized — always true on this synchronous backend.
func (b *metalBuf) Ready() bool { return true }

// metalUploadCache shares one device copy per distinct host buffer across all sessions — a
// model's weights are zero-copy views into one blob (the SAME pointer every call), so
// without this each session would re-upload the whole model. Keyed by host pointer; Free
// evicts (so per-token inputs, which have fresh pointers, don't accumulate). Mirrors the
// CUDA backend's uploadCache.
var metalUploadCache = map[uintptr]Tensor{}

type metalBackend struct {
	name string
	tier string
	// transient holds per-token op-output buffers (NOT weights or KV). Recycle() frees them
	// all at a token boundary so steady-state decode stops paying newBuffer per op. Guarded
	// by metalMu (every appender holds it).
	transient []*metalBuf
}

// Recycle frees every transient op-output buffer allocated since the last Recycle. The HAL
// calls it at each token boundary (after Read), where all intermediates are dead — the KV
// cache has already copied what it keeps, and weights are cached separately via Upload
// (never transient). cpu-ref has no Recycle, so this is a device-only fast path the HAL
// probes for.
func (c *metalBackend) Recycle() {
	metalMu.Lock()
	defer metalMu.Unlock()
	for _, b := range c.transient {
		if b.ptr != nil {
			C.fmetal_free(b.ptr)
			b.ptr = nil
		}
	}
	c.transient = c.transient[:0]
}

// Name is the registry id of this backend ("metal").
func (c *metalBackend) Name() string            { return c.name }
func (c *metalBackend) Tier() string            { return c.tier }
func (c *metalBackend) Class() CorrectnessClass { return Approx } // MPS GEMM != fdot order
func (c *metalBackend) Caps() Caps {
	// Device-resident tensors are not host-addressable; the KV cache lives in device buffers.
	// We do not yet advertise Async/FusedAttn/GraphCompile/UploadDtype — those are tracked seams.
	return Caps{DeviceMemory: true}
}

// ---- residency ------------------------------------------------------------------

func (c *metalBackend) dalloc(nbytes int) *metalBuf {
	p := C.fmetal_malloc(C.size_t(nbytes))
	if p == nil {
		panic("compute: metal device allocation failed")
	}
	return &metalBuf{ptr: unsafe.Pointer(p), n: nbytes}
}

func (c *metalBackend) dev(shape []int, dt Dtype) (Tensor, *metalBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := c.dalloc(n * dt.Bytes())
	return makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, buf), buf
}

// devTr is dev() for an op OUTPUT: the buffer is registered as transient so Recycle() can
// free it at the next token boundary. Weights (Upload) deliberately use dev, not devTr, so
// they are never recycled out from under the resident-weight cache.
func (c *metalBackend) devTr(shape []int, dt Dtype) (Tensor, *metalBuf) {
	t, b := c.dev(shape, dt)
	c.transient = append(c.transient, b)
	return t, b
}

// Upload copies host data resident -> device. Only F32 is implemented today; quantized
// upload (narrowing weights at H2D) is the UploadDtype seam, deferred.
func (c *metalBackend) Upload(t Tensor, as Dtype) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic("compute: metal Upload expects host data")
	}
	if t.Dtype != F32 {
		panic("compute: metal Upload supports only F32 today (got " + t.Dtype.String() + ")")
	}
	f := hb.F32()
	var hp uintptr
	if len(f) > 0 {
		hp = uintptr(unsafe.Pointer(&f[0]))
		if cached, ok := metalUploadCache[hp]; ok {
			return cached // same host buffer already resident; share it
		}
	}
	out, buf := c.dev(t.Shape, F32)
	if len(f) > 0 {
		C.fmetal_h2d(buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
		buf.host = hp
		metalUploadCache[hp] = out
	}
	return out
}

// Host returns a host-addressable f32 view only for a host-resident tensor; a
// device (MTLBuffer) tensor returns (nil,false), since device memory is not host-addressable.
func (c *metalBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(*hostBuf); ok && hb.f32 != nil {
		return hb.f32, true
	}
	return nil, false // device tensor: not host-addressable
}

// Read is the host fence: device -> host f32.
func (c *metalBackend) Read(t Tensor) []float32 {
	metalMu.Lock()
	defer metalMu.Unlock()
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32
	}
	db := t.buf.(*metalBuf)
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		C.fmetal_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
	}
	return out
}

// Free releases the tensor's device buffer, evicting it from the upload cache so a
// later re-upload of the same host buffer re-stages it. A no-op for a host tensor.
func (c *metalBackend) Free(t Tensor) {
	metalMu.Lock()
	defer metalMu.Unlock()
	if db, ok := t.buf.(*metalBuf); ok && db.ptr != nil {
		if db.host != 0 {
			delete(metalUploadCache, db.host) // evict so a re-upload of the same host buffer re-stages
		}
		C.fmetal_free(db.ptr)
		db.ptr = nil
	}
}

// ---- primitives -----------------------------------------------------------------

func (c *metalBackend) mb(t Tensor) unsafe.Pointer { return t.buf.(*metalBuf).ptr }

func (c *metalBackend) MatMul(w, x Tensor) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: metal MatMul supports only F32 weights today (got " + w.Dtype.String() + "); quantized device GEMM is a tracked follow-up")
	}
	y, _ := c.devTr([]int{out}, F32)
	C.fmetal_matmul_f32(c.mb(w), c.mb(x), c.mb(y), C.int(out), C.int(in), 1)
	return y
}

// BatchedMatMul is the prefill GEMM Y = X @ Wᵀ over P rows on the MPS path; it refuses
// non-F32 weights (quantized device GEMM is a tracked follow-up).
func (c *metalBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if w.Dtype != F32 {
		panic("compute: metal BatchedMatMul supports only F32 weights today (got " + w.Dtype.String() + ")")
	}
	y, _ := c.devTr([]int{P, out}, F32)
	C.fmetal_matmul_f32(c.mb(w), c.mb(X), c.mb(y), C.int(out), C.int(in), C.int(P))
	return y
}

// RMSNorm runs the per-row RMS normalization x*weight on the Metal compute kernel,
// returning a new device tensor of x's shape.
func (c *metalBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	n := weight.Numel()
	rows := x.Numel() / n
	y, _ := c.devTr(append([]int(nil), x.Shape...), F32)
	C.fmetal_rmsnorm_f32(c.mb(x), c.mb(weight), c.mb(y), C.int(rows), C.int(n), C.float(eps))
	return y
}

// RoPE returns a NEW tensor (value semantics, matching cpuref): copy then rotate in place.
func (c *metalBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	y, ybuf := c.devTr(append([]int(nil), x.Shape...), F32)
	C.fmetal_copy_at(ybuf.ptr, C.size_t(0), x.buf.(*metalBuf).ptr, C.size_t(0), C.size_t(x.Numel()*4))
	C.fmetal_rope_f32(ybuf.ptr, C.int(pos), C.int(nHeads), C.int(headDim), C.float(theta))
	return y
}

// SwiGLU computes silu(gate)*up elementwise on the Metal kernel, returning a new
// device tensor of gate's shape.
func (c *metalBackend) SwiGLU(gate, up Tensor) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	n := gate.Numel()
	y, _ := c.devTr(append([]int(nil), gate.Shape...), F32)
	C.fmetal_swiglu_f32(c.mb(gate), c.mb(up), c.mb(y), C.int(n))
	return y
}

// AddInPlace adds src into dst on the device (the residual dst += src).
func (c *metalBackend) AddInPlace(dst, src Tensor) {
	metalMu.Lock()
	defer metalMu.Unlock()
	C.fmetal_add_f32(c.mb(dst), c.mb(src), C.int(dst.Numel()))
}

// AddBias broadcasts bias across every row of dst on the device (dst += bias per row).
func (c *metalBackend) AddBias(dst, bias Tensor) {
	metalMu.Lock()
	defer metalMu.Unlock()
	width := bias.Numel()
	rows := dst.Numel() / width
	C.fmetal_add_bias_f32(c.mb(dst), c.mb(bias), C.int(rows), C.int(width))
}

// Attention runs the fused scaled-dot-product attention kernel over the layer's
// device-resident K/V for query q, returning the [nH*hd] context. It refuses a head
// dim above 256 (the kernel's accumulator bound).
func (c *metalBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	metalMu.Lock()
	defer metalMu.Unlock()
	mk := kv.(*metalKV)
	hd, nKV := mk.cfg.HeadDim, mk.cfg.NumKVHeads
	if hd > 256 {
		panic("compute: metal Attention headDim > 256 not supported (kernel acc[] bound); a tracked follow-up")
	}
	nH := grp * nKV
	w := nKV * hd
	nPos := mk.K[layer].len / w
	out, _ := c.devTr([]int{nH * hd}, F32)
	C.fmetal_attention_f32(c.mb(q),
		mk.K[layer].ptr, mk.V[layer].ptr, c.mb(out),
		C.int(nPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	return out
}

// Argmax returns the index of the largest logit, reduced on the device so greedy
// decode never copies the full logits vector host-ward.
func (c *metalBackend) Argmax(logits Tensor) int {
	metalMu.Lock()
	defer metalMu.Unlock()
	return int(C.fmetal_argmax_f32(c.mb(logits), C.int(logits.Numel())))
}

// ---- device-resident KV store ---------------------------------------------------

func (c *metalBackend) NewKV(cfg KVConfig) KVStore {
	return &metalKV{
		be:   c,
		cfg:  cfg,
		K:    make([]mslice, cfg.NumLayers),
		Kraw: make([]mslice, cfg.NumLayers),
		V:    make([]mslice, cfg.NumLayers),
	}
}

// mslice is a growable device float buffer (len/cap in floats).
type mslice struct {
	ptr      unsafe.Pointer
	len, cap int
}

func (c *metalBackend) growAppend(d *mslice, srcPtr unsafe.Pointer, nFloats int) {
	if d.len+nFloats > d.cap {
		ncap := d.cap*2 + nFloats
		np := C.fmetal_malloc(C.size_t(ncap * 4))
		if d.len > 0 {
			C.fmetal_copy_at(unsafe.Pointer(np), C.size_t(0), d.ptr, C.size_t(0), C.size_t(d.len*4))
		}
		if d.ptr != nil {
			C.fmetal_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(np)
		d.cap = ncap
	}
	C.fmetal_copy_at(d.ptr, C.size_t(d.len*4), srcPtr, C.size_t(0), C.size_t(nFloats*4))
	d.len += nFloats
}

type metalKV struct {
	be   *metalBackend
	cfg  KVConfig
	K    []mslice
	Kraw []mslice
	V    []mslice
	pos  []int
}

func (k *metalKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *metalKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	metalMu.Lock()
	defer metalMu.Unlock()
	w := k.stride()
	k.be.growAppend(&k.Kraw[layer], kRaw.buf.(*metalBuf).ptr, w)
	k.be.growAppend(&k.K[layer], kRoPE.buf.(*metalBuf).ptr, w)
	k.be.growAppend(&k.V[layer], v.buf.(*metalBuf).ptr, w)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// Len is the number of cached positions in this KV store.
func (k *metalKV) Len() int   { return len(k.pos) }
func (k *metalKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *metalKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := k.K[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &metalBuf{ptr: k.K[layer].ptr, n: k.K[layer].len * 4})
}

// ValuesView returns a [pos, nKV*hd] device-tensor handle over the layer's cached
// value rows (no copy — it aliases the underlying device buffer).
func (k *metalKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := k.V[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &metalBuf{ptr: k.V[layer].ptr, n: k.V[layer].len * 4})
}

// Evict: correctness-first host round-trip (read device -> host, compact + single-rotation
// re-RoPE survivors exactly as cpuKV.Evict, write back). The on-GPU Evict that keeps the
// quarantine witness without a host round-trip is a tracked follow-up; this preserves the
// numerics so the contract holds, just not the performance. Mirrors cudaKV.Evict.
func (k *metalKV) Evict(from, n int) int {
	metalMu.Lock()
	defer metalMu.Unlock()
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

// Clone deep-copies the device-resident K/Kraw/V buffers and positions into a fresh
// metalKV for prefix reuse, so the original and the copy share no device storage.
func (k *metalKV) Clone() KVStore {
	metalMu.Lock()
	defer metalMu.Unlock()
	n := &metalKV{be: k.be, cfg: k.cfg,
		K: make([]mslice, len(k.K)), Kraw: make([]mslice, len(k.Kraw)), V: make([]mslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	cp := func(dst, src *mslice) {
		if src.len == 0 {
			return
		}
		np := C.fmetal_malloc(C.size_t(src.len * 4))
		C.fmetal_copy_at(unsafe.Pointer(np), C.size_t(0), src.ptr, C.size_t(0), C.size_t(src.len*4))
		dst.ptr, dst.len, dst.cap = unsafe.Pointer(np), src.len, src.len
	}
	for l := range k.K {
		cp(&n.K[l], &k.K[l])
		cp(&n.Kraw[l], &k.Kraw[l])
		cp(&n.V[l], &k.V[l])
	}
	return n
}

// Free releases every K/Kraw/V device buffer this KV store holds and clears its positions.
func (k *metalKV) Free() {
	metalMu.Lock()
	defer metalMu.Unlock()
	free := func(d *mslice) {
		if d.ptr != nil {
			C.fmetal_free(d.ptr)
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

func (k *metalKV) readDS(d *mslice) []float32 {
	out := make([]float32, d.len)
	if d.len > 0 {
		C.fmetal_d2h(unsafe.Pointer(&out[0]), d.ptr, C.size_t(d.len*4))
	}
	return out
}

func (k *metalKV) writeDS(d *mslice, data []float32) {
	need := len(data)
	if need > d.cap {
		if d.ptr != nil {
			C.fmetal_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(C.fmetal_malloc(C.size_t(need * 4)))
		d.cap = need
	}
	if need > 0 {
		C.fmetal_h2d(d.ptr, unsafe.Pointer(&data[0]), C.size_t(need*4))
	}
	d.len = need
}
