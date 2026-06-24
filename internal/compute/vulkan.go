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
	"os"
	"strconv"
	"sync"
	"unsafe"
)

var vulkanMu sync.Mutex

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
	vulkanDev = &vulkanBackend{
		name:        "vulkan",
		tier:        tier + ":" + C.GoString(&name[0]),
		haveQ8:      C.fvk_have_q8() != 0,
		budgetBytes: vulkanBudgetBytes(),
	}
	Register(vulkanDev)
}

// vulkanBudgetBytes reads FAK_GPU_BUDGET_MB — the device-local weight budget in MiB. 0 / unset /
// invalid = unbounded (place every weight device-local, the prior behavior). A positive value
// caps device-local weight residency; weights past the cap go host-visible in upload order.
func vulkanBudgetBytes() int64 {
	s := os.Getenv("FAK_GPU_BUDGET_MB")
	if s == "" {
		return 0
	}
	mb, err := strconv.ParseInt(s, 10, 64)
	if err != nil || mb <= 0 {
		return 0
	}
	return mb * 1024 * 1024
}

var vulkanDev *vulkanBackend

type vulkanBuf struct {
	ptr      unsafe.Pointer
	n        int
	scalePtr unsafe.Pointer
	scaleN   int
}

// Ready always reports true: Vulkan dispatches are submitted synchronously, so a
// vulkanBuf handle is materialized as soon as it exists.
func (b *vulkanBuf) Ready() bool { return true }

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
}

const vulkanGoPoolBucketCap = 64

// Recycle returns every transient buffer from the current op cycle to the per-size
// free pool (recycling or freeing each) and trims the shim's device pool if oversized.
func (v *vulkanBackend) Recycle() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	for _, b := range v.transient {
		if b.ptr != nil {
			v.recycleTransientLocked(b)
			b.ptr = nil
		}
	}
	v.transient = v.transient[:0]
	C.fvk_trim_pool_if_over(512)
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
func (v *vulkanBackend) Caps() Caps              { return Caps{DeviceMemory: true, UploadDtype: v.haveQ8} }

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

func (v *vulkanBackend) dalloc(nbytes int) *vulkanBuf {
	p := C.fvk_malloc(C.size_t(nbytes))
	if p == nil {
		// Device-local (and the shim's own host-visible storage fallback) is exhausted. Rather
		// than crash the whole run, try a clean host-visible allocation as a last resort — slow
		// but alive. This is what makes a budgeted run degrade gracefully when KV/scratch (which
		// don't go through the weight budget) outgrow the remaining device-local headroom,
		// instead of the old hard panic. A nil here too is a genuine OOM with nowhere left.
		p = C.fvk_malloc_hostvis(C.size_t(nbytes))
		if p == nil {
			panic("compute: vulkan device allocation failed (device-local and host-visible both exhausted)")
		}
	}
	return &vulkanBuf{ptr: unsafe.Pointer(p), n: nbytes}
}

// dallocHostVis allocates a storage buffer in host-visible memory directly (no device-local
// attempt). Used by the residency-budget path for cold weights. Caller holds vulkanMu.
func (v *vulkanBackend) dallocHostVis(nbytes int) *vulkanBuf {
	p := C.fvk_malloc_hostvis(C.size_t(nbytes))
	if p == nil {
		panic("compute: vulkan host-visible allocation failed")
	}
	return &vulkanBuf{ptr: unsafe.Pointer(p), n: nbytes}
}

// dallocWeight places a weight buffer device-local while under the residency budget, else
// host-visible (deliberately, in upload order). budgetBytes==0 means unbounded -> always
// device-local. Caller holds vulkanMu.
func (v *vulkanBackend) dallocWeight(nbytes int) *vulkanBuf {
	if v.budgetBytes > 0 && v.dlUsed+int64(nbytes) > v.budgetBytes {
		v.hostvisN++
		return v.dallocHostVis(nbytes)
	}
	v.dlUsed += int64(nbytes)
	return v.dalloc(nbytes)
}

func (v *vulkanBackend) dallocTransient(nbytes int) *vulkanBuf {
	if v.freeTransient != nil {
		bucket := v.freeTransient[nbytes]
		if len(bucket) > 0 {
			b := bucket[len(bucket)-1]
			v.freeTransient[nbytes] = bucket[:len(bucket)-1]
			return b
		}
	}
	return v.dalloc(nbytes)
}

func (v *vulkanBackend) recycleTransientLocked(b *vulkanBuf) {
	if b == nil || b.ptr == nil {
		return
	}
	if v.freeTransient == nil {
		v.freeTransient = make(map[int][]*vulkanBuf)
	}
	bucket := v.freeTransient[b.n]
	owner := &vulkanBuf{ptr: b.ptr, n: b.n}
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
	buf := v.dalloc(n * dt.Bytes())
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
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		panic("compute: vulkan Upload expects host data")
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
	if as == Q8_0 {
		q := QuantizeQ8(Default(), t.Shape, f, 32)
		qh := q.buf.(HostBuffer)
		return v.uploadQ8Locked(q.Shape, qh.I8(), q.Quant.Scale, q.Quant.Block)
	}
	out, buf := v.dev(t.Shape, F32)
	if len(f) > 0 {
		C.fvk_h2d(buf.ptr, unsafe.Pointer(&f[0]), C.size_t(len(f)*4))
	}
	return out
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
	// The code buffer is the bulk of the weight (in*out bytes) — it's the budget's subject.
	// The scale buffer is ~1/32 the size; keep it device-local so the hot per-block scales
	// stay fast even when the codes spill host-visible.
	codeBuf := v.dallocWeight(len(codes))
	scaleBuf := v.dalloc(len(scales) * F32.Bytes())
	if len(codes) > 0 {
		C.fvk_h2d(codeBuf.ptr, unsafe.Pointer(&codes[0]), C.size_t(len(codes)))
	}
	if len(scales) > 0 {
		C.fvk_h2d(scaleBuf.ptr, unsafe.Pointer(&scales[0]), C.size_t(len(scales)*F32.Bytes()))
	}
	q := &QuantSpec{Block: block, Axis: 2, Bits: 8, Symmetric: true}
	buf := &vulkanBuf{ptr: codeBuf.ptr, n: codeBuf.n, scalePtr: scaleBuf.ptr, scaleN: scaleBuf.n}
	return makeTensor(v, Q8_0, RowMajor, append([]int(nil), shape...), q, buf)
}

// Host returns the host-addressable f32 view only when the tensor is backed by a host
// buffer; a device-resident vulkanBuf is not host-addressable, so it returns (nil, false).
func (v *vulkanBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(*hostBuf); ok && hb.f32 != nil {
		return hb.f32, true
	}
	return nil, false
}

// Read returns the tensor as host f32: a host-backed buffer is returned directly, a
// device buffer is copied D2H into a fresh slice (the device-to-host fence).
func (v *vulkanBackend) Read(t Tensor) []float32 {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32
	}
	db := t.buf.(*vulkanBuf)
	out := make([]float32, t.Numel())
	if len(out) > 0 {
		C.fvk_d2h(unsafe.Pointer(&out[0]), db.ptr, C.size_t(len(out)*4))
	}
	return out
}

// Free releases the tensor's device buffer (and its companion Q8 scale buffer, if any)
// back to the shim and nils the handle; it is a no-op for a non-device tensor.
func (v *vulkanBackend) Free(t Tensor) {
	if db, ok := t.buf.(*vulkanBuf); ok && db.ptr != nil {
		if db.scalePtr != nil {
			C.fvk_free(db.scalePtr)
			db.scalePtr = nil
			db.scaleN = 0
		}
		C.fvk_free(db.ptr)
		db.ptr = nil
	}
}

func (v *vulkanBackend) vp(t Tensor) unsafe.Pointer { return t.buf.(*vulkanBuf).ptr }

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
	default:
		panic("compute: vulkan MatMul unsupported weight dtype " + w.Dtype.String())
	}
	return y
}

func (v *vulkanBackend) q8MatMulLocked(w, x, y Tensor, out, in, P int) {
	wb := v.q8WeightBufLocked(w, in, "Q8 MatMul")
	C.fvk_q8_matmul_f32(wb.ptr, wb.scalePtr, v.vp(x), v.vp(y),
		C.int(out), C.int(in), C.int(P))
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
	if wb.scalePtr == nil {
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
	if w0.Dtype == Q8_0 || w1.Dtype == Q8_0 {
		if w0.Dtype != Q8_0 || w1.Dtype != Q8_0 {
			panic("compute: vulkan RMSNormMatMul2 requires either all F32 or all Q8_0 weights")
		}
		wb0 := v.q8WeightBufLocked(w0, in, "Q8 RMSNormMatMul2")
		wb1 := v.q8WeightBufLocked(w1, in, "Q8 RMSNormMatMul2")
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
	case Q8_0:
		wb := v.q8WeightBufLocked(w, in, "Q8 SwiGLUMatMulAddInPlace")
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

// NewKV creates an empty device-resident KV cache sized for cfg.NumLayers, with the
// pre-RoPE keys, post-RoPE keys, and values each held in their own per-layer slices.
func (v *vulkanBackend) NewKV(cfg KVConfig) KVStore {
	k := &vulkanKV{be: v, cfg: cfg}
	k.K = make([]vslice, cfg.NumLayers)
	k.Kraw = make([]vslice, cfg.NumLayers)
	k.V = make([]vslice, cfg.NumLayers)
	return k
}

type vslice struct {
	ptr      unsafe.Pointer
	len, cap int
}

func (v *vulkanBackend) growAppend(d *vslice, srcPtr unsafe.Pointer, nFloats int) {
	if d.len+nFloats > d.cap {
		ncap := d.cap*2 + nFloats
		np := C.fvk_malloc(C.size_t(ncap * 4))
		if d.len > 0 {
			C.fvk_d2d(unsafe.Pointer(np), d.ptr, C.size_t(d.len*4))
		}
		if d.ptr != nil {
			C.fvk_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(np)
		d.cap = ncap
	}
	// append the new row at byte offset d.len within the (possibly grown) layer buffer.
	// d.ptr is an OPAQUE Buffer* handle, not a base address, so the destination offset must
	// be expressed to the shim (fvk_d2d_off) — pointer arithmetic on d.ptr would be garbage.
	C.fvk_d2d_off(d.ptr, C.size_t(d.len*4), srcPtr, C.size_t(nFloats*4))
	d.len += nFloats
}

type vulkanKV struct {
	be   *vulkanBackend
	cfg  KVConfig
	K    []vslice
	Kraw []vslice
	V    []vslice
	pos  []int
}

func (k *vulkanKV) stride() int { return k.cfg.NumKVHeads * k.cfg.HeadDim }

func (k *vulkanKV) AppendKV(layer int, kRaw, kRoPE, val Tensor, pos int) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	w := k.stride()
	k.be.growAppend(&k.Kraw[layer], kRaw.buf.(*vulkanBuf).ptr, w)
	k.be.growAppend(&k.K[layer], kRoPE.buf.(*vulkanBuf).ptr, w)
	k.be.growAppend(&k.V[layer], val.buf.(*vulkanBuf).ptr, w)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// AppendKVRoPE appends one position, applying RoPE on-device: it stores the pre-RoPE key
// (so Evict can reposition it), rotates it in place to form the post-RoPE key, and stores
// that and the value row.
func (k *vulkanKV) AppendKVRoPE(layer int, kRaw, val Tensor, pos, nHeads, headDim int, theta float64) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if nHeads != k.cfg.NumKVHeads || headDim != k.cfg.HeadDim {
		panic("compute: vulkan AppendKVRoPE shape does not match KV config")
	}
	w := k.stride()
	kRawPtr := kRaw.buf.(*vulkanBuf).ptr
	k.be.growAppend(&k.Kraw[layer], kRawPtr, w)
	C.fvk_rope_f32(kRawPtr, C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	k.be.growAppend(&k.K[layer], kRawPtr, w)
	k.be.growAppend(&k.V[layer], val.buf.(*vulkanBuf).ptr, w)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

// Len reports the number of positions currently cached.
func (k *vulkanKV) Len() int   { return len(k.pos) }
func (k *vulkanKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *vulkanKV) KeysView(layer int) Tensor {
	w := k.stride()
	n := k.K[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &vulkanBuf{ptr: k.K[layer].ptr, n: k.K[layer].len * 4})
}

// ValuesView returns a flat [pos, nKV*hd] device tensor viewing the layer's cached value
// rows, without copying the underlying storage.
func (k *vulkanKV) ValuesView(layer int) Tensor {
	w := k.stride()
	n := k.V[layer].len / w
	return makeTensor(k.be, F32, RowMajor, []int{n, w}, nil, &vulkanBuf{ptr: k.V[layer].ptr, n: k.V[layer].len * 4})
}

// Evict removes [from, from+n) from every layer and compacts the survivors, re-RoPE-ing
// each shifted key from its stored pre-RoPE copy so the cache is byte-for-byte what it
// would be had the span never been seen; it returns the number of positions removed.
func (k *vulkanKV) Evict(from, n int) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
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
		K := k.readVS(&k.K[l])
		Kraw := k.readVS(&k.Kraw[l])
		V := k.readVS(&k.V[l])
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
		k.writeVS(&k.K[l], K)
		k.writeVS(&k.Kraw[l], Kraw)
		k.writeVS(&k.V[l], V)
	}
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return end - from
}

// Clone deep-copies the cache (each layer's key, pre-RoPE key, and value buffers copied
// D2D into fresh device allocations) so a forked decode can reuse a shared prefix.
func (k *vulkanKV) Clone() KVStore {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := &vulkanKV{be: k.be, cfg: k.cfg,
		K: make([]vslice, len(k.K)), Kraw: make([]vslice, len(k.Kraw)), V: make([]vslice, len(k.V)),
		pos: append([]int(nil), k.pos...)}
	cp := func(dst, src *vslice) {
		if src.len == 0 {
			return
		}
		np := C.fvk_malloc(C.size_t(src.len * 4))
		C.fvk_d2d(unsafe.Pointer(np), src.ptr, C.size_t(src.len*4))
		dst.ptr, dst.len, dst.cap = unsafe.Pointer(np), src.len, src.len
	}
	for l := range k.K {
		cp(&n.K[l], &k.K[l])
		cp(&n.Kraw[l], &k.Kraw[l])
		cp(&n.V[l], &k.V[l])
	}
	return n
}

// Free releases every per-layer key, pre-RoPE key, and value device buffer and clears
// the position list, returning all VRAM the cache held.
func (k *vulkanKV) Free() {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	free := func(d *vslice) {
		if d.ptr != nil {
			C.fvk_free(d.ptr)
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

func (k *vulkanKV) readVS(d *vslice) []float32 {
	out := make([]float32, d.len)
	if d.len > 0 {
		C.fvk_d2h(unsafe.Pointer(&out[0]), d.ptr, C.size_t(d.len*4))
	}
	return out
}

func (k *vulkanKV) writeVS(d *vslice, data []float32) {
	need := len(data)
	if need > d.cap {
		if d.ptr != nil {
			C.fvk_free(d.ptr)
		}
		d.ptr = unsafe.Pointer(C.fvk_malloc(C.size_t(need * 4)))
		d.cap = need
	}
	if need > 0 {
		C.fvk_h2d(d.ptr, unsafe.Pointer(&data[0]), C.size_t(need*4))
	}
	d.len = need
}
