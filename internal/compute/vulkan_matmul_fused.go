//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo !v41_indexer_witness LDFLAGS: -L${SRCDIR} -lfakvulkan
#cgo v41_indexer_witness LDFLAGS: -L${SRCDIR} -lfakvulkan_v41_indexer_witness
#include <stdlib.h>
#include "vulkan_backend.h"
// Issue-local adapter while the shared Vulkan C ABI remains stable: the fused Q2_K
// tail reuses the required q2k_matmul pipeline rather than adding an optional module.
void fvk_swiglu_q2k_matmul_add_f32(const void *dW, const void *dG, const void *dU,
                                   void *dD, int out, int in, int P);
void fvk_rmsnorm_q2k_matmul2_f32(const void* dW0, const void* dW1,
    const void* dX, const void* dNorm, void* dY0, void* dY1,
    int out0, int out1, int in, int P, float eps);

// Issue-local #12217 ABI pending the generic #11096 VMM contract. Keeping these declarations
// beside the only Go consumer avoids widening the public backend header before that contract lands.
void *fvk_malloc_weight(size_t bytes, uint64_t max_arena_bytes);
void fvk_weight_arena_stats(uint64_t *memory_allocations, uint64_t *buffer_bindings,
                            uint64_t *reserved_bytes, uint64_t *live_bytes,
                            uint64_t *peak_reserved_bytes);

// #12218 exact batch resource hazards: the Go lowering proves a window's declarations
// complete, then arms the shim and sets one ordinal-keyed sync verdict per declared
// dispatch. The ordinal is the shim's own recorded-op index, read at declaration time, so
// a verdict can only elide the exact dispatch it was computed for.
void fvk_batch_hazards_arm(int armed);
int fvk_batch_hazards_armed(void);
int fvk_batch_next_ordinal(void);
void fvk_batch_hazards_set(int ordinal, int needs_sync);
void fvk_batch_hazards_reset(void);
*/
import "C"

import "fmt"

// MatMulArgmax returns the final projection's largest-logit index without copying
// logits host-ward. F32 uses the fused shader; Q2_K stays packed for its device
// projection and composes the existing device argmax until the packed fused shader lands.
func (v *vulkanBackend) MatMulArgmax(w, x Tensor) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	out, in := w.Shape[0], w.Shape[1]
	if in == 0 || x.Numel() != in {
		panic("compute: vulkan MatMulArgmax expects one input row matching the weight input dim")
	}
	switch w.Dtype {
	case F32:
		return int(C.fvk_matmul_argmax_f32(v.vp(w), v.vp(x), C.int(out), C.int(in)))
	case Q2_K:
		logits, _ := v.devTr([]int{out}, F32)
		v.q2kMatMulLocked(w, x, logits, out, in, 1)
		return int(C.fvk_argmax_f32(v.vp(logits), C.int(out)))
	default:
		panic("compute: vulkan MatMulArgmax supports only F32 or Q2_K weights (got " + w.Dtype.String() + ")")
	}
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
// F32, Q8_0, Q4_K, Q6_K, Q5_K, Q3_K, or Q2_K shader by the weight's dtype.
func (v *vulkanBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if w.Dtype == Q6_K {
		v.validateQ6KMatMulInputs(w, X, P)
	}
	if w.Dtype == Q5_K {
		v.validateQ5KMatMulInputs(w, X, P)
	}
	if w.Dtype == Q3_K {
		v.validateQ3KMatMulInputs(w, X, P)
	}
	out, in := w.Shape[0], w.Shape[1]
	if P <= 0 || in <= 0 || X.Numel() != P*in {
		panic(fmt.Sprintf("compute: vulkan BatchedMatMul input numel=%d, want P*in=%d*%d", X.Numel(), P, in))
	}
	y, _ := v.devTr([]int{P, out}, F32)
	v.recordBatchDispatch(
		VulkanBufferAccess{BufferID: v.bufferIdentity(w), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(X), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(y), Size: 0, Mode: VulkanAccessWrite},
	)
	switch w.Dtype {
	case F32:
		C.fvk_matmul_f32(v.vp(w), v.vp(X), v.vp(y), C.int(out), C.int(in), C.int(P))
	case Q8_0:
		v.q8MatMulLocked(w, X, y, out, in, P)
	case Q4_K:
		v.q4kMatMulLocked(w, X, y, out, in, P)
	case Q6_K:
		v.q6kMatMulLocked(w, X, y, P)
	case Q5_K:
		v.q5kMatMulLocked(w, X, y, P)
	case Q3_K:
		v.q3kMatMulLocked(w, X, y, P)
	case Q2_K:
		v.q2kMatMulLocked(w, X, y, out, in, P)
	default:
		if !isRawIQ(w.Dtype) {
			panic("compute: vulkan BatchedMatMul unsupported weight dtype " + w.Dtype.String())
		}
		v.iqMatVecLocked(w, X, y, out, in, P)
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
	v.recordBatchDispatch(
		VulkanBufferAccess{BufferID: v.bufferIdentity(w), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(x), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(dst), Size: 0, Mode: VulkanAccessReadWrite},
	)
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
	if v.composeRawIQLocked([]Tensor{w0, w1}, x, []Tensor{y0, y1}, in) {
		return y0, y1
	}

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
	if v.composeRawIQLocked([]Tensor{wq, wk, wv}, x, []Tensor{q, k, val}, in) {
		return q, k, val
	}
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
// in one decode-only operation, returning both outputs. Pairs containing Q2_K
// compose normalization and the existing projection kernels without expanding weights.
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
	if w0.Dtype == Q2_K || w1.Dtype == Q2_K || isRawIQ(w0.Dtype) || isRawIQ(w1.Dtype) {
		// Refuse both operands before allocating or recording normalization. The
		// presence of a Q2 kernel does not admit Q5/Q6 or other unsupported formats.
		for _, w := range []Tensor{w0, w1} {
			switch {
			case w.Dtype == F32, w.Dtype == Q8_0, w.Dtype == Q4_K, w.Dtype == Q2_K, isRawIQ(w.Dtype):
			default:
				panic("compute: vulkan RMSNormMatMul2 unsupported companion weight dtype " + w.Dtype.String())
			}
		}
		y0, _ := v.devTr([]int{out0}, F32)
		y1, _ := v.devTr([]int{out1}, F32)
		if w0.Dtype == Q2_K && w1.Dtype == Q2_K && v.selectQ2KFusionLocked(P) {
			C.fvk_rmsnorm_q2k_matmul2_f32(v.vp(w0), v.vp(w1), v.vp(x), v.vp(normWeight), v.vp(y0), v.vp(y1),
				C.int(out0), C.int(out1), C.int(in), C.int(P), C.float(eps))
			return y0, y1
		}
		xn, _ := v.devTr([]int{in}, F32)
		C.fvk_rmsnorm_f32(v.vp(x), v.vp(normWeight), v.vp(xn), C.int(P), C.int(in), C.float(eps))
		if isRawIQ(w0.Dtype) && w1.Dtype == w0.Dtype {
			v.iqMatVecPairLocked(w0, w1, xn, y0, y1, out0, out1, in)
			return y0, y1
		}
		project := func(w, y Tensor, out int) {
			if isRawIQ(w.Dtype) {
				v.iqMatVecLocked(w, xn, y, out, in, P)
				return
			}
			switch w.Dtype {
			case Q2_K:
				v.q2kMatMulLocked(w, xn, y, out, in, P)
			case Q4_K:
				v.q4kMatMulLocked(w, xn, y, out, in, P)
			case Q8_0:
				v.q8MatMulLocked(w, xn, y, out, in, P)
			case F32:
				C.fvk_matmul_f32(v.vp(w), v.vp(xn), v.vp(y), C.int(out), C.int(in), C.int(P))
			}
		}
		project(w0, y0, out0)
		project(w1, y1, out1)
		return y0, y1
	}
	y0, _ := v.devTr([]int{out0}, F32)
	y1, _ := v.devTr([]int{out1}, F32)
	if w0.Dtype == Q4_K || w1.Dtype == Q4_K {
		if w0.Dtype != Q4_K || w1.Dtype != Q4_K {
			panic("compute: vulkan RMSNormMatMul2 requires either all F32, all Q8_0, or all Q4_K weights")
		}
		if v.selectQ4KFusionLocked(P) {
			v.q4kFusionRMSNormCalls++
			C.fvk_rmsnorm_q4k_matmul2_f32(v.vp(w0), v.vp(w1), v.vp(x), v.vp(normWeight), v.vp(y0), v.vp(y1),
				C.int(out0), C.int(out1), C.int(in), C.int(P), C.float(eps))
			return y0, y1
		}
		// Unchanged Q4_K composition: RMSNorm + 2 Q4_K GEMVs (three dispatches)
		v.q4kComposedRMSNormCalls++
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
	if isRawIQ(wq.Dtype) || isRawIQ(wk.Dtype) || isRawIQ(wv.Dtype) {
		xn, _ := v.devTr([]int{in}, F32)
		C.fvk_rmsnorm_f32(v.vp(x), v.vp(normWeight), v.vp(xn), C.int(P), C.int(in), C.float(eps))
		v.composeRawIQLocked([]Tensor{wq, wk, wv}, xn, []Tensor{q, k, val}, in)
		return q, k, val
	}
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
	if isRawIQ(w.Dtype) {
		sw, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
		C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(sw), C.int(gate.Numel()))
		if P == 1 {
			v.iqMatVecAddLocked(w, sw, dst, out, in)
			return
		}
		proj, _ := v.devTr([]int{P, out}, F32)
		v.iqMatVecLocked(w, sw, proj, out, in, P)
		C.fvk_add_f32(v.vp(dst), v.vp(proj), C.int(dst.Numel()))
		return
	}
	switch w.Dtype {
	case F32:
		C.fvk_swiglu_matmul_add_f32(v.vp(w), v.vp(gate), v.vp(up), v.vp(dst), C.int(out), C.int(in), C.int(P))
	case Q4_K, Q2_K:
		if w.Dtype == Q2_K && P == 1 {
			wb := w.buf.(*vulkanBuf)
			C.fvk_swiglu_q2k_matmul_add_f32(wb.ptr, v.vp(gate), v.vp(up), v.vp(dst), C.int(out), C.int(in), C.int(P))
			return
		}
		if w.Dtype == Q4_K && v.selectQ4KFusionLocked(P) {
			v.q4kFusionSwiGLUCalls++
			C.fvk_swiglu_q4k_matmul_add_f32(v.vp(w), v.vp(gate), v.vp(up), v.vp(dst), C.int(out), C.int(in), C.int(P))
			return
		}
		if w.Dtype == Q4_K {
			// Unchanged Q4_K composition: SwiGLU + Q4_K GEMV + Add (three dispatches)
			v.q4kComposedSwiGLUCalls++
		}
		sw, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
		C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(sw), C.int(gate.Numel()))
		projShape := []int{P, out}
		if P == 1 {
			projShape = []int{out}
		}
		proj, _ := v.devTr(projShape, F32)
		if w.Dtype == Q2_K {
			v.q2kMatMulLocked(w, sw, proj, out, in, P)
		} else {
			v.q4kMatMulLocked(w, sw, proj, out, in, P)
		}
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
