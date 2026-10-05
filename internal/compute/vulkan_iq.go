//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

// vulkan_iq.go — native llama.cpp i-quant weights on Vulkan (ticket 05f).
//
// A raw i-quant weight (quant_rawiq.go) is uploaded verbatim and multiplied by its
// iq*_matvec.spv kernel when that pipeline is available. When it is not (absent SPIR-V, or
// the format's FAK_VULKAN_<FMT>=0 kill switch read by the shim), Upload expands the weight to
// Q8_0 through the CPU dequant, so callers keep the pre-05f Q8 kernels and representation.

// iqNativeLocked reports whether dt is a raw i-quant dtype whose native kernel is built.
func (v *vulkanBackend) iqNativeLocked(dt Dtype) bool {
	f, ok := rawIQFormats[dt]
	return ok && C.fvk_iq_matvec_available(C.int(f.vulkanID)) != 0
}

func (v *vulkanBackend) uploadRawIQLocked(t Tensor) Tensor {
	f := rawIQFormats[t.Dtype]
	hb, ok := t.buf.(HostBuffer)
	if !ok || len(t.Shape) != 2 || t.Shape[1]%rawIQSuper != 0 {
		panic("compute: vulkan " + t.Dtype.String() + " upload requires host raw bytes and [out,in] with in divisible by 256")
	}
	raw := i8AsBytes(hb.I8())
	if len(raw) != t.Shape[0]*(t.Shape[1]/rawIQSuper)*f.blockBytes {
		panic("compute: vulkan " + t.Dtype.String() + " raw byte length does not match shape")
	}
	if !v.iqNativeLocked(t.Dtype) {
		// Kill switch / missing kernel: the pre-05f representation (dequant -> Q8_0).
		deq := DequantRawIQ(t.Dtype, t.Shape[0], t.Shape[1], raw)
		q := QuantizeQ8(Default(), t.Shape, deq, 32)
		qh := q.buf.(HostBuffer)
		return v.uploadQ8Locked(q.Shape, qh.I8(), q.Quant.Scale, q.Quant.Block)
	}
	buf := v.dallocWeightFor(len(raw), t.Dtype.String()+" weight buffer "+shapeText(t.Shape))
	if len(raw) > 0 {
		C.fvk_h2d(buf.ptr, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
	}
	return makeTensor(v, t.Dtype, RowMajor, append([]int(nil), t.Shape...), t.Quant, buf)
}

// iqMatVecLocked computes y[t] = W x[t] for t < tokens with a native i-quant weight.
func (v *vulkanBackend) iqMatVecLocked(w, x, y Tensor, out, in, tokens int) {
	f := rawIQFormats[w.Dtype]
	C.fvk_iq_matvec_f32(C.int(f.vulkanID), v.vp(w), v.vp(x), v.vp(y), nil, nil,
		C.int(out), 0, C.int(in), C.int(tokens), 0)
}

// iqMatVecAddLocked accumulates y += W x for one token.
func (v *vulkanBackend) iqMatVecAddLocked(w, x, y Tensor, out, in int) {
	f := rawIQFormats[w.Dtype]
	C.fvk_iq_matvec_f32(C.int(f.vulkanID), v.vp(w), v.vp(x), v.vp(y), nil, nil,
		C.int(out), 0, C.int(in), 1, 1)
}

// iqMatVecPairLocked computes y0 = W0 x and y1 = W1 x in one dispatch (same dtype).
func (v *vulkanBackend) iqMatVecPairLocked(w0, w1, x, y0, y1 Tensor, out0, out1, in int) {
	f := rawIQFormats[w0.Dtype]
	C.fvk_iq_matvec_f32(C.int(f.vulkanID), v.vp(w0), v.vp(x), v.vp(y0), v.vp(w1), v.vp(y1),
		C.int(out0), C.int(out1), C.int(in), 1, 2)
}

// composeRawIQLocked serves a fused multi-projection op (MatMul2/MatMul3 and the RMSNorm
// variants) whose weights include a raw i-quant: the fused kernels require uniform F32 or
// Q8_0 weights, so each projection runs on its own kernel against the shared single-token
// input x. It reports false (doing nothing) when no weight is a raw i-quant.
func (v *vulkanBackend) composeRawIQLocked(ws []Tensor, x Tensor, ys []Tensor, in int) bool {
	hit := false
	for _, w := range ws {
		hit = hit || isRawIQ(w.Dtype)
	}
	if !hit {
		return false
	}
	for i, w := range ws {
		out := w.Shape[0]
		switch {
		case isRawIQ(w.Dtype):
			v.iqMatVecLocked(w, x, ys[i], out, in, 1)
		case w.Dtype == Q8_0:
			v.q8MatMulLocked(w, x, ys[i], out, in, 1)
		case w.Dtype == Q4_K:
			v.q4kMatMulLocked(w, x, ys[i], out, in, 1)
		case w.Dtype == Q2_K:
			v.q2kMatMulLocked(w, x, ys[i], out, in, 1)
		case w.Dtype == F32:
			C.fvk_matmul_f32(v.vp(w), v.vp(x), v.vp(ys[i]), C.int(out), C.int(in), 1)
		default:
			panic("compute: vulkan fused projection unsupported companion weight dtype " + w.Dtype.String())
		}
	}
	return true
}
