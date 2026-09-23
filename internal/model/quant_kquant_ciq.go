package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// quant_kquant_ciq.go — the legacy ggml Q8_0 / Q4_0 resident decode GEMV via the compute CIQ
// (Compute-in-Quant) integer dot products (issue #12275, vllm.cpp borrow). The f32
// kQuantMatRowsRange path dequantizes every 32-weight block to f32 and dots it, which is
// compute- and memory-bus-bound on the packed-8-bit weight traffic. The CIQ path quantizes
// the activation ONCE to Q8_0 and reduces int8×int8 directly on the resident bytes, so
// Q8_0 streams 1.0625 B/weight and Q4_0 0.5625 B/weight instead of the 4 B/weight f32
// expansion — the same lever the K-quant int8 reducers already deliver for Q5_K/Q6_K.
//
// The integer reduction is APPROX compares to the f32 dequant-then-dot reference (it adds
// activation quantization), so like the Q4_K/Q5_K/Q6_K int8 paths it rides the same gate:
// kQuantSDOTEnabled (FAK_KQ_INT8 env or the test force), default OFF with the f32 reduction
// as the conservative floor. The f32 kQuantMatRowsRange is byte-unchanged.

// ciqMatRowsRangeInt8 is the int8 GEMV over output rows [lo,hi) for a legacy Q8_0/Q4_0
// tensor: one shared Q8_0 activation quantize (qv), then per row the CIQ integer dot. It
// mirrors q5kMatRowsRangeInt8's shape but delegates the row math to the compute CIQ kernels,
// which hold no model dependency.
func ciqMatRowsRangeInt8(qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	ciqMatRowsRangeInt8Raw(qt.raw, qt, qv, y, lo, hi)
}

func ciqMatRowsRangeInt8Raw(raw []byte, qt *kQuantTensor, qv q8Vec, y []float32, lo, hi int) {
	if len(raw) == 0 {
		raw = qt.raw
	}
	qx := qv.q
	dx := qv.d
	rowBytes := qt.rowBytes()
	for o := lo; o < hi; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		switch qt.kind {
		case kindQ4_0:
			y[o] = compute.VecDotQ4_0Q8_0(row, qt.nblk, qx, dx)
		default: // kindQ8_0
			y[o] = compute.VecDotQ8_0Q8_0(row, qt.nblk, qx, dx)
		}
	}
}
