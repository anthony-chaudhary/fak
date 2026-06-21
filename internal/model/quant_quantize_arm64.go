//go:build arm64

package model

// quantizeRowAsmNEON is the NEON activation-row quantizer (quant_quantize_arm64.s), bit-identical
// to quantizeRowQ8scalar (with q8round //go:noinline) — pinned by TestQuantizeRowAsmMatchesScalar.
//
//go:noescape
func quantizeRowAsmNEON(x *float32, q *int8, d *float32, nblk int)

// quantizeRowQ8 dispatches activation-row quantization to the NEON kernel when FEAT_DotProd is
// present (same gate as the dot/GEMM kernels — every Apple Silicon part has it), else the portable
// scalar reference. The scalar quantizer (~18% of prefill, ~13% of decode) was the largest
// remaining non-GEMM term on arm64; the NEON path is bit-identical, so this changes only speed.
func quantizeRowQ8(x []float32, q []int8, d []float32, nblk int) {
	if neonDot && nblk > 0 {
		quantizeRowAsmNEON(&x[0], &q[0], &d[0], nblk)
		return
	}
	quantizeRowQ8scalar(x, q, d, nblk)
}
