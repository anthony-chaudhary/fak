//go:build amd64

package model

// quantizeRowAsm512 quantizes one activation row of nblk Q8_0 blocks (nblk*32 floats at x)
// into codes q (nblk*32 int8) and per-block scales d (nblk float32), with AVX-512F+BW.
// Bit-identical to quantizeRowQ8scalar — pinned by TestQuantizeRowAsmMatchesScalar.
//
//go:noescape
func quantizeRowAsm512(x *float32, q *int8, d *float32, nblk int)

// quantizeRowQ8 dispatches the activation-row quantization: the AVX-512 kernel when the
// resolved tier has it (same gate as the dot/GEMM kernels), else the portable scalar
// reference. The AVX-512 path is bit-identical, so this changes only speed.
func quantizeRowQ8(x []float32, q []int8, d []float32, nblk int) {
	if nblk > 0 && qtier == tierAVX512 {
		quantizeRowAsm512(&x[0], &q[0], &d[0], nblk)
		return
	}
	quantizeRowQ8scalar(x, q, d, nblk)
}
