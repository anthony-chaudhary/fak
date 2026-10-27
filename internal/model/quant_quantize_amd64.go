//go:build amd64

package model

// quantizeRowAsm512 quantizes one activation row of nblk Q8_0 blocks (nblk*32 floats at x)
// into codes q (nblk*32 int8) and per-block scales d (nblk float32), with AVX-512F+BW.
// Bit-identical to quantizeRowQ8scalar — pinned by TestQuantizeRowAsmMatchesScalar.
//
//go:noescape
func quantizeRowAsm512(x *float32, q *int8, d *float32, nblk int)

// quantizeRowAsm256 processes a finite prefix of nblk blocks using AVX2. It
// returns the number fully written; the first exceptional block and all later
// blocks are untouched for the original scalar suffix. Input, codes and scales
// are disjoint in the activation-panel/vector producers.
//
//go:noescape
func quantizeRowAsm256(x *float32, q *int8, d *float32, nblk int) (done int)

// quantizeRowQ8 dispatches the activation-row quantization: the AVX-512 kernel when the
// resolved tier has it (same gate as the dot/GEMM kernels), or the AVX2 finite
// prefix kernel. Exceptional AVX2 blocks and unsupported shapes use the original
// scalar reference. Existing AVX-512 and scalar selections are unchanged.
func quantizeRowQ8(x []float32, q []int8, d []float32, nblk int) {
	if nblk > 0 && qtier == tierAVX512 {
		quantizeRowAsm512(&x[0], &q[0], &d[0], nblk)
		return
	}
	if nblk > 0 && qtier == tierAVX2 && nblk <= len(x)/qBlk && nblk <= len(q)/qBlk && nblk <= len(d) {
		done := quantizeRowAsm256(&x[0], &q[0], &d[0], nblk)
		if done == nblk {
			return
		}
		quantizeRowQ8scalar(x[done*qBlk:], q[done*qBlk:], d[done:], nblk-done)
		return
	}
	quantizeRowQ8scalar(x, q, d, nblk)
}
