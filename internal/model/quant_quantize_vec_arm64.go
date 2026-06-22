//go:build arm64

package model

// quantizeVecAsmNEON is the NEON decode activation-vector quantizer authored for issue #476; the
// kernel lives in quant_quantize_vec_arm64.s and is held bit-identical to quantizeRowQ8scalar.
//
//go:noescape
func quantizeVecAsmNEON(x *float32, q *int8, d *float32, nblk int)

// quantizeVecQ8NEON quantizes one activation vector to Q8_0 using the NEON kernel when the CPU has
// FEAT_DotProd (neonDot — the same gate the dot/GEMM kernels use), and the portable scalar
// reference (quantizeRowQ8scalar) on any arm64 part without it. This is the independently-authored
// vec-form deliverable issue #476 names: a NEON amax reduction per 32-block + a vectorized x*inv +
// FRINTA round (ties-away == q8round), with the scalar version retained as the reference/fallback.
//
// q16 stays empty: arm64's SDOT consumes int8 directly (q8PreextendVec()==false), so the decode
// path wants no int16 pre-extension — matching what quantizeVecQ8Into builds on arm64.
//
// The production decode quantizer (quantizeVecQ8 -> quantizeVecQ8Into -> quantizeRowQ8) already
// routes through the proven NEON row kernel (quantizeRowAsmNEON), so this is NOT rewired into the
// hot path — it is the #476 vec-form deliverable and a differential-correctness oracle, pinned
// BIT-IDENTICAL to both the scalar reference and that production kernel by
// TestQuantizeVecQ8NEONMatchesScalar. The acceptance gate that proves it on real silicon is
// tools/run_476_acceptance_on_arm64.sh.
func quantizeVecQ8NEON(x []float32) q8Vec {
	in := len(x)
	if in%qBlk != 0 {
		panic("model: Q8_0 activation length not a multiple of 32")
	}
	nblk := in / qBlk
	qv := q8Vec{q: make([]int8, in), d: make([]float32, nblk), nblk: nblk}
	if neonDot && nblk > 0 {
		quantizeVecAsmNEON(&x[0], &qv.q[0], &qv.d[0], nblk)
	} else {
		quantizeRowQ8scalar(x, qv.q, qv.d, nblk)
	}
	return qv
}
