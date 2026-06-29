//go:build !arm64 && !amd64

package model

// quant_noasm_q6k.go — the resident-Q6_K int8 decode reduction dispatch for archs with no SIMD Q6_K
// reducer (everything but amd64 and arm64). It is the scalar-fallback twin of quant_noasm_q4k.go:
// amd64 has the AVX2/VNNI kernel (quant_amd64_kquant.go) and arm64 the NEON SDOT kernel
// (quant_arm64_q6k.go); every other arch routes the per-group reduction to q6kReduceRowScalar, the
// arch-neutral reference in quant_kquant_int8_q6k.go.

// q6kReduceRow computes the per-group (I_g = Σ q6*qx, S_g = Σ qx) reductions for a Q6_K row.
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
