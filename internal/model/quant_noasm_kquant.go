//go:build !amd64

package model

// quant_noasm_kquant.go — the resident Q5_K/Q6_K int8 decode reduction dispatch for archs with no
// SIMD K-quant reducer (everything but amd64; arm64's K-quant reducers are a future slice and also
// land scalar here until then). q5kReduceRowScalar / q6kReduceRowScalar stay the arch-neutral
// reference in quant_kquant_int8*.go; this file just provides the q5kReduceRow / q6kReduceRow
// indirection the hot GEMV loops call, so amd64 can swap in an AVX2 kernel without touching them.

// q5kReduceRow computes the per-sub-block (I_s = Σ q5*qx, S_s = Σ qx) reductions for a Q5_K row.
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}

// q6kReduceRow computes the per-group (I_g = Σ q6*qx, S_g = Σ qx) reductions for a Q6_K row.
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
