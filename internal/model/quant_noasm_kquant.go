//go:build !amd64

package model

// quant_noasm_kquant.go — the resident Q5_K int8 decode reduction dispatch for archs with no SIMD
// Q5_K reducer (everything but amd64; arm64's Q5_K reducer is a future slice and also lands scalar
// here until then). Q6_K's dispatch lives in its own arch files now: quant_amd64_kquant.go
// (AVX2/VNNI), quant_arm64_q6k.go (NEON SDOT) and quant_noasm_q6k.go (scalar, other archs) —
// mirroring the Q4_K split. q5kReduceRowScalar stays the arch-neutral reference in
// quant_kquant_int8.go; this file just provides the q5kReduceRow indirection the hot GEMV loop
// calls, so amd64 can swap in an AVX2 kernel without touching it.

// q5kReduceRow computes the per-sub-block (I_s = Σ q5*qx, S_s = Σ qx) reductions for a Q5_K row.
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}
