//go:build amd64

package model

// quant_amd64_kquant.go — amd64 dispatch for the resident Q5_K/Q6_K int8 decode reductions. The
// q5k/q6k hot GEMV loops (quant_kquant_int8*.go) call q5kReduceRow / q6kReduceRow; this file owns
// the amd64 build of those (quant_noasm_kquant.go owns every other arch). The reductions stay scalar
// here until the AVX2 K-quant kernels land — q5kReduceRowScalar / q6kReduceRowScalar are the shared
// arch-neutral reference. The float combine is shared Go, so swapping a reducer in is bit-checked
// against the scalar reference (TestQ5KReduceAsmMatchesScalar / TestQ6K...).

// q5kReduceRow computes the per-sub-block (I_s = Σ q5*qx, S_s = Σ qx) reductions for a Q5_K row.
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}

// q6kReduceRow computes the per-group (I_g = Σ q6*qx, S_g = Σ qx) reductions for a Q6_K row.
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
