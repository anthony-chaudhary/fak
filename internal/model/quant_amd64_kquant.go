//go:build amd64

package model

// quant_amd64_kquant.go — amd64 dispatch for the resident Q5_K/Q6_K int8 decode reductions. The
// q5k/q6k hot GEMV loops (quant_kquant_int8*.go) call q5kReduceRow / q6kReduceRow; this file owns
// the amd64 build of those (quant_noasm_kquant.go owns every other arch). Q5_K has an AVX2 kernel
// (quant_amd64_kquant.s); Q6_K stays scalar until its kernel lands. q5kReduceRowScalar /
// q6kReduceRowScalar are the shared arch-neutral reference; the float combine is shared Go, so a
// reducer is bit-checked against the scalar reference (TestQ5KReduceAsmMatchesScalar).

//go:noescape
func q5kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)

// q5kReduceRow dispatches the Q5_K integer reduction to the AVX2 kernel when the resolved tier has
// it, else the scalar reference. IS/SS are sized nblk*8 (one I_s/S_s per sub-block).
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if nblk > 0 && qtier >= tierAVX2 {
		q5kReduceRowAsmAVX2(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}

// q6kReduceRow computes the per-group (I_g = Σ q6*qx, S_g = Σ qx) reductions for a Q6_K row.
// Scalar until the Q6_K AVX2 kernel lands (its 16-wide groups + position gather are a separate slice).
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
