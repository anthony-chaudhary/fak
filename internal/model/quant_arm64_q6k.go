//go:build arm64 && !(fakaccel && darwin && cgo)

package model

// quant_arm64_q6k.go — arm64 dispatch for the resident-Q6_K int8 decode reduction, the twin of
// quant_arm64_q4k.go. The NEON SDOT kernel (quant_arm64_q6k.s) computes the per-group integer
// reductions (I_g = Σ q6*qx via SDOT, S_g = Σ qx via SDOT vs ones) for a whole row; the float
// combine stays in shared Go (q6kCombineRow), so asm correctness reduces to "the int32 reductions
// match the scalar reference" (TestQ6KReduceAsmMatchesScalar) — integer SDOT is associative with no
// overflow on these ranges, so any lane order is bit-identical. Falls back to the scalar reference
// (q6kReduceRowScalar) on an arm64 part without FEAT_DotProd (neonDot), exactly as Q4_K does.
// FAK_QKERNEL=scalar pins neonDot off, so the scalar path stays exercised.

//go:noescape
func q6kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)

// q6kReduceRow dispatches the per-group Q6_K integer reduction to the NEON SDOT kernel when the part
// has FEAT_DotProd, else the scalar reference. IS/SS are sized nblk*16 (one I_g/S_g per group).
func q6kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if neonDot && nblk > 0 {
		q6kReduceRowAsm(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q6kReduceRowScalar(row, nblk, qx, IS, SS)
}
