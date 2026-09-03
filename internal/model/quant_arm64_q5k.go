//go:build arm64 && !(fakaccel && darwin && cgo)

package model

// Prior-art: llama.cpp ggml-quants.c (K-quant vec_dot SIMD kernels)
// quant_arm64_q5k.go — arm64 dispatch for the resident-Q5_K int8 decode reduction, the twin of
// quant_arm64_q4k.go and quant_arm64_q6k.go. The NEON SDOT kernel (quant_arm64_q5k.s) computes
// the per-sub-block integer reductions (I_s = Σ q5*qx via SDOT, S_s = Σ qx via SDOT vs ones) for
// a whole row; the float combine stays in shared Go (kQuantCombineRow), so asm correctness
// reduces to "the int32 reductions match the scalar reference" (TestQ5KReduceAsmMatchesScalar) —
// integer SDOT is associative with no overflow on these ranges, so any lane order is bit-identical.
// Falls back to the scalar reference (q5kReduceRowScalar) on an arm64 part without FEAT_DotProd
// (neonDot), exactly as Q4_K and Q6_K do. FAK_QKERNEL=scalar pins neonDot off, so the scalar
// path stays exercised.

//go:noescape
func q5kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)

// q5kReduceRow dispatches the per-sub-block Q5_K integer reduction to the NEON SDOT kernel when the
// part has FEAT_DotProd, else the scalar reference. IS/SS are sized nblk*8 (one I_s/S_s per sub-block).
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if neonDot && nblk > 0 {
		q5kReduceRowAsm(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}
