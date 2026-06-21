//go:build arm64

package model

// quant_arm64_q4k.go — arm64 dispatch for the resident-Q4_K int8 decode reduction. The NEON
// SDOT kernel (quant_arm64_q4k.s) computes the per-sub-block integer reductions (I_s = Σ
// nibble*qx via SDOT, S_s = Σ qx via SADDLV) for a whole row; the float combine stays in shared
// Go (q4kCombineRow), so asm correctness reduces to "the int32 reductions match the scalar
// reference" (TestQ4KReduceAsmMatchesScalar) — integer SDOT is associative with no overflow on
// these ranges, so any lane order is bit-identical. Falls back to the scalar reference on an
// arm64 part without FEAT_DotProd (neonDot). FAK_QKERNEL=scalar pins the f32 path (q4kSDOTEnabled
// observes neonDot, which observes FAK_QKERNEL).

//go:noescape
func q4kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)

// q4kSDOTEnabled reports whether the resident-Q4_K int8 decode path is active: arm64 with
// FEAT_DotProd (SDOT) and FAK_QKERNEL not pinning scalar, unless a test forces it off via
// setQ4KSDOTForTest. When false, q4kMatRowsInto keeps the byte-identical f32 scalar GEMV (the
// path TestQ4KMatRowsMatchesF32 pins).
func q4kSDOTEnabled() bool {
	if q4kSDOTForce != 0 {
		return q4kSDOTForce > 0
	}
	return neonDot
}

// q4kReduceRow dispatches the integer reduction to the NEON kernel when available, else the
// scalar reference. IS/SS are sized nblk*8 (one I_s/S_s per sub-block across all super-blocks).
func q4kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	if neonDot && nblk > 0 {
		q4kReduceRowAsm(&row[0], nblk, &qx[0], &IS[0], &SS[0])
		return
	}
	q4kReduceRowScalar(row, nblk, qx, IS, SS)
}
