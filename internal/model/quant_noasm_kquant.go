//go:build (!arm64 && !amd64) || (fakaccel && darwin && arm64 && cgo)

package model

// quant_noasm_kquant.go — the resident-Q5_K int8 decode reduction dispatch for archs with no SIMD
// Q5_K reducer (everything but amd64 and arm64). It is the scalar-fallback twin of quant_noasm_q4k.go
// and quant_noasm_q6k.go: amd64 has the AVX2/VNNI kernel (quant_amd64_kquant.go) and arm64 the NEON
// SDOT kernel (quant_arm64_q5k.go); every other arch routes the per-sub-block reduction to
// q5kReduceRowScalar, the arch-neutral reference in quant_kquant_int8.go.

// q5kReduceRow computes the per-sub-block (I_s = Σ q5*qx, S_s = Σ qx) reductions for a Q5_K row.
func q5kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q5kReduceRowScalar(row, nblk, qx, IS, SS)
}
