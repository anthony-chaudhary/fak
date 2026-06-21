//go:build !arm64

package model

// quant_noasm_q4k.go — the resident-Q4_K decode dispatch for archs without an SDOT Q4_K kernel
// (everything but arm64 today; an amd64 AVX twin is a future phase). q4kSDOTEnabled is false, so
// q4kMatRowsInto keeps the byte-identical f32 scalar GEMV and the int8 reduction is only ever
// reached by tests exercising the scalar reference directly. The reduction itself still resolves
// to the scalar reference here for completeness.

func q4kSDOTEnabled() bool {
	if q4kSDOTForce > 0 {
		return true // test hook only (no SIMD kernel on this arch)
	}
	return false
}

func q4kReduceRow(row []byte, nblk int, qx []int8, IS, SS []int32) {
	q4kReduceRowScalar(row, nblk, qx, IS, SS)
}
