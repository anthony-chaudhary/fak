//go:build amd64

package model

import "unsafe"

// awq_amd64_asm.go — AWQ SIMD kernels for amd64 (#1124 C4 / #1128).
//
// Until this file landed, awqDequantRow*/awqDotProduct* were placeholders that
// delegated straight to the scalar reference. They are now real AVX2 + AVX-512
// Go-assembly (awq_amd64_simd.s): nibble-unpack → sub zero-point → int→float →
// scale → store (dequant) / FMA-accumulate (dot). Pure Go-assembly — no cgo,
// links into the static binary, passes TestRequestPathDefaultBuildIsCgoFree.
//
// Correctness (the epic #1124 / #209 bar):
//   - DEQUANT is BIT-IDENTICAL to awqDequantRowScalar: every element is the same
//     int16(nibble)-8 → float32 → *scale, in the same order (no reduction), so
//     the asm and scalar produce bit-for-bit equal floats. Pinned by
//     TestAWQDequantRow{AVX2,AVX512}MatchesScalar (exact ==).
//   - DOT is COSINE-PARITY: the asm folds scale per element (matching the scalar
//     per-element scale) but accumulates across SIMD lanes, so the float
//     reduction order differs from the sequential scalar sum. Bit-identity is
//     not achievable for a reduction; cosine ≈ 1.0 and a tight relative bound
//     are the gate (TestAWQDotProduct{AVX2,AVX512}MatchesScalar).
//
// The asm cores below process whole SIMD blocks (4 src bytes → 8 weights for
// AVX2, 8 src bytes → 16 weights for AVX-512); the Go wrappers handle the sub-
// block tail with the shared scalar reference so any row length is exact. The
// dispatch (awxMatRowsRangeAVX / awxGemmRangeAVX in awq_amd64.go) is unchanged
// and still CPUID/FAK_AWQ_KERNEL-gated, so the default path is provably untouched
// where the feature is absent.

// ---- asm cores (whole-block only; bytes a multiple of the block) -------------

//go:noescape
func awqDequantRowAsmAVX2(dst *float32, scale float32, src *byte, nbytes int)

//go:noescape
func awqDequantRowAsmAVX512(dst *float32, scale float32, src *byte, nbytes int)

//go:noescape
func awqDotProductAsmAVX2(src *byte, scale float32, x *float32, nbytes int) float32

//go:noescape
func awqDotProductAsmAVX512(src *byte, scale float32, x *float32, nbytes int) float32

// ---- AVX2 wrappers -----------------------------------------------------------

// awqDequantRowAVX2 dequantizes one AWQ row using the AVX2 core for the whole
// 4-byte (8-weight) blocks and the scalar reference for the sub-block tail.
// Bit-identical to awqDequantRowScalar.
func awqDequantRowAVX2(dst []float32, scale float32, src *byte, n int) {
	nb := n / 2     // packed bytes (2 weights per byte)
	full := nb &^ 3 // whole 4-byte AVX2 blocks
	if full > 0 {
		awqDequantRowAsmAVX2(&dst[0], scale, src, full)
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		awqDequantRowScalar(dst[full*2:], scale, &srcSlice[full], (nb-full)*2)
	}
}

// awqDotProductAVX2 computes dot(scale*(code-8), x) using the AVX2 core for the
// whole 4-byte (8-weight) blocks and the scalar reference for the tail.
// Cosine-parity with awqDotProductScalar (lane-reduced sum).
func awqDotProductAVX2(src *byte, scale float32, x *float32, n int) float32 {
	nb := n / 2
	full := nb &^ 3
	var acc float32
	if full > 0 {
		acc = awqDotProductAsmAVX2(src, scale, x, full)
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		xSlice := unsafe.Slice(x, n)
		acc += awqDotProductScalar(&srcSlice[full], scale, &xSlice[full*2], (nb-full)*2)
	}
	return acc
}

// ---- AVX-512 wrappers --------------------------------------------------------

// awqDequantRowAVX512 dequantizes one AWQ row using the AVX-512 core for the
// whole 8-byte (16-weight) blocks and the scalar reference for the tail.
// Bit-identical to awqDequantRowScalar.
func awqDequantRowAVX512(dst []float32, scale float32, src *byte, n int) {
	nb := n / 2     // packed bytes
	full := nb &^ 7 // whole 8-byte AVX-512 blocks
	if full > 0 {
		awqDequantRowAsmAVX512(&dst[0], scale, src, full)
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		awqDequantRowScalar(dst[full*2:], scale, &srcSlice[full], (nb-full)*2)
	}
}

// awqDotProductAVX512 computes dot(scale*(code-8), x) using the AVX-512 core for
// the whole 8-byte (16-weight) blocks and the scalar reference for the tail.
// Cosine-parity with awqDotProductScalar (lane-reduced sum).
func awqDotProductAVX512(src *byte, scale float32, x *float32, n int) float32 {
	nb := n / 2
	full := nb &^ 7
	var acc float32
	if full > 0 {
		acc = awqDotProductAsmAVX512(src, scale, x, full)
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		xSlice := unsafe.Slice(x, n)
		acc += awqDotProductScalar(&srcSlice[full], scale, &xSlice[full*2], (nb-full)*2)
	}
	return acc
}
