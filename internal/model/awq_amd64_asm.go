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

// ---- shared block-then-tail drivers ------------------------------------------
//
// Both cores consume whole SIMD blocks only, so every row runs the same shape: the core
// over the whole blocks, then the scalar reference over the sub-block remainder [full,nb).
// The ISA contributes exactly two facts — the block width in packed bytes, and which core
// runs — so the drivers below take the width and the wrappers pass it. The core is chosen
// by a direct branch on that width, NOT a func value: the asm declarations are //go:noescape
// and an indirect call would forfeit that, pushing the caller's dst/x buffers onto the heap
// in a per-row loop.

// AWQ SIMD block widths, in packed source bytes (2 weights per byte).
const (
	awqBlockBytesAVX2   = 4 // 4 bytes = 8 weights per AVX2 block
	awqBlockBytesAVX512 = 8 // 8 bytes = 16 weights per AVX-512 block
)

// awqDequantRowSIMD dequantizes one AWQ row of n weights: whole blockBytes-wide blocks
// through the matching asm core, the sub-block tail through awqDequantRowScalar. Every
// element is the same int16(nibble)-8 → float32 → *scale in the same order as the scalar
// reference (dequant is elementwise, no reduction), so the result is bit-identical to
// awqDequantRowScalar for any n.
func awqDequantRowSIMD(dst []float32, scale float32, src *byte, n, blockBytes int) {
	nb := n / 2                    // packed bytes (2 weights per byte)
	full := nb &^ (blockBytes - 1) // whole SIMD blocks
	if full > 0 {
		if blockBytes == awqBlockBytesAVX512 {
			awqDequantRowAsmAVX512(&dst[0], scale, src, full)
		} else {
			awqDequantRowAsmAVX2(&dst[0], scale, src, full)
		}
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		awqDequantRowScalar(dst[full*2:], scale, &srcSlice[full], (nb-full)*2)
	}
}

// awqDotProductSIMD computes dot(scale*(code-8), x) over n weights: whole blockBytes-wide
// blocks through the matching asm core, the sub-block tail through awqDotProductScalar.
// The tail is ADDED to the core's accumulator only when it exists, rather than folding an
// unconditional +0 in (a zero add is not a no-op on a negatively-signed zero). Cosine-parity
// with awqDotProductScalar, not bit-identity: the core reduces across SIMD lanes.
func awqDotProductSIMD(src *byte, scale float32, x *float32, n, blockBytes int) float32 {
	nb := n / 2
	full := nb &^ (blockBytes - 1)
	var acc float32
	if full > 0 {
		if blockBytes == awqBlockBytesAVX512 {
			acc = awqDotProductAsmAVX512(src, scale, x, full)
		} else {
			acc = awqDotProductAsmAVX2(src, scale, x, full)
		}
	}
	if full < nb {
		srcSlice := unsafe.Slice(src, nb)
		xSlice := unsafe.Slice(x, n)
		acc += awqDotProductScalar(&srcSlice[full], scale, &xSlice[full*2], (nb-full)*2)
	}
	return acc
}

// ---- per-ISA entry points ----------------------------------------------------
//
// These are what awq_amd64.go's CPUID/FAK_AWQ_KERNEL dispatch selects; each names its
// block width and nothing else.

func awqDequantRowAVX2(dst []float32, scale float32, src *byte, n int) {
	awqDequantRowSIMD(dst, scale, src, n, awqBlockBytesAVX2)
}

func awqDequantRowAVX512(dst []float32, scale float32, src *byte, n int) {
	awqDequantRowSIMD(dst, scale, src, n, awqBlockBytesAVX512)
}

func awqDotProductAVX2(src *byte, scale float32, x *float32, n int) float32 {
	return awqDotProductSIMD(src, scale, x, n, awqBlockBytesAVX2)
}

func awqDotProductAVX512(src *byte, scale float32, x *float32, n int) float32 {
	return awqDotProductSIMD(src, scale, x, n, awqBlockBytesAVX512)
}
