//go:build !amd64

package model

// awq_noamd64.go — scalar fallback for AWQ kernels on non-AMD64 platforms.
// On ARM, RISC-V, and other architectures, we use the portable Go implementation.
// The tier labels (awqTierScalar/AVX2/AVX512) are shared in awq_scalar.go; non-amd64 is
// always the scalar tier.

var awqTier = awqTierScalar

// awxMatRowsRangeAVX is the scalar fallback (just calls the portable implementation).
func awxMatRowsRangeAVX(qt *awqTensor, x, y []float32, lo, hi int) {
	awxMatRowsRange(qt, x, y, lo, hi)
}

// awxGemmRangeAVX is the scalar fallback (just calls the portable implementation).
func awxGemmRangeAVX(qt *awqTensor, X []float32, P int, Y []float32, lo, hi int) {
	awxGemmRange(qt, X, P, Y, lo, hi)
}
