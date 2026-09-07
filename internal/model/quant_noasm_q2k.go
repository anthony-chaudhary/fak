//go:build !amd64

package model

// q2kDequantSuperBlockArch is the non-amd64 fallback: there is no SIMD Q2_K decode
// kernel for this architecture yet, so return false to let the caller take the
// scalar q2kDequantSuperBlock reference.
func q2kDequantSuperBlockArch(dst []float32, blk []byte) bool {
	return false
}
