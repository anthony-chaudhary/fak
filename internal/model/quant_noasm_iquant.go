//go:build !amd64

package model

// iq3xxsDequantSuperBlockArch is the non-amd64 fallback: there is no SIMD IQ3_XXS decode
// kernel here, so it returns false and kQuantDequantSuperBlock uses the scalar
// iq3xxsDequantSuperBlock reference (quant_iquant.go). The arm64 decode path keeps its own
// kernels for the k-quants; IQ3_XXS stays scalar off amd64 until a NEON port lands.
func iq3xxsDequantSuperBlockArch(dst []float32, blk []byte) bool { return false }
