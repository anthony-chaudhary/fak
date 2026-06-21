//go:build amd64

package model

// awq_amd64_asm.go — Go implementations of AWQ SIMD kernels.
// These serve as the base implementations that can be optimized with assembly.
// For now, we provide portable Go implementations that match the assembly signatures.
// The scalar reference kernels + zeroPoint they delegate to are shared (awq_scalar.go).

// awqDequantRowAVX512 dequantizes one AWQ row using portable Go (placeholder for assembly).
func awqDequantRowAVX512(dst []float32, scale float32, src *byte, n int) {
	awqDequantRowScalar(dst, scale, src, n)
}

// awqDotProductAVX512 computes dot product using portable Go (placeholder for assembly).
func awqDotProductAVX512(src *byte, scale float32, x *float32, n int) float32 {
	return awqDotProductScalar(src, scale, x, n)
}

// awqDequantRowAVX2 dequantizes one AWQ row using portable Go (placeholder for assembly).
func awqDequantRowAVX2(dst []float32, scale float32, src *byte, n int) {
	awqDequantRowScalar(dst, scale, src, n)
}

// awqDotProductAVX2 computes dot product using portable Go (placeholder for assembly).
func awqDotProductAVX2(src *byte, scale float32, x *float32, n int) float32 {
	return awqDotProductScalar(src, scale, x, n)
}
