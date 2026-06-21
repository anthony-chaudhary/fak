//go:build !cuda || !awqcuda

package model

// awq_nocuda.go — CPU fallback for AWQ ops. Compiled whenever the cgo AWQ-CUDA path
// (awq_cuda.go, `cuda && awqcuda`) is NOT in the build — i.e. the default build AND a
// plain `-tags cuda` build (so internal/model stays cgo-free there and its SIMD .s
// files compile; see awq_cuda.go). AWQ operations run on CPU only here.

// awqMatRowsCUDA is the CPU fallback when CUDA is not compiled in.
func awqMatRowsCUDA(qt *awqTensor, x []float32) []float32 {
	return awqMatRows(qt, x)
}

// awqGemmCUDA is the CPU fallback when CUDA is not compiled in.
func awqGemmCUDA(qt *awqTensor, X []float32, P int) []float32 {
	return awqGemm(qt, X, P)
}
