//go:build amd64

package model

import "os"

// awq_amd64.go — AVX2/AVX-512 optimized kernels for AWQ 4-bit dequantization and matmul.
// This module provides SIMD acceleration for AWQ operations on x86-64 processors.
// Assembly implementations are in awq_amd64_asm.go.

// ---- AWQ tier detection -------------------------------------------------------
// The tier labels (awqTierScalar/AVX2/AVX512) are shared in awq_scalar.go; this file only
// resolves which one is active on amd64.

var awqTier = resolveAWQTier()

func resolveAWQTier() int {
	want := awqTierAVX512
	switch os.Getenv("FAK_AWQ_KERNEL") {
	case "scalar":
		want = awqTierScalar
	case "avx2":
		want = awqTierAVX2
	case "avx512":
		want = awqTierAVX512
	}
	if want >= awqTierAVX512 && detectAVX512() {
		return awqTierAVX512
	}
	if want >= awqTierAVX2 && detectAVX2() {
		return awqTierAVX2
	}
	return awqTierScalar
}

// ---- AWQ kernel dispatch -----------------------------------------------------

// awxMatRowsRangeAVX is the AVX-accelerated version of awxMatRowsRange.
func awxMatRowsRangeAVX(qt *awqTensor, x, y []float32, lo, hi int) {
	rowBytes := qt.awqRowBytes()

	if awqTier == awqTierAVX512 {
		for o := lo; o < hi; o++ {
			row := qt.raw[o*rowBytes:]
			scale := qt.scales[o]
			y[o] = awqDotProductAVX512(&row[0], scale, &x[0], qt.in)
		}
		return
	}

	if awqTier == awqTierAVX2 {
		for o := lo; o < hi; o++ {
			row := qt.raw[o*rowBytes:]
			scale := qt.scales[o]
			y[o] = awqDotProductAVX2(&row[0], scale, &x[0], qt.in)
		}
		return
	}

	// Scalar fallback (already implemented in awxMatRowsRange)
	awxMatRowsRange(qt, x, y, lo, hi)
}

// awxGemmRangeAVX is the AVX-accelerated version of awxGemmRange.
func awxGemmRangeAVX(qt *awqTensor, X []float32, P int, Y []float32, lo, hi int) {
	rowBytes := qt.awqRowBytes()
	in := qt.in

	// Temporary buffer for dequantized row (reused)
	buf := make([]float32, in)

	if awqTier == awqTierAVX512 || awqTier == awqTierAVX2 {
		acc := make([]float32, P)
		for o := lo; o < hi; o++ {
			row := qt.raw[o*rowBytes:]
			scale := qt.scales[o]

			// Dequantize row
			if awqTier == awqTierAVX512 {
				awqDequantRowAVX512(buf, scale, &row[0], in)
			} else {
				awqDequantRowAVX2(buf, scale, &row[0], in)
			}

			// Compute dot products for all P tokens
			for t := 0; t < P; t++ {
				acc[t] = 0
				xs := X[t*in:]
				var sum float32
				for i := 0; i < in; i++ {
					sum += buf[i] * xs[i]
				}
				acc[t] = sum
			}

			for t := 0; t < P; t++ {
				Y[t*qt.out+o] = acc[t]
			}
		}
		return
	}

	// Scalar fallback
	awxGemmRange(qt, X, P, Y, lo, hi)
}
