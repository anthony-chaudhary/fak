package model

// awq_scalar.go — portable, arch-independent AWQ scalar code shared by every platform.
//
// The AWQ tier ENUM LABELS, the scalar dequant/dot reference kernels (the correctness oracle
// the AVX paths are checked against), and the small test helpers are plain Go — no arch
// intrinsics — so they belong in one untagged file, not behind //go:build amd64. They
// previously lived in the amd64-only awq_amd64_asm.go, which broke the internal/model TEST
// build on non-amd64 targets (arm64/darwin — the Apple-silicon fleet): awq_test.go is
// untagged and references awqDotProductScalar / awqDequantRowScalar / cosineSimilarity /
// awqTierAVX2 / awqTierAVX512, so on arm64 the whole package test binary failed to compile
// ("undefined: …"), taking every internal/model test down with it. Which tier is ACTIVE is
// still chosen per-arch (resolveAWQTier on amd64, awqTierScalar on every other arch); only
// the labels + the portable reference code are shared here.

import (
	"math"
	"unsafe"
)

// AWQ kernel tier labels. The active tier (var awqTier) is set per-arch: resolveAWQTier()
// in awq_amd64.go, awqTierScalar in awq_noamd64.go.
const (
	awqTierScalar = iota
	awqTierAVX2
	awqTierAVX512
)

const zeroPoint = 8 // symmetric 4-bit zero-point

// awqDequantRowScalar is the scalar reference implementation for AWQ dequantization.
func awqDequantRowScalar(dst []float32, scale float32, src *byte, n int) {
	// Convert byte pointer to slice for safe access (within the function)
	// This is safe because n/2 is the length of the packed data
	srcSlice := unsafe.Slice(src, (n+1)/2)

	for i := 0; i < n/2; i++ {
		b := srcSlice[i]
		lo, hi := unpack4bit(b)
		dst[i*2] = scale * float32(int16(lo)-zeroPoint)
		dst[i*2+1] = scale * float32(int16(hi)-zeroPoint)
	}
}

// awqDotProductScalar is the scalar reference implementation for AWQ dot product.
// Computes dot(scale * (code - 8), x) directly without dequantizing the full row.
func awqDotProductScalar(src *byte, scale float32, x *float32, n int) float32 {
	srcSlice := unsafe.Slice(src, (n+1)/2)
	xSlice := unsafe.Slice(x, n)

	var acc float32
	for i := 0; i < n/2; i++ {
		b := srcSlice[i]
		lo, hi := unpack4bit(b)
		acc += scale * float32(int16(lo)-zeroPoint) * xSlice[i*2]
		acc += scale * float32(int16(hi)-zeroPoint) * xSlice[i*2+1]
	}
	return acc
}

// ---- Test helpers for verifying assembly implementations ---------------------

// ---- Cosine similarity for oracle testing -------------------------------------

// cosineSimilarity computes cosine similarity between two float32 vectors.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float32
	for i := range a {
		ai, bi := a[i], b[i]
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
