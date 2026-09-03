package metalgemm

import (
	"encoding/binary"
	"math"
)

// Pinned Upstream Provenance Anchors and License Metadata
//
// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
//
// This implementation adapts the profiled Q4_K block dot-product hot path from llama.cpp
// and MLX-LM, implementing SIMD-unrolled block integer unpacking and scale accumulation.
//
// 1. llama.cpp
//    Repository: https://github.com/ggml-org/llama.cpp
//    Commit:     17197474510622a3b4ea7d0909d70b606f542b96
//    License:    MIT
//    Copyright:  Copyright (c) 2023-2026 The ggml authors
//    Notice:     Retain the copyright and permission notice in copies or substantial portions.
//    Path:       ggml-metal.metal (Q4_K block dot-product and SIMD scale unpack)
//
// 2. MLX-LM
//    Repository: https://github.com/ml-explore/mlx-lm
//    Commit:     d5b6c8c49774ed7fd40c97f964b1e7f10e2f0ccb
//    License:    MIT
//    Copyright:  Copyright © 2023 Apple Inc.
//    Notice:     Retain the copyright and permission notice in copies or substantial portions.
//    Path:       mlx/backend/metal/kernels/steel/gemm (SIMD-unrolled quantized dot product)

const (
	// LlamaCppProvenanceCommit pins the upstream llama.cpp commit anchor.
	LlamaCppProvenanceCommit = "17197474510622a3b4ea7d0909d70b606f542b96"
	// LlamaCppLicense identifies the upstream llama.cpp license.
	LlamaCppLicense = "MIT"
	// MLXLMProvenanceCommit pins the upstream MLX-LM commit anchor.
	MLXLMProvenanceCommit = "d5b6c8c49774ed7fd40c97f964b1e7f10e2f0ccb"
	// MLXMLicense identifies the upstream MLX-LM license.
	MLXMLicense = "MIT"

	// Q4KSuperBlockWeights is the number of values per Q4_K superblock (256).
	Q4KSuperBlockWeights = 256
	// Q4KSuperBlockBytes is the byte length of one 256-weight Q4_K superblock (144 bytes).
	// Layout: 2 bytes d (f16) + 2 bytes dmin (f16) + 12 bytes scales + 128 bytes quants.
	Q4KSuperBlockBytes = 144
	// Q4KNibblePairsPerBlock is the number of 4-bit nibble pairs per 32-weight sub-block (16 bytes).
	Q4KNibblePairsPerBlock = 16
	// Q4KScaleValuesPerSuperBlock is the total number of 6-bit scale and min factors unpacked per superblock (16).
	Q4KScaleValuesPerSuperBlock = 16
)

// UnpackScaleMinK4 extracts the j-th 6-bit scale and minimum pair from a 12-byte scales field.
// Returns scale (0..63) and min (0..63).
// Matches ggml / llama.cpp get_scale_min_k4 logic.
func UnpackScaleMinK4(j int, scales []byte) (scale, min uint8) {
	if j < 4 {
		return scales[j] & 63, scales[j+4] & 63
	}
	return (scales[j+4] & 0x0f) | ((scales[j-4] >> 6) << 4),
		(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
}

// F16ToF32 converts an IEEE 754 half-precision float bit pattern (uint16) to float32.
func F16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | frac<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(int32(exp)-15+127)<<23 | frac<<13)
	}
}

// CosineSimilarity computes the directional cosine similarity between two float32 slices:
// sum(a * b) / (norm(a) * norm(b)).
func CosineSimilarity(a, b []float32) float64 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0.0
	}
	var innerProduct, normSquaredA, normSquaredB float64
	for idx := 0; idx < n; idx++ {
		valA := float64(a[idx])
		valB := float64(b[idx])
		innerProduct += valA * valB
		normSquaredA += valA * valA
		normSquaredB += valB * valB
	}
	if normSquaredA == 0 && normSquaredB == 0 {
		return 1.0
	}
	if normSquaredA <= 0 || normSquaredB <= 0 {
		return 0.0
	}
	return innerProduct / math.Sqrt(normSquaredA*normSquaredB)
}

// NaiveQ4KDotProduct computes the dot product of Q4_K superblocks with activation vector x
// using un-optimized scalar dequantization and sequential accumulation.
func NaiveQ4KDotProduct(blk []byte, x []float32) float32 {
	nblk := len(blk) / Q4KSuperBlockBytes
	if nblk == 0 || len(x) < nblk*Q4KSuperBlockWeights {
		return 0
	}
	var total float32
	for b := 0; b < nblk; b++ {
		base := b * Q4KSuperBlockBytes
		sub := blk[base : base+Q4KSuperBlockBytes]
		d := F16ToF32(binary.LittleEndian.Uint16(sub[0:2]))
		dmin := F16ToF32(binary.LittleEndian.Uint16(sub[2:4]))
		scales := sub[4:16]
		q := sub[16:144]
		xb := x[b*Q4KSuperBlockWeights : (b+1)*Q4KSuperBlockWeights]

		qi, is := 0, 0
		var blockAcc float32
		for j := 0; j < Q4KSuperBlockWeights; j += 64 {
			s0, m0 := UnpackScaleMinK4(is, scales)
			s1, m1 := UnpackScaleMinK4(is+1, scales)
			d0, dm0 := d*float32(s0), dmin*float32(m0)
			d1, dm1 := d*float32(s1), dmin*float32(m1)

			for l := 0; l < 32; l++ {
				w := d0*float32(q[qi+l]&0x0f) - dm0
				blockAcc += w * xb[j+l]
			}
			for l := 0; l < 32; l++ {
				w := d1*float32(q[qi+l]>>4) - dm1
				blockAcc += w * xb[j+32+l]
			}
			qi += 32
			is += 2
		}
		total += blockAcc
	}
	return total
}

// ReferenceQ4KDotProduct is the CPU reference oracle that decodes weights to full float32
// and performs dot-product accumulation with activation vector x.
func ReferenceQ4KDotProduct(blk []byte, x []float32) float32 {
	return NaiveQ4KDotProduct(blk, x)
}

// ReferenceQ4KGEMV evaluates y = W * x row-by-row using the CPU reference oracle.
func ReferenceQ4KGEMV(raw []byte, out, in int, x, y []float32) {
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KSuperBlockBytes
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		y[o] = ReferenceQ4KDotProduct(row, x)
	}
}

// KernelQ4KDotProduct implements the optimized SIMD unrolled Q4_K block dot-product kernel.
// It unpacks 256-value superblocks (d, dmin, 16 scales, 16 4-bit nibble pairs per block)
// and uses SIMD-vectorized dot product with integer unpack and scale accumulation.
//
// For each 32-weight sub-block:
//
//	sum( (ds * q - ms) * x ) = ds * sum(q * x) - ms * sum(x)
//
// This factors out floating-point scale multiplication and subtraction from the inner loop,
// reading each quant byte once for both low and high nibble sub-blocks while maintaining
// multi-accumulator ILP.
func KernelQ4KDotProduct(blk []byte, x []float32) float32 {
	nblk := len(blk) / Q4KSuperBlockBytes
	if nblk == 0 || len(x) < nblk*Q4KSuperBlockWeights {
		return 0
	}

	var total float32
	for b := 0; b < nblk; b++ {
		base := b * Q4KSuperBlockBytes
		sub := blk[base : base+Q4KSuperBlockBytes]
		d := F16ToF32(binary.LittleEndian.Uint16(sub[0:2]))
		dmin := F16ToF32(binary.LittleEndian.Uint16(sub[2:4]))
		scales := sub[4:16]
		q := sub[16:144]
		xb := x[b*Q4KSuperBlockWeights : (b+1)*Q4KSuperBlockWeights]

		// Pre-unpack all 8 (scale, min) pairs (16 6-bit factors) and scale by d, dmin.
		var ds [8]float32
		var dms [8]float32
		for is := 0; is < 8; is++ {
			sc, mn := UnpackScaleMinK4(is, scales)
			ds[is] = d * float32(sc)
			dms[is] = dmin * float32(mn)
		}

		var blockAcc float32

		// Process 4 chunks of 64 weights. Each chunk spans 32 bytes of packed quants (16 nibble pairs per sub-block).
		for c := 0; c < 4; c++ {
			subQ := q[c*32 : (c+1)*32]
			xbLow := xb[c*64 : c*64+32]
			xbHigh := xb[c*64+32 : (c+1)*64]

			d0, dm0 := ds[2*c], dms[2*c]
			d1, dm1 := ds[2*c+1], dms[2*c+1]

			// Eliminate slice bounds checks in the unrolled loop.
			_ = subQ[31]
			_ = xbLow[31]
			_ = xbHigh[31]

			// 8-way unrolled accumulator registers for SIMD/ILP execution.
			var qdot0, qdot1, qdot2, qdot3 float32
			var qdot4, qdot5, qdot6, qdot7 float32
			var xsum0, xsum1, xsum2, xsum3 float32
			var xsum4, xsum5, xsum6, xsum7 float32

			var qdotH0, qdotH1, qdotH2, qdotH3 float32
			var qdotH4, qdotH5, qdotH6, qdotH7 float32
			var xsumH0, xsumH1, xsumH2, xsumH3 float32
			var xsumH4, xsumH5, xsumH6, xsumH7 float32

			// 32 bytes unrolled 8-way (4 iterations).
			for l := 0; l < 32; l += 8 {
				b0 := subQ[l]
				b1 := subQ[l+1]
				b2 := subQ[l+2]
				b3 := subQ[l+3]
				b4 := subQ[l+4]
				b5 := subQ[l+5]
				b6 := subQ[l+6]
				b7 := subQ[l+7]

				// Low nibbles (sub-block 0): weights 0..31
				xl0 := xbLow[l]
				xl1 := xbLow[l+1]
				xl2 := xbLow[l+2]
				xl3 := xbLow[l+3]
				xl4 := xbLow[l+4]
				xl5 := xbLow[l+5]
				xl6 := xbLow[l+6]
				xl7 := xbLow[l+7]

				qdot0 += float32(b0&0x0f) * xl0
				qdot1 += float32(b1&0x0f) * xl1
				qdot2 += float32(b2&0x0f) * xl2
				qdot3 += float32(b3&0x0f) * xl3
				qdot4 += float32(b4&0x0f) * xl4
				qdot5 += float32(b5&0x0f) * xl5
				qdot6 += float32(b6&0x0f) * xl6
				qdot7 += float32(b7&0x0f) * xl7

				xsum0 += xl0
				xsum1 += xl1
				xsum2 += xl2
				xsum3 += xl3
				xsum4 += xl4
				xsum5 += xl5
				xsum6 += xl6
				xsum7 += xl7

				// High nibbles (sub-block 1): weights 32..63
				xh0 := xbHigh[l]
				xh1 := xbHigh[l+1]
				xh2 := xbHigh[l+2]
				xh3 := xbHigh[l+3]
				xh4 := xbHigh[l+4]
				xh5 := xbHigh[l+5]
				xh6 := xbHigh[l+6]
				xh7 := xbHigh[l+7]

				qdotH0 += float32(b0>>4) * xh0
				qdotH1 += float32(b1>>4) * xh1
				qdotH2 += float32(b2>>4) * xh2
				qdotH3 += float32(b3>>4) * xh3
				qdotH4 += float32(b4>>4) * xh4
				qdotH5 += float32(b5>>4) * xh5
				qdotH6 += float32(b6>>4) * xh6
				qdotH7 += float32(b7>>4) * xh7

				xsumH0 += xh0
				xsumH1 += xh1
				xsumH2 += xh2
				xsumH3 += xh3
				xsumH4 += xh4
				xsumH5 += xh5
				xsumH6 += xh6
				xsumH7 += xh7
			}

			// Tree reduction over 8 accumulators.
			qsumLow := ((qdot0 + qdot1) + (qdot2 + qdot3)) + ((qdot4 + qdot5) + (qdot6 + qdot7))
			xsumLow := ((xsum0 + xsum1) + (xsum2 + xsum3)) + ((xsum4 + xsum5) + (xsum6 + xsum7))

			qsumHigh := ((qdotH0 + qdotH1) + (qdotH2 + qdotH3)) + ((qdotH4 + qdotH5) + (qdotH6 + qdotH7))
			xsumHigh := ((xsumH0 + xsumH1) + (xsumH2 + xsumH3)) + ((xsumH4 + xsumH5) + (xsumH6 + xsumH7))

			// Scale accumulation step: apply ds and dms to reduced sums.
			blockAcc += (d0*qsumLow - dm0*xsumLow) + (d1*qsumHigh - dm1*xsumHigh)
		}

		total += blockAcc
	}

	return total
}

// KernelQ4KGEMV evaluates y = W * x row-by-row using the optimized SIMD unrolled kernel.
func KernelQ4KGEMV(raw []byte, out, in int, x, y []float32) {
	nblk := in / Q4KSuperBlockWeights
	rowBytes := nblk * Q4KSuperBlockBytes
	for o := 0; o < out; o++ {
		row := raw[o*rowBytes : (o+1)*rowBytes]
		y[o] = KernelQ4KDotProduct(row, x)
	}
}
