// q2_0_ref_test.go — the CPU reference ternary (Q2_0) quantizer + GEMV shared by BOTH builds'
// witnesses. It carries NO build tag on purpose: the Apple-Silicon parity test (q2_0_test.go,
// darwin && arm64 && cgo) compares the Metal kernel against this reference, and the stub-build
// witness (q2_0_witness_test.go, the exact negation) pins this reference's own math obligations.
// One reference, one convention, both builds — so "Metal matches CPU-ref" and "the CPU-ref is
// correct" are asserted against the same code rather than two drifting copies.
//
// It lives in a _test.go file because package metalgemm never quantizes on the hot path: like
// UploadQ8/UploadQ4K, UploadQ2_0 takes an ALREADY-packed payload (the model side owns quantization
// — internal/model.quantizeQ2). This is test-only scaffolding to generate parity inputs and to
// state, executably, what the Metal kernel must reproduce.
//
// Format (internal/model/quant_q2.go's g32 form, verbatim). A weight is [out, in] row-major with
// in = nblk*32. Each 32-wide block is one f32 scale d = amax followed by 8 code bytes holding 32
// signed 2-bit codes, four per byte, LOW CODE FIRST. Code c (0..3) dequantizes to d*(c-2).

package metalgemm

import "math"

// q2_0QuantizeBlock stores the 32 weights of src as one (d, 8 packed bytes) block at dst and
// returns the per-block scale. d = amax, so the largest magnitude maps to code ±1; a zero block
// stays zero. Codes are round(w/d)+2 clamped to [0,3] (signed range [-2,1]). This is
// internal/model.quantizeQ2Block's convention exactly.
func q2_0QuantizeBlock(dst []byte, src []float32) float32 {
	var amax float32
	for _, v := range src {
		a := v
		if a < 0 {
			a = -a
		}
		if a > amax {
			amax = a
		}
	}
	if amax == 0 {
		for i := 0; i < Q2_0BlockBytes; i++ {
			dst[i] = 0
		}
		return 0
	}
	d := amax // = amax/(half-1), half = 1<<(2-1) = 2
	inv := 1.0 / float64(d)
	for i := 0; i < Q2_0BlockBytes; i++ {
		var bits byte
		for j := 0; j < 4; j++ {
			// Round-to-nearest: int() truncates toward zero and would bias every code by up to
			// half a quantum. Runs once per block in a test, so a float64 round is fine.
			c := int(math.Round(float64(src[4*i+j])*inv)) + 2
			if c < 0 {
				c = 0
			} else if c > 3 {
				c = 3
			}
			bits |= byte(c) << (2 * j)
		}
		dst[i] = bits
	}
	return d
}

// q2_0DequantBlock writes the 32 weights of one (d, 8 packed bytes) block into dst. It is the exact
// inverse of q2_0QuantizeBlock's packing, so the round trip is bounded only by 2-bit rounding error.
func q2_0DequantBlock(dst []float32, d float32, q []byte) {
	for i := 0; i < Q2_0BlockBytes; i++ {
		b := q[i]
		dst[4*i+0] = d * float32(int(b&0x3)-2)
		dst[4*i+1] = d * float32(int((b>>2)&0x3)-2)
		dst[4*i+2] = d * float32(int((b>>4)&0x3)-2)
		dst[4*i+3] = d * float32(int((b>>6)&0x3)-2)
	}
}

// q2_0Quantize builds the packed ternary payload (codes, scales) for an [out, in] f32 matrix. in
// must be a multiple of Q2_0BlockWeights. The returned pair is exactly what UploadQ2_0 accepts.
func q2_0Quantize(w []float32, out, in int) (codes []byte, scales []float32) {
	if in%Q2_0BlockWeights != 0 {
		panic("metalgemm: ternary reduction dim not a multiple of 32")
	}
	nblk := in / Q2_0BlockWeights
	codes = make([]byte, out*nblk*Q2_0BlockBytes)
	scales = make([]float32, out*nblk)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			off := o*in + b*Q2_0BlockWeights
			blkIdx := o*nblk + b
			scales[blkIdx] = q2_0QuantizeBlock(codes[blkIdx*Q2_0BlockBytes:], w[off:off+Q2_0BlockWeights])
		}
	}
	return codes, scales
}

// q2_0Dequantize reconstructs the dense [out, in] f32 matrix from a packed ternary payload — the
// whole-tensor inverse of q2_0Quantize. The dense product against this matrix is the ground truth
// the GEMV (CPU-ref and Metal alike) must reproduce.
func q2_0Dequantize(codes []byte, scales []float32, out, in int) []float32 {
	nblk := in / Q2_0BlockWeights
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blkIdx := o*nblk + b
			q2_0DequantBlock(w[o*in+b*Q2_0BlockWeights:], scales[blkIdx], codes[blkIdx*Q2_0BlockBytes:])
		}
	}
	return w
}

// q2_0RefGEMV is the CPU reference ternary decode GEMV: y[o] = dot(row o, x), computed by
// dequantizing each 32-wide block to d*(c-2) and dotting it against the f32 activation — the same
// shape as internal/model.q2MatRows. This is the "CPU-ref ternary GEMM" the Metal kernel is held
// against. The accumulation is a plain sequential float sum; the Metal kernel factors d out of the
// block and reduces via simd_sum, so agreement is within-tolerance rather than bit-exact.
func q2_0RefGEMV(codes []byte, scales []float32, x []float32, out, in int) []float32 {
	nblk := in / Q2_0BlockWeights
	y := make([]float32, out)
	blk := make([]float32, Q2_0BlockWeights)
	for o := 0; o < out; o++ {
		var s float32
		for b := 0; b < nblk; b++ {
			blkIdx := o*nblk + b
			q2_0DequantBlock(blk, scales[blkIdx], codes[blkIdx*Q2_0BlockBytes:])
			xs := x[b*Q2_0BlockWeights:]
			for i := 0; i < Q2_0BlockWeights; i++ {
				s += blk[i] * xs[i]
			}
		}
		y[o] = s
	}
	return y
}

// q2_0RefGEMVBatch is the CPU reference for GEMVBatch: n independent reference GEMVs of the same
// weight over n contiguous activation rows, concatenated into one n*out result.
func q2_0RefGEMVBatch(codes []byte, scales []float32, Xcat []float32, n, out, in int) []float32 {
	ycat := make([]float32, n*out)
	for i := 0; i < n; i++ {
		y := q2_0RefGEMV(codes, scales, Xcat[i*in:(i+1)*in], out, in)
		copy(ycat[i*out:], y)
	}
	return ycat
}
