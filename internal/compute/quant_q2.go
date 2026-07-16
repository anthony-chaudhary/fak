package compute

// quant_q2.go — the packed-ternary Q2_0 format (issue #4872). A ternary weight is
// t ∈ {-1, 0, +1} times a per-block f32 scale; it is the "runs on a laptop/phone GPU"
// density point BitNet / prism-ml ship, because the GEMM never expands the weight to
// FP16 — it multiply-accumulates the activation by a signed indicator (a select/add,
// not a multiply) and folds one f32 scale per block at the end.
//
// STORAGE (mirrors Q8_0's codes-plus-scale-side-channel so the device upload/GEMM
// machinery reuses the Q8 shape): 2 bits per weight, 4 weights per byte, LSB-first.
// The 2-bit field u ∈ {0,1,2} decodes to t = u-1; u=3 is unused. A [out,in] weight is
// out*in/4 packed code bytes plus out*(in/32) f32 block scales. That is 0.25 byte/elem
// of codes — 4× narrower than Q8_0's int8 codes, 16× narrower than f32.
//
// This file is pure Go (no build tag): it is the CPU reference for the ternary GEMM the
// CUDA k_q2_0_gemm kernel is held numerically equivalent to, so it compiles and its
// witness runs on every host, GPU or not.

// q2Block is the ternary block size — 32 weights share one f32 scale (Q8_0's grouping).
const q2Block = 32

// packQ2 packs ternary codes (one int8 in {-1,0,+1} per weight) into 2-bit fields, 4 per
// byte, LSB-first: weight i lands in byte i/4 at bit (i%4)*2, as u = t+1 ∈ {0,1,2}.
// Returns ceil(len/4) bytes.
func packQ2(tern []int8) []byte {
	out := make([]byte, (len(tern)+3)/4)
	for i, t := range tern {
		u := byte(t+1) & 0x3
		out[i>>2] |= u << uint((i&3)*2)
	}
	return out
}

// q2Code returns the ternary value (-1|0|+1) of weight index i in a packed row.
func q2Code(packed []byte, i int) int32 {
	return int32((packed[i>>2]>>uint((i&3)*2))&0x3) - 1
}

// bytesAsI8 reinterprets packed code bytes as the int8 slice a hostBuf carries (the
// two's-complement bit pattern is preserved), the inverse of i8AsBytes.
func bytesAsI8(b []byte) []int8 {
	s := make([]int8, len(b))
	for i, v := range b {
		s[i] = int8(v)
	}
	return s
}

// QuantizeQ2 builds a packed Q2_0 weight Tensor from an f32 [out,in] matrix. Each block of
// `block` weights takes a symmetric scale d = amax/1 (the max |w| in the block, so |t|≤1
// covers the range) and a ternary code t = round(w/d) clamped to {-1,0,+1} — the standard
// absmax ternary quantizer. The result's MatMul is Approx vs the f32 reference (the block
// shares one scale and every weight rounds to three levels), exactly the ternary lane.
func QuantizeQ2(be Backend, shape []int, w []float32, block int) Tensor {
	out, in := shape[0], shape[1]
	nblk := in / block
	tern := make([]int8, out*in)
	scale := make([]float32, out*nblk)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := o*in + b*block
			var amax float32
			for i := 0; i < block; i++ {
				if a := absf(w[base+i]); a > amax {
					amax = a
				}
			}
			scale[o*nblk+b] = amax
			if amax == 0 {
				continue
			}
			inv := 1.0 / amax
			for i := 0; i < block; i++ {
				tern[base+i] = ternRound(w[base+i] * inv)
			}
		}
	}
	return NewQ2(be, shape, packQ2(tern), scale, block)
}

// ternRound rounds a scaled weight in [-1,1]-ish to the nearest ternary level {-1,0,+1}
// (thresholds at ±0.5, half away from zero). Quantization is host-only; the device reads
// prepacked ternary codes, so there is no device counterpart to reproduce.
func ternRound(x float32) int8 {
	if x >= 0.5 {
		return 1
	}
	if x <= -0.5 {
		return -1
	}
	return 0
}

// NewQ2 wraps prepacked Q2_0 ternary codes (2-bit, 4/byte) + per-block(=block) f32 scales as
// a Tensor. shape [out,in]; len(packed)==out*in/4 (rounded up); len(scale)==out*(in/block).
func NewQ2(be Backend, shape []int, packed []byte, scale []float32, block int) Tensor {
	q := &QuantSpec{Block: block, Axis: 2, Bits: 2, Symmetric: true, Scale: scale}
	return makeTensor(be, Q2_0, RowMajor, shape, q, &hostBuf{i8: bytesAsI8(packed)})
}

// q2RowDot is the ternary GEMV inner product for one output row: y[o] = Σ_b scale[b] ·
// Σ_{i in block} t(o, b·block+i)·x[i]. The signed indicator dot accumulates as f32 and the
// single block scale folds in once per block — the exact arithmetic k_q2_0_gemm reproduces
// on device (only the reduction order differs, which is what makes the device lane Approx).
func q2RowDot(packedRow []byte, scale, x []float32, block int) float32 {
	nblk := len(x) / block
	var acc float32
	for b := 0; b < nblk; b++ {
		var s float32
		off := b * block
		for i := 0; i < block; i++ {
			s += float32(q2Code(packedRow, off+i)) * x[off+i]
		}
		acc += s * scale[b]
	}
	return acc
}
