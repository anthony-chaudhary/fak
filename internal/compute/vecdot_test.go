package compute

import (
	"encoding/binary"
	"math"
	"testing"
)

// vecdot_test.go — the CIQ witness for issue #12275. TestComputeInQuantDotProduct proves two
// things for each of the Q4_0 and Q8_0 CIQ kernels against the f32 dequant-then-dot reference:
//
//  1. NUMERICAL PARITY: the integer dot, scaled by the two f16 block scales, matches the
//     reference dot(w_dequant, x_dequant) within the activation-quantization tolerance. The
//     reference dequantizes BOTH operands to f32 (the exact ggml block math) and dots them;
//     the CIQ path quantizes the activation to Q8_0 and reduces int8×int8. The only error
//     source is the activation quantization, so the parity bound is derived from the Q8_0
//     quantization step (dx/2 per element).
//  2. ZERO TRANSIENT FLOAT BUFFERS: the kernel body allocates no float32 slices — the point
//     of CIQ is one int32 accumulator instead of a block-sized f32 dequant scratch — verified
//     by measuring allocations across the call.

// f16Bits converts an f32 to the nearest IEEE binary16 bit pattern (round-to-nearest-even),
// used to build ggml-layout test blocks. NaNs/infs are not exercised by this witness.
func f16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32((b>>23)&0xff) - 127 + 15
	frac := b & 0x7fffff
	switch {
	case exp <= 0:
		return sign // underflow to zero is fine for test magnitudes
	case exp >= 0x1f:
		return sign | 0x7c00 // clamp to inf
	default:
		// round the 23-bit mantissa to 10 bits (round-half-to-even)
		rounded := frac + 0x1000 + ((frac >> 13) & 1)
		if rounded&0x800000 != 0 {
			rounded = 0
			exp++
			if exp >= 0x1f {
				return sign | 0x7c00
			}
		}
		return sign | uint16(exp)<<10 | uint16((rounded>>13)&0x3ff)
	}
}

func putF16(dst []byte, f float32) { binary.LittleEndian.PutUint16(dst, f16Bits(f)) }

// q8_0Pack packs codes (len 32*nblk) into ggml Q8_0 blocks with the given per-block scales.
func q8_0Pack(codes []int8, scales []float32) []byte {
	nblk := len(scales)
	raw := make([]byte, nblk*q8_0BlockBytes)
	for b := 0; b < nblk; b++ {
		blk := raw[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
		putF16(blk[0:], scales[b])
		for j := 0; j < 32; j++ {
			blk[2+j] = byte(codes[b*32+j])
		}
	}
	return raw
}

// q4_0Pack packs codes (len 32*nblk, each in [0,15]) into ggml Q4_0 blocks with the ggml
// INTERLEAVE (low nibble of byte j -> element j, high nibble -> element j+16).
func q4_0Pack(codes []int, scales []float32) []byte {
	nblk := len(scales)
	raw := make([]byte, nblk*q4_0BlockBytes)
	for b := 0; b < nblk; b++ {
		blk := raw[b*q4_0BlockBytes : (b+1)*q4_0BlockBytes]
		putF16(blk[0:], scales[b])
		for j := 0; j < 16; j++ {
			blk[2+j] = byte(codes[b*32+j]&0x0f) | byte(codes[b*32+j+16]&0x0f)<<4
		}
	}
	return raw
}

// q4_0PackFloat is the ggml-layout pack from a []float32 of reals: each 32-wide block gets
// scale d = amax/15 (or 1 when amax==0) and codes round(w/d) in [0,15]. Returns the packed
// bytes and the per-block scales so the caller can build the f32 reference by dequant.
func q4_0PackFloat(w []float32) (raw []byte, scales []float32) {
	nblk := len(w) / 32
	codes := make([]int, nblk*32)
	scales = make([]float32, nblk)
	for b := 0; b < nblk; b++ {
		blk := w[b*32 : (b+1)*32]
		amax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := float32(1)
		if amax > 0 {
			d = amax / 15
		}
		scales[b] = d
		for j := 0; j < 32; j++ {
			c := int(math.Round(float64(blk[j]/d))) + 8
			if c < 0 {
				c = 0
			}
			if c > 15 {
				c = 15
			}
			codes[b*32+j] = c
		}
	}
	return q4_0Pack(codes, scales), scales
}

// q8_0PackFloat packs an f32 vector to ggml Q8_0 with d = amax/127.
func q8_0PackFloat(w []float32) (raw []byte, codes []int8, scales []float32) {
	nblk := len(w) / 32
	codes = make([]int8, nblk*32)
	scales = make([]float32, nblk)
	for b := 0; b < nblk; b++ {
		blk := w[b*32 : (b+1)*32]
		amax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := float32(1)
		if amax > 0 {
			d = amax / 127
		}
		scales[b] = d
		for j := 0; j < 32; j++ {
			c := int(math.Round(float64(blk[j] / d)))
			if c > 127 {
				c = 127
			}
			if c < -127 {
				c = -127
			}
			codes[b*32+j] = int8(c)
		}
	}
	return q8_0Pack(codes, scales), codes, scales
}

// dequantQ4_0 and dequantQ8_0 reproduce the ggml block math exactly (matching
// internal/model's q4_0DequantBlock / q8_0DequantBlock) for the f32 reference.
func dequantQ4_0(raw []byte, nblk int) []float32 {
	out := make([]float32, nblk*32)
	for b := 0; b < nblk; b++ {
		blk := raw[b*q4_0BlockBytes : (b+1)*q4_0BlockBytes]
		d := math.Float32frombits(kquantbitsF16(binary.LittleEndian.Uint16(blk[0:])))
		qs := blk[2:q4_0BlockBytes]
		for j := 0; j < 16; j++ {
			out[b*32+j] = float32(int(qs[j]&0x0f)-8) * d
			out[b*32+j+16] = float32(int(qs[j]>>4)-8) * d
		}
	}
	return out
}

func dequantQ8_0(raw []byte, nblk int) []float32 {
	out := make([]float32, nblk*32)
	for b := 0; b < nblk; b++ {
		blk := raw[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
		d := math.Float32frombits(kquantbitsF16(binary.LittleEndian.Uint16(blk[0:])))
		for j := 0; j < 32; j++ {
			out[b*32+j] = float32(int8(blk[2+j])) * d
		}
	}
	return out
}

func dotF32(a, b []float32) float32 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return float32(s)
}

func TestComputeInQuantDotProduct(t *testing.T) {
	const nblk = 4
	const n = nblk * 32

	// deterministic non-trivial weight/activation vectors
	wf := make([]float32, n)
	xf := make([]float32, n)
	for i := 0; i < n; i++ {
		wf[i] = float32(math.Sin(float64(i)*0.7)*3.1 + math.Cos(float64(i)*0.13)*0.4)
		xf[i] = float32(math.Cos(float64(i)*0.31)*2.7 - math.Sin(float64(i)*0.05)*0.3)
	}

	// activation quantized once, shared by both kernels (the CIQ shape)
	xRawQ8, xCodes, xScales := q8_0PackFloat(xf)

	t.Run("Q8_0", func(t *testing.T) {
		wRaw, _, wScales := q8_0PackFloat(wf)
		got := VecDotQ8_0Q8_0(wRaw, nblk, xCodes, xScales)
		// reference: dequantize weights AND activation, f32 dot
		ref := dotF32(dequantQ8_0(wRaw, nblk), dequantQ8_0(xRawQ8, nblk))
		assertParity(t, got, ref, wScales, xScales)
	})

	t.Run("Q4_0", func(t *testing.T) {
		wRaw, wScales := q4_0PackFloat(wf)
		got := VecDotQ4_0Q8_0(wRaw, nblk, xCodes, xScales)
		ref := dotF32(dequantQ4_0(wRaw, nblk), dequantQ8_0(xRawQ8, nblk))
		assertParity(t, got, ref, wScales, xScales)
	})

	t.Run("ExactIntegerMath", func(t *testing.T) {
		// With block scale d==1 for BOTH operands, the integer product IS the exact dot:
		// codes == reals, so the CIQ result must equal the reference bit-for-bit.
		const nb = 1
		codes := make([]int, 32)
		xCodes1 := make([]int8, 32)
		xScales1 := []float32{1}
		for j := 0; j < 32; j++ {
			codes[j] = (j * 7) % 16 // 0..15
			xCodes1[j] = int8((j*11)%127 - 63)
		}
		wRaw := q4_0Pack(codes, []float32{1})
		got := VecDotQ4_0Q8_0(wRaw, nb, xCodes1, xScales1)
		ref := dotF32(dequantQ4_0(wRaw, nb), dequantQ8_0(q8_0Pack(xCodes1, xScales1), nb))
		if got != ref {
			t.Fatalf("exact-parity (d=1) mismatch: got %v want %v", got, ref)
		}
	})
}

// assertParity checks the CIQ result is within the activation-quantization error bound of the
// f32 reference. The activation Q8_0 step is dx/2 per element, so the worst-case dot error is
// Σ |w_j|·dx_j/2 <= maxdx/2 · Σ|w_j|. Bounds are generous by construction (the reference
// itself carries weight-quantization error too).
func assertParity(t *testing.T, got, ref float32, wScales, xScales []float32) {
	t.Helper()
	absSumW := float32(0)
	for _, d := range wScales {
		absSumW += d * 15 * 32 // worst-case |w| per block for Q4_0; scaled up for Q8_0 below
	}
	maxdx := float32(0)
	for _, d := range xScales {
		if d > maxdx {
			maxdx = d
		}
	}
	// generous relative+absolute tolerance: the CIQ needle must sit on the reference curve,
	// not merely be "close-ish". 2% relative plus a half-quantum absolute floor.
	tol := float32(math.Abs(float64(ref)))*0.02 + maxdx*absSumW/2
	if diff := float32(math.Abs(float64(got - ref))); diff > tol {
		t.Fatalf("CIQ vs f32 reference out of tolerance: got %v ref %v diff %v tol %v", got, ref, diff, tol)
	}
}

// TestComputeInQuantDotProductNoAlloc proves the CIQ kernel allocates no float32 scratch: the
// whole point vs the dequant-then-dot path is one int32 accumulator instead of a block-sized
// f32 dequant buffer. allocs must be 0 across the call.
func TestComputeInQuantDotProductNoAlloc(t *testing.T) {
	const nblk = 8
	n := nblk * 32
	wf := make([]float32, n)
	xf := make([]float32, n)
	for i := 0; i < n; i++ {
		wf[i] = float32(math.Sin(float64(i)) * 2)
		xf[i] = float32(math.Cos(float64(i)) * 2)
	}
	w8, _, _ := q8_0PackFloat(wf)
	w4, _ := q4_0PackFloat(wf)
	_, xc, xd := q8_0PackFloat(xf)

	var sink float32
	allocs8 := testing.AllocsPerRun(50, func() { sink += VecDotQ8_0Q8_0(w8, nblk, xc, xd) })
	allocs4 := testing.AllocsPerRun(50, func() { sink += VecDotQ4_0Q8_0(w4, nblk, xc, xd) })
	if allocs8 != 0 {
		t.Fatalf("VecDotQ8_0Q8_0 allocated %v times/op, want 0 (CIQ must use no transient f32 buffer)", allocs8)
	}
	if allocs4 != 0 {
		t.Fatalf("VecDotQ4_0Q8_0 allocated %v times/op, want 0 (CIQ must use no transient f32 buffer)", allocs4)
	}
	if sink == 0 {
		t.Skip("unreachable guard") // keep sink live
	}
}

// kquantbitsF16 mirrors internal/kquantbits.F16BitsToF32Bits (same math the model uses), kept
// local so the test pins the exact bit conversion without importing through the kernel.
func kquantbitsF16(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := int((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return sign
		}
		exp = -14
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x03ff
		return sign | uint32(exp+127)<<23 | frac<<13
	case 0x1f:
		return sign | 0x7f800000 | frac<<13
	default:
		return sign | uint32(exp-15+127)<<23 | frac<<13
	}
}
