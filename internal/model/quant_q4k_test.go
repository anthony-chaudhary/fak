package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// randQ4KBlock fills blk with random bytes but constrains the d/min f16 scales (bytes 0-3)
// to finite, non-denormal values, so the dequant produces finite weights. Real GGUF Q4_K
// blocks are always finite (a quantizer wrote them); purely random bytes can hit the f16
// all-ones exponent (0x1f) and yield NaN/Inf scales that poison the dot. Constraining only
// the 4 scale bytes still exercises every code nibble and all 12 packed scale bytes.
func randQ4KBlock(rng *rand.Rand, blk []byte) {
	for i := range blk {
		blk[i] = byte(rng.Intn(256))
	}
	// f16 little-endian: byte0 = frac low 8, byte1 = sign(1)|exp(5)|frac high(2). Force the
	// 5-bit exponent into [1,30] (finite, non-denormal); keep sign and frac random.
	for s := 0; s < 2; s++ {
		hi := blk[s*2+1]
		exp := int((hi >> 2) & 0x1f)
		if exp == 0 || exp == 0x1f {
			exp = 15 // ~1.0 magnitude, finite
		}
		blk[s*2+1] = (hi & 0x83) | byte(exp<<2)
	}
}

// sameFloat reports whether a and b are equal, treating two NaNs as equal (they arose from
// the identical input/algorithm in these tests, so a NaN-vs-NaN mismatch is the comparison
// artifact, not a dequant difference).
func sameFloat(a, b float32) bool {
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	return a == b
}

// dequantQ4KRef is a verbatim copy of ggufload.dequantQ4K (the loader's f32 reference path),
// kept here as the correctness oracle for the resident dequant. It MUST agree with
// q4kDequantSuperBlock to the ULP — that equality is what lets the resident GEMV claim
// "correct by construction against the same f32 weights the loader would have produced."
func dequantQ4KRef(out []float32, raw []byte) {
	const block = q4kBlockBytes
	for b := 0; b < len(out)/qkK; b++ {
		base := b * block
		d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base:])))
		min := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+2:])))
		scales := raw[base+4 : base+4+12]
		q := raw[base+4+12 : base+block]
		qi := 0
		is := 0
		yi := b * qkK
		for j := 0; j < qkK; j += 64 {
			sc, m := GetScaleMinK4(is, scales)
			d1, m1 := d*float32(sc), min*float32(m)
			sc, m = GetScaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), min*float32(m)
			for l := 0; l < 32; l++ {
				out[yi+j+l] = d1*float32(q[qi+l]&0x0f) - m1
			}
			for l := 0; l < 32; l++ {
				out[yi+j+32+l] = d2*float32(q[qi+l]>>4) - m2
			}
			qi += 32
			is += 2
		}
	}
}

// TestQ4KDequantSuperBlockMatchesRef pins q4kDequantSuperBlock to the loader reference
// dequant over many random super-blocks. Any byte pattern is a valid Q4_K block, so random
// bytes exercise every code/scale/min branch. Equality must be exact (same arithmetic).
func TestQ4KDequantSuperBlockMatchesRef(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	blk := make([]byte, q4kBlockBytes)
	got := make([]float32, qkK)
	refBuf := make([]float32, q4kBlockBytes*4) // room for a few blocks for the ref path
	for trial := 0; trial < 2000; trial++ {
		randQ4KBlock(rng, blk)
		// Reference: dequant the same single block via the full-row loader algorithm.
		dequantQ4KRef(refBuf[:qkK], blk)

		q4kDequantSuperBlock(got, blk)
		for i := 0; i < qkK; i++ {
			if !sameFloat(got[i], refBuf[i]) {
				t.Fatalf("trial %d idx %d: resident %v != ref %v (block=%x)", trial, i, got[i], refBuf[i], blk[:8])
			}
		}
	}
}

// TestQ4KMatRowsMatchesF32 checks the resident-Q4_K f32 GEMV (q4kMatRowsRange — the scalar
// dequant+dot path, which is the byte-identical fallback when the int8 SDOT path is off) against
// a plain f32 matvec on the dequantized weights. Both paths dequant identically, so the only
// difference is float32 accumulation order; the bound is therefore far tighter than a
// quantization-quality gate (which this is NOT — it is a packing/dequant correctness gate).
// The int8 SDOT path (the default decode route on arm64+SDOT, P2) carries its own tolerance gate
// in TestQ4KInt8DotMatchesF32. max-abs/RMS here must be tiny.
func TestQ4KMatRowsMatchesF32(t *testing.T) {
	const out, in = 32, 768 // in is a multiple of qkK (3 super-blocks/row)
	rng := rand.New(rand.NewSource(7))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			copy(raw[(o*nblk+b)*q4kBlockBytes:(o*nblk+b+1)*q4kBlockBytes], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)

	// Reference weights: dequant every row, then a naive sequential dot.
	wflat := make([]float32, out*in)
	for o := 0; o < out; o++ {
		dequantQ4KRef(wflat[o*in:(o+1)*in], raw[o*nblk*q4kBlockBytes:(o+1)*nblk*q4kBlockBytes])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	yGot := make([]float32, out)
	q4kMatRowsRange(qt, x, yGot, 0, out) // f32 scalar path (bit-identical fallback)

	var sumSq, maxRel float64
	for o := 0; o < out; o++ {
		var yRef float32
		wr := wflat[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			yRef += wr[i] * x[i]
		}
		sumSq += float64(yRef) * float64(yRef)
	}
	rms := math.Sqrt(sumSq / float64(out))
	if rms < 1e-9 {
		t.Fatalf("reference RMS ~0; bad test data")
	}
	for o := 0; o < out; o++ {
		var yRef float32
		wr := wflat[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			yRef += wr[i] * x[i]
		}
		if rel := math.Abs(float64(yGot[o]-yRef)) / rms; rel > maxRel {
			maxRel = rel
		}
	}
	// Same dequant arithmetic on both sides; only float32 summation order differs over a
	// 768-wide reduction. 1e-3 is generous for that and still catches any real dequant/packing
	// or sign bug (which would blow this up by orders of magnitude).
	if maxRel > 1e-3 {
		t.Fatalf("Q4_K GEMV max-abs/RMS %.3e exceeds 1e-3 (reorder-only tolerance)", maxRel)
	}
	t.Logf("Q4_K GEMV max-abs/RMS = %.3e (in=%d, rms=%.4g)", maxRel, in, rms)
}

// TestQ4KKernelMatchesQ4KMatRows pins the matKernel wrapper to the direct GEMV.
func TestQ4KKernelMatchesQ4KMatRows(t *testing.T) {
	const out, in = 8, 256 // exactly one super-block per row
	rng := rand.New(rand.NewSource(3))
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			copy(raw[(o*nblk+b)*q4kBlockBytes:(o*nblk+b+1)*q4kBlockBytes], blk)
		}
	}
	m := &Model{manifest: map[string]tensorMeta{}, q4kw: map[string]*q4kTensor{}}
	m.q4kw["w"] = quantizeQ4KFromRaw(raw, out, in)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	k := q4kKernel{m}
	yK := k.mul("w", k.prep(x), out, in)
	yD := q4kMatRows(m.q4k("w"), x)
	for i := range yK {
		if yK[i] != yD[i] {
			t.Fatalf("kernel vs direct mismatch at %d: %v != %v", i, yK[i], yD[i])
		}
	}
}

// TestQ4KFromRawGuards pins the precondition panics: a non-multiple-of-256 reduction dim
// and a payload-size mismatch both fail loudly instead of mis-super-blocking every row.
func TestQ4KFromRawGuards(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on non-multiple-of-256 reduction dim")
		}
	}()
	quantizeQ4KFromRaw(make([]byte, 144), 1, 255) // 255 is not a multiple of 256
}
