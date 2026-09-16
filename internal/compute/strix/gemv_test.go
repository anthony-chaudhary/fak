package strix

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"
)

// gemv_test.go -- adversarial roofline-saturation acceptance test for the gfx1151 Wave32 decode
// GEMV (Q3_K, Q4_K, BF16, FP16), authored by an independent test context.
//
// BIPARTITE DISCIPLINE: criteria 3 and 4 of ticket #594 are [HW-WITNESSED] and can only be
// satisfied on physical gfx1151 silicon. This file asserts the SW-verifiable numeric parity and
// the roofline TARGET CONSTANT, and gates the physical >= 218.44 GB/s reading behind an explicit
// device opt-in. On any non-gfx1151 host the physical clause is [HW-WITNESSED], UNCHECKED and
// SKIPPED; no hardware number is fabricated here.
//
// All references below are independently written; the implementation's own *Stream functions are
// never used as the reference (that would be circular).

// --- independent scalar reference primitives -----------------------------------------------

// refF16BitsToF32 is an independent IEEE-754 half -> float32 widening, written from the format
// definition (sign/exponent/mantissa) rather than calling the package's converter.
func refF16BitsToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(sign << 31)
		}
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		man &= 0x3ff
		return math.Float32frombits(sign<<31 | e<<23 | man<<13)
	case 0x1f:
		return math.Float32frombits(sign<<31 | 0xff<<23 | man<<13)
	default:
		return math.Float32frombits(sign<<31 | (exp-15+127)<<23 | man<<13)
	}
}

// refF32ToF16Bits is an independent round-to-nearest-even float32 -> half converter, written from
// the format definition. Used to build FP16 fixtures without reusing the implementation's code.
func refF32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32((bits>>23)&0xff) - 127 + 15
	man := bits & 0x7fffff
	switch {
	case (bits & 0x7f800000) == 0x7f800000:
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 31:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		rounded := (man + half) >> shift
		return sign | uint16(rounded)
	default:
		man += 0x1000
		if man&0x800000 != 0 {
			man = 0
			exp++
			if exp >= 31 {
				return sign | 0x7c00
			}
		}
		return sign | uint16(exp)<<10 | uint16(man>>13)
	}
}

// refQ3KScales independently expands the 12 packed 6-bit Q3_K sub-scale bytes (llama.cpp k-quant
// get_scale_min_k4 packing), written from the layout rather than calling the mirror.
func refQ3KScales(raw []byte) [16]int8 {
	aux0 := uint32(raw[0]) | uint32(raw[1])<<8 | uint32(raw[2])<<16 | uint32(raw[3])<<24
	aux1 := uint32(raw[4]) | uint32(raw[5])<<8 | uint32(raw[6])<<16 | uint32(raw[7])<<24
	aux2 := uint32(raw[8]) | uint32(raw[9])<<8 | uint32(raw[10])<<16 | uint32(raw[11])<<24
	words := [4]uint32{
		(aux0 & 0x0f0f0f0f) | ((aux2 & 0x03030303) << 4),
		(aux1 & 0x0f0f0f0f) | (((aux2 >> 2) & 0x03030303) << 4),
		((aux0 >> 4) & 0x0f0f0f0f) | (((aux2 >> 4) & 0x03030303) << 4),
		((aux1 >> 4) & 0x0f0f0f0f) | (((aux2 >> 6) & 0x03030303) << 4),
	}
	var out [16]int8
	for i, w := range words {
		out[i*4+0] = int8(byte(w))
		out[i*4+1] = int8(byte(w >> 8))
		out[i*4+2] = int8(byte(w >> 16))
		out[i*4+3] = int8(byte(w >> 24))
	}
	return out
}

// refQ3KDot independently dequants one 110-byte Q3_K super-block and dots it with x, accumulating
// in float64. It re-derives the hmask/code layout from the format definition.
func refQ3KDot(blk []byte, x []float32) float64 {
	scales := refQ3KScales(blk[96:108])
	d := float64(refF16BitsToF32(uint16(blk[108]) | uint16(blk[109])<<8))
	var sum float64
	hmask := blk[:32]
	q := blk[32:96]
	qi, is := 0, 0
	mask := byte(1)
	for n := 0; n < 256; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			dl := d * float64(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+l] >> shift) & 3)
				if hmask[l]&mask == 0 {
					code -= 4
				}
				sum += dl * float64(code) * float64(x[n+j*32+l])
			}
			dl = d * float64(int(scales[is])-32)
			is++
			for l := 0; l < 16; l++ {
				code := int8((q[qi+16+l] >> shift) & 3)
				if hmask[16+l]&mask == 0 {
					code -= 4
				}
				sum += dl * float64(code) * float64(x[n+j*32+16+l])
			}
			shift += 2
			mask <<= 1
		}
		qi += 32
	}
	return sum
}

// refQ4KScaleMin independently unpacks the j-th 6-bit Q4_K scale/min pair.
func refQ4KScaleMin(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// refQ4KDot independently dequants one 144-byte Q4_K super-block and dots it with x.
func refQ4KDot(blk []byte, x []float32) float64 {
	d := float64(refF16BitsToF32(uint16(blk[0]) | uint16(blk[1])<<8))
	dmin := float64(refF16BitsToF32(uint16(blk[2]) | uint16(blk[3])<<8))
	scales := blk[4:16]
	q := blk[16:144]
	var sum float64
	qi, is := 0, 0
	for j := 0; j < 256; j += 64 {
		sc, m := refQ4KScaleMin(is, scales)
		d1, m1 := d*float64(sc), dmin*float64(m)
		sc, m = refQ4KScaleMin(is+1, scales)
		d2, m2 := d*float64(sc), dmin*float64(m)
		for l := 0; l < 32; l++ {
			sum += (d1*float64(q[qi+l]&0x0f) - m1) * float64(x[j+l])
		}
		for l := 0; l < 32; l++ {
			sum += (d2*float64(q[qi+l]>>4) - m2) * float64(x[j+32+l])
		}
		qi += 32
		is += 2
	}
	return sum
}

// refBF16Dot independently dots one BF16 row with x using the canonical bf16 -> f32 bit shift.
func refBF16Dot(row []uint16, x []float32) float64 {
	var sum float64
	for j := range row {
		bits := uint32(row[j]) << 16
		sum += float64(math.Float32frombits(bits)) * float64(x[j])
	}
	return sum
}

// --- independent fixture builders ----------------------------------------------------------

func refQ3KBlock(rng *rand.Rand, d float32) []byte {
	blk := make([]byte, Q3KSuperBlockBytes)
	for i := 0; i < 32; i++ {
		blk[i] = byte(rng.Intn(256))
	}
	for i := 32; i < 96; i++ {
		blk[i] = byte(rng.Intn(256))
	}
	for i := 96; i < 108; i++ {
		blk[i] = byte(rng.Intn(256))
	}
	h := refF32ToF16Bits(d)
	blk[108], blk[109] = byte(h), byte(h>>8)
	return blk
}

func refQ4KBlock(rng *rand.Rand, d, dmin float32) []byte {
	blk := make([]byte, Q4KSuperBlockBytes)
	dh := refF32ToF16Bits(d)
	dm := refF32ToF16Bits(dmin)
	blk[0], blk[1] = byte(dh), byte(dh>>8)
	blk[2], blk[3] = byte(dm), byte(dm>>8)
	for i := 4; i < Q4KSuperBlockBytes; i++ {
		blk[i] = byte(rng.Intn(256))
	}
	return blk
}

// --- host-side reference dot drivers ---------------------------------------------------------

func refQ3KRowDots(raw []byte, x []float32, out, in int) []float64 {
	rowBlocks := in / Q3KSuperBlockElems
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for b := 0; b < rowBlocks; b++ {
			off := (o*rowBlocks + b) * Q3KSuperBlockBytes
			want[o] += refQ3KDot(raw[off:off+Q3KSuperBlockBytes], x[b*Q3KSuperBlockElems:(b+1)*Q3KSuperBlockElems])
		}
	}
	return want
}

func refQ4KRowDots(raw []byte, x []float32, out, in int) []float64 {
	rowBlocks := in / Q4KSuperBlockElems
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		for b := 0; b < rowBlocks; b++ {
			off := (o*rowBlocks + b) * Q4KSuperBlockBytes
			want[o] += refQ4KDot(raw[off:off+Q4KSuperBlockBytes], x[b*Q4KSuperBlockElems:(b+1)*Q4KSuperBlockElems])
		}
	}
	return want
}

func refBF16RowDots(raw []uint16, x []float32, out, in int) []float64 {
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		want[o] = refBF16Dot(raw[o*in:(o+1)*in], x)
	}
	return want
}

func refFP16RowDots(raw []uint16, x []float32, out, in int) []float64 {
	want := make([]float64, out)
	for o := 0; o < out; o++ {
		row := raw[o*in : (o+1)*in]
		var sum float64
		for j := range row {
			sum += float64(refF16BitsToF32(row[j])) * float64(x[j])
		}
		want[o] = sum
	}
	return want
}

// --- the acceptance test ----------------------------------------------------------------------

func TestRooflineSaturation(t *testing.T) {
	t.Run("CosineParity", testRooflineCosineParity)
	t.Run("ReductionEdges", testRooflineReductionEdges)
	t.Run("ErrorLadder", testRooflineErrorLadder)
	t.Run("FailClosed", testRooflineFailClosed)
	t.Run("CoalescedBurstInvariant", testRooflineCoalescedBurstInvariant)
	t.Run("RooflineConstants", testRooflineConstants)
	t.Run("PhysicalBandwidthHWWitnessed", testRooflinePhysicalBandwidthHWWitnessed)
}

// testRooflineCosineParity checks the SW-verifiable clause: cosine similarity of every format's
// GEMV output against an independently written scalar reference exceeds 0.999900.
func testRooflineCosineParity(t *testing.T) {
	const out, in = 40, 768 // 3 super-blocks per row
	rng := rand.New(rand.NewSource(0x5A7A7A7A))

	raw3 := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
	for r := 0; r < out*(in/Q3KSuperBlockElems); r++ {
		copy(raw3[r*Q3KSuperBlockBytes:], refQ3KBlock(rng, 0.001+rng.Float32()*0.05))
	}
	raw4 := make([]byte, out*(in/Q4KSuperBlockElems)*Q4KSuperBlockBytes)
	for r := 0; r < out*(in/Q4KSuperBlockElems); r++ {
		copy(raw4[r*Q4KSuperBlockBytes:], refQ4KBlock(rng, 0.001+rng.Float32()*0.05, 0.0005+rng.Float32()*0.02))
	}
	rawBF := make([]uint16, out*in)
	for i := range rawBF {
		rawBF[i] = FP32ToBF16(rng.Float32()*4 - 2)
	}
	rawFP := make([]uint16, out*in)
	for i := range rawFP {
		rawFP[i] = refF32ToF16Bits(rng.Float32()*4 - 2)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	got3, err := Wave32Q3KGEMVStream(raw3, x, out, in)
	if err != nil {
		t.Fatalf("Wave32Q3KGEMVStream: %v", err)
	}
	got4, err := Wave32Q4KGEMVStream(raw4, x, out, in)
	if err != nil {
		t.Fatalf("Wave32Q4KGEMVStream: %v", err)
	}
	gotBF, err := Wave32BF16GEMVStream(rawBF, x, out, in)
	if err != nil {
		t.Fatalf("Wave32BF16GEMVStream: %v", err)
	}
	gotFP, err := Wave32FP16GEMVStream(rawFP, x, out, in)
	if err != nil {
		t.Fatalf("Wave32FP16GEMVStream: %v", err)
	}

	cases := []struct {
		name string
		got  []float32
		want []float64
	}{
		{"Q3_K_M", got3, refQ3KRowDots(raw3, x, out, in)},
		{"Q4_K_M", got4, refQ4KRowDots(raw4, x, out, in)},
		{"BF16", gotBF, refBF16RowDots(rawBF, x, out, in)},
		{"FP16", gotFP, refFP16RowDots(rawFP, x, out, in)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assertCosineAgainstReference(t, c.got, c.want)
		})
	}
}

// assertCosineAgainstReference enforces the ticket's > 0.999900 SW clause and reports the number.
func assertCosineAgainstReference(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output len %d != reference len %d", len(got), len(want))
	}
	ref := make([]float32, len(want))
	var refMax float64
	for i, w := range want {
		ref[i] = float32(w)
		if a := math.Abs(w); a > refMax {
			refMax = a
		}
	}
	if refMax < 1e-6 {
		t.Fatalf("degenerate all-zero reference (max |ref| = %.3e)", refMax)
	}
	cos, err := CosineSimilarity(got, ref)
	if err != nil {
		t.Fatalf("CosineSimilarity: %v", err)
	}
	if !(cos > 0.999900) {
		t.Fatalf("cosine similarity %.9f <= 0.999900 (SW-VERIFIED clause failed)", cos)
	}
	t.Logf("cosine=%.9f (> 0.999900) [SW-VERIFIED]", cos)
}

// testRooflineReductionEdges probes the pipelined reduction boundary shapes: a 256-multiple
// reduction, out=1, all-zero activation, all-zero weight row, a negative-heavy vector, and a
// large `in`. Each compares against the independent reference (except where the reference is
// collinear-by-zero and only structural assertions apply).
func testRooflineReductionEdges(t *testing.T) {
	rng := rand.New(rand.NewSource(0xED6E5))

	t.Run("SingleRow", func(t *testing.T) {
		const out, in = 1, 512
		raw := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
		for r := 0; r < out*(in/Q3KSuperBlockElems); r++ {
			copy(raw[r*Q3KSuperBlockBytes:], refQ3KBlock(rng, 0.002+rng.Float32()*0.03))
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		got, err := Wave32Q3KGEMVStream(raw, x, out, in)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		assertCosineAgainstReference(t, got, refQ3KRowDots(raw, x, out, in))
	})

	t.Run("FlatNonMultipleReduction", func(t *testing.T) {
		// BF16/FP16 have no 256-element super-block: a reduction of 7 elements must be computed
		// exactly (row dot over the whole flat row), not rejected and not silently zero-padded.
		const out, in = 4, 7
		rawBF := make([]uint16, out*in)
		rawFP := make([]uint16, out*in)
		for i := range rawBF {
			rawBF[i] = FP32ToBF16(rng.Float32()*2 - 1)
			rawFP[i] = refF32ToF16Bits(rng.Float32()*2 - 1)
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		gotBF, err := Wave32BF16GEMVStream(rawBF, x, out, in)
		if err != nil {
			t.Fatalf("BF16 non-multiple stream: %v", err)
		}
		assertCosineAgainstReference(t, gotBF, refBF16RowDots(rawBF, x, out, in))

		gotFP, err := Wave32FP16GEMVStream(rawFP, x, out, in)
		if err != nil {
			t.Fatalf("FP16 non-multiple stream: %v", err)
		}
		assertCosineAgainstReference(t, gotFP, refFP16RowDots(rawFP, x, out, in))
	})

	t.Run("AllZeroActivation", func(t *testing.T) {
		const out, in = 4, 256
		raw := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
		for r := 0; r < out*(in/Q3KSuperBlockElems); r++ {
			copy(raw[r*Q3KSuperBlockBytes:], refQ3KBlock(rng, 0.01))
		}
		got, err := Wave32Q3KGEMVStream(raw, make([]float32, in), out, in)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		for i, v := range got {
			if v != 0 {
				t.Fatalf("all-zero activation: got[%d] = %v, want 0", i, v)
			}
		}
	})

	t.Run("AllZeroWeightRow", func(t *testing.T) {
		const out, in = 3, 256
		raw := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
		zero := refQ3KBlockZeros()
		copy(raw[0:Q3KSuperBlockBytes], zero)
		copy(raw[Q3KSuperBlockBytes:2*Q3KSuperBlockBytes], refQ3KBlock(rng, 0.01))
		copy(raw[2*Q3KSuperBlockBytes:], refQ3KBlock(rng, 0.01))
		x := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		got, err := Wave32Q3KGEMVStream(raw, x, out, in)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if ref := refQ3KDot(raw[0:Q3KSuperBlockBytes], x); ref != 0 {
			t.Fatalf("zeroed fixture row 0 is not zero (ref=%v)", ref)
		}
		if got[0] != 0 {
			t.Fatalf("all-zero weight row produced %v, want exactly 0", got[0])
		}
		for o := 1; o < out; o++ {
			want := []float64{refQ3KDot(raw[o*Q3KSuperBlockBytes:(o+1)*Q3KSuperBlockBytes], x)}
			if !approxEqual(float64(got[o]), want[0], 1e-4) {
				t.Fatalf("row %d dot %v != reference %v", o, got[o], want[0])
			}
		}
	})

	t.Run("NegativeHeavyActivation", func(t *testing.T) {
		const out, in = 6, 512
		raw := make([]byte, out*(in/Q4KSuperBlockElems)*Q4KSuperBlockBytes)
		for r := 0; r < out*(in/Q4KSuperBlockElems); r++ {
			copy(raw[r*Q4KSuperBlockBytes:], refQ4KBlock(rng, 0.01, 0.01))
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = -(1.0 + rng.Float32()*3)
		}
		got, err := Wave32Q4KGEMVStream(raw, x, out, in)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		assertCosineAgainstReference(t, got, refQ4KRowDots(raw, x, out, in))
	})

	t.Run("LargeIn", func(t *testing.T) {
		const out, in = 2, 4096 // 16 super-blocks per row
		raw := make([]byte, out*(in/Q4KSuperBlockElems)*Q4KSuperBlockBytes)
		for r := 0; r < out*(in/Q4KSuperBlockElems); r++ {
			copy(raw[r*Q4KSuperBlockBytes:], refQ4KBlock(rng, 0.001+rng.Float32()*0.01, 0.0005))
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		got, err := Wave32Q4KGEMVStream(raw, x, out, in)
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		assertCosineAgainstReference(t, got, refQ4KRowDots(raw, x, out, in))
	})
}

// refQ3KBlockZeros builds a Q3_K super-block whose dequantized weights are all exactly zero: the
// f16 d scale is zero, so every dl = d*(scale-32) = 0 regardless of codes.
func refQ3KBlockZeros() []byte {
	blk := make([]byte, Q3KSuperBlockBytes)
	return blk
}

// testRooflineErrorLadder asserts every malformed dimension/payload returns an error instead of
// panicking. A wrong shape must never be silently accepted.
func testRooflineErrorLadder(t *testing.T) {
	const out, in = 3, 512
	validQ3Raw := make([]byte, out*(in/Q3KSuperBlockElems)*Q3KSuperBlockBytes)
	validQ4Raw := make([]byte, out*(in/Q4KSuperBlockElems)*Q4KSuperBlockBytes)
	validX := make([]float32, in)

	cases := []struct {
		name string
		call func() ([]float32, error)
		want error
	}{
		{"Q3K_out_zero", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX, 0, in) }, ErrInvalidDimensions},
		{"Q3K_out_negative", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX, -2, in) }, ErrInvalidDimensions},
		{"Q3K_in_zero", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX, out, 0) }, ErrInvalidDimensions},
		{"Q3K_in_negative", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX, out, -256) }, ErrInvalidDimensions},
		{"Q3K_in_not_multiple", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX, out, 255) }, ErrInvalidDimensions},
		{"Q3K_payload_short", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw[:len(validQ3Raw)-1], validX, out, in) }, ErrDimensionMismatch},
		{"Q3K_payload_long", func() ([]float32, error) { return Wave32Q3KGEMVStream(append(validQ3Raw, 0), validX, out, in) }, ErrDimensionMismatch},
		{"Q3K_x_short", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, validX[:in-1], out, in) }, ErrDimensionMismatch},
		{"Q3K_x_long", func() ([]float32, error) { return Wave32Q3KGEMVStream(validQ3Raw, append(validX, 0), out, in) }, ErrDimensionMismatch},

		{"Q4K_in_not_multiple", func() ([]float32, error) { return Wave32Q4KGEMVStream(validQ4Raw, validX, out, 300) }, ErrInvalidDimensions},
		{"Q4K_payload_short", func() ([]float32, error) { return Wave32Q4KGEMVStream(validQ4Raw[:len(validQ4Raw)-1], validX, out, in) }, ErrDimensionMismatch},
		{"Q4K_x_short", func() ([]float32, error) { return Wave32Q4KGEMVStream(validQ4Raw, validX[:in-1], out, in) }, ErrDimensionMismatch},

		{"BF16_out_zero", func() ([]float32, error) { return Wave32BF16GEMVStream(make([]uint16, out*in), validX, 0, in) }, ErrInvalidDimensions},
		{"BF16_in_zero", func() ([]float32, error) { return Wave32BF16GEMVStream(make([]uint16, out*in), validX, out, 0) }, ErrInvalidDimensions},
		{"BF16_payload_short", func() ([]float32, error) { return Wave32BF16GEMVStream(make([]uint16, out*in-1), validX, out, in) }, ErrDimensionMismatch},
		{"BF16_x_short", func() ([]float32, error) { return Wave32BF16GEMVStream(make([]uint16, out*in), validX[:in-1], out, in) }, ErrDimensionMismatch},

		{"FP16_out_zero", func() ([]float32, error) { return Wave32FP16GEMVStream(make([]uint16, out*in), validX, 0, in) }, ErrInvalidDimensions},
		{"FP16_in_zero", func() ([]float32, error) { return Wave32FP16GEMVStream(make([]uint16, out*in), validX, out, 0) }, ErrInvalidDimensions},
		{"FP16_payload_short", func() ([]float32, error) { return Wave32FP16GEMVStream(make([]uint16, out*in-1), validX, out, in) }, ErrDimensionMismatch},
		{"FP16_x_long", func() ([]float32, error) {
			return Wave32FP16GEMVStream(make([]uint16, out*in), append(validX, 0), out, in)
		}, ErrDimensionMismatch},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("malformed input panicked instead of erroring: %v", r)
					}
				}()
				_, err := c.call()
				if err == nil {
					t.Fatal("malformed input returned nil error")
				}
				if !errors.Is(err, c.want) {
					t.Fatalf("error %v is not %v", err, c.want)
				}
			}()
		})
	}

	t.Run("PayloadHelpers", func(t *testing.T) {
		if Q3KPayloadBytes(out, 255) != 0 {
			t.Fatal("Q3KPayloadBytes must be 0 when in is not a super-block multiple")
		}
		if Q4KPayloadBytes(out, 255) != 0 {
			t.Fatal("Q4KPayloadBytes must be 0 when in is not a super-block multiple")
		}
		if BF16PayloadBytes(0, in) != 0 || FP16PayloadBytes(-1, in) != 0 {
			t.Fatal("flat payload helpers must be 0 for non-positive dims")
		}
	})
}

// testRooflineFailClosed asserts a default kernel is NOT available and that Dispatch refuses with
// ErrWave32GEMVUnavailable rather than silently producing a CPU result.
func testRooflineFailClosed(t *testing.T) {
	kernels := []struct {
		name      string
		available func() bool
		dispatch  func() ([]float32, error)
	}{
		{
			"Q3_K",
			func() bool { return DefaultWave32Q3KGEMVDecodeKernel().Available() },
			func() ([]float32, error) {
				return DefaultWave32Q3KGEMVDecodeKernel().DispatchQ3KGEMV(make([]byte, 110), make([]float32, 256), 1, 256)
			},
		},
		{
			"Q4_K",
			func() bool { return DefaultWave32Q4KGEMVDecodeKernel().Available() },
			func() ([]float32, error) {
				return DefaultWave32Q4KGEMVDecodeKernel().DispatchQ4KGEMV(make([]byte, 144), make([]float32, 256), 1, 256)
			},
		},
		{
			"BF16",
			func() bool { return DefaultWave32BF16GEMVDecodeKernel().Available() },
			func() ([]float32, error) {
				return DefaultWave32BF16GEMVDecodeKernel().DispatchBF16GEMV(make([]uint16, 256), make([]float32, 256), 1, 256)
			},
		},
		{
			"FP16",
			func() bool { return DefaultWave32FP16GEMVDecodeKernel().Available() },
			func() ([]float32, error) {
				return DefaultWave32FP16GEMVDecodeKernel().DispatchFP16GEMV(make([]uint16, 256), make([]float32, 256), 1, 256)
			},
		},
	}
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
			if k.available() {
				t.Fatal("default kernel must be NOT available (fail-closed)")
			}
			got, err := k.dispatch()
			if !errors.Is(err, ErrWave32GEMVUnavailable) {
				t.Fatalf("dispatch error %v is not ErrWave32GEMVUnavailable", err)
			}
			if got != nil {
				t.Fatalf("refused dispatch returned %v, want nil (no silent CPU fallback)", got)
			}
		})
	}

	t.Run("AdmittedStillNumeric", func(t *testing.T) {
		k := NewWave32Q3KGEMVDecodeKernel(DefaultQ3KDecodeConfig(), true)
		if !k.Available() {
			t.Fatal("an explicitly admitted kernel must be available")
		}
		const out, in = 2, 256
		rng := rand.New(rand.NewSource(0xAD))
		raw := make([]byte, out*Q3KSuperBlockBytes)
		for r := 0; r < out; r++ {
			copy(raw[r*Q3KSuperBlockBytes:], refQ3KBlock(rng, 0.02))
		}
		x := make([]float32, in)
		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		got, err := k.DispatchQ3KGEMV(raw, x, out, in)
		if err != nil {
			t.Fatalf("admitted dispatch: %v", err)
		}
		assertCosineAgainstReference(t, got, refQ3KRowDots(raw, x, out, in))
	})
}

// testRooflineCoalescedBurstInvariant binds the 128-byte burst model and the 512-byte wavefront
// dwordx4 instruction count.
func testRooflineCoalescedBurstInvariant(t *testing.T) {
	if CoalescedBurstBytes != 128 {
		t.Fatalf("CoalescedBurstBytes = %d, want 128", CoalescedBurstBytes)
	}
	if BytesPerLaneDwordX4 != 16 {
		t.Fatalf("BytesPerLaneDwordX4 = %d, want 16", BytesPerLaneDwordX4)
	}
	for _, off := range []int{0, 128, 256, 1024, 4096} {
		if !CoalescedBurstAligned(off) {
			t.Fatalf("offset %d must be 128-byte burst aligned", off)
		}
	}
	for _, off := range []int{-1, -128, 1, 127, 129, 130, 4095} {
		if CoalescedBurstAligned(off) {
			t.Fatalf("offset %d must NOT be burst aligned", off)
		}
	}

	loads := []struct {
		payload int64
		want    int64
	}{{0, 0}, {-5, 0}, {1, 1}, {16, 1}, {511, 1}, {512, 1}, {513, 2}, {110, 1}, {144, 1}, {1024, 2}, {1025, 3}, {4096 * 4, 32}}
	for _, l := range loads {
		if got := GlobalLoadDwordX4Count(l.payload); got != l.want {
			t.Fatalf("GlobalLoadDwordX4Count(%d) = %d, want %d", l.payload, got, l.want)
		}
	}

	m := DefaultWave32LaneLoadModel()
	if err := m.Validate(); err != nil {
		t.Fatalf("default lane model invalid: %v", err)
	}
	if m.Lanes != RDNA35NativeWaveSize || m.PayloadBytes != WavefrontCoalescedBytes || m.Bursts != BurstsPerWavefrontTransaction {
		t.Fatalf("lane model shape = %d lanes / %d bytes / %d bursts", m.Lanes, m.PayloadBytes, m.Bursts)
	}
	if m.Loads[0].ByteOffset != 0 || m.Loads[31].ByteOffset != 31*16 {
		t.Fatalf("lane offsets must be lane*16 (got %d .. %d)", m.Loads[0].ByteOffset, m.Loads[31].ByteOffset)
	}
	if !CoalescedBurstAligned(m.BaseOffset) {
		t.Fatalf("default lane model base offset %d is not burst aligned", m.BaseOffset)
	}

	notAligned := DefaultWave32LaneLoadModel()
	notAligned.BaseOffset = 64
	if err := notAligned.Validate(); err == nil {
		t.Fatal("lane model with a 64-byte base must not validate")
	}
	overflow := DefaultWave32LaneLoadModel()
	overflow.Loads[31].ByteOffset = WavefrontCoalescedBytes
	if err := overflow.Validate(); err == nil {
		t.Fatal("lane read escaping the 512-byte payload must not validate")
	}
}

// testRooflineConstants pins the roofline target constant and asserts the unmeasured case cannot
// fabricate a device bandwidth number.
func testRooflineConstants(t *testing.T) {
	const wantFloor = 218.44
	const wantPeak = 273.056

	if math.Abs(StrixHaloPhysicalDRAMBandwidthGBs-wantPeak) > 1e-9 {
		t.Fatalf("StrixHaloPhysicalDRAMBandwidthGBs = %.6f, want %.6f", StrixHaloPhysicalDRAMBandwidthGBs, wantPeak)
	}
	if TargetDecodeBandwidthFloorGBps < 218.44 || TargetDecodeBandwidthFloorGBps > 218.45 {
		t.Fatalf("TargetDecodeBandwidthFloorGBps = %.6f, want within [218.44, 218.45]", TargetDecodeBandwidthFloorGBps)
	}
	if !approxEqual(TargetDecodeBandwidthFloorGBps, 0.80*wantPeak, 1e-9) {
		t.Fatalf("TargetDecodeBandwidthFloorGBps = %.6f, want 80%% of %.3f = %.6f",
			TargetDecodeBandwidthFloorGBps, wantPeak, 0.80*wantPeak)
	}
	if !MeetsDecodeBandwidthFloor(TargetDecodeBandwidthFloorGBps) {
		t.Fatal("the floor must satisfy itself")
	}
	if MeetsDecodeBandwidthFloor(TargetDecodeBandwidthFloorGBps - 0.001) {
		t.Fatal("just below the floor must NOT qualify")
	}
	if MeetsDecodeBandwidthFloor(0) {
		t.Fatal("an unmeasured (0) rate must NOT qualify")
	}

	if _, err := EstimateDecodeBandwidthGBps(1<<20, 0); err == nil {
		t.Fatal("unmeasured (elapsed == 0) must return an error, never a fabricated number")
	}
	if _, err := EstimateDecodeBandwidthGBps(1<<20, -time.Second); err == nil {
		t.Fatal("negative elapsed must return an error")
	}
	if _, err := EstimateDecodeBandwidthGBps(0, time.Second); err == nil {
		t.Fatal("zero weight bytes must return an error")
	}
	// A real timed run must report the payload over the elapsed interval.
	gbs, err := EstimateDecodeBandwidthGBps(1_000_000_000, time.Second)
	if err != nil {
		t.Fatalf("EstimateDecodeBandwidthGBps: %v", err)
	}
	if !approxEqual(gbs, 1.0, 1e-9) {
		t.Fatalf("1 GB in 1 s = %.6f GB/s, want 1.0", gbs)
	}
}

// testRooflinePhysicalBandwidthHWWitnessed is the [HW-WITNESSED] clause. It is gated behind the
// sibling lane's device opt-in (FAK_STRIX_WAVE32_GEMV_DECODE); without it the physical >= 218.44
// GB/s reading is UNCHECKED on this host and the subtest skips. It never asserts a hardware
// measurement on a non-gfx1151 host.
func testRooflinePhysicalBandwidthHWWitnessed(t *testing.T) {
	if !StrixDecodeGEMVDeviceRequested() {
		t.Skipf("[HW-WITNESSED] physical >= %.2f GB/s sustained LPDDR5X read is UNCHECKED on this host: "+
			"device opt-in %s is unset; run on live gfx1151 silicon with the opt-in and an asserted launch path (%s)",
			TargetDecodeBandwidthFloorGBps, EnvStrixWave32GEMVDecode, EnvStrixGEMVLaunchPath)
	}
	if strings.TrimSpace(StrixGEMVLaunchPath()) == "" {
		t.Skipf("[HW-WITNESSED] device opt-in is set but no validated gfx1151 launch path is asserted (%s); "+
			"physical >= %.2f GB/s remains UNCHECKED", EnvStrixGEMVLaunchPath, TargetDecodeBandwidthFloorGBps)
	}
	// The launch path is asserted. Even then this test owns only the SW side: it verifies the
	// admitted kernel computes, and that the roofline reader refuses an unmeasured interval. A
	// real device measurement must arrive as a timed launch on physical silicon; it is not
	// fabricated here.
	k := NewWave32Q3KGEMVDecodeKernel(DefaultQ3KDecodeConfig(), true)
	if !k.Available() {
		t.Fatalf("asserted launch path did not admit the Q3_K kernel (reason: %s)", k.Admission().Reason)
	}
	if _, err := EstimateDecodeBandwidthGBps(1<<20, 0); err == nil {
		t.Fatal("unmeasured interval must not fabricate a bandwidth reading even with a launch path")
	}
	_ = os.Getenv(EnvStrixGEMVLaunchPath)
}

// approxEqual reports whether a and b agree to within tol relative to the larger magnitude.
func approxEqual(a, b, tol float64) bool {
	d := math.Abs(a - b)
	m := math.Max(math.Abs(a), math.Abs(b))
	if m == 0 {
		return d <= tol
	}
	return d/m <= tol
}
