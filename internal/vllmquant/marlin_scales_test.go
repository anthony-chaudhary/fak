package vllmquant

import (
	"bytes"
	"math"
	"testing"
)

// packNibblePair packs two signed 4-bit integers into one byte (low nibble first, high nibble second).
func packNibblePair(q0, q1 int) byte {
	u0 := EncodeSignedInt4(q0)
	u1 := EncodeSignedInt4(q1)
	return u0 | (u1 << 4)
}

func TestFoldMarlinPositiveScalesUntouched(t *testing.T) {
	groupSize := 16
	scales := []float32{1.5, 0.75, 2.25}
	totalWeights := len(scales) * groupSize
	packed := make([]byte, totalWeights/2)

	// Fill with diverse int4 values in [-7, 7]
	for i := 0; i < len(packed); i++ {
		q0 := (i % 15) - 7
		q1 := ((i * 3) % 15) - 7
		packed[i] = packNibblePair(q0, q1)
	}

	foldedScales, foldedWeights, foldedCount := FoldMarlinNegativeGroupScales(scales, packed, groupSize)

	if foldedCount != 0 {
		t.Fatalf("foldedCount = %d, want 0", foldedCount)
	}
	for i, s := range foldedScales {
		if s != scales[i] {
			t.Errorf("scale[%d] = %f, want %f", i, s, scales[i])
		}
	}
	if !bytes.Equal(foldedWeights, packed) {
		t.Fatalf("foldedWeights modified when scales were strictly positive")
	}

	dequantOriginal := DequantizeW4(scales, packed, groupSize)
	dequantMarlin := DequantizeMarlinW4(foldedScales, foldedWeights, groupSize)

	if len(dequantMarlin) != len(dequantOriginal) {
		t.Fatalf("dequant length mismatch: got %d, want %d", len(dequantMarlin), len(dequantOriginal))
	}
	for i := range dequantOriginal {
		diff := math.Abs(float64(dequantMarlin[i] - dequantOriginal[i]))
		if diff > 1e-6 {
			t.Errorf("elem[%d]: marlin %f != original %f (diff %e)", i, dequantMarlin[i], dequantOriginal[i], diff)
		}
	}
}

func TestFoldMarlinNegativeScalesNegated(t *testing.T) {
	groupSize := 16
	scales := []float32{-0.5, -1.25}
	totalWeights := len(scales) * groupSize
	packed := make([]byte, totalWeights/2)

	// Fill with values spanning [-7, 7]
	for i := 0; i < len(packed); i++ {
		q0 := (i % 15) - 7
		q1 := ((i * 5) % 15) - 7
		packed[i] = packNibblePair(q0, q1)
	}

	foldedScales, foldedWeights, foldedCount := FoldMarlinNegativeGroupScales(scales, packed, groupSize)

	if foldedCount != len(scales) {
		t.Fatalf("foldedCount = %d, want %d", foldedCount, len(scales))
	}

	// Verify scales negated to positive
	for i, s := range foldedScales {
		if s <= 0 {
			t.Errorf("foldedScale[%d] = %f, want positive", i, s)
		}
		if s != -scales[i] {
			t.Errorf("foldedScale[%d] = %f, want %f", i, s, -scales[i])
		}
	}

	// Verify each nibble was negated
	for i := 0; i < len(packed); i++ {
		origLow := DecodeSignedInt4(packed[i] & 0x0F)
		origHigh := DecodeSignedInt4((packed[i] >> 4) & 0x0F)
		foldLow := DecodeSignedInt4(foldedWeights[i] & 0x0F)
		foldHigh := DecodeSignedInt4((foldedWeights[i] >> 4) & 0x0F)

		if foldLow != -origLow {
			t.Errorf("byte[%d] low nibble: got %d, want %d", i, foldLow, -origLow)
		}
		if foldHigh != -origHigh {
			t.Errorf("byte[%d] high nibble: got %d, want %d", i, foldHigh, -origHigh)
		}
	}

	// Verify dequantized outputs match within floating-point precision
	dequantOriginal := DequantizeW4(scales, packed, groupSize)
	dequantMarlin := DequantizeMarlinW4(foldedScales, foldedWeights, groupSize)

	if len(dequantMarlin) != len(dequantOriginal) {
		t.Fatalf("length mismatch: got %d, want %d", len(dequantMarlin), len(dequantOriginal))
	}
	for i := range dequantOriginal {
		diff := math.Abs(float64(dequantMarlin[i] - dequantOriginal[i]))
		if diff > 1e-6 {
			t.Errorf("elem[%d]: marlin %f != original %f (diff %e)", i, dequantMarlin[i], dequantOriginal[i], diff)
		}
	}

	// Verify that unfolded negative scales in Marlin produce numerical garbage
	dequantMarlinUnfolded := DequantizeMarlinW4(scales, packed, groupSize)
	mismatchCount := 0
	for i := range dequantOriginal {
		if dequantOriginal[i] != 0 {
			diff := math.Abs(float64(dequantMarlinUnfolded[i] - dequantOriginal[i]))
			if diff > 1e-3 {
				mismatchCount++
			}
		}
	}
	if mismatchCount == 0 {
		t.Errorf("expected Marlin without folding to produce numerical divergence on negative scales")
	}
}

func TestFoldMarlinMixedScalesMultiGroup(t *testing.T) {
	groupSize := 32
	scales := []float32{1.5, -0.75, 2.0, -1.25}
	totalWeights := len(scales) * groupSize
	packed := make([]byte, totalWeights/2)

	for i := 0; i < len(packed); i++ {
		q0 := ((i * 3) % 15) - 7
		q1 := ((i*7 + 2) % 15) - 7
		packed[i] = packNibblePair(q0, q1)
	}

	foldedScales, foldedWeights, foldedCount := FoldMarlinNegativeGroupScales(scales, packed, groupSize)

	if foldedCount != 2 {
		t.Fatalf("foldedCount = %d, want 2", foldedCount)
	}

	// Groups 0 and 2 (positive) must be untouched
	for _, g := range []int{0, 2} {
		if foldedScales[g] != scales[g] {
			t.Errorf("group %d scale changed: %f -> %f", g, scales[g], foldedScales[g])
		}
		startByte := (g * groupSize) / 2
		endByte := startByte + (groupSize / 2)
		if !bytes.Equal(foldedWeights[startByte:endByte], packed[startByte:endByte]) {
			t.Errorf("group %d weights changed but scale was positive", g)
		}
	}

	// Groups 1 and 3 (negative) must be negated
	for _, g := range []int{1, 3} {
		if foldedScales[g] != -scales[g] {
			t.Errorf("group %d scale not negated: %f, want %f", g, foldedScales[g], -scales[g])
		}
		startByte := (g * groupSize) / 2
		endByte := startByte + (groupSize / 2)
		for b := startByte; b < endByte; b++ {
			origLow := DecodeSignedInt4(packed[b] & 0x0F)
			origHigh := DecodeSignedInt4((packed[b] >> 4) & 0x0F)
			foldLow := DecodeSignedInt4(foldedWeights[b] & 0x0F)
			foldHigh := DecodeSignedInt4((foldedWeights[b] >> 4) & 0x0F)
			if foldLow != -origLow {
				t.Errorf("byte %d low: %d != -%d", b, foldLow, origLow)
			}
			if foldHigh != -origHigh {
				t.Errorf("byte %d high: %d != -%d", b, foldHigh, origHigh)
			}
		}
	}

	// Verify complete multi-group tensor dequantization matches
	dequantOriginal := DequantizeW4(scales, packed, groupSize)
	dequantMarlin := DequantizeMarlinW4(foldedScales, foldedWeights, groupSize)

	for i := range dequantOriginal {
		diff := math.Abs(float64(dequantMarlin[i] - dequantOriginal[i]))
		if diff > 1e-6 {
			t.Errorf("elem[%d]: marlin %f != original %f (diff %e)", i, dequantMarlin[i], dequantOriginal[i], diff)
		}
	}
}

func TestFoldMarlinEdgeCasesAndClamping(t *testing.T) {
	// Test clamping of -8: in signed int4 [-8, 7], -(-8) is clamped to +7
	negNibble := NegateInt4Nibble(0x08)
	if negNibble != 0x07 {
		t.Errorf("NegateInt4Nibble(0x08) = 0x%02X, want 0x07", negNibble)
	}

	// Test empty slices
	fs, fw, count := FoldMarlinNegativeGroupScales(nil, nil, 16)
	if count != 0 || len(fs) != 0 || len(fw) != 0 {
		t.Errorf("empty test failed: count=%d, len(fs)=%d, len(fw)=%d", count, len(fs), len(fw))
	}

	// Test zero groupSize (should infer group size from lengths)
	scales := []float32{-1.0, 2.0}
	packed := []byte{packNibblePair(2, -3), packNibblePair(-1, 0)} // 4 weights, 2 per group
	fs, fw, count = FoldMarlinNegativeGroupScales(scales, packed, 0)
	if count != 1 {
		t.Fatalf("foldedCount = %d, want 1", count)
	}
	if fs[0] != 1.0 || fs[1] != 2.0 {
		t.Errorf("scales = %v, want [1.0, 2.0]", fs)
	}
	origDequant := DequantizeW4(scales, packed)
	marlinDequant := DequantizeMarlinW4(fs, fw)
	for i := range origDequant {
		diff := math.Abs(float64(marlinDequant[i] - origDequant[i]))
		if diff > 1e-6 {
			t.Errorf("inferred groupSize elem[%d]: marlin %f != orig %f", i, marlinDequant[i], origDequant[i])
		}
	}
}
