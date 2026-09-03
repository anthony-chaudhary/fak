package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// rocmfp4_test.go — unit tests for ROCmFP4 block quantization layout (Q4_0_ROCMFP4_FAST, #10730).

// TestROCmFP4BitExactRoundTrip proves bit-exact round-trip when inputs are exact multiples
// of an FP16 scale by FP4 E2M1 code values.
func TestROCmFP4BitExactRoundTrip(t *testing.T) {
	// Construct 32 elements using the 16 exact E2M1 values, each scaled by 0.5 (exact in FP16).
	exactScale := float32(0.5)
	src := make([]float32, ROCmFP4BlockSize)
	for i := 0; i < ROCmFP4BlockSize; i++ {
		code := i % 16
		src[i] = exactScale * rocmfp4E2M1Values[code]
	}

	blk := QuantizeROCmFP4Block(src)
	scale, codes, values := UnpackROCmFP4Block(blk)

	if scale != exactScale {
		t.Fatalf("unpacked scale = %v, want %v", scale, exactScale)
	}

	for i := 0; i < ROCmFP4BlockSize; i++ {
		wantCode := byte(i % 16)
		if codes[i] != wantCode {
			t.Fatalf("element %d: got code %d (0x%x), want %d (0x%x)", i, codes[i], codes[i], wantCode, wantCode)
		}
		if math.Float32bits(values[i]) != math.Float32bits(src[i]) {
			t.Fatalf("element %d: float bits mismatch: got 0x%08x (%v), want 0x%08x (%v)",
				i, math.Float32bits(values[i]), values[i], math.Float32bits(src[i]), src[i])
		}
	}

	// Dequantize with DequantizeROCmFP4Block
	dequant := make([]float32, ROCmFP4BlockSize)
	DequantizeROCmFP4Block(blk, dequant)
	for i := 0; i < ROCmFP4BlockSize; i++ {
		if math.Float32bits(dequant[i]) != math.Float32bits(src[i]) {
			t.Fatalf("dequant element %d: got %v, want %v", i, dequant[i], src[i])
		}
	}
}

// TestROCmFP4NumericalAccuracyAndErrorBounds verifies that quantization errors on normal
// distributions respect the theoretical bound (max step between E2M1 points is 2.0, so
// worst-case error is <= scale * 1.0 = maxAbs / 6.0).
func TestROCmFP4NumericalAccuracyAndErrorBounds(t *testing.T) {
	rng := rand.New(rand.NewSource(20260903))
	const nBlocks = 16
	src := make([]float32, nBlocks*ROCmFP4BlockSize)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 2.5)
	}

	tensor, err := QuantizeROCmFP4(src, nBlocks, ROCmFP4BlockSize)
	if err != nil {
		t.Fatalf("QuantizeROCmFP4 failed: %v", err)
	}

	dequant := DequantizeROCmFP4(tensor)
	if len(dequant) != len(src) {
		t.Fatalf("dequant length %d != src length %d", len(dequant), len(src))
	}

	var maxErr float32
	for b := 0; b < nBlocks; b++ {
		offset := b * ROCmFP4BlockSize
		blkSrc := src[offset : offset+ROCmFP4BlockSize]
		var blkMax float32
		for _, v := range blkSrc {
			a := float32(math.Abs(float64(v)))
			if a > blkMax {
				blkMax = a
			}
		}
		bound := (blkMax / 6.0) * 1.05 // small FP16 rounding slack

		for i := 0; i < ROCmFP4BlockSize; i++ {
			diff := float32(math.Abs(float64(dequant[offset+i] - blkSrc[i])))
			if diff > maxErr {
				maxErr = diff
			}
			if diff > bound {
				t.Fatalf("block %d element %d: error %v exceeds bound %v (orig=%v, dequant=%v)",
					b, i, diff, bound, blkSrc[i], dequant[offset+i])
			}
		}
	}

	t.Logf("ROCmFP4 %d blocks max error: %v", nBlocks, maxErr)
}

// TestROCmFP4BlockStrideAlignment verifies RDNA 3/3.5 vector register stride enforcement:
// matrices must have columns aligned to 32 elements (one half-wave).
func TestROCmFP4BlockStrideAlignment(t *testing.T) {
	// Valid: cols is multiple of 32
	for _, cols := range []int{32, 64, 96, 128, 256, 1024} {
		if err := ValidateROCmFP4Dimensions(4, cols); err != nil {
			t.Fatalf("valid cols=%d failed validation: %v", cols, err)
		}
	}

	// Invalid: cols is not a multiple of 32
	for _, cols := range []int{1, 16, 31, 33, 48, 63, 65, 100} {
		err := ValidateROCmFP4Dimensions(4, cols)
		if err == nil {
			t.Fatalf("unaligned cols=%d should fail validation", cols)
		}
	}

	// Invalid rows or cols <= 0
	if err := ValidateROCmFP4Dimensions(0, 32); err == nil {
		t.Fatal("rows=0 should fail")
	}
	if err := ValidateROCmFP4Dimensions(4, 0); err == nil {
		t.Fatal("cols=0 should fail")
	}
	if err := ValidateROCmFP4Dimensions(-1, 32); err == nil {
		t.Fatal("rows=-1 should fail")
	}
}

// TestROCmFP4LayoutValidationAndRejection tests layout validation edge cases.
func TestROCmFP4LayoutValidationAndRejection(t *testing.T) {
	// Truncated buffer
	raw := make([]byte, ROCmFP4BlockBytes-1)
	if err := ValidateROCmFP4Layout(32, raw); err == nil {
		t.Fatal("truncated raw bytes should be rejected")
	}

	// Non-multiple of 32 elements
	validRaw := make([]byte, ROCmFP4BlockBytes)
	if err := ValidateROCmFP4Layout(31, validRaw); err == nil {
		t.Fatal("non-32 elements should be rejected")
	}
	if err := ValidateROCmFP4Layout(0, validRaw); err == nil {
		t.Fatal("zero elements should be rejected")
	}

	// Byte length mismatch (e.g. 64 elements = 2 blocks = 36 bytes, given 18)
	if err := ValidateROCmFP4Layout(64, validRaw); err == nil {
		t.Fatal("element/byte mismatch should be rejected")
	}

	// NaN scale rejection
	nanRaw := make([]byte, ROCmFP4BlockBytes)
	binary.LittleEndian.PutUint16(nanRaw[0:2], 0x7E00) // FP16 NaN
	if err := ValidateROCmFP4Layout(32, nanRaw); err == nil {
		t.Fatal("NaN scale should be rejected")
	}
}

// TestROCmFP4Unpacker verifies UnpackROCmFP4Bytes and UnpackROCmFP4Block.
func TestROCmFP4Unpacker(t *testing.T) {
	src := make([]float32, ROCmFP4BlockSize)
	for i := range src {
		src[i] = float32(i - 16)
	}

	blk := QuantizeROCmFP4Block(src)
	scale, codes, values := UnpackROCmFP4Block(blk)

	if scale <= 0 {
		t.Fatalf("scale should be positive, got %v", scale)
	}

	// Unpack through raw bytes
	raw := make([]byte, ROCmFP4BlockBytes)
	binary.LittleEndian.PutUint16(raw[0:2], blk.Scale)
	copy(raw[2:], blk.Data[:])

	scaleRaw, codesRaw, valuesRaw, err := UnpackROCmFP4Bytes(raw)
	if err != nil {
		t.Fatalf("UnpackROCmFP4Bytes failed: %v", err)
	}

	if scaleRaw != scale {
		t.Fatalf("scale mismatch: %v vs %v", scaleRaw, scale)
	}
	for i := 0; i < ROCmFP4BlockSize; i++ {
		if codesRaw[i] != codes[i] {
			t.Fatalf("code mismatch at %d: %d vs %d", i, codesRaw[i], codes[i])
		}
		if valuesRaw[i] != values[i] {
			t.Fatalf("value mismatch at %d: %v vs %v", i, valuesRaw[i], values[i])
		}
	}

	// Invalid byte length
	if _, _, _, err := UnpackROCmFP4Bytes(raw[:17]); err == nil {
		t.Fatal("short byte slice should fail unpack")
	}
}

// TestROCmFP4SerializationRoundTrip verifies byte serialization / deserialization.
func TestROCmFP4SerializationRoundTrip(t *testing.T) {
	const rows, cols = 4, 64 // 8 blocks
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i%50) * 0.1
	}

	tensor, err := QuantizeROCmFP4(src, rows, cols)
	if err != nil {
		t.Fatalf("QuantizeROCmFP4 failed: %v", err)
	}

	raw := tensor.Bytes()
	wantLen := (rows * cols / ROCmFP4BlockSize) * ROCmFP4BlockBytes
	if len(raw) != wantLen {
		t.Fatalf("tensor.Bytes() len = %d, want %d", len(raw), wantLen)
	}

	restored, err := ROCmFP4FromBytes(raw, rows, cols)
	if err != nil {
		t.Fatalf("ROCmFP4FromBytes failed: %v", err)
	}

	d1 := DequantizeROCmFP4(tensor)
	d2 := DequantizeROCmFP4(restored)

	for i := range d1 {
		if d1[i] != d2[i] {
			t.Fatalf("element %d mismatch after serialize/deserialize: %v vs %v", i, d1[i], d2[i])
		}
	}

	// Also test DequantizeROCmFP4Bytes
	d3, err := DequantizeROCmFP4Bytes(raw)
	if err != nil {
		t.Fatalf("DequantizeROCmFP4Bytes failed: %v", err)
	}
	for i := range d1 {
		if d1[i] != d3[i] {
			t.Fatalf("element %d mismatch from DequantizeROCmFP4Bytes: %v vs %v", i, d1[i], d3[i])
		}
	}
}

// TestROCmFP4MetadataPreset verifies integration with fp4meta.go contract.
func TestROCmFP4MetadataPreset(t *testing.T) {
	preset := ROCmFP4MetadataPreset()
	result := AdjudicateFP4Metadata(preset)

	if result.Disposition != FP4Accept {
		t.Fatalf("ROCmFP4MetadataPreset disposition = %v (want %v), reason=%s, detail=%s",
			result.Disposition, FP4Accept, result.Reason, result.Detail)
	}
	if result.Reason != FP4ReasonSupported {
		t.Fatalf("ROCmFP4MetadataPreset reason = %v (want %v)", result.Reason, FP4ReasonSupported)
	}

	// Verify alias method
	aliasPreset := ROCmFP4Metadata()
	if aliasPreset.Format != FP4FormatROCmFP4 || aliasPreset.BlockScale.Encoding != FP4ScaleFP16 {
		t.Fatalf("ROCmFP4Metadata alias mismatch: %+v", aliasPreset)
	}
}

// TestROCmFP4MatVec verifies matrix-vector GEMV decode calculation against FP32 baseline.
func TestROCmFP4MatVec(t *testing.T) {
	const rows, cols = 8, 64
	rng := rand.New(rand.NewSource(20260905))

	wSrc := make([]float32, rows*cols)
	for i := range wSrc {
		wSrc[i] = float32(rng.NormFloat64() * 0.5)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.2)
	}

	tensor, err := QuantizeROCmFP4(wSrc, rows, cols)
	if err != nil {
		t.Fatalf("QuantizeROCmFP4 failed: %v", err)
	}

	yQuant, err := ROCmFP4MatVec(tensor, x)
	if err != nil {
		t.Fatalf("ROCmFP4MatVec failed: %v", err)
	}

	if len(yQuant) != rows {
		t.Fatalf("expected y length %d, got %d", rows, len(yQuant))
	}

	// Compute FP32 reference using dequantized weights
	wDequant := DequantizeROCmFP4(tensor)
	for r := 0; r < rows; r++ {
		var expected float32
		for c := 0; c < cols; c++ {
			expected += wDequant[r*cols+c] * x[c]
		}
		diff := float32(math.Abs(float64(yQuant[r] - expected)))
		if diff > 1e-5 {
			t.Fatalf("row %d: ROCmFP4MatVec result %v != dequant matvec %v (diff=%v)", r, yQuant[r], expected, diff)
		}
	}

	// Error cases
	if _, err := ROCmFP4MatVec(nil, x); err == nil {
		t.Fatal("nil tensor should return error")
	}
	if _, err := ROCmFP4MatVec(tensor, x[:cols-1]); err == nil {
		t.Fatal("mismatched vector dimension should return error")
	}
}

// TestROCmFP4SliceAndByteSize tests QuantizeROCmFP4Slice, ByteSize(), and Inf scale rejection.
func TestROCmFP4SliceAndByteSize(t *testing.T) {
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i) * 0.1
	}

	tensor, err := QuantizeROCmFP4Slice(src)
	if err != nil {
		t.Fatalf("QuantizeROCmFP4Slice failed: %v", err)
	}

	if tensor.Rows != 1 || tensor.Cols != 64 {
		t.Fatalf("expected 1x64 tensor, got %dx%d", tensor.Rows, tensor.Cols)
	}
	if tensor.ByteSize() != 2*ROCmFP4BlockBytes {
		t.Fatalf("expected %d bytes, got %d", 2*ROCmFP4BlockBytes, tensor.ByteSize())
	}

	var nilTensor *ROCmFP4Tensor
	if nilTensor.ByteSize() != 0 {
		t.Fatal("nilTensor.ByteSize should return 0")
	}

	// Test Inf scale rejection in ValidateROCmFP4Layout
	infRaw := make([]byte, ROCmFP4BlockBytes)
	binary.LittleEndian.PutUint16(infRaw[0:2], 0x7C00) // FP16 Inf
	if err := ValidateROCmFP4Layout(32, infRaw); err == nil {
		t.Fatal("Inf scale should be rejected by ValidateROCmFP4Layout")
	}
}

// TestROCmFP4Float32ToFP16_RoundToNearestEven runs the RTNE tests under the ROCmFP4 test filter.
func TestROCmFP4Float32ToFP16_RoundToNearestEven(t *testing.T) {
	TestFloat32ToFP16_RoundToNearestEven(t)
}

// TestFloat32ToFP16_RoundToNearestEven verifies IEEE 754 Round-to-Nearest, Ties-to-Even (RTNE)
// rounding in Float32ToFP16 (#10912).
func TestFloat32ToFP16_RoundToNearestEven(t *testing.T) {
	t.Run("ExactMidpointTiesToEven", func(t *testing.T) {
		// Even LSB mantissa: mant16 = 0x010 (16), lower 13 bits = 0x1000 (exact midpoint).
		// RTNE preserves even LSB, so it rounds down to 0x010.
		midpointEven := math.Float32frombits((127 << 23) | (0x010 << 13) | 0x1000)
		gotEven := Float32ToFP16(midpointEven)
		wantEven := uint16((15 << 10) | 0x010)
		if gotEven != wantEven {
			t.Fatalf("even LSB midpoint: got 0x%04x, want 0x%04x", gotEven, wantEven)
		}

		// Negative even LSB midpoint
		gotEvenNeg := Float32ToFP16(-midpointEven)
		wantEvenNeg := (1 << 15) | wantEven
		if gotEvenNeg != wantEvenNeg {
			t.Fatalf("negative even LSB midpoint: got 0x%04x, want 0x%04x", gotEvenNeg, wantEvenNeg)
		}

		// Odd LSB mantissa: mant16 = 0x011 (17), lower 13 bits = 0x1000 (exact midpoint).
		// RTNE rounds up to nearest even LSB (0x012).
		midpointOdd := math.Float32frombits((127 << 23) | (0x011 << 13) | 0x1000)
		gotOdd := Float32ToFP16(midpointOdd)
		wantOdd := uint16((15 << 10) | 0x012)
		if gotOdd != wantOdd {
			t.Fatalf("odd LSB midpoint: got 0x%04x, want 0x%04x", gotOdd, wantOdd)
		}

		// Negative odd LSB midpoint
		gotOddNeg := Float32ToFP16(-midpointOdd)
		wantOddNeg := (1 << 15) | wantOdd
		if gotOddNeg != wantOddNeg {
			t.Fatalf("negative odd LSB midpoint: got 0x%04x, want 0x%04x", gotOddNeg, wantOddNeg)
		}
	})

	t.Run("SlightlyAboveMidpointRoundsUp", func(t *testing.T) {
		// Even base mant16 = 0x010, rem = 0x1001 (> 0x1000) -> rounds up to 0x011.
		aboveEven := math.Float32frombits((127 << 23) | (0x010 << 13) | 0x1001)
		gotEven := Float32ToFP16(aboveEven)
		wantEven := uint16((15 << 10) | 0x011)
		if gotEven != wantEven {
			t.Fatalf("above midpoint (even base): got 0x%04x, want 0x%04x", gotEven, wantEven)
		}

		// Odd base mant16 = 0x011, rem = 0x1001 (> 0x1000) -> rounds up to 0x012.
		aboveOdd := math.Float32frombits((127 << 23) | (0x011 << 13) | 0x1001)
		gotOdd := Float32ToFP16(aboveOdd)
		wantOdd := uint16((15 << 10) | 0x012)
		if gotOdd != wantOdd {
			t.Fatalf("above midpoint (odd base): got 0x%04x, want 0x%04x", gotOdd, wantOdd)
		}
	})

	t.Run("SlightlyBelowMidpointRoundsDown", func(t *testing.T) {
		// Even base mant16 = 0x010, rem = 0x0FFF (< 0x1000) -> rounds down to 0x010.
		belowEven := math.Float32frombits((127 << 23) | (0x010 << 13) | 0x0FFF)
		gotEven := Float32ToFP16(belowEven)
		wantEven := uint16((15 << 10) | 0x010)
		if gotEven != wantEven {
			t.Fatalf("below midpoint (even base): got 0x%04x, want 0x%04x", gotEven, wantEven)
		}

		// Odd base mant16 = 0x011, rem = 0x0FFF (< 0x1000) -> rounds down to 0x011.
		belowOdd := math.Float32frombits((127 << 23) | (0x011 << 13) | 0x0FFF)
		gotOdd := Float32ToFP16(belowOdd)
		wantOdd := uint16((15 << 10) | 0x011)
		if gotOdd != wantOdd {
			t.Fatalf("below midpoint (odd base): got 0x%04x, want 0x%04x", gotOdd, wantOdd)
		}
	})

	t.Run("MantissaOverflowIncrementsExponent", func(t *testing.T) {
		// Normal range: mant16 = 0x3FF (odd), rem = 0x1000.
		// Rounds up to 0x400 -> mantissa overflows: exp16 becomes 16, mant16 becomes 0.
		overflowNormal := math.Float32frombits((127 << 23) | (0x3FF << 13) | 0x1000)
		gotNormal := Float32ToFP16(overflowNormal)
		wantNormal := uint16((16 << 10) | 0)
		if gotNormal != wantNormal {
			t.Fatalf("normal mantissa overflow: got 0x%04x, want 0x%04x", gotNormal, wantNormal)
		}

		// Overflow to Infinity: exp16 = 30 (exp = 142), mant16 = 0x3FF, rem = 0x1000.
		// Rounds up: exp16 becomes 31 -> overflows to Inf (0x7C00).
		overflowInf := math.Float32frombits((142 << 23) | (0x3FF << 13) | 0x1000)
		gotInf := Float32ToFP16(overflowInf)
		wantInf := uint16(0x7C00)
		if gotInf != wantInf {
			t.Fatalf("overflow to infinity: got 0x%04x, want 0x%04x", gotInf, wantInf)
		}

		// Subnormal to normal transition:
		// exp16 = 0 (exp = 112). Exact midpoint between largest subnormal (0x03FF) and smallest normal (0x0400).
		// Smallest normal is even, so tie rounds up into normal range (exp16=1, mant16=0 => 0x0400).
		bitsMid := uint32(112<<23) | 0x7FE000
		gotSubMid := Float32ToFP16(math.Float32frombits(bitsMid))
		wantSubMid := uint16(0x0400)
		if gotSubMid != wantSubMid {
			t.Fatalf("subnormal midpoint to normal: got 0x%04x, want 0x%04x", gotSubMid, wantSubMid)
		}

		// Slightly below midpoint rounds down to largest subnormal (0x03FF)
		bitsBelow := uint32(112<<23) | 0x7FDFFF
		gotSubBelow := Float32ToFP16(math.Float32frombits(bitsBelow))
		wantSubBelow := uint16(0x03FF)
		if gotSubBelow != wantSubBelow {
			t.Fatalf("subnormal below midpoint: got 0x%04x, want 0x%04x", gotSubBelow, wantSubBelow)
		}

		// Slightly above midpoint rounds up to normal (0x0400)
		bitsAbove := uint32(112<<23) | 0x7FE001
		gotSubAbove := Float32ToFP16(math.Float32frombits(bitsAbove))
		wantSubAbove := uint16(0x0400)
		if gotSubAbove != wantSubAbove {
			t.Fatalf("subnormal above midpoint: got 0x%04x, want 0x%04x", gotSubAbove, wantSubAbove)
		}
	})

	t.Run("BlockScaleReconstructsMaxAbsAccurately", func(t *testing.T) {
		testMaxVals := []float32{0.001, 0.05, 0.5, 1.0, 2.71828, 3.14159, 6.0, 15.2, 42.0, 100.0, 1024.0, 65000.0}
		for _, maxVal := range testMaxVals {
			src := make([]float32, ROCmFP4BlockSize)
			src[0] = maxVal
			src[1] = -maxVal * 0.5
			for i := 2; i < ROCmFP4BlockSize; i++ {
				src[i] = maxVal * float32(i) / float32(ROCmFP4BlockSize*2)
			}

			blk := QuantizeROCmFP4Block(src)
			expectedScaleFP16 := Float32ToFP16(maxVal / 6.0)
			if blk.Scale != expectedScaleFP16 {
				t.Fatalf("maxVal %v: block scale 0x%04x != Float32ToFP16(maxVal/6.0) 0x%04x",
					maxVal, blk.Scale, expectedScaleFP16)
			}

			scale := FP16ToFloat32(blk.Scale)
			reconstructedMax := scale * 6.0
			relErr := float32(math.Abs(float64(reconstructedMax-maxVal))) / maxVal
			if relErr > 0.001 {
				t.Fatalf("maxVal %v: reconstructedMax %v has relative error %v > 0.001",
					maxVal, reconstructedMax, relErr)
			}

			dequant := make([]float32, ROCmFP4BlockSize)
			DequantizeROCmFP4Block(blk, dequant)
			dequantErr := float32(math.Abs(float64(dequant[0]-maxVal))) / maxVal
			if dequantErr > 0.001 {
				t.Fatalf("maxVal %v: dequant[0] %v has relative error %v > 0.001",
					maxVal, dequant[0], dequantErr)
			}
		}
	})
}
