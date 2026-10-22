package compute

import (
	"fmt"
	"math"
)

// QuantizeNVFP4 returns a canonical logical CPU matrix using our deterministic
// reference policy, not a vendor training recipe, device layout, or GPU bit-parity
// claim. It does not enable generic dtype, loader, or hardware support.
//
// globalScale is a caller-supplied finite positive FP32 decode multiplier. Only
// all-zero input permits +0; -0 is never canonical. Each consecutive block of 16
// values uses absmax/(6*globalScale), rounded nearest, ties to even, to canonical
// nonnegative E4M3FN with saturation at 448. Each magnitude is then divided by
// the rounded block scale times globalScale and rounded nearest, ties to even,
// to E2M1 with saturation at 6. A block scale that rounds to zero emits signed
// zeros. All input signs are preserved, including zero and underflow signs.
//
// The factors and block-scale formula follow NVIDIA's format description:
// https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html
// The deterministic policy here is distinct from its stochastic training recipe.
// Comparisons use exact FP64 midpoint products, without rounded FP32 quotients.
// Shape/length errors, nonfinite input, invalid global scales, or any encoded
// value overflowing FP32 on decode return a zero matrix, never partial output.
// Input is not modified and must not be modified concurrently by the caller.
func QuantizeNVFP4(values []float32, rows, cols int, globalScale float32) (NVFP4Matrix, error) {
	if rows <= 0 || cols <= 0 || cols%16 != 0 || rows > int(^uint(0)>>1)/cols {
		return NVFP4Matrix{}, fmt.Errorf("NVFP4 quantize: invalid or overflowing shape %dx%d", rows, cols)
	}
	n := rows * cols
	if len(values) != n {
		return NVFP4Matrix{}, fmt.Errorf("NVFP4 quantize: input length %d, want %d", len(values), n)
	}
	if !nvfp4Finite(globalScale) || math.Signbit(float64(globalScale)) {
		return NVFP4Matrix{}, fmt.Errorf("NVFP4 quantize: global decode scale must be canonical nonnegative finite FP32")
	}
	for i, v := range values {
		if !nvfp4Finite(v) || (globalScale == 0 && v != 0) {
			return NVFP4Matrix{}, fmt.Errorf("NVFP4 quantize: invalid input at element %d for global decode scale", i)
		}
	}

	// Reuse the frozen oracle's exact codebook, avoiding a second decoder.
	var scales [127]float64
	var magnitudes [8]float64
	probe := NVFP4Matrix{Packed: []byte{2}, BlockScales: []byte{0}, GlobalScale: 1}
	for i := range scales {
		probe.BlockScales[0] = byte(i)
		scales[i] = float64(probe.decode(0))
	}
	probe.BlockScales[0] = 0x38 // E4M3FN 1.
	for i := range magnitudes {
		probe.Packed[0] = byte(i)
		magnitudes[i] = float64(probe.decode(0))
	}

	m := NVFP4Matrix{Rows: rows, Cols: cols, Packed: make([]byte, n/2), BlockScales: make([]byte, n/16), GlobalScale: globalScale}
	for block := range m.BlockScales {
		start := block * 16
		var absmax float64
		for _, v := range values[start : start+16] {
			absmax = math.Max(absmax, math.Abs(float64(v)))
		}
		var scaleCode byte
		if absmax != 0 {
			scaleCode = nvfp4NearestCode(absmax, 6*float64(globalScale), scales[:])
		}
		m.BlockScales[block] = scaleCode
		multiplier := scales[scaleCode] * float64(globalScale)
		for j, v := range values[start : start+16] {
			var code byte
			if scaleCode != 0 {
				code = nvfp4NearestCode(math.Abs(float64(v)), multiplier, magnitudes[:])
			}
			if math.Signbit(float64(v)) {
				code |= 8
			}
			i := start + j
			m.Packed[i/2] |= code << uint(4*(i%2))
			if !nvfp4Finite(m.decode(i)) {
				return NVFP4Matrix{}, fmt.Errorf("NVFP4 quantize: decoded element %d overflows FP32", i)
			}
		}
	}
	return m, nil
}

// nvfp4NearestCode rounds value/multiplier without division. Adjacent positive
// codebook entries are ordered; ties select the even code (even significand).
// Products are exact: FP32 has at most 24 significand bits; these small dyadic
// codebooks and their midpoints keep every intermediate well within FP64's 53
// bits and exponent range. End codes naturally implement finite saturation.
func nvfp4NearestCode(value, multiplier float64, levels []float64) byte {
	lo, hi := 0, len(levels)-1
	for lo < hi {
		mid := lo + (hi-lo)/2
		threshold := ((levels[mid] + levels[mid+1]) / 2) * multiplier
		if value < threshold || (value == threshold && mid%2 == 0) {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	return byte(lo)
}
