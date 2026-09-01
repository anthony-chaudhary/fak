package metalgemm

import "math"

// NVFP4BlockWeights is the number of E2M1 weights sharing one E4M3FN scale.
const NVFP4BlockWeights = 16

// NVFP4PayloadBytes returns the packed weight plus scale bytes for an [out,in]
// NVFP4 matrix. Invalid shapes return zero; the M5 GEMV candidate accepts M=1
// only and requires K (in) to be a positive multiple of 16.
func NVFP4PayloadBytes(out, in int) int {
	if out <= 0 || in <= 0 || in%NVFP4BlockWeights != 0 {
		return 0
	}
	return out*in/2 + out*in/NVFP4BlockWeights
}

func nvfp4ValidPayload(packed, scales []byte, out, in int) bool {
	if NVFP4PayloadBytes(out, in) == 0 {
		return false
	}
	return len(packed) == out*in/2 && len(scales) == out*in/NVFP4BlockWeights
}

var nvfp4E2M1 = [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6}

// nvfp4E4M3FN decodes the finite-only FP8 scale encoding. The sole NaN code
// (magnitude 0x7f) is rejected by returning NaN so upload can fail closed.
func nvfp4E4M3FN(raw byte) float32 {
	sign := float32(1)
	if raw&0x80 != 0 {
		sign = -1
	}
	mag := raw & 0x7f
	exp, mant := mag>>3, mag&7
	if exp == 15 && mant == 7 {
		return float32(math.NaN())
	}
	if exp == 0 {
		return sign * float32(mant) / 512
	}
	return sign * float32(math.Ldexp(8+float64(mant), int(exp)-10))
}

func nvfp4Reference(packed, scales []byte, out, in int, x []float32) ([]float32, bool) {
	if !nvfp4ValidPayload(packed, scales, out, in) || len(x) != in {
		return nil, false
	}
	y := make([]float32, out)
	blocks := in / NVFP4BlockWeights
	for row := 0; row < out; row++ {
		var sum float32
		for k := 0; k < in; k++ {
			b := packed[(row*in+k)/2]
			nibble := b & 0xf
			if k&1 != 0 {
				nibble = b >> 4
			}
			scale := nvfp4E4M3FN(scales[row*blocks+k/NVFP4BlockWeights])
			if math.IsNaN(float64(scale)) {
				return nil, false
			}
			sum += nvfp4E2M1[nibble] * float32(math.Abs(float64(scale))) * x[k]
		}
		y[row] = sum
	}
	return y, true
}
