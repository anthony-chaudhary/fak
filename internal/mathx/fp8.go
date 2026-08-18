package mathx

import "math"

// DecodeE4M3 decodes one OCP float8_e4m3fn byte exactly into float32.
func DecodeE4M3(b byte) float32 {
	sign := 1.0
	if b&0x80 != 0 {
		sign = -1
	}
	exp, mantissa := int(b>>3)&0x0f, int(b)&0x07
	switch {
	case exp == 0x0f && mantissa == 0x07:
		return float32(math.NaN())
	case exp == 0:
		return float32(sign * math.Ldexp(float64(mantissa)/8, 1-7))
	default:
		return float32(sign * math.Ldexp(1+float64(mantissa)/8, exp-7))
	}
}
