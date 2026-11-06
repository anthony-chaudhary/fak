package kquantbits

import (
	"math"
	"testing"
)

// TestF16BitsToF32BitsExhaustive checks the shared k-quant scale decoder's full domain.
// fak-test:runtime fast est=10ms lane=default
func TestF16BitsToF32BitsExhaustive(t *testing.T) {
	t.Parallel()
	for raw := uint32(0); raw < 1<<16; raw++ {
		fraction := raw & 0x03ff
		exponent := (raw >> 10) & 0x1f
		var want uint32
		if exponent == 0x1f {
			// Inf has zero payload. NaNs preserve all ten payload bits,
			// including the quiet/signaling bit, without floating-point arithmetic.
			want = 0x7f800000 | fraction<<13
		} else {
			// Independent arithmetic oracle: all finite binary16 values are
			// exact in float32. Do not mirror the decoder's normalization loop
			// or construct its rebiased float32 exponent field.
			var value float64
			if exponent == 0 {
				value = math.Ldexp(float64(fraction), -24)
			} else {
				value = math.Ldexp(float64(1024+fraction), int(exponent)-25)
			}
			want = math.Float32bits(float32(value))
		}
		// Applying the sign to bits also distinguishes negative zero.
		want |= (raw & 0x8000) << 16
		if got := F16BitsToF32Bits(uint16(raw)); got != want {
			t.Fatalf("F16BitsToF32Bits(%04x) = %08x, want %08x", raw, got, want)
		}
	}
}
