package compute

import (
	"math"
	"testing"
)

// TestDecodeE5M2Golden pins the FP8-E5M2 element decode two ways: (1) the load-bearing identity
// that all 256 byte values decode bit-for-bit to f16(b<<8) — the exact oracle a future device
// kernel must match — and (2) hand-checked anchors across every regime of the format.
func TestDecodeE5M2Golden(t *testing.T) {
	// (1) exact identity over the whole 8-bit domain, compared on bits so NaN payloads match too.
	for i := 0; i < 256; i++ {
		b := byte(i)
		got := math.Float32bits(DecodeE5M2(b))
		want := f16bitsToF32(uint16(b) << 8)
		if got != want {
			t.Fatalf("DecodeE5M2(%#02x) bits = %#08x, want f16(b<<8) = %#08x", b, got, want)
		}
	}

	// (2) hand-checked anchors: ±0, unit normals, ±Inf, NaN, and the smallest sub/normal.
	if v := DecodeE5M2(0x00); v != 0 || math.Signbit(float64(v)) {
		t.Errorf("0x00 = %v (signbit %v), want +0", v, math.Signbit(float64(v)))
	}
	if v := DecodeE5M2(0x80); v != 0 || !math.Signbit(float64(v)) {
		t.Errorf("0x80 = %v (signbit %v), want -0", v, math.Signbit(float64(v)))
	}
	if v := DecodeE5M2(0x3C); v != 1 { // exp 15 (unbiased 0), mantissa 0 → 2^0 * 1.0
		t.Errorf("0x3C = %v, want 1.0", v)
	}
	if v := DecodeE5M2(0x40); v != 2 { // exp 16 (unbiased 1), mantissa 0 → 2^1
		t.Errorf("0x40 = %v, want 2.0", v)
	}
	if v := DecodeE5M2(0xBC); v != -1 { // sign + exp 15, mantissa 0 → -1.0
		t.Errorf("0xBC = %v, want -1.0", v)
	}
	if v := DecodeE5M2(0x7C); !math.IsInf(float64(v), 1) { // exp 31, mantissa 0 → +Inf
		t.Errorf("0x7C = %v, want +Inf", v)
	}
	if v := DecodeE5M2(0xFC); !math.IsInf(float64(v), -1) { // sign + exp 31, mantissa 0 → -Inf
		t.Errorf("0xFC = %v, want -Inf", v)
	}
	if v := DecodeE5M2(0x7F); !math.IsNaN(float64(v)) { // exp 31, mantissa != 0 → NaN
		t.Errorf("0x7F = %v, want NaN", v)
	}
	// smallest positive subnormal: exp 0, mantissa 01 → 2^-14 * (1/4) = 2^-16.
	if v := DecodeE5M2(0x01); v != float32(math.Ldexp(1, -16)) {
		t.Errorf("0x01 = %v, want 2^-16 = %v", v, math.Ldexp(1, -16))
	}
	// smallest positive normal: exp 1, mantissa 0 → 2^-14.
	if v := DecodeE5M2(0x04); v != float32(math.Ldexp(1, -14)) {
		t.Errorf("0x04 = %v, want 2^-14 = %v", v, math.Ldexp(1, -14))
	}
}
