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

// TestDecodeE4M3Golden pins the FP8-E4M3FN element decode with hand-checked anchors across every
// regime AND the defining "fn" property: NO byte decodes to infinity (unlike E5M2), and the only
// non-finite encoding is the single NaN pattern S.1111.111. Values are exact in f32, so these are
// equalities, not tolerances — the exact oracle a device kernel must match. Anchors mirror the
// safetensors-side golden (internal/model fp8_blockscale) so the two decoders cannot drift apart.
func TestDecodeE4M3Golden(t *testing.T) {
	// The e4m3fn contract: no infinities anywhere in the 8-bit domain, and exactly two NaN bytes.
	nans := 0
	for i := 0; i < 256; i++ {
		v := DecodeE4M3(byte(i))
		if math.IsInf(float64(v), 0) {
			t.Fatalf("DecodeE4M3(%#02x) = %v — e4m3fn has no infinities", byte(i), v)
		}
		if math.IsNaN(float64(v)) {
			nans++
		}
	}
	if nans != 2 { // 0x7F and 0xFF only
		t.Errorf("e4m3fn NaN byte count = %d, want 2 (0x7F, 0xFF)", nans)
	}

	// Signed zeros.
	if v := DecodeE4M3(0x00); v != 0 || math.Signbit(float64(v)) {
		t.Errorf("0x00 = %v (signbit %v), want +0", v, math.Signbit(float64(v)))
	}
	if v := DecodeE4M3(0x80); v != 0 || !math.Signbit(float64(v)) {
		t.Errorf("0x80 = %v (signbit %v), want -0", v, math.Signbit(float64(v)))
	}
	// Unit normals and their neighbours: exp bias 7.
	if v := DecodeE4M3(0x30); v != 0.5 { // exp 6 (unbiased -1), mantissa 0 → 2^-1
		t.Errorf("0x30 = %v, want 0.5", v)
	}
	if v := DecodeE4M3(0x38); v != 1 { // exp 7 (unbiased 0), mantissa 0 → 2^0
		t.Errorf("0x38 = %v, want 1.0", v)
	}
	if v := DecodeE4M3(0x40); v != 2 { // exp 8 (unbiased 1), mantissa 0 → 2^1
		t.Errorf("0x40 = %v, want 2.0", v)
	}
	if v := DecodeE4M3(0xB8); v != -1 { // sign + exp 7, mantissa 0 → -1.0
		t.Errorf("0xB8 = %v, want -1.0", v)
	}
	// Max finite: exp 15 with mantissa 6 (mantissa 7 is the NaN slot) → 2^8 * 1.75 = 448.
	if v := DecodeE4M3(0x7E); v != 448 {
		t.Errorf("0x7E = %v, want 448 (max finite)", v)
	}
	// NaN pattern S.1111.111.
	if v := DecodeE4M3(0x7F); !math.IsNaN(float64(v)) {
		t.Errorf("0x7F = %v, want NaN", v)
	}
	// Smallest positive subnormal: exp 0, mantissa 1 → 2^(1-7) * 1/8 = 2^-9.
	if v := DecodeE4M3(0x01); v != float32(math.Ldexp(1, -9)) {
		t.Errorf("0x01 = %v, want 2^-9 = %v", v, math.Ldexp(1, -9))
	}
	// Smallest positive normal: exp 1, mantissa 0 → 2^(1-7) = 2^-6.
	if v := DecodeE4M3(0x08); v != float32(math.Ldexp(1, -6)) {
		t.Errorf("0x08 = %v, want 2^-6 = %v", v, math.Ldexp(1, -6))
	}
}
