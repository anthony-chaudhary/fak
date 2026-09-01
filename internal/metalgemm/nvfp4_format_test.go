package metalgemm

import (
	"math"
	"testing"
)

func TestNVFP4FormatAndIndependentOracle(t *testing.T) {
	wantE2M1 := []float32{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}
	for code, want := range wantE2M1 {
		if got := nvfp4E2M1[code]; got != want {
			t.Fatalf("E2M1 code %x = %g, want %g", code, got, want)
		}
	}
	for raw, want := range map[byte]float32{0x00: 0, 0x01: 1.0 / 512, 0x08: 1.0 / 64, 0x38: 1, 0x3c: 1.5, 0x7e: 448, 0xbc: -1.5} {
		if got := nvfp4E4M3FN(raw); got != want {
			t.Fatalf("E4M3FN %02x = %g, want %g", raw, got, want)
		}
	}
	if !math.IsNaN(float64(nvfp4E4M3FN(0x7f))) {
		t.Fatal("E4M3FN NaN code accepted")
	}

	const out, in = 2, 16
	packed := make([]byte, out*in/2)
	for i := range packed {
		packed[i] = byte((2*i+1)&15) | byte((2*i+2)&15)<<4
	}
	scales := []byte{0x3c, 0xbc} // abs(scale) makes both rows use 1.5.
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(i-7) / 8
	}
	got, ok := nvfp4Reference(packed, scales, out, in, x)
	if !ok {
		t.Fatal("valid source-format payload rejected")
	}
	for row := 0; row < out; row++ {
		var want float32
		for k := 0; k < in; k++ {
			code := byte((row*in + k + 1) & 15)
			want += wantE2M1[code] * 1.5 * x[k]
		}
		if got[row] != want {
			t.Fatalf("row %d = %g, want independent oracle %g", row, got[row], want)
		}
	}
}

func TestNVFP4FormatFailsClosedOutsideEnvelope(t *testing.T) {
	if NVFP4PayloadBytes(2, 16) != 18 {
		t.Fatalf("payload bytes = %d, want 18", NVFP4PayloadBytes(2, 16))
	}
	for _, shape := range [][2]int{{0, 16}, {2, 0}, {2, 15}, {-1, 16}} {
		if got := NVFP4PayloadBytes(shape[0], shape[1]); got != 0 {
			t.Fatalf("shape %v payload bytes = %d, want 0", shape, got)
		}
	}
	if _, ok := nvfp4Reference(make([]byte, 8), []byte{0x38}, 1, 16, make([]float32, 15)); ok {
		t.Fatal("short activation accepted")
	}
	if _, ok := nvfp4Reference(make([]byte, 8), []byte{0x7f}, 1, 16, make([]float32, 16)); ok {
		t.Fatal("NaN scale accepted")
	}
}
