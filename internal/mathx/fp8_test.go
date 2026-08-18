package mathx

import (
	"math"
	"testing"
)

func TestDecodeE4M3(t *testing.T) {
	for _, tc := range []struct {
		bits byte
		want float32
	}{{0x00, 0}, {0x38, 1}, {0xB8, -1}, {0x7E, 448}} {
		if got := DecodeE4M3(tc.bits); got != tc.want {
			t.Errorf("DecodeE4M3(%02x) = %g, want %g", tc.bits, got, tc.want)
		}
	}
	if !math.IsNaN(float64(DecodeE4M3(0x7F))) {
		t.Fatal("DecodeE4M3(7f) must be NaN")
	}
}
