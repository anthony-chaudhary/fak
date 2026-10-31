package model

import (
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"math"
	"testing"
)

func TestV41KVNeutralDecoderAll256BitIdentical(t *testing.T) {
	for code := 0; code < 256; code++ {
		got, want := mathx.DecodeE4M3(byte(code)), FP8E4M3ToF32(byte(code))
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("code=%02x neutral=%08x production=%08x", code, math.Float32bits(got), math.Float32bits(want))
		}
		if math.IsInf(float64(got), 0) {
			t.Fatalf("E4M3FN has no infinity: %02x", code)
		}
		if (code&0x7f == 0x7f) != math.IsNaN(float64(got)) {
			t.Fatalf("NaN policy: %02x", code)
		}
	}
}
