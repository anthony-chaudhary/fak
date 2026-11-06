package mathx

import (
	"math"
	"testing"
)

// TestDecodeE4M3Exhaustive pins every E4M3FN code used by the KV reference decoder.
// fak-test:runtime fast est=1ms lane=default
func TestDecodeE4M3Exhaustive(t *testing.T) {
	t.Parallel()
	// Positive finite codes 0x00..0x7e, in exact units of 1/512.
	// This literal table is independent of DecodeE4M3's Ldexp/exponent branches.
	// Every integer and the power-of-two division are exact in float32.
	units := [127]uint32{
		0, 1, 2, 3, 4, 5, 6, 7,
		8, 9, 10, 11, 12, 13, 14, 15,
		16, 18, 20, 22, 24, 26, 28, 30,
		32, 36, 40, 44, 48, 52, 56, 60,
		64, 72, 80, 88, 96, 104, 112, 120,
		128, 144, 160, 176, 192, 208, 224, 240,
		256, 288, 320, 352, 384, 416, 448, 480,
		512, 576, 640, 704, 768, 832, 896, 960,
		1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
		2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
		4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680,
		8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360,
		16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720,
		32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440,
		65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880,
		131072, 147456, 163840, 180224, 196608, 212992, 229376,
	}
	for code := 0; code < 256; code++ {
		got := DecodeE4M3(byte(code))
		magnitude := code & 0x7f
		if magnitude == 0x7f {
			// E4M3FN has two NaN codes and no infinity. Payload/sign are not promised.
			if !math.IsNaN(float64(got)) {
				t.Fatalf("DecodeE4M3(%02x) = %08x, want NaN", code, math.Float32bits(got))
			}
			continue
		}
		want := math.Float32bits(float32(units[magnitude]) / 512)
		want |= uint32(code&0x80) << 24
		if bits := math.Float32bits(got); bits != want {
			t.Fatalf("DecodeE4M3(%02x) = %08x, want %08x", code, bits, want)
		}
	}
}
