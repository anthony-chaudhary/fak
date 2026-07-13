package model

import (
	"math"
	"testing"
)

// TestFP8E4M3ToF32Golden pins the float8_e4m3fn decode byte-for-byte. Every e4m3
// value is exactly representable in f32, so these are equality assertions, not
// tolerances. The table covers ±0, the smallest subnormal, a mid subnormal, the
// normal/subnormal boundary, unit and negative values, the max finite (448), and the
// sole NaN pattern (0x7F / 0xFF).
func TestFP8E4M3ToF32Golden(t *testing.T) {
	cases := []struct {
		b    byte
		want float32
	}{
		{0x00, 0},           // +0
		{0x01, 0.001953125}, // smallest subnormal: 2^-9
		{0x04, 0.0078125},   // subnormal: 2^-6 * 4/8
		{0x07, 0.013671875}, // largest subnormal: 2^-6 * 7/8
		{0x08, 0.015625},    // smallest normal: 2^-6
		{0x30, 0.5},         // exp=6: 2^-1
		{0x38, 1.0},         // exp=7,man=0: 1.0
		{0x3A, 1.25},        // exp=7,man=2
		{0x3F, 1.875},       // exp=7,man=7
		{0x40, 2.0},         // exp=8
		{0x78, 256.0},       // exp=15,man=0
		{0x7E, 448.0},       // exp=15,man=6: max finite
		{0xB8, -1.0},        // sign,exp=7,man=0
	}
	for _, c := range cases {
		if got := fp8E4M3ToF32(c.b); got != c.want {
			t.Errorf("fp8E4M3ToF32(0x%02x) = %v, want %v", c.b, got, c.want)
		}
	}

	// Signed zero: 0x80 is -0, distinct from +0 only by the sign bit.
	if got := fp8E4M3ToF32(0x80); got != 0 || !math.Signbit(float64(got)) {
		t.Errorf("fp8E4M3ToF32(0x80) = %v, want -0", got)
	}
	// The only NaN pattern, both signs.
	for _, b := range []byte{0x7f, 0xff} {
		if got := fp8E4M3ToF32(b); !math.IsNaN(float64(got)) {
			t.Errorf("fp8E4M3ToF32(0x%02x) = %v, want NaN", b, got)
		}
	}
}

// TestDecodeFP8BlockScaleSingleBlock is a hand-golden 2x2 tensor whose whole extent
// fits one 128x128 block, so every element multiplies the same scale.
func TestDecodeFP8BlockScaleSingleBlock(t *testing.T) {
	// weight = [[1.0, 2.0], [0.5, -1.0]], scaleInv = [3.0].
	weight := []byte{0x38, 0x40, 0x30, 0xB8}
	got, err := decodeFP8BlockScale("t", 2, 2, weight, []float32{3.0})
	if err != nil {
		t.Fatalf("decodeFP8BlockScale: %v", err)
	}
	want := []float32{3.0, 6.0, 1.5, -3.0}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestDecodeFP8BlockScaleTiling exercises the 128x128 tiling and ragged-edge crop: a
// 130x130 all-ones (0x38) weight spans a 2x2 grid of scale blocks whose final row and
// column are only 2 wide/tall. Each element must pick up its own block's scale, which
// proves both the block-index math (o/128, i/128) and the crop of the short edge tiles.
func TestDecodeFP8BlockScaleTiling(t *testing.T) {
	const O, I = 130, 130
	weight := make([]byte, O*I)
	for i := range weight {
		weight[i] = 0x38 // 1.0, so dequant == the block scale
	}
	// scaleInv row-major [2,2] — one distinct scale per block so a mis-indexed
	// element lands on the wrong value and fails the assertion.
	scaleInv := []float32{10, 20, 30, 40}
	got, err := decodeFP8BlockScale("t", O, I, weight, scaleInv)
	if err != nil {
		t.Fatalf("decodeFP8BlockScale: %v", err)
	}
	for o := 0; o < O; o++ {
		for i := 0; i < I; i++ {
			want := scaleInv[(o/fp8BlockDim)*2+i/fp8BlockDim]
			if g := got[o*I+i]; g != want {
				t.Fatalf("out[%d,%d] = %v, want %v (block %d,%d)", o, i, g, want, o/fp8BlockDim, i/fp8BlockDim)
			}
		}
	}
}

// TestDecodeFP8BlockScaleFailClosed: a length or shape that disagrees with the tensor
// is a hard error, never a silent mis-load.
func TestDecodeFP8BlockScaleFailClosed(t *testing.T) {
	cases := []struct {
		name     string
		O, I     int
		weight   []byte
		scaleInv []float32
	}{
		{"non-positive O", 0, 4, nil, nil},
		{"non-positive I", 4, 0, nil, nil},
		{"weight too short", 2, 2, []byte{0x38, 0x38}, []float32{1}},
		{"weight too long", 2, 2, []byte{0x38, 0x38, 0x38, 0x38, 0x38}, []float32{1}},
		{"scaleInv wrong count", 2, 2, []byte{0x38, 0x38, 0x38, 0x38}, []float32{1, 2}},
		{"scaleInv empty", 130, 130, make([]byte, 130*130), []float32{1}},
	}
	for _, c := range cases {
		if _, err := decodeFP8BlockScale(c.name, c.O, c.I, c.weight, c.scaleInv); err == nil {
			t.Errorf("%s: want error, got nil", c.name)
		}
	}
}
