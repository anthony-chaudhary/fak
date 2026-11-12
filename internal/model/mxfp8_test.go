package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// MXFP8 (OCP microscaling) decode/encode witnesses. The layout under test is
// the one NVIDIA ModelOpt emits for nvidia/Kimi-K3.1-Pro-NVFP4's attention and
// shared-expert tensors: E4M3 data bytes with one E8M0 scale byte per
// 32-element block along the last dim, out[i] = e4m3(data[i]) * 2^(scale-127).
// The E4M3 element decode is the package's golden-pinned fp8E4M3ToF32, so these
// tests pin the block geometry, the E8M0 scale rule, and the encoder's agreement
// with that decoder — not the E4M3 table itself.
//
// Hand-picked finite codes used below: 0x00=+0, 0x80=-0, 0x01=2^-9 (min
// subnormal), 0x07=7*2^-9 (max subnormal), 0x08=2^-6, 0x30=0.5, 0x38=1.0,
// 0x44=3.0, 0x48=4.0, 0x78=256, 0x7C=384, 0x7E=448, 0xFE=-448, 0xB8=-1.0,
// 0xC0=-2.0.

func TestMXFP8DecodeHandVector(t *testing.T) {
	// One row of 32: four real codes then zeros. Scale byte 127 -> scale 1.0.
	data := make([]byte, 32)
	copy(data, []byte{0x38, 0xB8, 0x40, 0x30}) // 1, -1, 2, 0.5
	got, err := DecodeMXFP8ToF32("hand", []int{1, 32}, data, []byte{127})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	want := make([]float32, 32)
	want[0], want[1], want[2], want[3] = 1, -1, 2, 0.5
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("scale 2^0: got[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Byte 129 -> 2^2 = 4.0; byte 125 -> 2^-2 = 0.25; byte 140 -> 2^13.
	// Every product is exact.
	for _, tc := range []struct {
		scale byte
		want  [4]float32
	}{
		{129, [4]float32{4, -4, 8, 2}},
		{125, [4]float32{0.25, -0.25, 0.5, 0.125}},
		{140, [4]float32{8192, -8192, 16384, 4096}}, // 2^13
	} {
		got, err := DecodeMXFP8ToF32("hand", []int{1, 32}, data, []byte{tc.scale})
		if err != nil {
			t.Fatalf("scale byte %d: %v", tc.scale, err)
		}
		for j, w := range tc.want {
			if got[j] != w {
				t.Fatalf("scale byte %d: got[%d] = %v, want %v", tc.scale, j, got[j], w)
			}
		}
	}
}

func TestMXFP8DecodeBlockBoundaries(t *testing.T) {
	// Two rows, two different scale bytes: block indexing must not bleed
	// across the 32-element row boundary.
	data := make([]byte, 64)
	for i := range data {
		data[i] = 0x38 // 1.0
	}
	got, err := DecodeMXFP8ToF32("rows", []int{2, 32}, data, []byte{127, 130})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i := 0; i < 32; i++ {
		if got[i] != 1 {
			t.Fatalf("row 0 [%d] = %v, want 1 (scale byte 127)", i, got[i])
		}
		if got[32+i] != 8 {
			t.Fatalf("row 1 [%d] = %v, want 8 (scale byte 130 = 2^3)", i, got[32+i])
		}
	}

	// A block of 32 zeros decodes to zeros under any scale byte, including
	// the extreme E8M0 exponents.
	zeros := make([]byte, 32)
	for _, sb := range []byte{0, 127, 254} {
		got, err := DecodeMXFP8ToF32("zeros", []int{1, 32}, zeros, []byte{sb})
		if err != nil {
			t.Fatalf("zero block scale %d: %v", sb, err)
		}
		for i, v := range got {
			if v != 0 {
				t.Fatalf("zero block scale %d: got[%d] = %v, want 0", sb, i, v)
			}
		}
	}
}

func TestMXFP8RoundTripExact(t *testing.T) {
	// Exactness class: a value round-trips exactly iff its scaled magnitude is
	// exactly an E4M3 value. The block scale is a power of two and power-of-two
	// division is exact in f32, so every E4M3 value times an in-range power of
	// two round-trips bit-exactly (f32 and bf16); arbitrary f32 values only
	// round-trip to within E4M3 round-to-nearest.
	codes := []byte{0x00, 0x80, 0x01, 0x07, 0x08, 0x30, 0x38, 0x44, 0x48, 0x78, 0x7C, 0x7E, 0xFE, 0xB8, 0xC0}
	base := make([]float32, 32)
	for i := range base {
		base[i] = fp8E4M3ToF32(codes[i%len(codes)])
	}
	for _, tc := range []struct {
		name string
		mag  float32
		want byte
	}{
		{"s1", 1, 127},                // amax 448 -> s = 1
		{"s2^-10", 0.0009765625, 117}, // amax 0.4375 -> s = 2^-10
		{"s2^9", 512, 136},            // amax 448*2^9 -> s = 2^9
	} {
		n := len(base)
		values := make([]float32, n)
		for i, v := range base {
			values[i] = v * tc.mag
		}
		data, scales, scaleShape, err := QuantizeMXFP8ToBytes(tc.name, []int{1, n}, values)
		if err != nil {
			t.Fatalf("%s: quantize: %v", tc.name, err)
		}
		if len(scaleShape) != 2 || scaleShape[0] != 1 || scaleShape[1] != 1 {
			t.Fatalf("%s: scale shape %v, want [1 1]", tc.name, scaleShape)
		}
		if scales[0] != tc.want {
			t.Fatalf("%s: scale byte %d, want %d (smallest power of two fitting 448*%v)", tc.name, scales[0], tc.want, tc.mag)
		}
		for i := range data {
			if want := codes[i%len(codes)]; data[i] != want {
				t.Fatalf("%s: data[%d] = %#02x, want %#02x (bit-exact round trip)", tc.name, i, data[i], want)
			}
		}
		got, err := DecodeMXFP8ToF32(tc.name, []int{1, n}, data, scales)
		if err != nil {
			t.Fatalf("%s: decode: %v", tc.name, err)
		}
		for i := range values {
			if math.Float32bits(got[i]) != math.Float32bits(values[i]) {
				t.Fatalf("%s: decoded[%d] = %v, want %v", tc.name, i, got[i], values[i])
			}
		}
		// The bf16 form is bit-exact too: every E4M3 value and every power-of-two
		// scale is exactly representable in bf16's 8-bit significand.
		bf, _, err := DecodeMXFP8ToBF16Bytes(tc.name, []int{1, n}, data, scales)
		if err != nil {
			t.Fatalf("%s: bf16 decode: %v", tc.name, err)
		}
		for i, v := range values {
			if w := uint16(math.Float32bits(v) >> 16); binary.LittleEndian.Uint16(bf[i*2:]) != w {
				t.Fatalf("%s: bf16[%d] = %#04x, want %#04x", tc.name, i, binary.LittleEndian.Uint16(bf[i*2:]), w)
			}
		}
	}

	// RTNE tie and rounding behavior at scale 1.0: amax 1.0625 -> e = -8, so
	// the scaled values are 272 (halfway between 256 and 288 -> ties-to-even
	// picks 256) and 256 (exact).
	data, scales, _, err := QuantizeMXFP8ToBytes("rtne", []int{1, 2}, []float32{1.0625, 1.0})
	if err != nil {
		t.Fatalf("rtne: %v", err)
	}
	if scales[0] != 119 {
		t.Fatalf("rtne scale byte %d, want 119 (s = 2^-8)", scales[0])
	}
	if data[0] != 0x78 || data[1] != 0x78 {
		t.Fatalf("rtne data = %#02x %#02x, want 0x78 0x78 (272 ties to even 256)", data[0], data[1])
	}

	// Scale-byte rule spot checks: smallest power of two with amax/s <= 448.
	for _, tc := range []struct {
		amax float32
		want byte
	}{
		{448, 127},
		{449, 128}, // s = 2
		{0.25, 117},
		{500, 128},
	} {
		values := make([]float32, 32)
		values[0] = tc.amax
		_, scales, _, err := QuantizeMXFP8ToBytes("rule", []int{1, 32}, values)
		if err != nil {
			t.Fatalf("rule amax %v: %v", tc.amax, err)
		}
		if scales[0] != tc.want {
			t.Fatalf("rule amax %v: scale byte %d, want %d", tc.amax, scales[0], tc.want)
		}
	}

	// Zero block: neutral scale byte 127, all-zero codes.
	data, scales, _, err = QuantizeMXFP8ToBytes("zero", []int{1, 32}, make([]float32, 32))
	if err != nil {
		t.Fatalf("zero block: %v", err)
	}
	if scales[0] != 127 {
		t.Fatalf("zero block scale byte %d, want 127 (scale 1.0)", scales[0])
	}
	for i, b := range data {
		if b != 0 {
			t.Fatalf("zero block data[%d] = %#02x, want 0", i, b)
		}
	}
}

// TestMXFP8EncodeAgreesWithDecode is the exhaustive encoder/decoder agreement
// witness: for every finite E4M3 code, re-encoding its decoded value must
// reproduce the same byte (an exactly representable value rounds to itself,
// including -0 and the subnormal band). This is what makes it impossible for
// mxfp8EncodeE4M3 and fp8E4M3ToF32 to disagree about what a code means.
func TestMXFP8EncodeAgreesWithDecode(t *testing.T) {
	for c := 0; c < 256; c++ {
		b := byte(c)
		if b&0x7f == 0x7f {
			continue // 0x7f / 0xff are the NaN codes, never emitted
		}
		if got := mxfp8EncodeE4M3(fp8E4M3ToF32(b)); got != b {
			t.Fatalf("encode(decode(%#02x)) = %#02x", b, got)
		}
	}
}

func TestMXFP8TailRule(t *testing.T) {
	// Last dim 33: block 0 covers cols 0..31, block 1 covers col 32 alone —
	// its scale is computed over the 1 real element (the 31 zero pads
	// contribute no magnitude) and col 32 decodes with that scale.
	values := make([]float32, 33)
	values[0] = 1
	values[32] = 3
	data, scales, scaleShape, err := QuantizeMXFP8ToBytes("tail33", []int{1, 33}, values)
	if err != nil {
		t.Fatalf("tail33 quantize: %v", err)
	}
	if len(data) != 33 || len(scales) != 2 {
		t.Fatalf("tail33: data %d bytes, %d scales, want 33 and 2", len(data), len(scales))
	}
	if scaleShape[0] != 1 || scaleShape[1] != 2 {
		t.Fatalf("tail33 scale shape %v, want [1 2]", scaleShape)
	}
	if scales[0] != 119 { // amax 1 -> s = 2^-8 -> 256
		t.Fatalf("tail33 scales[0] = %d, want 119", scales[0])
	}
	if scales[1] != 120 { // amax 3 -> s = 2^-7 -> 384
		t.Fatalf("tail33 scales[1] = %d, want 120", scales[1])
	}
	got, err := DecodeMXFP8ToF32("tail33", []int{1, 33}, data, scales)
	if err != nil {
		t.Fatalf("tail33 decode: %v", err)
	}
	if got[0] != 1 || got[32] != 3 {
		t.Fatalf("tail33 decoded[0]=%v [32]=%v, want 1 and 3", got[0], got[32])
	}

	// Last dim 31: the whole row is one short block; one scale byte.
	values31 := make([]float32, 31)
	values31[30] = 0.25
	data31, scales31, shape31, err := QuantizeMXFP8ToBytes("tail31", []int{1, 31}, values31)
	if err != nil {
		t.Fatalf("tail31 quantize: %v", err)
	}
	if len(scales31) != 1 || shape31[1] != 1 {
		t.Fatalf("tail31: %d scales, shape %v, want 1 scale and [1 1]", len(scales31), shape31)
	}
	if scales31[0] != 117 { // amax 0.25 -> s = 2^-10 -> 256 (smallest fitting)
		t.Fatalf("tail31 scale byte %d, want 117", scales31[0])
	}
	got31, err := DecodeMXFP8ToF32("tail31", []int{1, 31}, data31, scales31)
	if err != nil {
		t.Fatalf("tail31 decode: %v", err)
	}
	if got31[30] != 0.25 {
		t.Fatalf("tail31 decoded[30] = %v, want 0.25", got31[30])
	}

	// Two ragged rows: row r owns scale blocks [r*2, r*2+2); a flat 32-step
	// would mix row 0's tail with row 1's head, so pin the isolation.
	values2 := make([]float32, 66)
	values2[32] = 3 // row 0, col 32 (short block)
	values2[65] = 6 // row 1, col 32 (short block)
	data2, scales2, _, err := QuantizeMXFP8ToBytes("tail2x33", []int{2, 33}, values2)
	if err != nil {
		t.Fatalf("tail2x33 quantize: %v", err)
	}
	if len(scales2) != 4 {
		t.Fatalf("tail2x33: %d scales, want 4", len(scales2))
	}
	if scales2[1] != 120 || scales2[3] != 121 { // 3 -> 2^-7; 6 -> 2^-6 (384)
		t.Fatalf("tail2x33 row-tail scales = %v, want [120 121] at [1 3]", scales2)
	}
	got2, err := DecodeMXFP8ToF32("tail2x33", []int{2, 33}, data2, scales2)
	if err != nil {
		t.Fatalf("tail2x33 decode: %v", err)
	}
	if got2[32] != 3 || got2[65] != 6 {
		t.Fatalf("tail2x33 decoded tails = %v %v, want 3 and 6", got2[32], got2[65])
	}

	// Raw decode with a tail: 33 data bytes, 2 scale bytes, col 32 scaled by
	// its own block byte.
	raw := make([]byte, 33)
	raw[0] = 0x38  // 1.0, scaled by block 0's byte
	raw[32] = 0x44 // 3.0, scaled by block 1's byte
	gotRaw, err := DecodeMXFP8ToF32("tailraw", []int{1, 33}, raw, []byte{129, 127})
	if err != nil {
		t.Fatalf("tailraw decode: %v", err)
	}
	if gotRaw[0] != 4 || gotRaw[32] != 3 {
		t.Fatalf("tailraw decoded[0]=%v [32]=%v, want 4 (2^2) and 3", gotRaw[0], gotRaw[32])
	}
}

// TestMXFP8CrossCheckFP8BlockScale pins the scale application against the
// package's existing 128x128 FP8 block-scale decode: with a power-of-two f32
// scaleInv = 2^-e, decodeFP8BlockScale's e4m3*scaleInv must equal the MXFP8
// decode at scale byte 127-e for every element (same E4M3 element decode, same
// multiply-by-power-of-two convention).
func TestMXFP8CrossCheckFP8BlockScale(t *testing.T) {
	const O, I = 2, 128
	weight := make([]byte, O*I)
	for i := range weight {
		weight[i] = []byte{0x38, 0x40, 0xB8, 0x30, 0x7E, 0x01, 0x44, 0x80}[i%8]
	}
	scaleInv := []float32{0.5} // one 128x128 tile: 2^-1
	want, err := decodeFP8BlockScale("cross", O, I, weight, scaleInv)
	if err != nil {
		t.Fatalf("fp8 block-scale decode: %v", err)
	}
	scales := make([]byte, O*(I/32))
	for i := range scales {
		scales[i] = 126 // 2^-1, matching scaleInv
	}
	got, err := DecodeMXFP8ToF32("cross", []int{O, I}, weight, scales)
	if err != nil {
		t.Fatalf("mxfp8 decode: %v", err)
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("cross[%d]: mxfp8 %v vs fp8-block-scale %v", i, got[i], want[i])
		}
	}
}

func TestMXFP8Refusals(t *testing.T) {
	data := make([]byte, 32)
	scales := []byte{127}
	if _, err := DecodeMXFP8ToF32("bad-data", []int{1, 32}, data[:31], scales); err == nil {
		t.Fatal("short data must be refused")
	}
	if _, err := DecodeMXFP8ToF32("bad-scales", []int{1, 32}, data, scales[:0]); err == nil {
		t.Fatal("missing scale must be refused")
	}
	if _, _, err := DecodeMXFP8ToBytes("nan-scale", []int{1, 32}, data, []byte{255}); err == nil {
		t.Fatal("E8M0 NaN scale byte 255 must be refused")
	}
	if _, err := DecodeMXFP8ToF32("empty", nil, nil, nil); err == nil {
		t.Fatal("empty shape must be refused")
	}
	if _, err := DecodeMXFP8ToF32("zero-dim", []int{1, 0}, nil, nil); err == nil {
		t.Fatal("non-positive dim must be refused")
	}
	if _, _, _, err := QuantizeMXFP8ToBytes("nan-in", []int{1, 32}, []float32{float32(math.NaN())}); err == nil {
		t.Fatal("NaN input must be refused")
	}
	if _, _, _, err := QuantizeMXFP8ToBytes("inf-in", []int{1, 32}, []float32{float32(math.Inf(1))}); err == nil {
		t.Fatal("Inf input must be refused")
	}
	if _, _, _, err := QuantizeMXFP8FromBF16("bad-bf16", []int{1, 32}, make([]byte, 63)); err == nil {
		t.Fatal("short bf16 source must be refused")
	}
}

// TestMXFP8BytesForms pins the two byte-form adapters: little-endian f32 out of
// DecodeMXFP8ToBytes (the safetensors-load form) and bf16 out of
// DecodeMXFP8ToBF16Bytes.
func TestMXFP8BytesForms(t *testing.T) {
	data := make([]byte, 32)
	copy(data, []byte{0x38, 0xB8, 0x40, 0x30})
	f32b, shape, err := DecodeMXFP8ToBytes("forms", []int{1, 32}, data, []byte{127})
	if err != nil {
		t.Fatalf("f32 bytes: %v", err)
	}
	if shape[0] != 1 || shape[1] != 32 || len(f32b) != 128 {
		t.Fatalf("f32 bytes: shape %v len %d, want [1 32] and 128", shape, len(f32b))
	}
	for i, want := range []uint32{
		math.Float32bits(1), math.Float32bits(-1), math.Float32bits(2), math.Float32bits(0.5),
	} {
		if got := binary.LittleEndian.Uint32(f32b[i*4:]); got != want {
			t.Fatalf("f32 bytes[%d] = %#08x, want %#08x", i, got, want)
		}
	}
	bf16b, _, err := DecodeMXFP8ToBF16Bytes("forms", []int{1, 32}, data, []byte{127})
	if err != nil {
		t.Fatalf("bf16 bytes: %v", err)
	}
	for i, want := range []uint16{0x3f80, 0xbf80, 0x4000, 0x3f00} {
		if got := binary.LittleEndian.Uint16(bf16b[i*2:]); got != want {
			t.Fatalf("bf16 bytes[%d] = %#04x, want %#04x", i, got, want)
		}
	}
}
