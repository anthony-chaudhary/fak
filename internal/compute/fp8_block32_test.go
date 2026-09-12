package compute

import (
	"math"
	"testing"
)

// fp8_block32_test.go - the witnessed oracle for the V4.1 32x32 FP8 block-scale decode.
// The oracle below re-derives E4M3 (sign/exponent/mantissa arithmetic) and E8M0 (2^(b-127))
// independently of the production decoder, so a drift in either cannot pass silently.

// oracleE4M3 re-derives one OCP float8_e4m3fn byte arithmetically: 1 sign / 4 exponent
// (bias 7) / 3 mantissa; the fn variant has no infinities and S.1111.111 is NaN.
func oracleE4M3(t *testing.T, b byte) float32 {
	t.Helper()
	sign := float32(1)
	if b&0x80 != 0 {
		sign = -1
	}
	exp := int((b >> 3) & 0x0f)
	man := int(b & 0x07)
	if exp == 0x0f && man == 0x07 {
		return float32(math.NaN())
	}
	var mag float32
	if exp == 0 {
		mag = float32(math.Ldexp(float64(man), -9))
	} else {
		mag = float32(math.Ldexp(1+float64(man)/8, exp-7))
	}
	return sign * mag
}

// oracleE8M0 re-derives a raw F8_E8M0 exponent byte as 2^(b-127).
func oracleE8M0(b byte) float32 {
	return float32(math.Ldexp(1, int(b)-127))
}

func oracleBlock32(t *testing.T, O, I int, weight []byte, scales []byte) []float32 {
	t.Helper()
	sI := (I + FP8Block32Dim - 1) / FP8Block32Dim
	out := make([]float32, O*I)
	for o := 0; o < O; o++ {
		for i := 0; i < I; i++ {
			s := oracleE8M0(scales[(o/FP8Block32Dim)*sI+i/FP8Block32Dim])
			out[o*I+i] = oracleE4M3(t, weight[o*I+i]) * s
		}
	}
	return out
}

func bitEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len(got)=%d want %d", label, len(got), len(want))
	}
	for k := range got {
		if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
			t.Fatalf("%s: [%d] bits %#08x != want %#08x (%v vs %v)", label, k, math.Float32bits(got[k]), math.Float32bits(want[k]), got[k], want[k])
		}
	}
}

// TestV41FP8Block32DecodeSemantics decodes O=64,I=96 (two full row tiles, three column
// tiles, exercising a ragged edge in I and multiple 32x32 tiles) and asserts exact
// Float32bits against the independent arithmetic oracle.
func TestV41FP8Block32DecodeSemantics(t *testing.T) {
	O, I := 64, 96
	sO, sI := (O+31)/32, (I+31)/32
	weight := make([]byte, O*I)
	for k := range weight {
		weight[k] = byte((k*7 + 3) & 0x7e)
	}
	weight[5] = 0x38
	weight[40] = 0xb8
	weight[95] = 0x30
	scales := make([]byte, sO*sI)
	for k := range scales {
		scales[k] = byte(120 + k)
	}

	got, err := DecodeFP8Block32E8M0("sem", O, I, weight, scales)
	if err != nil {
		t.Fatalf("DecodeFP8Block32E8M0: %v", err)
	}
	want := oracleBlock32(t, O, I, weight, scales)
	bitEqual(t, "decode-e8m0", got, want)

	f32scales := make([]float32, len(scales))
	for k, b := range scales {
		f32scales[k] = oracleE8M0(b)
	}
	gotF32, err := DecodeFP8Block32("sem-f32", O, I, weight, f32scales)
	if err != nil {
		t.Fatalf("DecodeFP8Block32: %v", err)
	}
	bitEqual(t, "decode-f32", gotF32, want)

	if v := got[5*I+5]; math.Float32bits(v) != math.Float32bits(want[5*I+5]) {
		t.Fatalf("anchor tile crossing: %v != %v", v, want[5*I+5])
	}
}

// TestV41FP8Block32MalformedFailsBeforeAllocation checks every failure mode returns an
// error and a nil output slice.
func TestV41FP8Block32MalformedFailsBeforeAllocation(t *testing.T) {
	ok := make([]byte, 32*32)
	for i := range ok {
		ok[i] = 0x38
	}
	okScale := []byte{127}

	cases := []struct {
		name string
		call func() ([]float32, error)
	}{
		{"zero-dim", func() ([]float32, error) { return DecodeFP8Block32E8M0("m", 0, 32, nil, nil) }},
		{"neg-dim", func() ([]float32, error) { return DecodeFP8Block32E8M0("m", -1, 32, nil, nil) }},
		{"weight-len", func() ([]float32, error) { return DecodeFP8Block32E8M0("m", 32, 32, ok[:100], okScale) }},
		{"scale-len", func() ([]float32, error) { return DecodeFP8Block32E8M0("m", 32, 32, ok, []byte{127, 128}) }},
		{"e8m0-nan", func() ([]float32, error) { return DecodeFP8Block32E8M0("m", 32, 32, ok, []byte{0xff}) }},
		{"e4m3-nan", func() ([]float32, error) {
			w := make([]byte, 32*32)
			copy(w, ok)
			w[10] = 0x7f
			return DecodeFP8Block32E8M0("m", 32, 32, w, okScale)
		}},
		{"f32-weight-len", func() ([]float32, error) { return DecodeFP8Block32("m", 32, 32, ok[:10], []float32{1}) }},
		{"f32-scale-len", func() ([]float32, error) { return DecodeFP8Block32("m", 32, 32, ok, []float32{1, 2}) }},
		{"gemv-x-len", func() ([]float32, error) { return GemvFP8Block32E8M0("m", 32, 32, ok, okScale, make([]float32, 31)) }},
	}
	for _, tc := range cases {
		out, err := tc.call()
		if err == nil {
			t.Fatalf("%s: expected error, got nil", tc.name)
		}
		if out != nil {
			t.Fatalf("%s: expected nil output, got len %d", tc.name, len(out))
		}
	}
}

// TestV41FP8Block32GemvMatchesScalarDecode proves the inline-decode GEMV equals the scalar
// decoded matrix times x within float tolerance on a reduced fixture.
func TestV41FP8Block32GemvMatchesScalarDecode(t *testing.T) {
	O, I := 64, 64
	weight := make([]byte, O*I)
	for k := range weight {
		weight[k] = byte((k*5 + 1) & 0x7e)
	}
	scales := []byte{126, 128, 130, 125}
	x := make([]float32, I)
	for i := range x {
		x[i] = float32(i%7) - 3
	}

	mat, err := DecodeFP8Block32E8M0("gemv", O, I, weight, scales)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	want := make([]float32, O)
	for o := 0; o < O; o++ {
		var acc float32
		for i := 0; i < I; i++ {
			acc += mat[o*I+i] * x[i]
		}
		want[o] = acc
	}

	got, err := GemvFP8Block32E8M0("gemv", O, I, weight, scales, x)
	if err != nil {
		t.Fatalf("gemv: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for o := range got {
		if diff := math.Abs(float64(got[o] - want[o])); diff > 1e-4 {
			t.Fatalf("y[%d]=%v want %v diff %v", o, got[o], want[o], diff)
		}
	}
}
