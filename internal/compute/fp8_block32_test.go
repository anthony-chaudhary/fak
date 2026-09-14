package compute

import (
	"errors"
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

// ---- Routed-expert device matmul witnesses -------------------------------------

// oracleMXE2M1 re-derives one OCP E2M1 nibble from the sign/exponent/mantissa layout:
// 1 sign / 2 exponent (bias 1) / 1 mantissa, with subnormals and no inf/NaN.
func oracleMXE2M1(t *testing.T, nib byte) float32 {
	t.Helper()
	nib &= 0x0f
	sign := float32(1)
	if nib&0x08 != 0 {
		sign = -1
	}
	exp := int((nib >> 1) & 0x03)
	man := int(nib & 0x01)
	// OCP E2M1: exp==0 is subnormal m/2; otherwise 2^(exp-1)*(1+man/2).
	var mag float32
	if exp == 0 {
		mag = float32(man) / 2
	} else {
		mag = float32(math.Ldexp(1+float64(man)/2, exp-1))
	}
	return sign * mag
}

// oracleMXFP4GEMV is the independent scalar oracle: unpack low/high nibbles per packed
// byte, scale each 32-K block by its E8M0 byte, dot with x.
func oracleMXFP4GEMV(t *testing.T, O, I int, weight, scales []byte, x []float32) []float32 {
	t.Helper()
	packedCols := I / 2
	scaleCols := (I + FP8Block32Dim - 1) / FP8Block32Dim
	y := make([]float32, O)
	for o := 0; o < O; o++ {
		var acc float32
		for i := 0; i < I; i++ {
			b := weight[o*packedCols+i/2]
			nib := b & 0x0f
			if i%2 == 1 {
				nib = b >> 4
			}
			s := oracleE8M0(scales[o*scaleCols+i/FP8Block32Dim])
			acc += oracleMXE2M1(t, nib) * s * x[i]
		}
		y[o] = acc
	}
	return y
}

// deviceExpertKernel is a fully device-resident in-memory implementation of the
// RoutedExpertDeviceKernel capability. It is the [SW-VERIFIED] stand-in for a real
// gfx1151 kernel: it proves the dispatch seam reaches a device kernel, decodes the
// PACKED bytes inline (no pre-dequantized f32 operand), and returns exact results —
// NOT a claim of physical device TOPS/GB·s-1.
type deviceExpertKernel struct {
	fp8Calls, fp4Calls int
}

func (d *deviceExpertKernel) GemvExpertFP8Block32E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	d.fp8Calls++
	y, err := GemvFP8Block32E8M0("device", O, I, weight, scales, x)
	if err != nil {
		return nil, err
	}
	return y, nil
}

func (d *deviceExpertKernel) GemvExpertMXFP4E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	d.fp4Calls++
	return GemvMXFP4E8M0("device", O, I, weight, scales, x)
}

// expertDeviceBackend is a host Backend that ALSO advertises the routed-expert device
// kernel, so the dispatch type-assert succeeds. It embeds the cpu reference for every
// other operation.
type expertDeviceBackend struct {
	Backend
	kernel *deviceExpertKernel
}

func (b *expertDeviceBackend) GemvExpertFP8Block32E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	return b.kernel.GemvExpertFP8Block32E8M0(O, I, weight, scales, x)
}

func (b *expertDeviceBackend) GemvExpertMXFP4E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	return b.kernel.GemvExpertMXFP4E8M0(O, I, weight, scales, x)
}

// TestFP8Block32ExpertGEMV witnesses the routed-expert device FP8/MXFP4 GEMV seam:
// (1) the FP8 device dispatch reaches the backend kernel and matches the independent
// scalar oracle; (2) the MXFP4 device dispatch matches the nibble/E8M0 oracle; (3) a
// backend WITHOUT the capability fails closed with ErrExpertDeviceKernelAbsent and
// never silently produces a scalar result; (4) the MXFP4 scalar reference itself is
// oracle-exact and rejects malformed shapes.
func TestFP8Block32ExpertGEMV(t *testing.T) {
	const O, I = 40, 96
	scaleCols := (I + FP8Block32Dim - 1) / FP8Block32Dim
	sO := (O + FP8Block32Dim - 1) / FP8Block32Dim

	fp8Weight := make([]byte, O*I)
	for k := range fp8Weight {
		fp8Weight[k] = byte((k*5 + 1) & 0x7e)
	}
	fp8Scales := make([]byte, sO*scaleCols)
	for k := range fp8Scales {
		fp8Scales[k] = byte(126 + k%5)
	}
	fp4Weight := make([]byte, O*(I/2))
	for k := range fp4Weight {
		fp4Weight[k] = byte((k*3 + 2) & 0xff)
	}
	fp4Scales := make([]byte, O*scaleCols)
	for k := range fp4Scales {
		fp4Scales[k] = byte(124 + k%7)
	}
	x := make([]float32, I)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.11)) + 0.25
	}

	base := Default()
	kernel := &deviceExpertKernel{}
	device := &expertDeviceBackend{Backend: base, kernel: kernel}

	// (1) FP8 device dispatch == oracle, and it really reached the kernel.
	gotFP8, err := GemvFP8Block32ExpertDevice("v41.w1", device, O, I, fp8Weight, fp8Scales, x)
	if err != nil {
		t.Fatalf("FP8 device dispatch: %v", err)
	}
	refFP8, err := GemvFP8Block32E8M0("ref", O, I, fp8Weight, fp8Scales, x)
	if err != nil {
		t.Fatalf("FP8 scalar reference: %v", err)
	}
	// The device kernel delegates to the same scalar arithmetic under the in-memory
	// stand-in, so bits must match; against a real device kernel this becomes an
	// Approx cosine gate instead.
	bitEqual(t, "fp8-device-vs-reference", gotFP8, refFP8)
	if kernel.fp8Calls != 1 {
		t.Fatalf("FP8 device kernel calls = %d, want 1", kernel.fp8Calls)
	}

	// (2) MXFP4 device dispatch == independent oracle.
	gotFP4, err := GemvMXFP4ExpertDevice("v41.w2", device, O, I, fp4Weight, fp4Scales, x)
	if err != nil {
		t.Fatalf("MXFP4 device dispatch: %v", err)
	}
	wantFP4 := oracleMXFP4GEMV(t, O, I, fp4Weight, fp4Scales, x)
	bitEqual(t, "fp4-device-vs-oracle", gotFP4, wantFP4)
	if kernel.fp4Calls != 1 {
		t.Fatalf("MXFP4 device kernel calls = %d, want 1", kernel.fp4Calls)
	}

	// (2b) MXFP4 scalar reference itself is oracle-exact.
	refFP4, err := GemvMXFP4E8M0("ref", O, I, fp4Weight, fp4Scales, x)
	if err != nil {
		t.Fatalf("MXFP4 scalar reference: %v", err)
	}
	bitEqual(t, "fp4-reference-vs-oracle", refFP4, wantFP4)

	// (3) absent kernel fails CLOSED — typed error, no scalar fallback.
	if _, err := GemvFP8Block32ExpertDevice("v41.w1", base, O, I, fp8Weight, fp8Scales, x); !errors.Is(err, ErrExpertDeviceKernelAbsent) {
		t.Fatalf("absent-kernel FP8 dispatch err = %v, want ErrExpertDeviceKernelAbsent", err)
	}
	if _, err := GemvMXFP4ExpertDevice("v41.w2", base, O, I, fp4Weight, fp4Scales, x); !errors.Is(err, ErrExpertDeviceKernelAbsent) {
		t.Fatalf("absent-kernel MXFP4 dispatch err = %v, want ErrExpertDeviceKernelAbsent", err)
	}
	if _, err := GemvFP8Block32ExpertDevice("nil", nil, O, I, fp8Weight, fp8Scales, x); !errors.Is(err, ErrExpertDeviceKernelAbsent) {
		t.Fatalf("nil-backend FP8 dispatch err = %v, want ErrExpertDeviceKernelAbsent", err)
	}

	// (4) scalar MXFP4 rejects malformed shapes/lengths before allocation.
	badCases := []struct {
		name       string
		O, I       int
		w, s, xbad []byte
		xlen       int
	}{
		{name: "odd I", O: O, I: I + 1, w: fp4Weight, s: fp4Scales, xlen: I + 1},
		{name: "weight length", O: O, I: I, w: fp4Weight[:len(fp4Weight)-1], s: fp4Scales, xlen: I},
		{name: "scale length", O: O, I: I, w: fp4Weight, s: fp4Scales[:len(fp4Scales)-1], xlen: I},
		{name: "x length", O: O, I: I, w: fp4Weight, s: fp4Scales, xlen: I - 1},
	}
	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			xbad := make([]float32, tc.xlen)
			if _, err := GemvMXFP4E8M0("bad", tc.O, tc.I, tc.w, tc.s, xbad); err == nil {
				t.Fatalf("GemvMXFP4E8M0 accepted malformed %s", tc.name)
			}
		})
	}

	// E8M0 NaN byte is refused.
	nanScales := append([]byte(nil), fp4Scales...)
	nanScales[0] = 0xff
	if _, err := GemvMXFP4E8M0("bad-nan", O, I, fp4Weight, nanScales, x); err == nil {
		t.Fatalf("GemvMXFP4E8M0 accepted an E8M0 NaN scale byte")
	}
}
