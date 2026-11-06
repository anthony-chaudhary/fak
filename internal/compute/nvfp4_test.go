package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4Dtype(t *testing.T) {
	ids := []struct {
		name string
		got  Dtype
		want Dtype
	}{
		{"FP8", FP8, 6},
		{"FP4", FP4, 16},
		{"IQ1_S", IQ1_S, 20},
		{"NVFP4", NVFP4, 21},
	}
	for _, id := range ids {
		if id.got != id.want {
			t.Fatalf("%s ID = %d, want frozen ID %d", id.name, id.got, id.want)
		}
	}
	if got := NVFP4.String(); got != "nvfp4" {
		t.Fatalf("NVFP4.String() = %q, want nvfp4", got)
	}
	if !NVFP4.Quantized() {
		t.Fatal("NVFP4.Quantized() = false")
	}
	if FP4.String() != "fp4" || !FP4.Quantized() {
		t.Fatal("existing FP4 dtype contract changed")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4Reference(t *testing.T) {
	t.Run("all E2M1 codes and nibble order", func(t *testing.T) {
		packed := make([]byte, 8)
		for i := range packed {
			packed[i] = byte(2*i) | byte(2*i+1)<<4
		}
		got, err := DecodeNVFP4Reference(1, 16, packed, []byte{0x38}, 1.5)
		if err != nil {
			t.Fatal(err)
		}
		for code := 0; code < 16; code++ {
			want := testNVFP4E2M1(byte(code)) * 1.5
			if got[code] != want {
				t.Fatalf("code %#x = %v, want %v", code, got[code], want)
			}
		}
	})

	t.Run("row and block boundaries", func(t *testing.T) {
		const rows, cols = 2, 32
		packed := make([]byte, rows*cols/2)
		for i := range packed {
			packed[i] = 0x32 // low=1, high=1.5
		}
		scales := []byte{0x38, 0x40, 0x30, 0x48} // 1, 2, 0.5, 4
		x := make([]float32, cols)
		for k := range x {
			x[k] = float32(k%5 - 2)
		}

		got, err := GemvNVFP4Reference(rows, cols, packed, scales, 0.25, x)
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < rows; row++ {
			var want float32
			for k := 0; k < cols; k++ {
				code := packed[(row*cols+k)/2] & 0x0f
				if k%2 == 1 {
					code = packed[(row*cols+k)/2] >> 4
				}
				weight := testNVFP4E2M1(code) * testNVFP4E4M3(scales[row*2+k/16])
				weight *= 0.25
				want += weight * x[k]
			}
			if got[row] != want {
				t.Fatalf("row %d = %v, want %v", row, got[row], want)
			}
		}
	})

	t.Run("zero scales", func(t *testing.T) {
		packed := []byte{0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77}
		got, err := GemvNVFP4Reference(1, 16, packed, []byte{0}, 0, make([]float32, 16))
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || got[0] != 0 {
			t.Fatalf("zero-scaled GEMV = %v, want [0]", got)
		}
	})

	t.Run("code precedes scale multiplication", func(t *testing.T) {
		packed := make([]byte, 8)
		packed[0] = 0x01 // low nibble 0.5; all other codes are zero
		global := float32(1e36)
		got, err := DecodeNVFP4Reference(1, 16, packed, []byte{0x7e}, global)
		if err != nil {
			t.Fatal(err)
		}
		want := float32(0.5) * float32(448)
		want *= global
		if math.IsInf(float64(got[0]), 0) || got[0] != want {
			t.Fatalf("ordered scaled value = %v, want finite %v", got[0], want)
		}
	})

	t.Run("closed error contracts", func(t *testing.T) {
		validPacked := make([]byte, 8)
		maxPacked := []byte{0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77}
		validScales := []byte{0x38}
		validX := make([]float32, 16)
		maxX := []float32{
			math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32,
			math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32,
			math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32,
			math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32,
		}
		maxInt := int(^uint(0) >> 1)
		cases := []struct {
			name   string
			rows   int
			cols   int
			packed []byte
			scales []byte
			global float32
			x      []float32
			want   error
		}{
			{"zero rows", 0, 16, validPacked, validScales, 1, validX, ErrNVFP4Shape},
			{"partial block", 1, 15, validPacked, validScales, 1, validX, ErrNVFP4Shape},
			{"shape overflow", maxInt, 16, nil, nil, 1, validX, ErrNVFP4Shape},
			{"short packed", 1, 16, validPacked[:7], validScales, 1, validX, ErrNVFP4Shape},
			{"short scales", 1, 16, validPacked, nil, 1, validX, ErrNVFP4Scale},
			{"negative block scale", 1, 16, validPacked, []byte{0xb8}, 1, validX, ErrNVFP4Scale},
			{"nonfinite block scale", 1, 16, validPacked, []byte{0x7f}, 1, validX, ErrNVFP4Scale},
			{"negative global scale", 1, 16, validPacked, validScales, -1, validX, ErrNVFP4Scale},
			{"NaN global scale", 1, 16, validPacked, validScales, float32(math.NaN()), validX, ErrNVFP4Scale},
			{"nonfinite global scale", 1, 16, validPacked, validScales, float32(math.Inf(1)), validX, ErrNVFP4Scale},
			{"decoded scale overflow", 1, 16, maxPacked, validScales, math.MaxFloat32, validX, ErrNVFP4Scale},
			{"short vector", 1, 16, validPacked, validScales, 1, validX[:15], ErrNVFP4Input},
			{"nonfinite vector", 1, 16, validPacked, validScales, 1, append([]float32{float32(math.NaN())}, validX[1:]...), ErrNVFP4Input},
			{"vector product overflow", 1, 16, maxPacked, validScales, 1, maxX, ErrNVFP4Input},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := GemvNVFP4Reference(tc.rows, tc.cols, tc.packed, tc.scales, tc.global, tc.x)
				if !errors.Is(err, tc.want) {
					t.Fatalf("error = %v, want errors.Is(_, %v)", err, tc.want)
				}
			})
		}
	})
}

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4SpecialValuesAndMismatch(t *testing.T) {
	t.Run("per-stack globals against hand-decoded goldens", func(t *testing.T) {
		// Two experts of [1, 32]: two scale groups per row; byte 0 holds columns 0 (low) and 1 (high).
		codes := make([]byte, 32)
		codes[0] = 0xF2                          // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
		codes[8] = 0x07                          // column 16: e2m1 7 -> 6
		codes[16+15] = 0x30                      // expert 1, column 31: e2m1 3 -> 1.5
		scales := []byte{0x38, 0x40, 0x38, 0x30} // 1, 2 | 1, 0.5
		globals := []float32{0.25, 2}

		got, err := DecodeNVFP4Stacks(2, 1, 32, codes, scales, globals)
		if err != nil {
			t.Fatal(err)
		}
		if wantValues := len(codes) * 2; len(got) != wantValues {
			t.Fatalf("decoded %d values, want %d from %d packed bytes", len(got), wantValues, len(codes))
		}
		want := map[int]float32{
			0:       0.25, // 1.0 * 1 * 0.25
			1:       -1.5, // -6 * 1 * 0.25
			16:      3,    // 6 * 2 * 0.25
			32 + 31: 1.5,  // 1.5 * 0.5 * 2
		}
		for i, v := range got {
			if w, ok := want[i]; ok && v != w {
				t.Fatalf("value %d = %v, want golden %v", i, v, w)
			}
		}
		for i, v := range got {
			if _, ok := want[i]; ok {
				continue
			}
			if v != 0 {
				t.Fatalf("value %d = %v, want zero code", i, v)
			}
		}
	})

	t.Run("subnormal and maximal e4m3 scales decode exactly", func(t *testing.T) {
		packed := make([]byte, 8)
		packed[0] = 0x01 // column 0 (low nibble): e2m1 1 -> 0.5
		got, err := DecodeNVFP4Stacks(1, 1, 16, packed, []byte{0x01}, []float32{1})
		if err != nil {
			t.Fatal(err)
		}
		if want := float32(math.Ldexp(1, -10)); got[0] != want { // 0.5 * 2^-9
			t.Fatalf("subnormal-scale value = %v, want exact %v", got[0], want)
		}

		maxPacked := []byte{0x07, 0, 0, 0, 0, 0, 0, 0} // column 0: e2m1 7 -> 6
		got, err = DecodeNVFP4Stacks(1, 1, 16, maxPacked, []byte{0x7e}, []float32{1})
		if err != nil {
			t.Fatal(err)
		}
		if want := float32(6) * 448; got[0] != want { // largest finite e4m3 scale
			t.Fatalf("max-scale value = %v, want exact %v", got[0], want)
		}
	})

	t.Run("closed error contracts", func(t *testing.T) {
		validPacked := make([]byte, 16) // two stacks of [1, 16]
		validScales := []byte{0x38, 0x38}
		validGlobals := []float32{1, 1}
		maxInt := int(^uint(0) >> 1)
		cases := []struct {
			name    string
			stacks  int
			rows    int
			cols    int
			packed  []byte
			scales  []byte
			globals []float32
			wantErr error
		}{
			{"zero stacks", 0, 1, 16, validPacked, validScales, nil, ErrNVFP4Shape},
			{"zero rows", 2, 0, 16, validPacked, validScales, validGlobals, ErrNVFP4Shape},
			{"partial block", 2, 1, 15, validPacked, validScales, validGlobals, ErrNVFP4Shape},
			{"shape overflow", 2, maxInt, 16, nil, nil, validGlobals, ErrNVFP4Shape},
			{"short packed", 2, 1, 16, validPacked[:15], validScales, validGlobals, ErrNVFP4Shape},
			{"long packed", 2, 1, 16, append(validPacked, 0), validScales, validGlobals, ErrNVFP4Shape},
			{"short block scales", 2, 1, 16, validPacked, validScales[:1], validGlobals, ErrNVFP4Scale},
			{"NaN block scale", 2, 1, 16, validPacked, []byte{0x7f, 0x38}, validGlobals, ErrNVFP4Scale},
			{"signed NaN block scale", 2, 1, 16, validPacked, []byte{0xff, 0x38}, validGlobals, ErrNVFP4Scale},
			{"negative block scale", 2, 1, 16, validPacked, []byte{0xb8, 0x38}, validGlobals, ErrNVFP4Scale},
			{"fewer globals than stacks", 2, 1, 16, validPacked, validScales, validGlobals[:1], ErrNVFP4Shape},
			{"more globals than stacks", 1, 1, 16, validPacked[:8], validScales[:1], validGlobals, ErrNVFP4Shape},
			{"negative global", 2, 1, 16, validPacked, validScales, []float32{-1, 1}, ErrNVFP4Scale},
			{"NaN global", 2, 1, 16, validPacked, validScales, []float32{float32(math.NaN()), 1}, ErrNVFP4Scale},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := DecodeNVFP4Stacks(tc.stacks, tc.rows, tc.cols, tc.packed, tc.scales, tc.globals)
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("error = %v, want errors.Is(_, %v)", err, tc.wantErr)
				}
			})
		}
	})
}

func testNVFP4E2M1(code byte) float32 {
	sign := float32(1)
	if code&8 != 0 {
		sign = -1
	}
	exponent := (code >> 1) & 3
	mantissa := code & 1
	if exponent == 0 {
		return sign * float32(mantissa) * 0.5
	}
	return sign * float32(math.Ldexp(float64(2+mantissa), int(exponent)-2))
}

func testNVFP4E4M3(raw byte) float32 {
	sign := float32(1)
	if raw&0x80 != 0 {
		sign = -1
	}
	exponent := (raw >> 3) & 0x0f
	mantissa := raw & 7
	if exponent == 0 {
		return sign * float32(mantissa) / 512
	}
	return sign * float32(8+mantissa) * float32(math.Ldexp(1, int(exponent)-10))
}
