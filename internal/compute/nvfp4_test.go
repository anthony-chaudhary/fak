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
