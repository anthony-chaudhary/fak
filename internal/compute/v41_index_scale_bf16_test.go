package compute

import (
	"math"
	"testing"
)

// The host oracle separates binary32 multiplication from integer BF16 RNE.
// Faulted output bytes are never claimed as published NaN payloads.
func v41IndexScaleBF16Oracle(input, scale float32) (uint32, bool) {
	product := float32(input * scale)
	bits := math.Float32bits(product)
	rounded := (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
	return rounded, bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000
}

func TestV41IndexScaleBF16Oracle(t *testing.T) {
	scales := []float32{0, math.Float32frombits(0x80000000), 1, -1, 0.5, math.SmallestNonzeroFloat32, math.MaxFloat32, float32(1 / math.Sqrt(128*64))}
	for bits := uint32(0); bits < 65536; bits++ {
		for _, scale := range scales {
			input := math.Float32frombits(bits << 16)
			out, fault := v41IndexScaleBF16Oracle(input, scale)
			product := float32(float64(input) * float64(scale))
			p := math.Float32bits(product)
			want := (p + 0x7fff + ((p >> 16) & 1)) & 0xffff0000
			invalid := math.IsNaN(float64(product)) || math.IsInf(float64(product), 0) || math.IsInf(float64(math.Float32frombits(want)), 0)
			if fault != invalid || !fault && out != want {
				t.Fatalf("input=%08x scale=%08x got=%08x/%v want=%08x/%v", bits<<16, math.Float32bits(scale), out, fault, want, invalid)
			}
		}
	}
	for _, bits := range []uint32{1, 0x80000001, 0x007fffff, 0x00800000, 0x3f807fff, 0x3f808000, 0x3f808001, 0x7f7f7fff, 0x7f7f8000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001} {
		for _, scale := range scales {
			out, bad := v41IndexScaleBF16Oracle(math.Float32frombits(bits), scale)
			product := float32(float64(math.Float32frombits(bits)) * float64(scale))
			pb := math.Float32bits(product)
			want := (pb + 0x7fff + ((pb >> 16) & 1)) & 0xffff0000
			fault := pb&0x7f800000 == 0x7f800000 || want&0x7f800000 == 0x7f800000
			if bad != fault || !bad && out != want {
				t.Fatalf("edge %08x scale %08x", bits, math.Float32bits(scale))
			}
		}
	}
}
func TestV41IndexScaleBF16Geometry(t *testing.T) {
	good := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{64}}
	if n, e := validateV41ScaleExact(good, 1); e != nil || n != 256 {
		t.Fatal(n, e)
	}
	for _, n := range []int{0, -1} {
		bad := good
		bad.Shape = []int{n}
		if _, e := validateV41ScaleExact(bad, 1); e == nil {
			t.Fatal(n)
		}
	}
	for _, scale := range []float32{float32(math.Inf(1)), float32(math.NaN())} {
		if _, e := validateV41ScaleExact(good, scale); e == nil {
			t.Fatal("nonfinite scale")
		}
	}
}
