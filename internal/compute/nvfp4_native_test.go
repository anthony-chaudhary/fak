package compute

import (
	"errors"
	"math"
	"testing"
)

type nvfp4NativeFixture struct {
	w, a             Tensor
	wPacked, aPacked []byte
	wScales, aScales []byte
	wGlobal, aGlobal float32
}

func newNVFP4NativeFixture(t *testing.T) nvfp4NativeFixture {
	t.Helper()
	be := Default()
	f := nvfp4NativeFixture{
		wPacked: []byte{
			0x32, 0x54, 0x76, 0x21, 0x43, 0x65, 0x17, 0x25,
			0xa2, 0xc4, 0xe6, 0x91, 0xb3, 0xd5, 0x8f, 0x16,
			0x73, 0x25, 0x41, 0x56, 0x12, 0x67, 0x34, 0x52,
			0x47, 0x61, 0x23, 0x75, 0x54, 0x32, 0x16, 0x27,
		},
		aPacked: []byte{
			0x24, 0x53, 0x71, 0x36, 0x42, 0x15, 0x67, 0x32,
			0x42, 0x76, 0x13, 0x54, 0x21, 0x63, 0x35, 0x17,
			0x07, 0x31, 0x52, 0x64, 0x25, 0x43, 0x16, 0x72,
			0xa1, 0x35, 0x47, 0x62, 0x14, 0x56, 0x73, 0x21,
			0x32, 0x14, 0x65, 0x27, 0x53, 0x71, 0x46, 0x12,
			0xe7, 0x24, 0x51, 0x36, 0x72, 0x15, 0x43, 0x67,
		},
		wScales: []byte{0x38, 0x40, 0x30, 0x44},
		aScales: []byte{0x34, 0x3c, 0x40, 0x30, 0x38, 0x44},
		wGlobal: 0.1,
		aGlobal: 0.3,
	}
	var err error
	f.w, err = NewNVFP4(be, []int{2, 32}, f.wPacked, f.wScales, f.wGlobal)
	if err != nil {
		t.Fatalf("NewNVFP4(weight): %v", err)
	}
	f.a, err = NewNVFP4(be, []int{3, 32}, f.aPacked, f.aScales, f.aGlobal)
	if err != nil {
		t.Fatalf("NewNVFP4(activation): %v", err)
	}
	return f
}

func nvfp4NativeCode(packed []byte, row, cols, k int) byte {
	code := packed[(row*cols+k)/2]
	if k&1 == 0 {
		return code & 0x0f
	}
	return code >> 4
}

func nvfp4NativeLocal(packed, scales []byte, row, cols, k int) float32 {
	code := testNVFP4E2M1(nvfp4NativeCode(packed, row, cols, k))
	scale := testNVFP4E4M3(scales[row*(cols/NVFP4BlockSize)+k/NVFP4BlockSize])
	return code * scale
}

type nvfp4LegacyOnlyBackend struct {
	fakeDevice
	calls int
}

func (b *nvfp4LegacyOnlyBackend) MatMul(_, _ Tensor) Tensor {
	b.calls++
	return Tensor{}
}

// fak-test:runtime fast est=2ms lane=nvfp4-native-tests
func TestNVFP4NativeMatMulGolden(t *testing.T) {
	f := newNVFP4NativeFixture(t)
	gotTensor, err := MatMulNVFP4(Default(), f.w, f.a)
	if err != nil {
		t.Fatal(err)
	}
	if gotTensor.Dtype != F32 || gotTensor.Layout != RowMajor ||
		len(gotTensor.Shape) != 2 || gotTensor.Shape[0] != 3 || gotTensor.Shape[1] != 2 {
		t.Fatalf("output metadata = dtype %v layout %v shape %v, want F32 row-major [3 2]",
			gotTensor.Dtype, gotTensor.Layout, gotTensor.Shape)
	}
	got := Default().Read(gotTensor)
	if len(got) != 6 {
		t.Fatalf("output length = %d, want 6", len(got))
	}

	alpha := f.wGlobal * f.aGlobal
	earlyDiff := false
	for p := 0; p < 3; p++ {
		for o := 0; o < 2; o++ {
			var local, early float32
			for k := 0; k < 32; k++ {
				wv := nvfp4NativeLocal(f.wPacked, f.wScales, o, 32, k)
				av := nvfp4NativeLocal(f.aPacked, f.aScales, p, 32, k)
				product := wv * av
				local += product
				early += product * alpha
			}
			want := local * alpha
			index := p*2 + o
			if math.Float32bits(got[index]) != math.Float32bits(want) {
				t.Fatalf("output[%d,%d] = %g (%08x), want late-scaled %g (%08x)",
					p, o, got[index], math.Float32bits(got[index]), want, math.Float32bits(want))
			}
			if math.Float32bits(want) != math.Float32bits(early) {
				earlyDiff = true
			}
		}
	}
	if !earlyDiff {
		t.Fatal("fixture does not distinguish late global scaling from per-product scaling")
	}
}

// fak-test:runtime fast est=3ms lane=nvfp4-native-tests
func TestNVFP4NativeMatMulRejectsInvalidContracts(t *testing.T) {
	f := newNVFP4NativeFixture(t)
	for _, tc := range []struct {
		name string
		be   Backend
	}{
		{"nil backend", nil},
		{"backend without native capability", &nvfp4LegacyOnlyBackend{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := MatMulNVFP4(tc.be, f.w, f.a)
			if !errors.Is(err, ErrNVFP4GEMMUnsupported) {
				t.Fatalf("error = %v, want errors.Is(_, ErrNVFP4GEMMUnsupported)", err)
			}
			if out.buf != nil || out.be != nil || out.Shape != nil || out.Quant != nil {
				t.Fatalf("unsupported output = %#v, want zero Tensor", out)
			}
			if legacy, ok := tc.be.(*nvfp4LegacyOnlyBackend); ok && legacy.calls != 0 {
				t.Fatalf("legacy MatMul called %d times, want no fallback", legacy.calls)
			}
		})
	}

	cases := []struct {
		name string
		want error
		mut  func(*Tensor, *Tensor)
	}{
		{"weight dtype", ErrNVFP4Shape, func(w, _ *Tensor) { w.Dtype = F32 }},
		{"activation layout", ErrNVFP4Shape, func(_, a *Tensor) { a.Layout = ColMajor }},
		{"activation rank", ErrNVFP4Shape, func(_, a *Tensor) { a.Shape = []int{3, 1, 32} }},
		{"mismatched K", ErrNVFP4Shape, func(_, a *Tensor) { a.Shape = []int{3, 16} }},
		{"short payload", ErrNVFP4Shape, func(w, _ *Tensor) {
			buf := w.buf.(*hostBuf)
			buf.i8 = buf.i8[:len(buf.i8)-1]
		}},
		{"missing quant metadata", ErrNVFP4Scale, func(_, a *Tensor) { a.Quant = nil }},
		{"negative block scale", ErrNVFP4Scale, func(w, _ *Tensor) {
			w.buf.(*hostBuf).i8[len(f.wPacked)] = -72 // raw E4M3 0xb8
		}},
		{"nonfinite block scale", ErrNVFP4Scale, func(_, a *Tensor) {
			a.buf.(*hostBuf).i8[len(f.aPacked)] = 0x7f
		}},
		{"nonfinite alpha", ErrNVFP4Scale, func(w, _ *Tensor) {
			w.Quant.Scale[0] = float32(math.NaN())
		}},
		{"alpha overflow", ErrNVFP4Scale, func(w, a *Tensor) {
			w.Quant.Scale[0], a.Quant.Scale[0] = math.MaxFloat32, 2
		}},
		{"alpha underflow", ErrNVFP4Scale, func(w, a *Tensor) {
			w.Quant.Scale[0], a.Quant.Scale[0] = math.SmallestNonzeroFloat32, 0.5
		}},
		{"foreign activation owner", ErrNVFP4Input, func(_, a *Tensor) { a.be = fakeDevice{} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pair := newNVFP4NativeFixture(t)
			tc.mut(&pair.w, &pair.a)
			out, err := MatMulNVFP4(Default(), pair.w, pair.a)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want errors.Is(_, %v)", err, tc.want)
			}
			if out.buf != nil {
				t.Fatalf("error output has storage %T, want zero Tensor", out.buf)
			}
		})
	}
}
