package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"testing"
)

// rowViewCase is one weight dtype for the RowView witnesses: build wraps rows
// [lo, hi) of a single random [out, in] payload as a fresh host tensor, so the
// whole-tensor view and the standalone slice read the same bytes.
type rowViewCase struct {
	dt    Dtype
	in    int
	build func(be Backend, lo, hi int) Tensor
}

// rowViewRawScaleOffsets are the f16 scale offsets inside one super-block (or,
// for IQ4_NL, inside each 18-byte block) pinned small so every value is finite.
//
//enumlint:exempt only the dtypes this test builds raw payloads for have f16 scale offsets to pin; any other dtype reads nil and is never looked up.
var rowViewRawScaleOffsets = map[Dtype][]int{
	Q3_K: {108}, Q4_K: {0, 2}, Q5_K: {0, 2}, Q6_K: {208},
	IQ2_XS: {0}, IQ2_S: {0}, IQ3_XXS: {0}, IQ3_S: {0}, IQ4_XS: {0},
}

func rowViewCases(t testing.TB, r *rand.Rand, out int) []rowViewCase {
	t.Helper()
	f16 := func() uint16 { return 0x2000 + uint16(r.Intn(0x0C00)) } // ~[0.008, 0.06]
	var cases []rowViewCase
	{
		const in = 96
		f := make([]float32, out*in)
		for i := range f {
			f[i] = float32(r.NormFloat64())
		}
		cases = append(cases, rowViewCase{F32, in, func(be Backend, lo, hi int) Tensor {
			return NewF32(be, []int{hi - lo, in}, append([]float32(nil), f[lo*in:hi*in]...))
		}})
	}
	{
		const in, blk = 128, 32
		codes := make([]int8, out*in)
		for i := range codes {
			codes[i] = int8(r.Intn(255) - 127)
		}
		scale := make([]float32, out*in/blk)
		for i := range scale {
			scale[i] = 0.001 + r.Float32()*0.01
		}
		cases = append(cases, rowViewCase{Q8_0, in, func(be Backend, lo, hi int) Tensor {
			n := in / blk
			return NewQ8(be, []int{hi - lo, in}, append([]int8(nil), codes[lo*in:hi*in]...),
				append([]float32(nil), scale[lo*n:hi*n]...), blk)
		}})
	}
	{
		const in = 64
		packed := make([]byte, out*in/2)
		r.Read(packed)
		scales := make([]byte, out*in/NVFP4BlockSize)
		for i := range scales {
			scales[i] = byte(0x20 + r.Intn(0x20)) // small finite E4M3FN
		}
		cases = append(cases, rowViewCase{NVFP4, in, func(be Backend, lo, hi int) Tensor {
			p, s := in/2, in/NVFP4BlockSize
			w, err := NewNVFP4(be, []int{hi - lo, in}, packed[lo*p:hi*p], scales[lo*s:hi*s], 1.5)
			if err != nil {
				t.Fatal(err)
			}
			return w
		}})
	}
	const in = 512
	for _, dt := range []Dtype{Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS} {
		rowBytes, _, reason := rowViewGeometry(Tensor{Dtype: dt, Shape: []int{out, in}})
		if reason != "" {
			t.Fatalf("%s: %s", dt, reason)
		}
		raw := make([]byte, out*rowBytes)
		r.Read(raw)
		if dt == IQ4_NL {
			for b := 0; b < len(raw); b += 18 {
				binary.LittleEndian.PutUint16(raw[b:], f16())
			}
		} else {
			bb := rowBytes / (in / q4kSuper)
			for b := 0; b < len(raw); b += bb {
				for _, o := range rowViewRawScaleOffsets[dt] {
					binary.LittleEndian.PutUint16(raw[b+o:], f16())
				}
			}
		}
		cases = append(cases, rowViewCase{dt, in, func(be Backend, lo, hi int) Tensor {
			part := append([]byte(nil), raw[lo*rowBytes:hi*rowBytes]...)
			shape := []int{hi - lo, in}
			switch dt {
			case Q3_K:
				return NewQ3K(be, shape, part)
			case Q4_K:
				return NewQ4K(be, shape, part)
			case Q5_K:
				return NewQ5K(be, shape, part)
			case Q6_K:
				return NewQ6K(be, shape, part)
			default:
				return NewRawIQ(be, dt, shape, part)
			}
		}})
	}
	return cases
}

func rowViewTestX(r *rand.Rand, in int) []float32 {
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

func TestRowViewMatVecEqualsStandaloneSlice(t *testing.T) {
	be := Default()
	r := rand.New(rand.NewSource(5))
	const out = 12
	for _, c := range rowViewCases(t, r, out) {
		whole := c.build(be, 0, out)
		x := NewF32(be, []int{c.in}, rowViewTestX(r, c.in))
		for _, span := range [][2]int{{0, 4}, {4, 8}, {8, 12}, {3, 5}, {0, out}} {
			lo, hi := span[0], span[1]
			view, err := RowView(be, whole, lo, hi-lo)
			if err != nil {
				t.Fatalf("%s [%d,%d): %v", c.dt, lo, hi, err)
			}
			if view.Dtype != c.dt || view.Shape[0] != hi-lo || view.Shape[1] != c.in {
				t.Fatalf("%s view is %s %v", c.dt, view.Dtype, view.Shape)
			}
			got := be.Read(be.MatMul(view, x))
			want := be.Read(be.MatMul(c.build(be, lo, hi), x))
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) || math.IsNaN(float64(want[i])) {
					t.Fatalf("%s [%d,%d) row %d: view %v, standalone %v", c.dt, lo, hi, i, got[i], want[i])
				}
			}
			be.Free(view)
		}
		// Freeing a view leaves the parent usable.
		got := be.Read(be.MatMul(whole, x))
		if len(got) != out {
			t.Fatalf("%s parent after view Free: %d rows", c.dt, len(got))
		}
	}
}

func TestRowViewRefusesBadRequests(t *testing.T) {
	be := Default()
	w := NewF32(be, []int{4, 8}, make([]float32, 32))
	var rv *RowViewError
	for _, c := range []struct {
		name           string
		w              Tensor
		rowStart, rows int
	}{
		{"negative start", w, -1, 2},
		{"zero rows", w, 0, 0},
		{"past end", w, 3, 2},
		{"rank one", NewF32(be, []int{8}, make([]float32, 8)), 0, 1},
		{"unsupported dtype", makeTensor(be, BF16, RowMajor, []int{4, 8}, nil, &hostBuf{i8: make([]int8, 64)}), 0, 1},
		{"Q4_K width", makeTensor(be, Q4_K, RowMajor, []int{4, 8}, nil, &hostBuf{i8: make([]int8, 64)}), 0, 1},
	} {
		if _, err := RowView(be, c.w, c.rowStart, c.rows); !errors.As(err, &rv) || rv.Backend != be.Name() {
			t.Fatalf("%s: err = %v, want *RowViewError naming %s", c.name, err, be.Name())
		}
	}
}
