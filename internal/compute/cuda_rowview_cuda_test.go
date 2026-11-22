//go:build cuda && cgo

package compute

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// TestCUDARowViewMatchesStandalone holds a device RowView of one uploaded
// [out, in] weight to a standalone upload of the same rows, for MatMul (P=1)
// and BatchedMatMul (P=3), and checks Free on a view leaves the parent intact.
// fak-test:runtime fast est=3s lane=default
func TestCUDARowViewMatchesStandalone(t *testing.T) {
	cb := cudaOrSkip(t)
	r := rand.New(rand.NewSource(9))
	const out = 48
	upload := func(h Tensor) Tensor { return cb.Upload(h, h.Dtype) }
	for _, c := range rowViewCases(t, r, out) {
		whole := upload(c.build(cb, 0, out))
		if (c.dt == Q5_K || c.dt == Q6_K) && cudaQ56KMMVQQualification {
			var rv *RowViewError
			if _, err := RowView(cb, whole, 16, 16); !errors.As(err, &rv) {
				t.Fatalf("%s: qualified MMVQ arm accepted a view (err %v)", c.dt, err)
			}
			cb.Free(whole)
			continue
		}
		x1 := rowViewTestX(r, c.in)
		x3 := append(append(rowViewTestX(r, c.in), rowViewTestX(r, c.in)...), rowViewTestX(r, c.in)...)
		dx1 := cb.Upload(NewF32(cb, []int{c.in}, x1), F32)
		dx3 := cb.Upload(NewF32(cb, []int{3, c.in}, x3), F32)
		before := cb.Read(cb.MatMul(whole, dx1))
		for _, span := range [][2]int{{0, 16}, {16, 32}, {32, 48}, {8, 40}} {
			lo, hi := span[0], span[1]
			view, err := RowView(cb, whole, lo, hi-lo)
			if err != nil {
				t.Fatalf("%s [%d,%d): %v", c.dt, lo, hi, err)
			}
			alone := upload(c.build(cb, lo, hi))
			for _, p := range []int{1, 3} {
				var got, want []float32
				if p == 1 {
					got, want = cb.Read(cb.MatMul(view, dx1)), cb.Read(cb.MatMul(alone, dx1))
				} else {
					got, want = cb.Read(cb.BatchedMatMul(view, dx3, 3)), cb.Read(cb.BatchedMatMul(alone, dx3, 3))
				}
				if len(got) != p*(hi-lo) || len(want) != len(got) {
					t.Fatalf("%s [%d,%d) P=%d: view %d outputs, standalone %d", c.dt, lo, hi, p, len(got), len(want))
				}
				for i := range want {
					d := math.Abs(float64(got[i] - want[i]))
					if math.IsNaN(float64(got[i])) || d > 1e-5*math.Max(1, math.Abs(float64(want[i]))) {
						t.Fatalf("%s [%d,%d) P=%d out %d: view %v, standalone %v", c.dt, lo, hi, p, i, got[i], want[i])
					}
				}
			}
			cb.Free(view)
			cb.Free(alone)
			cb.Recycle()
		}
		after := cb.Read(cb.MatMul(whole, dx1))
		for i := range before {
			if math.Float32bits(after[i]) != math.Float32bits(before[i]) {
				t.Fatalf("%s: parent row %d changed after view Free: %v -> %v", c.dt, i, before[i], after[i])
			}
		}
		cb.Free(dx1)
		cb.Free(dx3)
		cb.Free(whole)
		cb.Recycle()
	}
}

// TestCUDARowViewRefusesMisalignedRows: an IQ2_XS row is 74 bytes per 256
// weights, so row 1 starts off the 16-byte grid the kernels' loads assume.
func TestCUDARowViewRefusesMisalignedRows(t *testing.T) {
	cb := cudaOrSkip(t)
	bb, _ := RawIQBlockBytes(IQ2_XS)
	w := cb.Upload(NewRawIQ(cb, IQ2_XS, []int{4, 256}, make([]byte, 4*bb)), IQ2_XS)
	defer cb.Free(w)
	var rv *RowViewError
	if _, err := RowView(cb, w, 1, 2); !errors.As(err, &rv) {
		t.Fatalf("misaligned IQ2_XS view: err %v, want *RowViewError", err)
	}
	if _, err := RowView(cb, w, 0, 2); err != nil {
		t.Fatalf("aligned IQ2_XS view: %v", err)
	}
}
