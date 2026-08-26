//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

func TestQ8NoCopyOwnerAliasesAndSurvivesKernels(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ8()
	const out, in = 3, 64
	codes := make([]int8, out*in)
	scales := make([]float32, out*(in/32))
	for i := range codes {
		codes[i] = int8(i%17 - 8)
	}
	for i := range scales {
		scales[i] = float32(i+1) / 32
	}
	w, q, d := UploadQ8NoCopyOwned(codes, scales, out, in)
	if w == nil {
		t.Fatal("no-copy upload declined")
	}
	if len(q) != len(codes) || len(d) != len(scales) {
		t.Fatal("owner aliases have wrong dimensions")
	}
	for i := range codes {
		if q[i] != codes[i] {
			t.Fatalf("code alias mismatch at %d", i)
		}
	}
	for i := range scales {
		if d[i] != scales[i] {
			t.Fatalf("scale alias mismatch at %d", i)
		}
	}
	xq := make([]int8, in)
	xd := []float32{0.25, 0.5}
	for i := range xq {
		xq[i] = int8(i%11 - 5)
	}
	y := make([]float32, out)
	w.GEMV(xq, xd, y)
	for row := 0; row < out; row++ {
		var want float32
		for b := 0; b < in/32; b++ {
			var dot int32
			for i := 0; i < 32; i++ {
				dot += int32(q[row*in+b*32+i]) * int32(xq[b*32+i])
			}
			want += float32(dot) * d[row*(in/32)+b] * xd[b]
		}
		if math.Abs(float64(y[row]-want)) > 1e-3*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("row %d got %g want %g", row, y[row], want)
		}
	}
}
