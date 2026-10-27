//go:build amd64

package model

import (
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=200ms
func TestQ8SIMDTailsMatchCellReferenceWithoutMutatingInputs(t *testing.T) {
	if qtier < tierAVX2 {
		t.Skip("resolved tier does not admit SIMD tails")
	}
	oldMode := qgemmMode
	qgemmMode = qgemmModeTile
	t.Cleanup(func() { qgemmMode = oldMode })
	for _, wide := range []bool{false, true} {
		if wide && qtier != tierAVX512 {
			continue
		}
		lanes := 8
		if wide {
			lanes = 16
		}
		for _, shape := range []struct{ out, P int }{{1, 1}, {2, 3}, {7, 5}, {8, 7}, {17, 16}, {41, 19}} {
			const in = 576
			qt := quantizeQ8(mkVec(shape.out*in, uint64(shape.out+900)), shape.out, in)
			qp := quantizeBatchPanel(mkVec(shape.P*in, uint64(shape.P+1200)), shape.P, in)
			beforeW, beforeD := slices.Clone(qt.q), slices.Clone(qt.d)
			beforeX, beforeS := slices.Clone(qp.q), slices.Clone(qp.d)
			want := qGemm8scalar(qt, qp, lanes)
			storage := make([]float32, len(want)+2)
			storage[0], storage[len(storage)-1] = 12345, -6789
			got := storage[1 : len(storage)-1]
			for i := range got {
				got[i] = float32(math.NaN())
			}
			if wide {
				// Ordinary dispatcher plus IntoMany's shared remainder helper.
				qGemm8IntoMany(qp, qgemm8Target{qt: qt, Y: got})
			} else {
				qGemm8avx2Into(qt, qp, got)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("wide=%t out=%d P=%d cell%d=%g want%g", wide, shape.out, shape.P, i, got[i], want[i])
				}
			}
			if storage[0] != 12345 || storage[len(storage)-1] != -6789 {
				t.Fatal("padded tile escaped output boundary")
			}
			if !slices.Equal(qt.q, beforeW) || !slices.Equal(qt.d, beforeD) || !slices.Equal(qp.q, beforeX) || !slices.Equal(qp.d, beforeS) {
				t.Fatal("SIMD tail padding mutated caller-owned operands")
			}
		}
	}
	// Unsupported geometry must decline without writing any output.
	qt := &q8Tensor{out: 1, in: 0, nblk: 0}
	qp := &q8Panel{P: 1}
	y := []float32{123}
	if qGemm8SIMDTokenTail(qt, qp, y, 0, false) || y[0] != 123 {
		t.Fatal("unsupported token tail did not decline cleanly")
	}
}
