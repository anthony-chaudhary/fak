//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

// cuda_q8_rowtile_test.go is the device witness for the row-tiled prompt-panel Q8_0 GEMM
// (k_q8_gemm_rowtile behind PromptPanelMatMul): every output is bitwise the decode GEMV's
// (MatMul, k_q8_gemm at P=1) for ragged shapes and every tile tail.

// q8RowTileRowCounts covers one row, partial and full 8-row tiles, the 34-row Qwen2.5-3B chunk,
// the old P>=128 dequant boundary and the 512-row panel cap.
var q8RowTileRowCounts = []int{1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 34, 35, 63, 64, 65, 127, 128, 129, 255, 257, 511, 513, 599, 600}

// q8RowTileActivations fills rows x in with mixed magnitudes: every 11th row from 3 is all
// zero (activation scale 0), every 11th from 7 carries one large outlier.
func q8RowTileActivations(g *lcg, rows, in int) []float32 {
	x := make([]float32, rows*in)
	for r := 0; r < rows; r++ {
		row := x[r*in : (r+1)*in]
		switch r % 11 {
		case 3:
		case 7:
			copy(row, rscale(g, in, 64))
			row[(r*37)%in] = 4096
		default:
			copy(row, rscale(g, in, float32(1+r%5)))
		}
	}
	return x
}

// poisonNextTransient leaves a NaN-filled buffer of rows*out floats at the top of its pool size
// class, so the GEMM's output allocation starts as NaN and a row the kernel never writes fails.
func poisonNextTransient(cb *cudaBackend, rows, out int) {
	nan := make([]float32, rows*out)
	for i := range nan {
		nan[i] = float32(math.NaN())
	}
	cb.Free(mkResident(cb, []int{rows, out}, nan))
}

// fak-test:runtime integration est=8s lane=optin
func TestCUDAQ8PanelGEMMBitExactVsDecodeGEMV(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x9e3779b9
	g := &seed
	const maxRows = 600
	shapes := []struct{ out, in int }{
		{1, 32}, {9, 96}, {33, 1056}, {257, 4128}, {256, 2048}, {2048, 2048}, {11008, 2048}, {2048, 11008},
	}
	for _, sh := range shapes {
		t.Run(fmt.Sprintf("out%d_in%d", sh.out, sh.in), func(t *testing.T) {
			w := cb.Upload(NewF32(cb, []int{sh.out, sh.in}, rscale(g, sh.out*sh.in, 1.0)), Q8_0)
			defer cb.Free(w)
			x := q8RowTileActivations(g, maxRows, sh.in)
			want := make([]float32, maxRows*sh.out)
			for r := 0; r < maxRows; r++ {
				xr := mkResident(cb, []int{sh.in}, append([]float32(nil), x[r*sh.in:(r+1)*sh.in]...))
				y := cb.MatMul(w, xr)
				copy(want[r*sh.out:], cb.Read(y))
				cb.Free(y)
				cb.Free(xr)
			}
			for _, rows := range q8RowTileRowCounts {
				X := mkResident(cb, []int{rows, sh.in}, append([]float32(nil), x[:rows*sh.in]...))
				poisonNextTransient(cb, rows, sh.out)
				y := cb.PromptPanelMatMul(w, X, rows)
				got := cb.Read(y)
				cb.Free(y)
				cb.Free(X)
				if len(got) != rows*sh.out {
					t.Fatalf("rows=%d: %d outputs, want %d", rows, len(got), rows*sh.out)
				}
				for i, v := range got {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Fatalf("rows=%d row %d out %d: panel %v (%#08x), decode GEMV %v (%#08x)",
							rows, i/sh.out, i%sh.out, v, math.Float32bits(v), want[i], math.Float32bits(want[i]))
					}
				}
			}
		})
	}
}
