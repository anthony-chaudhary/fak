//go:build cuda && cgo

package compute

import (
	"math"
	"strconv"
	"testing"
)

// TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef holds the resident IQ2_S / IQ3_XXS / IQ3_S device GEMV
// (MatMul P=1 and BatchedMatMul P=3) to an f64-accumulated dot over the host dequant
// (DequantRawIQ) of the SAME random super-block bytes. 2304x2304 is one Inkling-2-Mini
// gate_up expert slice; the odd shapes exercise the partial last 8-row block.
// fak-test:runtime fast est=3s lane=default
func TestCUDAIQ2SIQ3XXSIQ3SGemvMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	shapes := []struct{ out, in int }{{2304, 2304}, {321, 512}, {77, 1280}}
	for _, dt := range []Dtype{IQ2_S, IQ3_XXS, IQ3_S} {
		bb, ok := RawIQBlockBytes(dt)
		if !ok {
			t.Fatalf("%s is not a registered raw i-quant", dt)
		}
		for _, sh := range shapes {
			name := dt.String() + "/" + strconv.Itoa(sh.out) + "x" + strconv.Itoa(sh.in)
			raw := iqGEMVTestRaw(sh.out, sh.in, bb, uint64(sh.out*7919+sh.in)^uint64(dt))
			wf := DequantRawIQ(dt, sh.out, sh.in, raw)
			devW := cb.Upload(NewRawIQ(cb, dt, []int{sh.out, sh.in}, raw), dt)
			if devW.Dtype != dt {
				t.Fatalf("%s: resident dtype %s, want verbatim %s", name, devW.Dtype, dt)
			}
			for _, P := range []int{1, 3} {
				x := make([]float32, 0, P*sh.in)
				for tok := 0; tok < P; tok++ {
					x = append(x, iqGEMVTestX(sh.in, tok)...)
				}
				want := make([]float32, P*sh.out)
				for tok := 0; tok < P; tok++ {
					xs := x[tok*sh.in : (tok+1)*sh.in]
					for o := 0; o < sh.out; o++ {
						row := wf[o*sh.in : (o+1)*sh.in]
						var acc float64
						for i, w := range row {
							acc += float64(w) * float64(xs[i])
						}
						want[tok*sh.out+o] = float32(acc)
					}
				}
				xShape := []int{P, sh.in}
				if P == 1 {
					xShape = []int{sh.in}
				}
				devX := cb.Upload(NewF32(cb, xShape, x), F32)
				var devY Tensor
				if P == 1 {
					devY = cb.MatMul(devW, devX)
				} else {
					devY = cb.BatchedMatMul(devW, devX, P)
				}
				got := cb.Read(devY)
				cb.Free(devY)
				cb.Free(devX)
				if len(got) != len(want) {
					t.Fatalf("%s P=%d: len %d != %d", name, P, len(got), len(want))
				}
				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("%s P=%d: non-finite output[%d]=%g", name, P, i, v)
					}
				}
				if e := iqGEMVRelErr(got, want); e > 1e-4 {
					t.Errorf("%s P=%d: rel err %g > 1e-4", name, P, e)
				}
			}
			cb.Free(devW)
		}
	}
}
