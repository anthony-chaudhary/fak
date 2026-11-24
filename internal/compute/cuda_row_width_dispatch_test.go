//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

// Every width class must agree with an F64 reference, the auto selector must pick
// the documented class, and a narrow row must match the warp and block kernels.
// fak-test:runtime fast est=3s lane=default
func TestCUDARowKernelWidthDispatch(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	const eps = float32(1e-6)
	for _, width := range []int{1, 2, 3, 5, 8, 13, 16, 17, 32, 128, 1000, 1024, 1025, 5120} {
		for _, rows := range []int{1, 5, 37, 300} {
			t.Run(fmt.Sprintf("w%d_rows%d", width, rows), func(t *testing.T) {
				xh, wh := make([]float32, rows*width), make([]float32, width)
				for i := range xh {
					xh[i] = float32((i*7919)%257-128) / 37
				}
				for i := range wh {
					wh[i] = float32(i%19-9)/7 + 0.25
				}
				want := make([]float64, rows*width)
				for r := 0; r < rows; r++ {
					var ss float64
					for i := 0; i < width; i++ {
						v := float64(xh[r*width+i])
						ss += v * v
					}
					inv := 1 / math.Sqrt(ss/float64(width)+float64(eps))
					for i := 0; i < width; i++ {
						want[r*width+i] = float64(xh[r*width+i]) * inv * float64(wh[i])
					}
				}
				x := be.UploadClass(NewF32(Default(), []int{rows, width}, xh), F32, MemoryScratchpad, "row-width-dispatch")
				w := be.UploadClass(NewF32(Default(), []int{width}, wh), F32, MemoryScratchpad, "row-width-dispatch")
				t.Cleanup(func() { be.Free(x); be.Free(w) })
				run := func(class int) ([]float32, int) {
					t.Helper()
					cudaMu.Lock()
					y, _ := be.devTr([]int{rows, width}, F32)
					status := be.rmsNormWidthDispatchLocked(class, x, w, y, rows, width, eps)
					cudaMu.Unlock()
					if status < 0 || status > cudaRMSNormWidthBlock {
						t.Fatalf("class %d: status %d", class, status)
					}
					out := be.Read(y)
					be.Free(y)
					return out, status
				}
				wantClass := cudaRMSNormWidthBlock
				switch {
				case width <= 16:
					wantClass = cudaRMSNormWidthSubwarp
				case width <= 1024:
					wantClass = cudaRMSNormWidthWarp
				}
				if got := cudaRMSNormWidthClass(width); got != wantClass {
					t.Fatalf("width class %d, want %d", got, wantClass)
				}
				got, class := run(cudaRMSNormWidthAuto)
				if class != wantClass {
					t.Fatalf("auto launched class %d, want %d", class, wantClass)
				}
				check := func(name string, got []float32) {
					t.Helper()
					for i := range got {
						tol := 1e-5 * math.Max(1, math.Abs(want[i]))
						if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i])-want[i]) > tol {
							t.Fatalf("%s[%d]=%v want %v", name, i, got[i], want[i])
						}
					}
				}
				check("auto", got)
				again, _ := run(cudaRMSNormWidthAuto)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(again[i]) {
						t.Fatalf("repeat drifted at %d", i)
					}
				}
				if width <= 1024 {
					warp, _ := run(cudaRMSNormWidthWarp)
					check("warp", warp)
				}
				block, _ := run(cudaRMSNormWidthBlock)
				check("block", block)
				if width > 16 {
					cudaMu.Lock()
					y, _ := be.devTr([]int{rows, width}, F32)
					status := be.rmsNormWidthDispatchLocked(cudaRMSNormWidthSubwarp, x, w, y, rows, width, eps)
					cudaMu.Unlock()
					be.Free(y)
					if status != -1 {
						t.Fatalf("segment class must refuse width %d, got status %d", width, status)
					}
				}
			})
		}
	}
}
