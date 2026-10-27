//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// One physical witness for the ordinary candidate's exact existing kernel;
// required mode refuses skipped CUDA or an inactive experimental selector.
// fak-test:runtime integration est=5s lane=optin
func TestCUDAQwen35ResidualNormMatchesComposition(t *testing.T) {
	be := cudaGDNBackend(t)
	if !cudaQwen35ResidualNormEnabled {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatal("set FAK_CUDA_QWEN35_RESIDUAL_NORM=1")
		}
		t.Skip("experimental CUDA residual norm disabled")
	}
	t.Cleanup(be.Recycle)
	for _, shape := range [][2]int{{2, 1}, {7, 255}, {3, 257}, {128, 5120}} {
		t.Run(fmt.Sprintf("P%d-n%d", shape[0], shape[1]), func(t *testing.T) {
			rows, width := shape[0], shape[1]
			xh, dh, wh := make([]float32, rows*width), make([]float32, rows*width), make([]float32, width)
			for i := range xh {
				switch i % 5 {
				case 0:
					xh[i], dh[i] = 1048576, 0.0625 // half ULP, must round before RMS
				case 1:
					xh[i], dh[i] = -1048576, -0.0625
				case 2:
					xh[i], dh[i] = 7.25, -7.25
				case 3:
					xh[i], dh[i] = float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))
				default:
					xh[i], dh[i] = float32(i%29-14)/17, float32(i%13-6)/11
				}
			}
			for i := range wh {
				wh[i] = float32(i%19-9) / 7
			}
			upload := func(shape []int, values []float32) Tensor {
				x := be.UploadClass(NewF32(Default(), shape, values), F32, MemoryScratchpad, "cuda-residual-norm-witness")
				t.Cleanup(func() { be.Free(x) })
				return x
			}
			x, wantX := upload([]int{rows, width}, xh), upload([]int{rows, width}, xh)
			delta, weight := upload([]int{rows, width}, dh), upload([]int{width}, wh)
			const eps = float32(1e-6)
			be.ResetHostXfer()
			be.ResetH2DXfer()
			be.AddInPlace(wantX, delta)
			var want, got Tensor
			err := func() error {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				var err error
				want, _, err = be.qwen35SequenceRMSNormLocked(wantX, weight, rows, eps, "residual-norm-oracle")
				if err != nil {
					return err
				}
				before := cudaQwen35ResidualNormLaunchesLocked()
				// Intentional X==ResidualOut is supported, X==delta is not.
				_, _, selected, aliasErr := be.qwen35SequenceResidualNormLocked(x, x, weight, rows, eps)
				if !selected || aliasErr == nil || cudaQwen35ResidualNormLaunchesLocked() != before {
					return fmt.Errorf("alias did not refuse before enqueue: %v", aliasErr)
				}
				var used bool
				got, _, used, err = be.qwen35SequenceResidualNormLocked(x, delta, weight, rows, eps)
				if err != nil {
					return err
				}
				if !used || cudaQwen35ResidualNormLaunchesLocked()-before != 1 {
					return fmt.Errorf("candidate must enqueue exactly once")
				}
				return be.qwen35SequenceSyncLocked()
			}()
			if err != nil {
				t.Fatal(err)
			}
			if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatal("resident operation transferred host activation bytes")
			}
			check := func(name string, got, want []float32) {
				t.Helper()
				if len(got) != len(want) {
					t.Fatalf("%s length mismatch", name)
				}
				for i := range got {
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("%s[%d] %08x != %08x", name, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			}
			check("updated residual", be.Read(x), be.Read(wantX))
			check("normalized output", be.Read(got), be.Read(want))
			check("read-only delta", be.Read(delta), dh)
			check("read-only norm", be.Read(weight), wh)
		})
	}
}
