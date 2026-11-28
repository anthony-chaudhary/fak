//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"strings"
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
	for _, shape := range [][2]int{{1, 1}, {1, 255}, {1, 257}, {1, 5120}, {2, 1}, {7, 255}, {3, 257}, {128, 5120}} {
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
				if !selected || aliasErr == nil || !strings.Contains(aliasErr.Error(), "CUDA status -4") || cudaQwen35ResidualNormLaunchesLocked() != before {
					return fmt.Errorf("alias did not refuse before enqueue: %v", aliasErr)
				}
				if rows == 1 {
					// One row makes W panel-sized: it may not be the residual it normalizes.
					selfNorm := x
					selfNorm.Shape = []int{width}
					_, _, selected, aliasErr = be.qwen35SequenceResidualNormLocked(x, delta, selfNorm, rows, eps)
					if !selected || aliasErr == nil || !strings.Contains(aliasErr.Error(), "CUDA status -4") || cudaQwen35ResidualNormLaunchesLocked() != before {
						return fmt.Errorf("norm weight aliasing the residual did not refuse before enqueue: %v", aliasErr)
					}
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
			if rows != 1 {
				return
			}
			// The one permitted overlap: a read-only branch that is also the norm weight.
			x2, wantX2 := upload([]int{rows, width}, xh), upload([]int{rows, width}, xh)
			shared := delta
			shared.Shape = []int{width}
			be.AddInPlace(wantX2, delta)
			err = func() error {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				var err error
				want, _, err = be.qwen35SequenceRMSNormLocked(wantX2, shared, rows, eps, "residual-norm-oracle")
				if err != nil {
					return err
				}
				before := cudaQwen35ResidualNormLaunchesLocked()
				var used bool
				got, _, used, err = be.qwen35SequenceResidualNormLocked(x2, delta, shared, rows, eps)
				if err != nil {
					return err
				}
				if !used || cudaQwen35ResidualNormLaunchesLocked()-before != 1 {
					return fmt.Errorf("shared read-only operand must enqueue exactly once")
				}
				return be.qwen35SequenceSyncLocked()
			}()
			if err != nil {
				t.Fatal(err)
			}
			check("shared-operand residual", be.Read(x2), be.Read(wantX2))
			check("shared-operand output", be.Read(got), be.Read(want))
			check("shared read-only operand", be.Read(delta), dh)
		})
	}
}

// Host enqueue cost of the fused launch against the two launches it replaces;
// "drain" includes the stream fence so GPU execution is counted too.
func BenchmarkCUDAQwen35ResidualNormEnqueue(b *testing.B) {
	registered, ok := Lookup("cuda")
	be, isCUDA := registered.(*cudaBackend)
	if !ok || !isCUDA {
		b.Skip("exact cuda backend not registered")
	}
	if !cudaQwen35ResidualNormEnabled {
		b.Skip("experimental CUDA residual norm disabled")
	}
	const eps = float32(1e-6)
	for _, shape := range [][2]int{{1, 5120}, {2, 5120}, {8, 5120}, {1, 2048}} {
		rows, width := shape[0], shape[1]
		upload := func(dims []int, fill float32) Tensor {
			values := make([]float32, dims[0]*dims[len(dims)-1])
			if len(dims) == 1 {
				values = values[:dims[0]]
			}
			for i := range values {
				values[i] = fill
			}
			x := be.UploadClass(NewF32(Default(), dims, values), F32, MemoryScratchpad, "cuda-residual-norm-bench")
			b.Cleanup(func() { be.Free(x) })
			return x
		}
		// A zero branch keeps the in-place residual stable across iterations.
		x, delta, weight := upload([]int{rows, width}, 0.5), upload([]int{rows, width}, 0), upload([]int{width}, 1)
		fence := func(b testing.TB) {
			cudaMu.Lock()
			err := be.qwen35SequenceSyncLocked()
			cudaMu.Unlock()
			if err != nil {
				b.Fatal(err)
			}
		}
		arms := []struct {
			name string
			step func(b testing.TB) bool
		}{
			{"capability", func(b testing.TB) bool {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				return cudaQwen35ResidualNormCapabilityLocked(rows, width) == 1
			}},
			{"fused", func(b testing.TB) bool {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				_, buf, used, err := be.qwen35SequenceResidualNormLocked(x, delta, weight, rows, eps)
				if err != nil {
					b.Fatal(err)
				}
				if used {
					be.releaseTransientBuffers([]*cudaBuf{buf})
				}
				return used
			}},
			{"norm-only", func(b testing.TB) bool {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				_, buf, err := be.qwen35SequenceRMSNormLocked(x, weight, rows, eps, "residual-norm-bench")
				if err != nil {
					b.Fatal(err)
				}
				be.releaseTransientBuffers([]*cudaBuf{buf})
				return true
			}},
			{"unfused", func(b testing.TB) bool {
				be.AddInPlace(x, delta)
				cudaMu.Lock()
				defer cudaMu.Unlock()
				_, buf, err := be.qwen35SequenceRMSNormLocked(x, weight, rows, eps, "residual-norm-bench")
				if err != nil {
					b.Fatal(err)
				}
				be.releaseTransientBuffers([]*cudaBuf{buf})
				return true
			}},
		}
		for _, arm := range arms {
			for _, mode := range []string{"enqueue", "drain"} {
				if arm.name == "capability" && mode == "drain" {
					continue
				}
				b.Run(fmt.Sprintf("P%d-n%d/%s/%s", rows, width, arm.name, mode), func(b *testing.B) {
					if !arm.step(b) {
						b.Skip("arm declined for this shape")
					}
					fence(b)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						arm.step(b)
						if i%256 == 255 {
							if mode == "enqueue" {
								b.StopTimer()
							}
							fence(b)
							if mode == "enqueue" {
								b.StartTimer()
							}
						}
					}
					if mode == "drain" {
						fence(b)
					}
					b.StopTimer()
					fence(b)
				})
			}
		}
	}
}

// gatedResidualRMSNormRef is a closed host oracle pinning the serial 4-step gated
// residual RMSNorm boundary, alpha/beta identities, and temperature T scaling.
func gatedResidualRMSNormRef(residual, branch, postW, nextPreW []float32, gate, temp, eps float64) (nextPre []float32, alpha, beta float64) {
	n := len(residual)
	scaled := make([]float64, n)
	var sumSq float64
	for i := 0; i < n; i++ {
		scaled[i] = float64(branch[i]) * (1.0 + float64(postW[i]))
		sumSq += scaled[i] * scaled[i]
	}
	invRMS := 1.0 / math.Sqrt(sumSq/float64(n)+eps)
	normed := make([]float64, n)
	for i := 0; i < n; i++ {
		normed[i] = scaled[i] * invRMS
	}
	sig := func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
	beta = sig(gate / temp)
	alpha = math.Sqrt(math.Max(0, sig(-gate/temp)*(1.0+beta)))
	mixed := make([]float64, n)
	var mixSq float64
	for i := 0; i < n; i++ {
		mixed[i] = alpha*float64(residual[i]) + beta*normed[i]
		mixSq += mixed[i] * mixed[i]
	}
	mixInvRMS := 1.0 / math.Sqrt(mixSq/float64(n)+eps)
	nextPre = make([]float32, n)
	for i := 0; i < n; i++ {
		nextPre[i] = float32(mixed[i] * mixInvRMS * float64(nextPreW[i]))
	}
	return nextPre, alpha, beta
}

// fak-test:runtime fast est=1ms lane=default
func TestGatedResidualRMSNormOracle(t *testing.T) {
	t.Parallel()
	residual := []float32{1.0, -2.0, 0.5, 3.0}
	branch := []float32{0.5, 1.0, -1.5, 2.0}
	postW := []float32{0.1, -0.2, 0.0, 0.3}
	nextPreW := []float32{1.0, 1.5, 0.8, 1.2}
	const eps = 1e-6

	out0, alpha0, beta0 := gatedResidualRMSNormRef(residual, branch, postW, nextPreW, 0, 1.0, eps)
	if math.Abs(beta0-0.5) > 1e-12 || math.Abs(alpha0-math.Sqrt(0.75)) > 1e-12 {
		t.Fatalf("gate=0 identities: alpha=%v beta=%v", alpha0, beta0)
	}
	if len(out0) != len(residual) {
		t.Fatalf("len=%d, want %d", len(out0), len(residual))
	}

	outA, aA, bA := gatedResidualRMSNormRef(residual, branch, postW, nextPreW, 1.5, 2.0, eps)
	outB, aB, bB := gatedResidualRMSNormRef(residual, branch, postW, nextPreW, 3.0, 4.0, eps)
	if math.Abs(aA-aB) > 1e-12 || math.Abs(bA-bB) > 1e-12 {
		t.Fatalf("T scaling broke alpha/beta: (%v,%v) vs (%v,%v)", aA, bA, aB, bB)
	}
	for i := range outA {
		if outA[i] != outB[i] {
			t.Fatalf("T scaling idx %d: %v != %v", i, outA[i], outB[i])
		}
	}
}
