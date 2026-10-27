//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// UNRUN. The exact original AddInPlace then RMSNorm operations are the oracle.
func TestVulkanQwen35ResidualNormMatchesSeparate(t *testing.T) {
	if os.Getenv("FAK_VULKAN_QWEN35_RESIDUAL_NORM") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires residual norm candidate and dispatch profiling at initialization")
	}
	v := vk(t)
	if !v.haveQwenResidualNorm {
		t.Fatal("requested optional residual norm module unavailable, not qualified")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: type %d", kind)
		}
	}
	for _, shape := range [][2]int{{2, 1}, {3, 255}, {17, 257}, {2, 5120}} {
		for _, batch := range []bool{false, true} {
			t.Run(fmt.Sprintf("P%d_n%d_batch%v", shape[0], shape[1], batch), func(t *testing.T) {
				rows, n := shape[0], shape[1]
				xh, dh, wh := make([]float32, rows*n), make([]float32, rows*n), make([]float32, n)
				for i := range xh {
					switch i % 5 {
					case 0:
						xh[i], dh[i] = 1048576, 0.0625 // half-ULP add must round before norm
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
					x := v.Upload(NewF32(Default(), shape, values), F32)
					t.Cleanup(func() { v.Free(x) })
					return x
				}
				x := upload([]int{rows, n}, xh)
				wantX := upload([]int{rows, n}, xh)
				delta := upload([]int{rows, n}, dh)
				weight := upload([]int{n}, wh)
				const eps = float32(1e-6)
				v.AddInPlace(wantX, delta)
				wantY := v.RMSNorm(wantX, weight, eps)
				defer v.Free(wantY)
				call := func(source, add, norm Tensor, p int) (Tensor, bool, error) {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					return v.tryQwen35ResidualNormLocked(source, add, norm, p, eps)
				}
				before := v.VulkanDebugDispatchProfileSnapshot()
				if batch {
					v.BeginBatch()
				}
				y, used, err := call(x, delta, weight, rows)
				if batch {
					v.FlushBatch()
				}
				defer v.Free(y)
				if !used || err != nil {
					t.Fatalf("candidate not selected: used=%v err=%v", used, err)
				}
				after := v.VulkanDebugDispatchProfileSnapshot()
				if after.ComputeDispatches-before.ComputeDispatches != 1 {
					t.Fatal("residual+norm must record one compute dispatch")
				}
				assertBits := func(name string, got, want []float32) {
					t.Helper()
					if len(got) != len(want) {
						t.Fatalf("%s extent %d want %d", name, len(got), len(want))
					}
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("%s[%d]=%08x want %08x", name, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
						}
					}
				}
				assertBits("updated residual", v.Read(x), v.Read(wantX))
				assertBits("normalized output", v.Read(y), v.Read(wantY))
				assertBits("read-only delta", v.Read(delta), dh)
				assertBits("read-only norm", v.Read(weight), wh)
				// Pre-record extent/alias rejection must leave the writable X intact
				// and retire only the newly allocated normalized output.
				short := upload([]int{n}, dh[:n])
				short.Shape = []int{rows, n}
				before = v.VulkanDebugDispatchProfileSnapshot()
				start := len(v.transient)
				for _, add := range []Tensor{short, x} {
					_, selected, failure := call(x, add, weight, rows)
					if selected || failure == nil || len(v.transient) != start {
						t.Fatal("selected invalid/alias operand did not fail closed with output cleanup")
					}
					assertBits("unchanged on admission failure", v.Read(x), v.Read(wantX))
				}
				wrongNorm := weight
				wrongNorm.Dtype = F16
				_, selected, failure := call(x, delta, wrongNorm, rows)
				if selected || failure != nil || len(v.transient) != start {
					t.Fatal("unsupported dtype must decline mutation-free")
				}
				_, selected, failure = call(x, delta, weight, 1)
				if selected || failure != nil || len(v.transient) != start {
					t.Fatal("decode must retain the original composition")
				}
				after = v.VulkanDebugDispatchProfileSnapshot()
				if after.ComputeDispatches != before.ComputeDispatches {
					t.Fatal("invalid/refused operands recorded compute")
				}
			})
		}
	}
}
