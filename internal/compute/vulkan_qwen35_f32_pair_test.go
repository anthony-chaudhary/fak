//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// The paired F32 panel must retain the ordinary native F32 reduction's bits,
// including a tail beyond its 1024-element shared-input window. No Q8 route is
// admitted here; a skipped device/profile test is not a qualification witness.
func TestVulkanQwen35SequenceF32PairMatchesSeparate(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("start with FAK_VULKAN_DISPATCH_PROFILE=1 for paired dispatch accounting")
	}
	v := vk(t)
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: device type %d", kind)
		}
	}
	const out0, out1 = 269, 17
	for _, in := range []int{64, 1025} {
		for _, tokens := range []int{1, 4} {
			t.Run(fmt.Sprintf("in%d_tokens%d", in, tokens), func(t *testing.T) {
				rng := rand.New(rand.NewSource(12390))
				values := func(n int) []float32 {
					x := make([]float32, n)
					for i := range x {
						x[i] = rng.Float32()*2 - 1
					}
					return x
				}
				w0 := v.Upload(NewF32(Default(), []int{out0, in}, values(out0*in)), F32)
				w1 := v.Upload(NewF32(Default(), []int{out1, in}, values(out1*in)), F32)
				x := v.Upload(NewF32(Default(), []int{tokens, in}, values(tokens*in)), F32)
				defer v.Free(w0)
				defer v.Free(w1)
				defer v.Free(x)
				ref0, ref1 := func() (Tensor, Tensor) {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					return v.qwen35VulkanSequenceMatMulLocked(w0, x, tokens), v.qwen35VulkanSequenceMatMulLocked(w1, x, tokens)
				}()
				defer v.Free(ref0)
				defer v.Free(ref1)
				before := v.VulkanDebugDispatchProfileSnapshot()
				v.BeginBatch()
				got0, got1 := func() (Tensor, Tensor) {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					return v.qwen35VulkanSequenceMatMulPairLocked(w0, w1, x, tokens)
				}()
				v.FlushBatch()
				defer v.Free(got0)
				defer v.Free(got1)
				after := v.VulkanDebugDispatchProfileSnapshot()
				wantDispatches := uint64(1)
				if tokens == 1 {
					wantDispatches = 2 // preserve the specialized decode route
				}
				if dispatches := after.ComputeDispatches - before.ComputeDispatches; dispatches != wantDispatches {
					t.Fatalf("dispatches=%d want=%d", dispatches, wantDispatches)
				}
				for i, pair := range [][2]Tensor{{got0, ref0}, {got1, ref1}} {
					got, want := v.Read(pair[0]), v.Read(pair[1])
					if len(got) != len(want) {
						t.Fatalf("output %d length=%d want=%d", i, len(got), len(want))
					}
					for j := range want {
						if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
							t.Fatalf("output %d element %d: got %08x want %08x", i, j, math.Float32bits(got[j]), math.Float32bits(want[j]))
						}
					}
				}
			})
		}
	}
}
