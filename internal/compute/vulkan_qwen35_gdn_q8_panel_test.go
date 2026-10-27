//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// UNRUN development witness. Start the process with both candidate and dispatch
// profiling enabled; a skip is not physical or numerical qualification.
func TestVulkanQwen35GDNQ8PanelMatchesSeparate(t *testing.T) {
	if os.Getenv("FAK_VULKAN_QWEN35_GDN_Q8_PANEL") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires FAK_VULKAN_QWEN35_GDN_Q8_PANEL=1 and FAK_VULKAN_DISPATCH_PROFILE=1 at initialization")
	}
	v := vk(t)
	if !v.haveQ8 || v.q8CooperativeMatrixActive(2) {
		t.Skip("candidate must not replace the device's cooperative Q8 precision route")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: device type %d", kind)
		}
	}
	for _, shape := range [][2]int{{64, 2}, {2080, 17}, {64, 257}} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("in%d_P%d_batch%v", shape[0], shape[1], batched), func(t *testing.T) {
				in, tokens := shape[0], shape[1]
				outs := [4]int{17, 9, 3, 5} // matrix boundaries cross eight-row workgroups
				var weights, want [4]Tensor
				for i, out := range outs {
					w := make([]float32, out*in)
					for j := range w {
						w[j] = float32((j*13+i*7)%47-23) / 31
					}
					weights[i] = v.Upload(NewF32(Default(), []int{out, in}, w), Q8_0)
					defer v.Free(weights[i])
				}
				input := make([]float32, tokens*in)
				for row := 0; row < tokens; row++ {
					if row%5 == 0 {
						continue // explicit zero-code writes after prior window/row reuse
					}
					for col := 0; col < in; col++ {
						input[row*in+col] = float32((col*19+row*11)%61-30) / 17
					}
				}
				x := v.Upload(NewF32(Default(), []int{tokens, in}, input), F32)
				defer v.Free(x)
				for i := range want {
					want[i] = func() Tensor {
						vulkanMu.Lock()
						defer vulkanMu.Unlock()
						return v.qwen35VulkanSequenceMatMulLocked(weights[i], x, tokens)
					}()
					defer v.Free(want[i])
				}
				before := v.VulkanDebugDispatchProfileSnapshot()
				if batched {
					v.BeginBatch()
				}
				var got [4]Tensor
				var fused bool
				var err error
				func() {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					got[0], got[1], got[2], got[3], fused, err = v.tryQwen35GDNQ8PanelLocked(
						x, weights[0], weights[1], weights[2], weights[3], tokens,
					)
				}()
				if batched {
					v.FlushBatch()
				}
				for _, y := range got {
					defer v.Free(y)
				}
				if err != nil || !fused {
					t.Fatalf("candidate not selected: fused=%v err=%v; missing module is not a witness", fused, err)
				}
				after := v.VulkanDebugDispatchProfileSnapshot()
				if n := after.ComputeDispatches - before.ComputeDispatches; n != 1 {
					t.Fatalf("panel recorded %d dispatches, want one", n)
				}
				for i := range got {
					a, b := v.Read(got[i]), v.Read(want[i])
					if len(a) != len(b) {
						t.Fatalf("output %d length=%d want=%d", i, len(a), len(b))
					}
					for j := range a {
						if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
							t.Fatalf("output %d index %d: got %08x want %08x", i, j, math.Float32bits(a[j]), math.Float32bits(b[j]))
						}
					}
				}

				// A short native input with forged panel metadata passes the shape
				// capability, then must fail before recording and release new outputs.
				short := v.Upload(NewF32(Default(), []int{in}, input[:in]), F32)
				defer v.Free(short)
				short.Shape = []int{tokens, in}
				before = v.VulkanDebugDispatchProfileSnapshot()
				func() {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					start := len(v.transient)
					_, _, _, _, selected, failure := v.tryQwen35GDNQ8PanelLocked(
						short, weights[0], weights[1], weights[2], weights[3], tokens,
					)
					if selected || failure == nil || len(v.transient) != start {
						t.Errorf("invalid native extent did not fail/clean up: selected=%v err=%v", selected, failure)
					}
					wrongType := weights[0]
					wrongType.Dtype = F32
					_, _, _, _, selected, failure = v.tryQwen35GDNQ8PanelLocked(
						x, wrongType, weights[1], weights[2], weights[3], tokens,
					)
					if selected || failure != nil || len(v.transient) != start {
						t.Errorf("unsupported dtype must decline before mutation: selected=%v err=%v", selected, failure)
					}
				}()
				after = v.VulkanDebugDispatchProfileSnapshot()
				if after.ComputeDispatches != before.ComputeDispatches {
					t.Fatal("refused/invalid operands recorded compute work")
				}
			})
		}
	}
}
