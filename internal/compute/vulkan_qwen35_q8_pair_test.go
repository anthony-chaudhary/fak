//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=3s lane=optin
// UNRUN. Ordinary paired sequence consumer vs two real BatchedMatMul calls.
func TestVulkanQwen35Q8PairMatchesSeparate(t *testing.T) {
	if os.Getenv("FAK_VULKAN_QWEN35_Q8_PAIR") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires Qwen Q8 pair and dispatch profile at initialization")
	}
	v := vk(t)
	if !v.haveQ8 || v.q8CooperativeMatrixActive(2) {
		t.Skip("noncooperative Q8 is required; do not alter device selection")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("physical GPU not established: %d", kind)
		}
	}
	//1024->3584 is the supported Qwen3.5-0.8B FFN geometry; other cases stress
	// the shared cap and the boundary between output matrices/workgroups.
	for _, shape := range [][4]int{{32, 257, 259, 2}, {1024, 3584, 3584, 2}, {2048, 17, 9, 17}} {
		for _, batch := range []bool{false, true} {
			t.Run(fmt.Sprintf("in%d_out%d_%d_P%d_batch%v", shape[0], shape[1], shape[2], shape[3], batch), func(t *testing.T) {
				in, out0, out1, tokens := shape[0], shape[1], shape[2], shape[3]
				var weights, want [2]Tensor
				for j, out := range []int{out0, out1} {
					values := make([]float32, out*in)
					for i := range values {
						values[i] = float32((i*13+j*7)%47-23) / 31
					}
					weights[j] = v.Upload(NewF32(Default(), []int{out, in}, values), Q8_0)
					defer v.Free(weights[j])
				}
				input := make([]float32, tokens*in)
				for row := 1; row < tokens; row++ {
					for col := 0; col < in; col++ {
						input[row*in+col] = float32((col*19+row*11)%61-30) / 17
					}
				}
				x := v.Upload(NewF32(Default(), []int{tokens, in}, input), F32)
				defer v.Free(x)
				for i := range want {
					want[i] = v.BatchedMatMul(weights[i], x, tokens)
					defer v.Free(want[i])
				}
				before := v.VulkanDebugDispatchProfileSnapshot()
				if batch {
					v.BeginBatch()
				}
				var got [2]Tensor
				func() {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					got[0], got[1] = v.qwen35VulkanSequenceMatMulPairLocked(weights[0], weights[1], x, tokens)
				}()
				if batch {
					v.FlushBatch()
				}
				after := v.VulkanDebugDispatchProfileSnapshot()
				for _, y := range got {
					defer v.Free(y)
				}
				if after.ComputeDispatches-before.ComputeDispatches != 1 {
					t.Fatal("ordinary pair must select one existing native dispatch")
				}
				for j := range got {
					a, b := v.Read(got[j]), v.Read(want[j])
					if len(a) != len(b) {
						t.Fatal("pair extent differs")
					}
					for i := range a {
						if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
							t.Fatalf("output%d[%d] %08x want%08x", j, i, math.Float32bits(a[i]), math.Float32bits(b[i]))
						}
					}
				}
				short := v.Upload(NewF32(Default(), []int{in}, input[:in]), F32)
				defer v.Free(short)
				short.Shape = []int{tokens, in}
				before = v.VulkanDebugDispatchProfileSnapshot()
				func() {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					start := len(v.transient)
					_, _, used, err := v.tryQwen35Q8PairLocked(short, weights[0], weights[1], tokens)
					if used || err == nil || len(v.transient) != start {
						t.Errorf("short native input did not fail/clean up: used%v err%v", used, err)
					}
					wrong := weights[0]
					wrong.Dtype = F32
					_, _, used, err = v.tryQwen35Q8PairLocked(x, wrong, weights[1], tokens)
					if used || err != nil || len(v.transient) != start {
						t.Error("wrong dtype must decline without mutation")
					}
					_, _, used, err = v.tryQwen35Q8PairLocked(x, weights[0], weights[1], 1)
					if used || err != nil || len(v.transient) != start {
						t.Error("decode must decline without mutation")
					}
					wide := x
					wide.Shape = []int{tokens, 2080}
					_, _, used, err = v.tryQwen35Q8PairLocked(wide, weights[0], weights[1], tokens)
					if used || err != nil || len(v.transient) != start {
						t.Error("oversized single window must decline without mutation")
					}
					// Copy metadata only: the real weight owner and buffers remain untouched.
					chunked := weights[0]
					b := *chunked.buf.(*vulkanBuf)
					b.q8Chunks = make([]vulkanQ8Chunk, 1)
					chunked.buf = &b
					_, _, used, err = v.tryQwen35Q8PairLocked(x, chunked, weights[1], tokens)
					if used || err != nil || len(v.transient) != start {
						t.Error("chunked weight must decline without mutation")
					}
				}()
				after = v.VulkanDebugDispatchProfileSnapshot()
				if after.ComputeDispatches != before.ComputeDispatches {
					t.Fatal("invalid/refused inputs recorded work")
				}
			})
		}
	}
}
