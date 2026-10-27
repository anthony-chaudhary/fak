//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// This witnesses the ordinary Q8 chunk-output helper, not a quantization or
// throughput claim. Start with native profiling; a skipped test is no witness.
func TestVulkanQ8ChunkOutputStridedCopy(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("start with FAK_VULKAN_DISPATCH_PROFILE=1 for native copy accounting")
	}
	v := vk(t)
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: device type %d", kind)
		}
	}
	const out, start, width = 11, 3, 5
	for _, tokens := range []int{1, 7, 1025} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("tokens%d_batch%t", tokens, batched), func(t *testing.T) {
				values := make([]float32, tokens*width)
				want := make([]float32, tokens*out)
				for i := range values {
					values[i] = float32(i) * .125
				}
				values[0], values[1] = math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234)
				for i := range want {
					want[i] = -999
				}
				src := v.Upload(NewF32(Default(), []int{tokens, width}, values), F32)
				dst := v.Upload(NewF32(Default(), []int{tokens, out}, want), F32)
				defer v.Free(src)
				defer v.Free(dst)
				copyPanel := func(p int) {
					vulkanMu.Lock()
					defer vulkanMu.Unlock()
					v.copyQ8ChunkOutputLocked(dst.buf.(*vulkanBuf), src.buf.(*vulkanBuf), out, start, width, p)
				}
				before, err := v.BackendCounters()
				if err != nil {
					t.Fatal(err)
				}
				if batched {
					v.BeginBatch()
				}
				profileBefore := v.VulkanDebugDispatchProfileSnapshot()
				// One extra row exceeds both buffers. The checked panel path must
				// refuse before recording even the otherwise-valid first rows.
				func() {
					defer func() {
						if recover() == nil {
							t.Fatal("oversized strided copy was not rejected")
						}
					}()
					copyPanel(tokens + 1)
				}()
				if p := v.VulkanDebugDispatchProfileSnapshot(); p.D2DCopies != profileBefore.D2DCopies || p.ComputeBarriers != profileBefore.ComputeBarriers {
					t.Fatal("invalid panel copy recorded partial work")
				}
				copyPanel(tokens)
				profileAfter := v.VulkanDebugDispatchProfileSnapshot()
				if copies := profileAfter.D2DCopies - profileBefore.D2DCopies; copies != 1 {
					t.Fatalf("native copies=%d want 1", copies)
				}
				if batched {
					if profileAfter.BatchSubmits != profileBefore.BatchSubmits || profileAfter.OneShotSubmits != profileBefore.OneShotSubmits {
						t.Fatal("strided copy introduced an intermediate submission")
					}
					v.FlushBatch()
				}
				after, err := v.BackendCounters()
				if err != nil {
					t.Fatal(err)
				}
				if after.D2DCopies-before.D2DCopies != uint64(tokens) || after.D2DBytes-before.D2DBytes != uint64(tokens*width*4) {
					t.Fatalf("logical rows/bytes changed: before=%+v after=%+v", before, after)
				}
				for row := 0; row < tokens; row++ {
					copy(want[row*out+start:row*out+start+width], values[row*width:(row+1)*width])
				}
				got := v.Read(dst)
				if len(got) != len(want) {
					t.Fatalf("output length=%d want=%d", len(got), len(want))
				}
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("element %d: got %08x want %08x (including untouched chunk gaps)", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
			})
		}
	}
}
