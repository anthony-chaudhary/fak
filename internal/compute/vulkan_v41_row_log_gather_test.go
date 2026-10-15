//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// The profile is read at native process startup. A software Vulkan device can
// establish copy/recording correctness, but never physical performance.
// fak-test:runtime integration est=1s lane=optin
func TestVulkanV41RowLogGatherRegions(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("start with FAK_VULKAN_DISPATCH_PROFILE=1 for the gather command witness")
	}
	v := vk(t)
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: device type %d", kind)
		}
	}
	const width = 4
	for _, batched := range []bool{false, true} {
		t.Run(fmt.Sprintf("batch%t", batched), func(t *testing.T) {
			aValues, bValues := make([]float32, 5*width), make([]float32, 8*width)
			for i := range aValues {
				aValues[i] = float32(i + 1)
			}
			for i := range bValues {
				bValues[i] = float32(100 + i)
			}
			bValues[width] = float32(math.Copysign(0, -1))
			a := v.Upload(NewF32(Default(), []int{5, width}, aValues), F32)
			b := v.Upload(NewF32(Default(), []int{8, width}, bValues), F32)
			defer v.Free(a)
			defer v.Free(b)
			if batched {
				if !v.BeginV41StepBatch() {
					t.Fatal("step recording window unavailable")
				}
				defer func() {
					if err := v.EndV41StepBatch(); err != nil {
						t.Error(err)
					}
				}()
			}
			// The gather must depend on this earlier write, including in a batch.
			if err := v.V41RowLogCopy(a, 0, b, 1, 1); err != nil {
				t.Fatal(err)
			}
			copy(aValues[:width], bValues[width:2*width])
			spans := []V41RowSpan{
				{Src: a, Row: 4, Rows: 1}, {Src: a, Row: 0, Rows: 2},
				{Src: b, Row: 6, Rows: 1}, {Src: b, Row: 1, Rows: 1},
				{Src: b, Row: 6, Rows: 1}, {Src: a, Row: 0, Rows: 1},
			}
			var want []float32
			for _, span := range spans {
				values := aValues
				if span.Src.Buf() == b.Buf() {
					values = bValues
				}
				want = append(want, values[span.Row*width:(span.Row+span.Rows)*width]...)
			}
			before := v.VulkanDebugDispatchProfileSnapshot()
			out, err := v.V41RowLogGather(spans, width)
			if err != nil {
				t.Fatal(err)
			}
			defer v.Free(out)
			after := v.VulkanDebugDispatchProfileSnapshot()
			if got := after.D2DCopies - before.D2DCopies; got != 2 {
				t.Fatalf("gather recorded %d copy commands, want two source-buffer groups", got)
			}
			wantBarriers := uint64(1)
			if !batched {
				wantBarriers++ // publish writes for the next unbatched consumer
			}
			if got := after.ComputeBarriers - before.ComputeBarriers; got != wantBarriers {
				t.Fatalf("gather recorded %d dependency barriers, want %d", got, wantBarriers)
			}
			if batched {
				if after.BatchSubmits != before.BatchSubmits || after.OneShotSubmits != before.OneShotSubmits {
					t.Fatal("gather introduced an intermediate submission")
				}
			} else if after.OneShotD2DSubmits-before.OneShotD2DSubmits != 1 {
				t.Fatal("unbatched gather did not use one checked submission")
			}
			got := v.Read(out)
			if len(got) != len(want) {
				t.Fatalf("got %d values, want %d", len(got), len(want))
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("value %d bits %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			bad := append(append([]V41RowSpan(nil), spans...), V41RowSpan{Src: b, Row: 8, Rows: 1})
			before = v.VulkanDebugDispatchProfileSnapshot()
			if invalid, err := v.V41RowLogGather(bad, width); err == nil || invalid.Buf() != nil {
				t.Fatal("late invalid region produced an output")
			}
			after = v.VulkanDebugDispatchProfileSnapshot()
			if after.D2DCopies != before.D2DCopies || after.ComputeBarriers != before.ComputeBarriers ||
				after.BatchSubmits != before.BatchSubmits || after.OneShotSubmits != before.OneShotSubmits {
				t.Fatal("invalid gather recorded partial work")
			}
		})
	}
}
