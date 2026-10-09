//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1 and
// FAK_VULKAN_DISPATCH_PROFILE=1 before process startup. A skipped test is no
// physical witness. Tolerances are fixed before execution: prefix and the
// cancellation control are bit-exact; ordinary tail abs/rel bounds are 2e-6.
func TestVulkanV41TailRoPEQK(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required V4.1 witness needs FAK_VULKAN_DISPATCH_PROFILE=1 before startup")
		}
		t.Skip("V4.1 physical dispatch witness is opt-in")
	}
	v := vk(t)
	if !v.SupportsV41TailRoPE() {
		t.Fatal("registered current Vulkan build lacks mandatory V4.1 tail RoPE")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		// Vulkan enum: integrated=1, discrete=2, virtual=3, CPU=4, other=0.
		// A virtual/unknown type needs separate physical qualification evidence.
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	for _, dims := range [][3]int{{2, 7, 4}, {4, 512, 64}} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("h%d_d%d_r%d_batch%t", dims[0], dims[1], dims[2], batched), func(t *testing.T) {
				head, width, rotary := dims[0], dims[1], dims[2]
				q, kv, table := make([]float32, head*width), make([]float32, width), make([]float32, rotary)
				fill := func(x []float32) {
					for i := range x {
						x[i] = float32((i%29)-13)*0.03125 + 0.00390625
					}
					for base := 0; base < len(x); base += width {
						x[base] = math.Float32frombits(0x80000000)
						x[base+1] = math.Float32frombits(0x7fc01234)
						x[base+width-rotary] = math.Float32frombits(0x3f800001)
						x[base+width-rotary+1] = 1
					}
				}
				fill(q)
				fill(kv)
				for j := 0; j < rotary/2; j++ {
					table[2*j], table[2*j+1] = float32(j+1)*0.0625, float32(j%3-1)*0.375+0.125
				}
				table[0], table[1] = 1, math.Float32frombits(0x3f7ffffe)
				upload := func(shape []int, x []float32) Tensor {
					d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 RoPE witness input")
					t.Cleanup(func() { v.Free(d) })
					return d
				}
				dq, dk, dt := upload([]int{head, width}, q), upload([]int{width}, kv), upload([]int{rotary / 2, 2}, table)
				if head == 2 && !batched {
					outQ := upload([]int{head, width}, make([]float32, head*width))
					outK := upload([]int{width}, make([]float32, width))
					short := upload([]int{1}, []float32{1})
					before, snapshotErr := v.BackendExecutionSnapshot()
					if snapshotErr != nil {
						t.Fatal(snapshotErr)
					}
					badShape := dq
					badShape.Shape = []int{head * width}
					badCapacity := dq
					badCapacity.buf = &vulkanBuf{ptr: dq.buf.(*vulkanBuf).ptr, n: 4}
					for _, badQ := range []Tensor{NewF32(Default(), []int{head, width}, q), badShape, badCapacity} {
						x, y, reject := v.V41TailRoPEQK(badQ, dk, dt, head, width, rotary)
						if reject == nil || x.Buf() != nil || y.Buf() != nil {
							t.Fatal("malformed Go input did not fail before publishing outputs")
						}
					}
					for _, tc := range []struct {
						qOut, kOut, table Tensor
					}{
						{dq, outK, dt},      // output aliases input
						{outQ, outQ, dt},    // outputs alias one another
						{short, outK, dt},   // short native output allocation
						{outQ, outK, short}, // short native table allocation
					} {
						status := func() int {
							vulkanMu.Lock()
							defer vulkanMu.Unlock()
							return v41TailRoPEQKStatusLocked(dq.buf.(*vulkanBuf), dk.buf.(*vulkanBuf), tc.qOut.buf.(*vulkanBuf), tc.kOut.buf.(*vulkanBuf), tc.table.buf.(*vulkanBuf), head, width, rotary)
						}()
						if status != 2 {
							t.Fatalf("native alias/extent refusal status=%d want=2", status)
						}
					}
					after, snapshotErr := v.BackendExecutionSnapshot()
					if snapshotErr != nil || before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
						t.Fatalf("rejected inputs dispatched, transferred, or allocated: before=%+v after=%+v err=%v", before, after, snapshotErr)
					}
				}
				before, err := v.BackendExecutionSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				profileBefore := v.VulkanDebugDispatchProfileSnapshot()
				if batched {
					v.BeginBatch()
				}
				qo, ko, err := v.V41TailRoPEQK(dq, dk, dt, head, width, rotary)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { v.Free(qo); v.Free(ko) })
				if qo.Buf() == dq.Buf() || ko.Buf() == dk.Buf() || qo.Buf() == ko.Buf() {
					t.Fatal("operation aliased an input or its other output")
				}
				gotQ, gotK := v.Read(qo), v.Read(ko) // checked read fences the marked batch
				if batched {
					v.FlushBatch() // close the Go ledger after readback ended the native batch
				}
				after, err := v.BackendExecutionSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				observation, err := BackendExecutionDelta(before, after)
				if err != nil {
					t.Fatal(err)
				}
				c := observation.Counters
				if !observation.TransferCountersObserved || c.ComputeDispatches != 1 || c.OtherDispatches != 1 || c.H2DCount != 0 || c.H2DBytes != 0 ||
					c.D2HCount != 2 || c.D2HBytes != uint64((len(q)+len(kv))*4) || c.D2DCopies != 0 || c.D2DBytes != 0 || c.Fallbacks != 0 {
					t.Fatalf("unexpected operation/transfer accounting: %+v", observation)
				}
				profileAfter := v.VulkanDebugDispatchProfileSnapshot()
				if profileAfter.OtherRoPEDispatches-profileBefore.OtherRoPEDispatches != 1 {
					t.Fatal("dedicated operation was not classified as otherRope")
				}
				check := func(label string, got, input []float32) {
					want := v41TailRoPEReference(input, table, width, rotary)
					if len(got) != len(want) {
						t.Fatalf("%s output length=%d want=%d", label, len(got), len(want))
					}
					for i := range want {
						dim := i % width
						if dim < width-rotary || dim == width-rotary {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								t.Fatalf("%s exact prefix/cancellation lane %d: bits=%08x want=%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
							}
							continue
						}
						delta := math.Abs(float64(got[i]) - float64(want[i]))
						if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || delta > 2e-6+2e-6*math.Abs(float64(want[i])) {
							t.Fatalf("%s tail lane %d: got=%g want=%g delta=%g", label, i, got[i], want[i], delta)
						}
					}
				}
				check("Q", gotQ, q)
				check("KV", gotK, kv)
				for _, input := range []struct {
					t    Tensor
					want []float32
				}{{dq, q}, {dk, kv}, {dt, table}} {
					got := v.Read(input.t)
					for i, want := range input.want {
						if math.Float32bits(got[i]) != math.Float32bits(want) {
							t.Fatalf("input mutated at lane %d", i)
						}
					}
				}
				t.Logf("V4.1 operation observation: %+v", observation)
			})
		}
	}
}
