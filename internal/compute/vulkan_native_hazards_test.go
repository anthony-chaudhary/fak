//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"os"
	"testing"
)

// fak-test:runtime fast est=1ms
// UNRUN. This invokes only the fixed native range-proof algorithm, not a device.
func TestVulkanNativeBatchRangeProof(t *testing.T) {
	if code := vulkanNativeHazardsSelfCheck(); code != 0 {
		t.Fatalf("native conservative range proof case %d failed", code)
	}
}

// fak-test:runtime integration est=2s lane=optin
// UNRUN. Uses actual ordinary BatchedMatMul F32 panel consumers, not a synthetic command plan.
func TestVulkanNativeBatchHazardsPanels(t *testing.T) {
	if os.Getenv("FAK_VULKAN_NATIVE_BATCH_HAZARDS") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires native hazard candidate and dispatch profile at initialization")
	}
	v := vk(t)
	if !vulkanNativeHazardsEnabled() {
		t.Fatal("native candidate unavailable")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("physical GPU not established: %d", kind)
		}
	}
	weight := make([]float32, 64)
	input := make([]float32, 16)
	norm := make([]float32, 8)
	for i := range weight {
		weight[i] = float32(i%11-5) / 8
	}
	for i := range input {
		input[i] = float32(i-7) / 4
	}
	for i := range norm {
		norm[i] = 1
	}
	w := v.Upload(NewF32(Default(), []int{8, 8}, weight), F32)
	defer v.Free(w)
	x := v.Upload(NewF32(Default(), []int{2, 8}, input), F32)
	defer v.Free(x)
	n := v.Upload(NewF32(Default(), []int{8}, norm), F32)
	defer v.Free(n)
	run := func(batch bool) []Tensor {
		if batch {
			v.BeginBatch()
		}
		a := v.BatchedMatMul(w, x, 2)
		b := v.BatchedMatMul(w, x, 2) // shared reads only; independent outputs
		c := v.BatchedMatMul(w, a, 2) // nonadjacent RAW must see a despite intervening b
		d := v.RMSNorm(x, n, 1e-5)    // independent of the c operation
		v.AddInPlace(d, b)            // unknown operation: preserve dependency, invalidate proof
		e := v.BatchedMatMul(w, x, 2) // first known operation after unknown stays fenced
		if batch {
			v.FlushBatch()
		}
		return []Tensor{a, b, c, d, e}
	}
	want := run(false)
	defer func() {
		for _, t := range want {
			v.Free(t)
		}
	}()
	// Run twice: the same native command allocation may be reused at ordinal zero.
	for cycle := 0; cycle < 2; cycle++ {
		before := v.VulkanDebugDispatchProfileSnapshot()
		got := run(true)
		after := v.VulkanDebugDispatchProfileSnapshot()
		if after.ComputeDispatches-before.ComputeDispatches != 6 || after.ComputeBarriers-before.ComputeBarriers != 3 {
			t.Fatalf("six ordinary ops require exactly three retained and two elided barriers: before=%+v after=%+v", before, after)
		}
		for j := range got {
			a, b := v.Read(got[j]), v.Read(want[j])
			if len(a) != len(b) {
				t.Fatal("output extent differs")
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("cycle%d output%d[%d] differs", cycle, j, i)
				}
			}
			v.Free(got[j])
		}
	}
	// Interposed native transfer has no admitted access declaration: both its
	// own dependency and the first following compute dependency must remain.
	copyDst := v.Upload(NewF32(Default(), []int{2, 8}, make([]float32, 16)), F32)
	defer v.Free(copyDst)
	before := v.VulkanDebugDispatchProfileSnapshot()
	v.BeginBatch()
	a := v.BatchedMatMul(w, x, 2)
	v.nativeHazardCopyForTest(copyDst, a)
	b := v.BatchedMatMul(w, copyDst, 2)
	c := v.BatchedMatMul(w, x, 2)
	v.FlushBatch()
	after := v.VulkanDebugDispatchProfileSnapshot()
	defer v.Free(a)
	defer v.Free(b)
	defer v.Free(c)
	if after.ComputeDispatches-before.ComputeDispatches != 3 || after.D2DCopies-before.D2DCopies != 1 || after.ComputeBarriers-before.ComputeBarriers != 2 {
		t.Fatalf("interposed native copy did not retain both dependencies: before=%+v after=%+v", before, after)
	}
	for j, pair := range [][2]Tensor{{copyDst, want[0]}, {b, want[2]}, {c, want[0]}} {
		got, ref := v.Read(pair[0]), v.Read(pair[1])
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(ref[i]) {
				t.Fatalf("copy chain output%d[%d] differs", j, i)
			}
		}
	}

}
