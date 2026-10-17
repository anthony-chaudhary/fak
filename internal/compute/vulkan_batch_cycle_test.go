//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"testing"
)

// A cycled window stays in flight while the next records; transients freed mid-window
// recycle into later windows, and host fences, frees outside a window and flushes drain it.
func TestVulkanBatchCycleKeepsResultsAndDrainsAtHostFences(t *testing.T) {
	v := vk(t)
	pipelined := os.Getenv("FAK_VULKAN_PIPELINE_FLUSH") != "0"
	c := cpu()
	const n, steps = 256, 64
	ident := make([]float32, n*n)
	ones := make([]float32, n)
	for i := 0; i < n; i++ {
		ident[i*n+i] = 1
		ones[i] = 1
	}
	w := v.Upload(NewF32(c, []int{n, n}, ident), F32)
	one := v.Upload(NewF32(c, []int{n}, ones), F32)
	acc := v.Upload(NewF32(c, []int{n}, make([]float32, n)), F32)
	defer v.Free(w)
	defer v.Free(one)
	defer v.Free(acc)

	check := func(at int, got []float32) {
		t.Helper()
		for i, x := range got {
			if x != float32(at) {
				t.Fatalf("after %d steps acc[%d] = %v, want %d", at, i, x, at)
			}
		}
	}
	v.BeginBatch()
	for step := 1; step <= steps; step++ {
		y := v.MatMul(w, one)
		v.AddInPlace(acc, y)
		v.Free(y)
		if step%2 == 0 {
			v.CycleBatch()
			if got := v.VulkanDebugBatchInflight(); got != pipelined {
				v.FlushBatch()
				t.Fatalf("step %d in-flight = %v, want %v", step, got, pipelined)
			}
		}
		if step == steps/2+1 {
			check(step, v.Read(acc))
			if v.VulkanDebugBatchInflight() {
				t.Fatal("host readback left a batch in flight")
			}
			v.BeginBatch()
		}
	}
	v.FlushBatch()
	if v.VulkanDebugBatchInflight() || v.VulkanDebugBatchActive() {
		t.Fatal("flush left a batch open or in flight")
	}
	check(steps, v.Read(acc))
	v.Recycle()
}
