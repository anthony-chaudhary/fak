//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
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

// A window that records a checked V4.1 op still cycles: the closed batch goes in
// flight with its checked bookkeeping stashed, the host readback drains it through
// the checked retire, and results stay bit-exact — including against the same work
// run through the blocking flush path (greedy-sha parity). An open step window stays
// serialized by the eligibility exclusion, pinned by the source-contract test.
// [SW-VERIFIED] only; no HW timing claim.
// fak-test:runtime fast est=1s lane=default
func TestVulkanBatchCyclePipelinesV41CheckedBatches(t *testing.T) {
	v := vk(t)
	const width = 256
	if !v.SupportsV41ExactRMSNorm(width, 1e-6) {
		t.Skip("V4.1 exact RMSNorm module unavailable on this device")
	}
	pipelined := vulkanPipelinedCycles()
	c := cpu()
	host := make([]float32, width)
	for i := range host {
		host[i] = float32(i%5) + 1
	}
	x := v.Upload(NewF32(c, []int{width}, host), F32)
	w := v.Upload(NewF32(c, []int{width}, host), F32)
	defer v.Free(x)
	defer v.Free(w)
	_ = v.Read(w)

	v.BeginBatch()
	out, err := v.V41ExactRMSNorm(x, w, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Free(out)
	v.AddInPlace(w, w)
	v.CycleBatch()
	if got := v.VulkanDebugBatchInflight(); got != pipelined {
		v.FlushBatch()
		t.Fatalf("after cycling a V4.1-checked batch in-flight = %v, want %v", got, pipelined)
	}
	got := v.Read(out)
	want := v41ExactRMSNormHostRow(host, host, 1e-6)
	for i, wv := range want {
		if math.Float32bits(got[i]) != math.Float32bits(wv) {
			t.Fatalf("out[%d] = %#08x (%v), want %#08x (%v)", i, math.Float32bits(got[i]), got[i], math.Float32bits(wv), wv)
		}
	}
	if v.VulkanDebugBatchInflight() {
		t.Fatal("host readback left a batch in flight")
	}
	v.FlushBatch()
	if v.VulkanDebugBatchInflight() || v.VulkanDebugBatchActive() {
		t.Fatal("flush left a batch open or in flight")
	}

	// Blocking-path parity: the identical recorded work flushed through the checked
	// end-submit-wait must come out bit-identical to the cycled result.
	wb := v.Upload(NewF32(c, []int{width}, host), F32)
	defer v.Free(wb)
	v.BeginBatch()
	outB, err := v.V41ExactRMSNorm(x, wb, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Free(outB)
	v.AddInPlace(wb, wb)
	v.FlushBatch()
	gotB := v.Read(outB)
	for i := range gotB {
		if math.Float32bits(gotB[i]) != math.Float32bits(got[i]) {
			t.Fatalf("blocking path out[%d] = %#08x (%v), pipelined %#08x (%v)",
				i, math.Float32bits(gotB[i]), gotB[i], math.Float32bits(got[i]), got[i])
		}
	}
	digest := make([]byte, width*4)
	for i, f := range got {
		binary.LittleEndian.PutUint32(digest[i*4:], math.Float32bits(f))
	}
	t.Logf("[SW-VERIFIED] V4.1 batch-cycle vs blocking-path greedy-sha parity sha256=%x", sha256.Sum256(digest))
	v.Recycle()
}
