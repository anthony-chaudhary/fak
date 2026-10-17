//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// Drives packed K-quant operands through the sequence-prefill projection switch
// (the same call the resident prefill makes) against the CPU dequant oracles.
func TestVulkanQwen35SequenceKQuantOperands(t *testing.T) {
	v := vk(t)
	const tokens, out, in = 3, 67, 512
	for _, tc := range []struct {
		name   string
		dt     Dtype
		have   bool
		host   func(raw []byte) Tensor
		raw    []byte
		oracle func(out, in, tokens int, raw []byte, x []float32) []float32
	}{
		{"Q6_K", Q6_K, v.SupportsQ6KMatMul(), func(r []byte) Tensor { return NewQ6K(Default(), []int{out, in}, r) }, q6KStrixFixture(out, in), q6KDequantF32Oracle},
		{"Q5_K", Q5_K, v.SupportsQ5KMatMul(), func(r []byte) Tensor { return NewQ5K(Default(), []int{out, in}, r) }, q5KStrixFixture(out, in), q5KDequantF32Oracle},
		{"Q3_K", Q3_K, v.SupportsQ3KMatMul(), func(r []byte) Tensor { return NewQ3K(Default(), []int{out, in}, r) }, q3kStrixFixture(out, in), q3kDequantF32Oracle},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.have {
				t.Skipf("loaded Vulkan bundle lacks the optional %s matmul pipeline", tc.dt)
			}
			x := q6KStrixInput(tokens, in)
			want := tc.oracle(out, in, tokens, tc.raw, x)
			dw := v.Upload(tc.host(tc.raw), tc.dt)
			defer v.Free(dw)
			if dw.Dtype != tc.dt {
				t.Fatalf("resident dtype=%s, want packed %s", dw.Dtype, tc.dt)
			}
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer v.Free(dx)
			vulkanMu.Lock()
			start := len(v.transient)
			y := v.qwen35VulkanSequenceMatMulLocked(dw, dx, tokens)
			vulkanMu.Unlock()
			got := v.Read(y)
			vulkanMu.Lock()
			v.qwen35VulkanSequenceReleaseLocked(start)
			vulkanMu.Unlock()
			if len(got) != tokens*out {
				t.Fatalf("panel size=%d want %d", len(got), tokens*out)
			}
			var maxDelta, maxRef float64
			for i := range want {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("element %d = %g, want finite", i, got[i])
				}
				maxDelta = math.Max(maxDelta, math.Abs(float64(got[i]-want[i])))
				maxRef = math.Max(maxRef, math.Abs(float64(want[i])))
			}
			if maxDelta > 2e-5*math.Max(1, maxRef) {
				t.Fatalf("%s sequence panel max-abs delta=%g exceeds 2e-5 relative gate (reference max=%g)", tc.dt, maxDelta, maxRef)
			}
		})
	}
}
