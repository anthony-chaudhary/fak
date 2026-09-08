package compute

import (
	"math"
	"strings"
	"testing"
)

func TestVulkanElementwiseChainLowering(t *testing.T) {
	chain := recurrentSwiGLUElementwiseChain()
	dispatch, err := lowerVulkanElementwiseChain(chain)
	if err != nil {
		t.Fatalf("lower recurrent SwiGLU chain: %v", err)
	}
	const wantDigest = "ef1d3b0c4169d1dc7feb4c03365737238941e2f5aa1f76d129d6493cdea493b2"
	if dispatch.Digest != wantDigest {
		t.Fatalf("chain digest = %q, want %q", dispatch.Digest, wantDigest)
	}
	if dispatch.Kernel != vulkanElementwiseKernelSwiGLU || dispatch.Dispatches != 1 || dispatch.IntermediateBarriers != 0 {
		t.Fatalf("lowered dispatch = %+v, want one SwiGLU dispatch without an intermediate barrier", dispatch)
	}

	gate := []float32{-7, -1, 0, 0.5, 3, 9}
	up := []float32{2, -4, 0.25, 8, -1.5, 0.125}
	got, err := evaluateElementwiseChain(chain, gate, up)
	if err != nil {
		t.Fatalf("evaluate recurrent SwiGLU chain: %v", err)
	}
	var maxAbs float64
	for i := range gate {
		want := gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
		delta := math.Abs(float64(got[i] - want))
		if delta > maxAbs {
			maxAbs = delta
		}
	}
	if maxAbs > 1e-6 {
		t.Fatalf("scalar oracle max_abs=%g, want <=1e-6", maxAbs)
	}

	unsupported := elementwiseChain{
		Inputs: 2,
		Ops: append(append([]elementwiseInstruction(nil), chain.Ops...), elementwiseInstruction{
			Op: elementwiseAdd, A: 4, B: 0,
		}),
	}
	_, err = lowerVulkanElementwiseChain(unsupported)
	if err == nil || !strings.Contains(err.Error(), "unsupported chain") {
		t.Fatalf("lower chain with trailing add error = %v, want unsupported chain", err)
	}
}
