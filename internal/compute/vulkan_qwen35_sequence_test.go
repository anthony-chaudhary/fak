//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// Three distinct rows and an output width crossing one workgroup expose token
// addressing errors that a one-token decode or repeated input cannot detect.
func TestVulkanQwen35SequenceQuantizedPanelsMatchCPU(t *testing.T) {
	v := vk(t)
	const tokens, out, in = 3, 67, 512
	rng := rand.New(rand.NewSource(11999))
	x := make([]float32, tokens*in)
	w := make([]float32, out*in)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * float32(1+i/in)
	}
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * .1
	}
	raw4 := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < len(raw4)/q4kSuperBlock; b++ {
		randQ4KBlockC(rng, raw4[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	raw2 := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < len(raw2)/q2kSuperBlock; b++ {
		block := raw2[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 80; i++ {
			block[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(block[80:82], .015625)
		binaryPutFloat16(block[82:84], .0078125)
	}
	for _, host := range []Tensor{
		NewF32(Default(), []int{out, in}, w),
		QuantizeQ8(Default(), []int{out, in}, w, 32),
		NewQ4K(Default(), []int{out, in}, raw4),
		NewQ2K(Default(), []int{out, in}, raw2),
	} {
		t.Run(host.Dtype.String(), func(t *testing.T) {
			dw := v.Upload(host, host.Dtype)
			defer v.Free(dw)
			dx := v.Upload(NewF32(Default(), []int{tokens, in}, x), F32)
			defer v.Free(dx)
			dy := v.BatchedMatMul(dw, dx, tokens)
			defer v.Free(dy)
			got := v.Read(dy)
			if len(got) != tokens*out {
				t.Fatalf("panel size=%d want %d", len(got), tokens*out)
			}
			for token := 0; token < tokens; token++ {
				// Independent CPU scalar matvec also checks the panel's row layout.
				want := Default().Read(Default().MatMul(host, NewF32(Default(), []int{in}, x[token*in:(token+1)*in])))
				for row, expected := range want {
					actual := got[token*out+row]
					if math.IsNaN(float64(actual)) || math.IsInf(float64(actual), 0) || math.Abs(float64(actual-expected)) > 2e-3+2e-4*math.Abs(float64(expected)) {
						t.Fatalf("token=%d row=%d got=%g want=%g", token, row, actual, expected)
					}
				}
			}
			t.Logf("engine=fak-native backend=vulkan dtype=%s tokens=%d in=%d out=%d cpu_parity=true", host.Dtype, tokens, in, out)
		})
	}
}

func TestVulkanQwen35SequenceGeometryValidation(t *testing.T) {
	v := &vulkanBackend{}
	req := Qwen35SequencePrefillRequest{
		Path:              Qwen35SequencePrefillPath,
		TokenIDs:          []int{1, 2},
		StartPos:          0,
		Hidden:            Qwen35DenseHidden,
		Intermediate:      Qwen35DenseIntermediate,
		NumHeads:          Qwen35DenseQueryHeads,
		NumKVHeads:        Qwen35DenseKVHeads,
		HeadDim:           Qwen35DenseHeadDim,
		RotaryDim:         Qwen35DenseHeadDim / 4,
		NumKeyHeads:       Qwen35DenseGDNGroups,
		NumValueHeads:     Qwen35DenseGDNRank,
		KeyHeadDim:        Qwen35DenseGDNState,
		ValueHeadDim:      Qwen35DenseGDNState,
		ConvKernel:        Qwen35DenseGDNConv,
		RMSNormEpsilon:    1e-6,
		Layers:            make([]Qwen35SequenceLayer, 4),
		States:            make([]Qwen35SequenceState, 4),
		RoPEThetaForLayer: make([]float64, 4),
	}
	for l := range req.Layers {
		req.Layers[l].Linear = (l+1)%4 != 0
		req.RoPEThetaForLayer[l] = 1e7
	}

	// 1. Wrong path
	badPath := req
	badPath.Path = "wrong-path"
	if _, err := v.validateQwen35VulkanSequence(badPath); err == nil {
		t.Fatal("expected error on wrong path")
	}

	// 2. Empty tokens
	badTokens := req
	badTokens.TokenIDs = nil
	if _, err := v.validateQwen35VulkanSequence(badTokens); err == nil {
		t.Fatal("expected error on empty token IDs")
	}

	// 3. Negative start pos
	badPos := req
	badPos.StartPos = -1
	if _, err := v.validateQwen35VulkanSequence(badPos); err == nil {
		t.Fatal("expected error on negative start pos")
	}

	// 4. Odd rotary dimension
	badRotary := req
	badRotary.RotaryDim = 63
	if _, err := v.validateQwen35VulkanSequence(badRotary); err == nil {
		t.Fatal("expected error on odd rotary dim")
	}

	// 5. Rotary exceeds head dim
	badRotaryOver := req
	badRotaryOver.RotaryDim = req.HeadDim + 2
	if _, err := v.validateQwen35VulkanSequence(badRotaryOver); err == nil {
		t.Fatal("expected error on rotary exceeding head dim")
	}

	// 6. 65 layers (64 text + 1 MTP metadata layer) must be rejected
	mtp65 := req
	mtp65.Layers = make([]Qwen35SequenceLayer, 65)
	mtp65.States = make([]Qwen35SequenceState, 65)
	mtp65.RoPEThetaForLayer = make([]float64, 65)
	for l := range mtp65.Layers {
		mtp65.Layers[l].Linear = (l+1)%4 != 0
		mtp65.RoPEThetaForLayer[l] = 1e7
	}
	if _, err := v.validateQwen35VulkanSequence(mtp65); err == nil {
		t.Fatal("expected error rejecting 65th MTP metadata layer")
	}
}
