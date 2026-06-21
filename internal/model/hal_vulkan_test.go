//go:build vulkan && windows && cgo

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// hal_vulkan_test.go — the on-box witness that the REAL in-kernel model (a synthetic Llama
// checkpoint with the genuine weight layout) runs its decode forward pass on the AMD GPU via
// the compute HAL (the peer's NewBackendSession path), and matches the native cpuref path
// within the Approx gate: greedy argmax-exact + prefill-logit cosine ≥ 0.999. Compiled only
// under -tags vulkan; skips if no Vulkan device. The CPU-ref bit-equality witness lives in
// hal_test.go (TestHALSessionMatchesLegacyCPUReference). This mirrors hal_cuda_test.go.

func TestHALVulkanForwardMatchesNative(t *testing.T) {
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	if compute.RequireReference(be) {
		t.Fatal("vulkan backend must be Approx")
	}
	t.Logf("vulkan backend tier=%s (dgpu = real discrete GPU; igpu = integrated)", be.Tier())

	cfg := Config{
		HiddenSize: 96, NumLayers: 4, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 256, VocabSize: 128, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompt := []int{3, 9, 44, 1, 77, 22}
	const n = 10

	// prefill-logit cosine: native (cpuref f32) vs device (Approx)
	nativeLogits := m.NewSession().Prefill(prompt)
	devLogits := m.NewBackendSession(be).Prefill(prompt)
	var dot, na, nd float64
	for i := range nativeLogits {
		dot += float64(nativeLogits[i]) * float64(devLogits[i])
		na += float64(nativeLogits[i]) * float64(nativeLogits[i])
		nd += float64(devLogits[i]) * float64(devLogits[i])
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nd))
	if cos < 0.999 {
		t.Fatalf("device prefill logit cosine %.6f < 0.999", cos)
	}

	// greedy decode argmax-exact
	nativeTokens := m.NewSession().Generate(prompt, n)
	devTokens := m.NewBackendSession(be).Generate(prompt, n)
	if len(nativeTokens) != len(devTokens) {
		t.Fatalf("token count: native=%d vulkan=%d", len(nativeTokens), len(devTokens))
	}
	for i := range nativeTokens {
		if nativeTokens[i] != devTokens[i] {
			t.Fatalf("greedy token %d: native=%d vulkan=%d (prefill cosine=%.6f, native=%v vulkan=%v)",
				i, nativeTokens[i], devTokens[i], cos, nativeTokens, devTokens)
		}
	}
	t.Logf("HAL device (%s/%s): real-model decode argmax-exact over %d tokens, prefill cosine=%.8f",
		be.Name(), be.Tier(), len(devTokens), cos)
}

func TestHALVulkanQ8ForwardMatchesComputeQ8(t *testing.T) {
	be := compute.Pick("vulkan")
	if be.Name() != "vulkan" {
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	if !be.Caps().UploadDtype {
		t.Skip("vulkan device does not expose Q8 upload/matmul support")
	}

	cfg := Config{
		HiddenSize: 96, NumLayers: 4, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 256, VocabSize: 128, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	prompt := []int{3, 9, 44, 1, 77, 22}

	cpu := m.NewSession()
	cpu.Quant = true
	cpuLogits := cpu.Prefill(prompt)
	dev := m.NewBackendSession(be)
	dev.Quant = true
	devLogits := dev.Prefill(prompt)
	if len(cpuLogits) != len(devLogits) {
		t.Fatalf("prefill logits length: cpu=%d vulkan=%d", len(cpuLogits), len(devLogits))
	}
	prefillCos := cosine(cpuLogits, devLogits)
	if prefillCos < 0.999 {
		t.Fatalf("Q8 prefill logit cosine %.6f < 0.999", prefillCos)
	}

	const nextID = 17
	cpuStep := m.NewSession()
	cpuStep.Quant = true
	cpuStep.Prefill(prompt)
	devStep := m.NewBackendSession(be)
	devStep.Quant = true
	devStep.Prefill(prompt)
	cpuNext := cpuStep.Step(nextID)
	devNext := devStep.Step(nextID)
	stepCos := cosine(cpuNext, devNext)
	if stepCos < 0.999 {
		t.Fatalf("Q8 step logit cosine %.6f < 0.999", stepCos)
	}

	devFull := m.NewBackendSession(be)
	devFull.Quant = true
	devFull.Prefill(prompt)
	wantNext := argmaxF32(devFull.Step(nextID))
	devFast := m.NewBackendSession(be)
	devFast.Quant = true
	devFast.Prefill(prompt)
	if gotNext := devFast.tokenHALArgmax(nextID, devFast.halKV.Len()); gotNext != wantNext {
		t.Fatalf("Q8 tokenHALArgmax=%d want device full-logit argmax %d", gotNext, wantNext)
	}

	t.Logf("HAL device (%s/%s): Q8 model path prefill cosine=%.8f step cosine=%.8f",
		be.Name(), be.Tier(), prefillCos, stepCos)
}
