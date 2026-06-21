//go:build cuda

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// hal_cuda_test.go — the on-box witness that the REAL in-kernel model (a synthetic Llama
// checkpoint with the genuine weight layout) runs its decode forward pass on the GPU via
// the compute HAL (the peer's NewBackendSession path), and matches the native cpuref path
// within the Approx gate: greedy argmax-exact + prefill-logit cosine ≥ 0.999. Compiled
// only under -tags cuda; skips if no CUDA device. The CPU-ref bit-equality witness lives
// in hal_test.go (TestHALSessionMatchesLegacyCPUReference).

func TestHALDeviceForwardMatchesNative(t *testing.T) {
	be := compute.Pick("cuda")
	if be.Name() != "cuda" {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	if compute.RequireReference(be) {
		t.Fatal("cuda backend must be Approx")
	}
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
		t.Fatalf("token count: native=%d cuda=%d", len(nativeTokens), len(devTokens))
	}
	for i := range nativeTokens {
		if nativeTokens[i] != devTokens[i] {
			t.Fatalf("greedy token %d: native=%d cuda=%d (prefill cosine=%.6f, native=%v cuda=%v)",
				i, nativeTokens[i], devTokens[i], cos, nativeTokens, devTokens)
		}
	}
	t.Logf("HAL device (%s/%s): real-model decode argmax-exact over %d tokens, prefill cosine=%.8f",
		be.Name(), be.Tier(), len(devTokens), cos)
}
