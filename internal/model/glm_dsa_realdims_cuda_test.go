//go:build cuda

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// glm_dsa_realdims_cuda_test.go — the DEVICE half of the GLM-5.2 context-blind hunt. The CPU test
// (glm_dsa_realdims_test.go) proved the f32/Q8 HOST forward is context-dependent at real asymmetric
// per-head dims. But the real serve runs --backend cuda, so GLM-DSA's sparse-attention + index
// selection execute on the DEVICE kernels (k_dsa_sparse_attend, k_dsa_index_score, k_dsa_index_topk),
// and every existing cuda GLM-DSA test uses TINY SYMMETRIC dims + the degenerate sequenceFloats ramp,
// checking only device==host AGREEMENT on ONE prompt. Two gaps that a real-model context-blind bug
// would slip through: (1) asymmetric dims (qkNope != qkRope) are never exercised on the device; (2)
// no test asserts the DEVICE output DEPENDS ON THE INPUT.
//
// This closes both: asymmetric dims + random weights, on the cuda backend, asserting BOTH
//   (a) the device logits track the host logits (argmax-exact, cosine >= floor), AND
//   (b) two different prompts give DIFFERENT device logits (context-dependent).
// If the device DSA path goes context-blind at asymmetric dims (the suspected real-model 'apel' bug),
// (b) fails on the cuda backend while the host stays context-dependent. A skip (no GPU) is NOT a pass.

func TestCUDAGLMDsaAsymmetricDimsContextDependent(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	if _, ok := be.(compute.DSASparseBackend); !ok {
		t.Fatalf("cuda backend does not implement DSASparseBackend — DSA sparse attention would run host-resident")
	}

	cfg := tinyGLMDsaAsymmetricCfg() // qkNope=64 != qkRope=32, vHead=96 — real GLM-5.2's asymmetry
	tensors := buildGLMDsaTensorsFromCfg(t, "F32", cfg)
	path := writeTinySafetensors(t, tensors)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}

	promptA := []int{3, 17, 5}
	promptB := []int{29, 7, 31}

	hostLogits := func(prompt []int) []float32 {
		s := lean.NewSession()
		s.Quant = true
		return s.Prefill(prompt)
	}
	devLogits := func(prompt []int) []float32 {
		s := lean.NewBackendSession(be)
		s.Quant = true
		l := s.Prefill(prompt)
		s.Close()
		return l
	}

	hA, hB := hostLogits(promptA), hostLogits(promptB)
	dA, dB := devLogits(promptA), devLogits(promptB)
	if len(dA) != cfg.VocabSize || len(dB) != cfg.VocabSize {
		t.Fatalf("device logits shape A=%d B=%d want vocab=%d", len(dA), len(dB), cfg.VocabSize)
	}

	// (a) device tracks host (argmax-exact + cosine floor) per prompt.
	const floor = 0.99
	for _, tc := range []struct {
		name string
		h, d []float32
	}{{"A", hA, dA}, {"B", hB, dB}} {
		c := glmDsaCosine(tc.h, tc.d)
		ah, ad := glmDsaArgmax(tc.h), glmDsaArgmax(tc.d)
		t.Logf("prompt-%s: host argmax=%d device argmax=%d cosine(host,device)=%.6f", tc.name, ah, ad, c)
		if c < floor {
			t.Fatalf("prompt-%s: device vs host cosine %.6f < %.4f at asymmetric dims", tc.name, c, floor)
		}
		if ah != ad {
			t.Fatalf("prompt-%s: device argmax %d != host argmax %d at asymmetric dims", tc.name, ad, ah)
		}
	}

	// (b) the DEVICE output is context-dependent: two different prompts -> different device logits.
	devCos := glmDsaCosine(dA, dB)
	t.Logf("DEVICE context-dependence: argmax A=%d B=%d cosine(devA,devB)=%.6f", glmDsaArgmax(dA), glmDsaArgmax(dB), devCos)
	if devCos > 0.9999 {
		t.Fatalf("DEVICE CONTEXT-BLIND: two different prompts gave device cosine %.6f (identical) at asymmetric dims — the cuda DSA path ignores its input (the real GLM-5.2 'apel' bug, isolated to the device kernels)", devCos)
	}
}
