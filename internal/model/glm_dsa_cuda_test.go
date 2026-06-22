//go:build cuda

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAGLMMoeDsaBackendForward is the on-device #86 (partial) witness: a lean (Q8-resident)
// GLM-MoE-DSA Prefill with the cuda backend runs GLM-5.2's MoE/FFN experts+router and the vocab head
// through k_q8_gemm on the GPU (backendKernel + glmDsaHead), while the DSA index-scoring + sparse
// attention stay host-resident. It must match the all-host CPU Q8 forward argmax-exact within the
// recorded Approx cosine floor (k_q8_gemm's reduction order differs from the host qMatRows). A skip
// (no reachable GPU) is NOT a pass. Run on an sm_80+ node via tools/dgx_glm_gpu_witness.sh.
func TestCUDAGLMMoeDsaBackendForward(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	path, cfg := writeTinyGLMDsaSafetensors(t)
	lean, err := LoadSafetensorsQuant(path, cfg) // q8-resident GLM-DSA -> k_q8_gemm on the backend
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}

	// All-host CPU Q8 reference.
	sCPU := lean.NewSession()
	sCPU.Quant = true
	lCPU := sCPU.Prefill(prompt)

	// cuda backend: GLM-5.2 MoE/FFN + head GEMMs on k_q8_gemm (the GPU pure kernel).
	sCu := lean.NewBackendSession(be)
	sCu.Quant = true
	lCu := sCu.Prefill(prompt)
	sCu.Close()

	if len(lCPU) != cfg.VocabSize || len(lCu) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d cu=%d want vocab=%d", len(lCPU), len(lCu), cfg.VocabSize)
	}
	c := glmDsaCosine(lCPU, lCu)
	const floor = 0.99 // tiny synthetic + Q8-on-device vs Q8-on-host reduction-order Approx
	if c < floor {
		t.Fatalf("GLM-MoE-DSA cuda-backend forward cosine %.6f < %.4f vs CPU Q8", c, floor)
	}
	aCPU, aCu := glmDsaArgmax(lCPU), glmDsaArgmax(lCu)
	t.Logf("GLM-MoE-DSA forward with MoE/FFN+head GEMMs on cuda backend (k_q8_gemm): cosine=%.6f argmax cpu=%d cuda=%d tier=%s class=%s",
		c, aCPU, aCu, be.Tier(), be.Class())
	if aCPU != aCu {
		t.Fatalf("GLM-MoE-DSA cuda-backend argmax %d != CPU Q8 argmax %d (cosine=%.6f)", aCu, aCPU, c)
	}
}

func glmDsaCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
