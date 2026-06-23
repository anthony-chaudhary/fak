package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestGLMMoeDsaBackendGEMMMatchesCPU is the #86 (partial) witness: a GLM-MoE-DSA Prefill with a
// compute.Backend attached routes its dense GEMMs — MoE/FFN experts + router (via the matKernel
// swap in decodeBandGLMDsa), the vocab head (glmDsaHead), AND the DSA attention's dense projections
// (q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj, o_proj — mat threaded into glmDsaAttentionStep)
// — through the backend, while only the sparse DSA glue (index-score dots, top-k, sparse softmax/
// ΣwV) + KV stay host-resident. Run against the cpu-ref backend it must reproduce the all-host CPU
// forward argmax-exact (the GEMMs differ only in f32 reduction order: cpu-ref MatMul's fdot tree vs
// the host matRows sequential dot). The SAME code path, with the cuda backend + lean Q8 weights,
// runs those GEMMs on k_q8_gemm — GLM-5.2's MoE/FFN/head AND its attention projections on the GPU
// pure kernel (the on-device run is witnessed by tools/dgx_glm_gpu_witness.sh on an sm_80 node). The
// shape-level proof that the attention projections reach the backend is
// TestGLMMoeDsaBackendRoutesAttentionProjections.
func TestGLMMoeDsaBackendGEMMMatchesCPU(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("model family = %q, want glm_moe_dsa", m.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}

	// All-host reference (residentKernel; no backend).
	lCPU := m.NewSession().Prefill(prompt)

	// Backend path: MoE/FFN + head GEMMs go through backendKernel on the cpu-ref backend.
	be := compute.Default()
	sBE := m.NewBackendSession(be)
	if sBE.Backend == nil {
		t.Fatalf("NewBackendSession produced a nil-backend session")
	}
	lBE := sBE.Prefill(prompt)

	if len(lCPU) != cfg.VocabSize || len(lBE) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d be=%d want vocab=%d", len(lCPU), len(lBE), cfg.VocabSize)
	}
	aCPU, aBE := glmDsaArgmax(lCPU), glmDsaArgmax(lBE)
	if aCPU != aBE {
		t.Fatalf("GLM-MoE-DSA backend-GEMM argmax %d != CPU argmax %d (routing diverged, not just f32 order)", aBE, aCPU)
	}
	d, at := maxAbsDiff(lCPU, lBE)
	if d > 1e-3 {
		t.Fatalf("GLM-MoE-DSA backend-GEMM max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — routing bug", d, at)
	}
	t.Logf("GLM-MoE-DSA forward with MoE/FFN+head GEMMs on compute backend %q: argmax-exact (%d), max|Δ|=%.3e vs all-host CPU",
		be.Name(), aBE, d)
}

func glmDsaArgmax(v []float32) int {
	bi, bv := 0, float32(-1e38)
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
