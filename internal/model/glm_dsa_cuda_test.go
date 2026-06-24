//go:build cuda

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAGLMMoeDsaBackendForward is the on-device #86/#413 witness: a lean (Q8-resident)
// GLM-MoE-DSA Prefill with the cuda backend runs GLM-5.2's MoE/FFN experts+router, the vocab head,
// AND the DSA attention's dense projections (q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj,
// o_proj — mat threaded into glmDsaAttentionStep) through k_q8_gemm on the GPU, AND now the DSA
// SPARSE-ATTENTION compute itself — the per-head softmax(scale·q·k)·ΣwV over the top-k selected keys
// — through k_dsa_sparse_attend (the compute.DSASparseBackend seam). So the attention math runs on
// the kernel; only the genuinely sparse SELECTION (the f64 index-score dots + top-k, which picks
// WHICH keys attend) + the DSA KV cache stay host-resident — the selection is host-computed by
// design so the device attends the SAME keys (no flipped top-k), keeping the forward argmax-exact.
// It must match the all-host CPU Q8 forward argmax-exact within the recorded Approx cosine floor
// (k_q8_gemm + k_dsa_sparse_attend reduction order differs from the host). A skip (no reachable GPU)
// is NOT a pass. Run on an sm_80+ node via tools/dgx_glm_gpu_witness.sh.
func TestCUDAGLMMoeDsaBackendForward(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	// The sparse attention must route to the device kernel, not silently fall back to the host loop.
	if _, ok := be.(compute.DSASparseBackend); !ok {
		t.Fatalf("cuda backend does not implement DSASparseBackend — GLM-DSA sparse attention would run host-resident, not on k_dsa_sparse_attend")
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
	t.Logf("GLM-MoE-DSA forward with MoE/FFN+head + DSA attention projections (k_q8_gemm) + DSA sparse attention (k_dsa_sparse_attend) on cuda backend: cosine=%.6f argmax cpu=%d cuda=%d tier=%s class=%s",
		c, aCPU, aCu, be.Tier(), be.Class())
	if aCPU != aCu {
		t.Fatalf("GLM-MoE-DSA cuda-backend argmax %d != CPU Q8 argmax %d (cosine=%.6f)", aCu, aCPU, c)
	}
}

// TestCUDAGLMMoeDsaIndexSelectMatches is the on-device witness for the LAST GLM-5.2 compute that was
// host-resident even after the projections AND the sparse-attention math moved to the kernel: the
// learned-indexer SCORE + top-k SELECTION (k_dsa_index_score + k_dsa_index_topk, the compute.
// DSAIndexBackend seam). A lean GLM-DSA DECODE STEP on the cuda backend runs the indexer score (f64
// accumulated on-device) and the top-k selection on the GPU; the gate is NOT a cosine but SET
// EQUALITY — because the device scores in f64 over the same losslessly-widened-f32 operands and
// selects with the identical total order, the selected key set is bit-identical to the host, so the
// decode logits must be argmax-exact vs the all-host Q8 decode. A flipped selection would diverge the
// output far past the Approx floor, so argmax-exactness here directly witnesses selection stability on
// real hardware. A skip (no reachable GPU) is NOT a pass. Run on an sm_80+ node.
func TestCUDAGLMMoeDsaIndexSelectMatches(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	// The index selection must route to the device kernel, not silently fall back to the host loop.
	if _, ok := be.(compute.DSAIndexBackend); !ok {
		t.Fatalf("cuda backend does not implement DSAIndexBackend — GLM-DSA index selection would run host-resident, not on k_dsa_index_score + k_dsa_index_topk")
	}
	path, cfg := writeTinyGLMDsaSafetensors(t)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}
	const next = 11 // the token decoded after the prompt (this is where glmDsaIndexStep runs)

	// All-host CPU Q8 reference: prefill then one decode step.
	sCPU := lean.NewSession()
	sCPU.Quant = true
	sCPU.Prefill(prompt)
	lCPU := sCPU.Step(next)

	// cuda backend: the decode step's index selection runs on k_dsa_index_score + k_dsa_index_topk.
	sCu := lean.NewBackendSession(be)
	sCu.Quant = true
	sCu.Prefill(prompt)
	lCu := sCu.Step(next)
	sCu.Close()

	if len(lCPU) != cfg.VocabSize || len(lCu) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d cu=%d want vocab=%d", len(lCPU), len(lCu), cfg.VocabSize)
	}
	c := glmDsaCosine(lCPU, lCu)
	const floor = 0.99
	if c < floor {
		t.Fatalf("GLM-MoE-DSA cuda-backend decode cosine %.6f < %.4f vs CPU Q8 — index selection may have flipped a key", c, floor)
	}
	aCPU, aCu := glmDsaArgmax(lCPU), glmDsaArgmax(lCu)
	t.Logf("GLM-MoE-DSA decode with index SELECTION on cuda backend (k_dsa_index_score + k_dsa_index_topk, f64-accumulated): cosine=%.6f argmax cpu=%d cuda=%d tier=%s class=%s",
		c, aCPU, aCu, be.Tier(), be.Class())
	if aCPU != aCu {
		t.Fatalf("GLM-MoE-DSA cuda-backend decode argmax %d != CPU Q8 argmax %d — the device selection picked a different key set (NOT selection-stable)", aCu, aCPU)
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
