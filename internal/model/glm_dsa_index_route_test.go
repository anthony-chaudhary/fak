package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestGLMMoeDsaBackendRoutesIndexSelection is the witness for the LAST GLM-5.2 compute that stayed
// host-resident even after the dense projections AND the sparse-attention math moved to the kernel:
// the learned-indexer SCORE + top-k SELECTION that picks WHICH keys a query attends. Threading the
// dsaIndexKernel type-assert into glmDsaIndexStep (the per-token decode hot path) routes the score +
// top-k through backendKernel.indexSelect — so on the cuda backend it runs on k_dsa_index_score +
// k_dsa_index_topk (the pure GPU kernel, f64-accumulated scoring). A recordingBackend wrapping cpu-ref
// proves a decode step's index selection reached be.DSAIndexSelect over real cached keys, and the
// decode logits stay argmax-exact vs the all-host decode — because the device accumulates the score
// dot in f64 over the same losslessly-widened-f32 operands, the selected key SET is bit-identical, so
// the selection-stability boundary that kept this host-side is satisfied, not bypassed.
func TestGLMMoeDsaBackendRoutesIndexSelection(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("model family = %q, want glm_moe_dsa", m.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}
	const next = 11 // the token decoded after the prompt

	rec := newRecordingBackend(compute.Default())
	if _, ok := compute.Backend(rec).(compute.DSAIndexBackend); !ok {
		t.Fatalf("recordingBackend over %q is not a DSAIndexBackend — index selection would fall back to host", rec.Name())
	}
	sBE := m.NewBackendSession(rec)
	if sBE.Backend == nil {
		t.Fatalf("NewBackendSession produced a nil-backend session")
	}
	sBE.Prefill(prompt)
	callsBefore, _ := rec.index()
	lBE := sBE.Step(next) // one decode step: this is where glmDsaIndexStep runs

	// The index selection must have run on the backend during the decode step, over real cached keys.
	calls, keys := rec.index()
	if calls == callsBefore {
		t.Fatalf("GLM-DSA index selection never reached the backend DSAIndexSelect during decode (still host-resident)")
	}
	if keys == 0 {
		t.Fatalf("GLM-DSA index selection reached the backend but scored 0 keys (empty/no-op routing)")
	}

	// Routing correctness: the backend decode (index selection on cpu-ref) must match the all-host
	// decode argmax-exact — proving the device-shaped selection picked the SAME keys.
	sCPU := m.NewSession()
	sCPU.Prefill(prompt)
	lCPU := sCPU.Step(next)
	if len(lCPU) != cfg.VocabSize || len(lBE) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d be=%d want vocab=%d", len(lCPU), len(lBE), cfg.VocabSize)
	}
	aCPU, aBE := glmDsaArgmax(lCPU), glmDsaArgmax(lBE)
	if aCPU != aBE {
		t.Fatalf("GLM-DSA backend decode (incl. index selection) argmax %d != CPU argmax %d (selection flipped a key)", aBE, aCPU)
	}
	d, at := maxAbsDiff(lCPU, lBE)
	if d > 1e-3 {
		t.Fatalf("GLM-DSA backend decode max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — index-selection routing bug", d, at)
	}
	t.Logf("GLM-DSA index selection on backend %q: %d DSAIndexSelect calls over %d scored keys during decode; argmax-exact (%d), max|Δ|=%.3e vs all-host",
		rec.Name(), calls-callsBefore, keys, aBE, d)
}
