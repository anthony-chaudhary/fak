package model

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// recordingBackend wraps a compute.Backend and records the [out,in] shape of every weight
// handed to MatMul/BatchedMatMul, delegating the actual arithmetic unchanged. It is a
// behavioral probe: a GLM-MoE-DSA dense projection executes on the backend if and only if
// its weight reaches be.MatMul, so the recorded shape set is direct, author-independent
// evidence of WHICH GEMMs ran on the kernel (vs host residentMatRows). It is transparent —
// cpu-ref's MatMul reads the host buffer, never the backend identity — so wrapping it does
// not perturb numerics (the argmax check below pins that).
type recordingBackend struct {
	compute.Backend
	mu          sync.Mutex
	shapes      map[[2]int]int // [out,in] -> MatMul call count
	sparseCalls int            // DSASparseAttend call count (the sparse-attention device op)
	sparseSel   int            // total selected keys handed to DSASparseAttend (proves real work, not an empty call)
}

func newRecordingBackend(be compute.Backend) *recordingBackend {
	return &recordingBackend{Backend: be, shapes: map[[2]int]int{}}
}

// DSASparseAttend records that GLM-DSA's sparse attention reached the backend device op (rather
// than staying host-resident in glmDsaAttendCached) and delegates to the wrapped backend. It
// makes recordingBackend itself a compute.DSASparseBackend, so the model's type-assert routes
// the sparse attend here — direct, author-independent evidence the attention math ran on the
// backend. Transparent: it forwards to the wrapped backend's exact arithmetic.
func (r *recordingBackend) DSASparseAttend(q, selK, selV compute.Tensor, nSel, nH, qkHead, vHead int, scale float32) compute.Tensor {
	r.mu.Lock()
	r.sparseCalls++
	r.sparseSel += nSel
	r.mu.Unlock()
	return r.Backend.(compute.DSASparseBackend).DSASparseAttend(q, selK, selV, nSel, nH, qkHead, vHead, scale)
}

func (r *recordingBackend) sparse() (calls, sel int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sparseCalls, r.sparseSel
}

func (r *recordingBackend) record(w compute.Tensor) {
	if len(w.Shape) != 2 {
		return
	}
	r.mu.Lock()
	r.shapes[[2]int{w.Shape[0], w.Shape[1]}]++
	r.mu.Unlock()
}

func (r *recordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	r.record(w)
	return r.Backend.MatMul(w, x)
}

func (r *recordingBackend) BatchedMatMul(w, X compute.Tensor, P int) compute.Tensor {
	r.record(w)
	return r.Backend.BatchedMatMul(w, X, P)
}

func (r *recordingBackend) saw(out, in int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.shapes[[2]int{out, in}] > 0
}

// TestGLMMoeDsaBackendRoutesAttentionProjections is the #86 (partial, next slice) witness that
// GLM-5.2's DSA-attention DENSE projections now execute on the compute.Backend, not just the
// MoE/FFN+head. Threading the matKernel into glmDsaAttentionStep routes q_a/q_b, kv_a/kv_b, the
// learned-indexer wq_b/wk/weights_proj, and o_proj through backendKernel — so on the cuda backend
// they run on k_q8_gemm (the GPU pure kernel). A recordingBackend wrapping cpu-ref proves each of
// those projection shapes reaches be.MatMul; the only DSA work that stays host-resident is the
// genuinely sparse glue (index-score dots, top-k, sparse softmax/ΣwV) and the DSA KV cache. The
// forward also stays argmax-exact vs the all-host reference, so the routing is correct, not just
// present.
func TestGLMMoeDsaBackendRoutesAttentionProjections(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("model family = %q, want glm_moe_dsa", m.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}

	rec := newRecordingBackend(compute.Default())
	sBE := m.NewBackendSession(rec)
	if sBE.Backend == nil {
		t.Fatalf("NewBackendSession produced a nil-backend session")
	}
	lBE := sBE.Prefill(prompt)

	// Every DSA-attention dense projection must have reached the backend MatMul.
	nH := cfg.NumHeads
	qkHead := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	want := []struct {
		name    string
		out, in int
	}{
		{"q_a_proj", cfg.QLoraRank, cfg.HiddenSize},
		{"q_b_proj", nH * qkHead, cfg.QLoraRank},
		{"kv_a_proj_with_mqa", cfg.KVLoraRank + cfg.QKRopeHeadDim, cfg.HiddenSize},
		{"kv_b_proj", nH * (cfg.QKNopeHeadDim + cfg.VHeadDim), cfg.KVLoraRank},
		{"o_proj", cfg.HiddenSize, nH * cfg.VHeadDim},
		{"indexer.wq_b", cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank},
		{"indexer.wk", cfg.IndexHeadDim, cfg.HiddenSize},
		{"indexer.weights_proj", cfg.IndexNHeads, cfg.HiddenSize},
	}
	for _, w := range want {
		if !rec.saw(w.out, w.in) {
			t.Errorf("GLM-DSA attention projection %s [%d,%d] never reached backend MatMul (still host-resident)", w.name, w.out, w.in)
		}
	}

	// Routing correctness: backend forward must match the all-host CPU forward argmax-exact.
	lCPU := m.NewSession().Prefill(prompt)
	if len(lCPU) != cfg.VocabSize || len(lBE) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d be=%d want vocab=%d", len(lCPU), len(lBE), cfg.VocabSize)
	}
	aCPU, aBE := glmDsaArgmax(lCPU), glmDsaArgmax(lBE)
	if aCPU != aBE {
		t.Fatalf("GLM-DSA backend (incl. attention projections) argmax %d != CPU argmax %d", aBE, aCPU)
	}
	d, at := maxAbsDiff(lCPU, lBE)
	if d > 1e-3 {
		t.Fatalf("GLM-DSA backend forward max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — routing bug", d, at)
	}
	t.Logf("GLM-DSA attention projections on backend %q: all 8 reached MatMul; argmax-exact (%d), max|Δ|=%.3e vs all-host",
		rec.Name(), aBE, d)
}

// TestGLMMoeDsaBackendRoutesSparseAttention is the witness that GLM-5.2's DSA SPARSE-ATTENTION
// compute — the per-head softmax(scale·q·k)·V over the top-k selected keys, the last piece of the
// attention sublayer that stayed host-resident after the projections moved to the backend — now
// executes on the compute.Backend via the DSASparseBackend seam (on the cuda backend:
// k_dsa_sparse_attend, the pure GPU kernel). A recordingBackend wrapping cpu-ref proves the sparse
// attend reached be.DSASparseAttend (real work: sparseSel > 0 selected keys), and the forward stays
// argmax-exact vs the all-host reference — because the key SELECTION (index scores + top-k) is
// host-computed, the device attends the SAME keys, so its only divergence is f32 reduction order.
// What remains host-resident is now only the genuinely sparse SELECTION (the f64 index-score dots +
// top-k) and the O(topK) key gather — the attention math itself is on the kernel.
func TestGLMMoeDsaBackendRoutesSparseAttention(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("model family = %q, want glm_moe_dsa", m.Cfg.archFamilyKey())
	}
	prompt := []int{3, 17, 5, 23}

	rec := newRecordingBackend(compute.Default())
	if _, ok := compute.Backend(rec).(compute.DSASparseBackend); !ok {
		t.Fatalf("recordingBackend over %q is not a DSASparseBackend — sparse attend would fall back to host", rec.Name())
	}
	sBE := m.NewBackendSession(rec)
	if sBE.Backend == nil {
		t.Fatalf("NewBackendSession produced a nil-backend session")
	}
	lBE := sBE.Prefill(prompt)

	// The sparse attention must have run on the backend (not host), over real selected keys.
	calls, sel := rec.sparse()
	if calls == 0 {
		t.Fatalf("GLM-DSA sparse attention never reached the backend DSASparseAttend (still host-resident)")
	}
	if sel == 0 {
		t.Fatalf("GLM-DSA sparse attention reached the backend but over 0 selected keys (empty/no-op routing)")
	}

	// Routing correctness: the backend forward (sparse attend on cpu-ref) must match all-host argmax-exact.
	lCPU := m.NewSession().Prefill(prompt)
	if len(lCPU) != cfg.VocabSize || len(lBE) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d be=%d want vocab=%d", len(lCPU), len(lBE), cfg.VocabSize)
	}
	aCPU, aBE := glmDsaArgmax(lCPU), glmDsaArgmax(lBE)
	if aCPU != aBE {
		t.Fatalf("GLM-DSA backend (incl. sparse attention) argmax %d != CPU argmax %d", aBE, aCPU)
	}
	d, at := maxAbsDiff(lCPU, lBE)
	if d > 1e-3 {
		t.Fatalf("GLM-DSA backend forward max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — sparse-attend routing bug", d, at)
	}
	t.Logf("GLM-DSA sparse attention on backend %q: %d DSASparseAttend calls over %d selected keys; argmax-exact (%d), max|Δ|=%.3e vs all-host",
		rec.Name(), calls, sel, aBE, d)
}
