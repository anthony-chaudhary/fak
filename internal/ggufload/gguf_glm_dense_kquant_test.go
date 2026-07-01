package ggufload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGLMMoeDsaDenseKQuantStaysOffResidentStore is the regression gate for the GLM-5.2
// cpu-offload serve panic (2026-07-01, sm_80 box): the real UD-Q4_K_M GGUF quantizes DENSE
// MLA projections (q_a_proj et al.) to residentable k-quants (Q8_0/Q6_K), and the dense
// resident-k-quant fast path admitted them into kqw — but the glm_moe_dsa device serve
// (glmDsaWeightHAL) uploads every dense weight from the f32/q8/q4kw stores and has no kqw
// kernels, so the first request panicked "got resident raw expert-quant weight
// model.layers.0.self_attn.q_a_proj.weight on the device path". A glm_moe_dsa dense k-quant
// must take the dequant→Q8 route; ONLY the routed experts stay raw-resident in kqw.
func TestGLMMoeDsaDenseKQuantStaysOffResidentStore(t *testing.T) {
	const (
		H, V                = 256, 8
		qLora, kvLora       = 32, 32
		qkNope, qkRope, vHd = 16, 16, 16
		nH                  = 2
		idxHeads, idxDim    = 2, 16
		E, I, sharedI       = 3, 256, 256
	)
	const denseName = "blk.0.attn_q_a.weight" // canonical: model.layers.0.self_attn.q_a_proj.weight
	gguf := glmMoeDsaFullGGUFWithTypes(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI,
		func(name string) TensorType {
			switch {
			case strings.Contains(name, "_exps.weight"):
				return TensorQ6_K // routed experts: raw-residentable, must stay in kqw
			case name == denseName:
				return TensorQ8_0 // dense MLA projection: the real GGUF's layout that panicked
			default:
				return TensorF32
			}
		})
	path := filepath.Join(t.TempDir(), "glm_dense_kq.gguf")
	if err := os.WriteFile(path, gguf, 0o644); err != nil {
		t.Fatal(err)
	}

	prof := NewLoadProfiler()
	m, err := LoadModelQ4KProfile(path, prof)
	if err != nil {
		t.Fatalf("LoadModelQ4KProfile: %v", err)
	}

	// The dense Q8_0 projection must NOT be raw-resident; only the E*3 routed experts are.
	if m.HasKQuant("model.layers.0.self_attn.q_a_proj.weight") {
		t.Fatalf("dense q_a_proj landed in the resident raw k-quant store — the glm_moe_dsa device serve has no kqw kernels for dense weights (the 2026-07-01 serve panic)")
	}
	if got := m.KQuantCount(); got != E*3 {
		t.Fatalf("resident k-quant tensors = %d, want %d (routed experts only)", got, E*3)
	}

	// The load-path breakdown must show the dense Q8_0 tensor on the dequant route.
	var denseRow *LoadPathStat
	for i := range prof.loadPathRows() {
		r := prof.loadPathRows()[i]
		if r.QuantType == "Q8_0" && !r.Expert {
			denseRow = &r
		}
	}
	if denseRow == nil || denseRow.DequantTensors != 1 || denseRow.ResidentTensors != 0 {
		t.Fatalf("dense Q8_0 load-path row = %+v, want dequant=1 resident=0", denseRow)
	}

	// The forward still runs to finite logits over the dequanted dense projection.
	ids := []int{0, 1, 2, 3}
	act := m.Forward(ids)
	if act == nil || len(act.Logits) != len(ids) {
		t.Fatalf("Forward returned %d logit rows, want %d", lenLogits(act), len(ids))
	}
	if p, i, bad := glmQuantNonFinite(act.Logits); bad {
		t.Fatalf("logit[%d][%d] non-finite after dense dequant routing", p, i)
	}
}
