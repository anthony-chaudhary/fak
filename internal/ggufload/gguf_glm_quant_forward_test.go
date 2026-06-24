package ggufload

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// gguf_glm_quant_forward_test.go — the native-753B P2 "full-model quant forward" rung in its honest
// form: a complete block-32-aligned glm_moe_dsa GGUF, loaded through the memory-lean Q8 quant path
// (LoadModelQuant — weights narrowed to int8 at load, the >VRAM lever), produces a model whose
// native glm_dsa forward (MLA + DSA indexer + MoE routing + shared experts) and Session decode RUN
// to finite logits of the right width. This is the model-level twin of P2's recorded device-GEMM
// Q8 cosines, end to end from a GGUF.
//
// It does NOT assert Q8-vs-f32 logit closeness: the codebase deliberately does not guarantee that on
// arbitrary weights — the model-package quant glm gate (TestGLMMoeDsaQuantLoadRunsResidentDSAProjections)
// pins Q8 SELF-consistency + finiteness, not Q8-vs-f32 faithfulness, and on a tiny random model the
// two are uncorrelated. So this mirrors the guarantee that actually exists.

func glmQuantNonFinite(rows [][]float32) (int, int, bool) {
	for p, row := range rows {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return p, i, true
			}
		}
	}
	return 0, 0, false
}

func TestGLMMoeDsaGGUFQuantLoadForwards(t *testing.T) {
	const V = 8
	// every matmul contraction dim is ÷32 so the Q8 builder's per-block (32) quant is whole:
	// H, qLora, kvLora, I, sharedI = 32; nH*vHead = 2*16 = 32.
	gguf := glmMoeDsaFullGGUF(32, V, 32, 32, 16, 16, 16, 2, 2, 16, 3, 32, 32)
	path := filepath.Join(t.TempDir(), "glm_q8.gguf")
	if err := os.WriteFile(path, gguf, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	qm, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant (memory-lean Q8 path) on a complete glm_moe_dsa GGUF: %v", err)
	}

	ids := []int{0, 1, 2, 3, 4}
	act := qm.Forward(ids)
	if len(act.Logits) != len(ids) {
		t.Fatalf("Q8 forward rows=%d, want %d", len(act.Logits), len(ids))
	}
	for pos, row := range act.Logits {
		if len(row) != V {
			t.Fatalf("Q8 forward row %d width=%d, want vocab %d", pos, len(row), V)
		}
	}
	if p, i, bad := glmQuantNonFinite(act.Logits); bad {
		t.Fatalf("Q8 forward logit[%d][%d] is non-finite — the quant glm_moe_dsa forward produced garbage", p, i)
	}

	// the Q8 Session decode path also runs to finite last-position logits (the serve path).
	s := qm.NewSession()
	s.Quant = true
	pre := s.Prefill(ids)
	if len(pre) != V {
		t.Fatalf("Q8 Prefill logits width=%d, want vocab %d", len(pre), V)
	}
	if _, _, bad := glmQuantNonFinite([][]float32{pre}); bad {
		t.Fatalf("Q8 Session Prefill produced non-finite logits")
	}
}
