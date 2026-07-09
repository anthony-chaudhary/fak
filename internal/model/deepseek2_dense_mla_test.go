package model

import (
	"math"
	"testing"
)

// deepseek2_dense_mla_test.go — hermetic witness for the DeepSeek-V2/V3 "dense-MLA seam": fak serves
// DeepSeek by reusing the glm_moe_dsa MLA + MoE forward MINUS the DSA lightning indexer. A real
// deepseek2 checkpoint has IndexNHeads==0, so every query attends its FULL causal prefix instead of a
// learned top-k. The seam lives in two independently-written places — glmDsaAttnSeqShared (cacheless
// Forward) and glmDsaAttentionStep (Session decode) — each of which, when IndexNHeads==0, sets
// topK = glmDsaPositions(...) and lets glmDsaSelectedCausalKeys filter it to exactly the causal prefix.
//
// Two witnesses, both with NO external oracle:
//   - TestDeepSeek2DenseSeamSelectsFullCausalPrefix pins the CRUX purely: topK = all-positions reduces
//     to exactly [0..t] — no causal key dropped, none acausal — for both the cacheless and decode forms.
//   - TestDeepSeek2DenseMLASeam builds a tiny deepseek2 dense-MLA model at ASYMMETRIC per-head dims and
//     pins the properties a correct forward MUST have: it takes the MLA layout branch (not the DSA
//     indexer branch); its Session decode is exactly self-consistent (incremental Prefill+Step ==
//     whole-sequence Prefill, the KV-cache invariant a broken seam would violate); and both the
//     cacheless and decode paths are context-DEPENDENT (the attention actually attends its input — the
//     guard against a context-blind forward, cf. the GLM-5.2 "apel" bug).
//
// NB: cacheless Forward and Session Prefill are NOT bit-parity on the MLA family (they are distinct
// implementations; the same gap exists for GLM-DSA), so this does not assert cross-path logit equality.

func tinyDeepSeek2DenseCfg() Config {
	// deepseek2 == the asymmetric GLM-DSA geometry MINUS the DSA indexer (IndexNHeads==0, no
	// IndexerTypes). Asymmetric per-head dims (qkNope 64 != qkRope 32, vHead 96 != qkNope 64) so a
	// nope/rope boundary or qkHead/vHead stride bug on the deepseek2 path is exercised, not masked.
	return Config{
		HiddenSize:        64,
		NumLayers:         2,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           96,
		IntermediateSize:  64,
		VocabSize:         41,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		EOSTokenID:        -1,
		ModelType:         "deepseek2",
		Architectures:     []string{"DeepseekV2ForCausalLM"},
		QLoraRank:         64,
		KVLoraRank:        32,
		QKNopeHeadDim:     64,
		QKRopeHeadDim:     32,
		VHeadDim:          96,
		IndexNHeads:       0, // the dense-MLA seam: no DSA lightning indexer
		TieWordEmbeddings: false,
	}
}

// TestDeepSeek2DenseSeamSelectsFullCausalPrefix pins the crux of the seam with no weights: when the
// dense branch feeds topK = glmDsaPositions(...), glmDsaSelectedCausalKeys must return exactly the
// causal prefix [0..t] in order — every causal key present (dense), none acausal.
func TestDeepSeek2DenseSeamSelectsFullCausalPrefix(t *testing.T) {
	const seq = 6

	// Cacheless form: topK[t] = glmDsaPositions(seq) for every t; selection reduces it to [0..t].
	full := glmDsaPositions(seq)
	if len(full) != seq {
		t.Fatalf("glmDsaPositions(%d) len=%d, want %d", seq, len(full), seq)
	}
	for q := 0; q < seq; q++ {
		got, ok := glmDsaSelectedCausalKeys(full, q, seq)
		if !ok {
			t.Fatalf("cacheless selection rejected at queryPos=%d", q)
		}
		if len(got) != q+1 {
			t.Fatalf("cacheless queryPos=%d selected %d keys, want %d (full causal prefix)", q, len(got), q+1)
		}
		for j, kp := range got {
			if kp != j {
				t.Fatalf("cacheless queryPos=%d selected[%d]=%d, want %d (exactly 0..t, in order)", q, j, kp, j)
			}
		}
	}

	// Decode form: at step pos the branch feeds topK = glmDsaPositions(pos+1); selection is [0..pos].
	for pos := 0; pos < seq; pos++ {
		got, ok := glmDsaSelectedCausalKeys(glmDsaPositions(pos+1), pos, seq)
		if !ok || len(got) != pos+1 {
			t.Fatalf("decode dense selection at pos=%d: ok=%v n=%d, want %d", pos, ok, len(got), pos+1)
		}
		for j, kp := range got {
			if kp != j {
				t.Fatalf("decode pos=%d selected[%d]=%d, want %d", pos, j, kp, j)
			}
		}
	}
}

func TestDeepSeek2DenseMLASeam(t *testing.T) {
	cfg := tinyDeepSeek2DenseCfg()
	tensors := buildGLMDsaTensorsFromCfg(t, "F32", cfg)
	path := writeTinySafetensors(t, tensors)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	// (1) It takes the shared MLA+MoE layout branch, but NOT the DSA-indexer branch.
	if !lean.Cfg.usesMLAMoELayout() {
		t.Fatalf("usesMLAMoELayout()=false, want true — deepseek2 must route through the MLA+MoE forward")
	}
	if lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("isGLMMoeDsa()=true, want false — deepseek2 must NOT trip the DSA-indexer gate")
	}

	promptA := []int{3, 17, 5, 29}
	promptB := []int{7, 31, 2, 19}

	// (2) Session decode self-consistency: incremental PrefillNoLogits(prefix)+Step(last) reproduces
	// the whole-sequence Prefill bit-for-bit. This is the KV-cache invariant of the decode dense
	// branch (glmDsaAttentionStep) — a seam that dropped a key or mis-set topK would break it.
	whole := lean.NewSession().Prefill(promptA)
	if len(whole) != cfg.VocabSize {
		t.Fatalf("Prefill logits len=%d, want vocab=%d", len(whole), cfg.VocabSize)
	}
	s := lean.NewSession()
	s.PrefillNoLogits(promptA[:len(promptA)-1])
	step := s.Step(promptA[len(promptA)-1])
	if d, at := maxAbsDiff(whole, step); d > 1e-6 {
		t.Fatalf("decode self-consistency: PrefillNoLogits(prefix)+Step vs Prefill(whole) max|Δlogit|=%.3e at vocab %d (>1e-6)", d, at)
	}

	// (3) The decode dense-MLA attention actually attends its input: different prompts -> different logits.
	other := lean.NewSession().Prefill(promptB)
	if cos := realDimsCosine(whole, other); cos > 0.9999 {
		t.Fatalf("decode context-blind: two different prompts gave cosine %.6f (~1) — attention ignores input", cos)
	}
	if std := stddev(whole); std < 1e-6 {
		t.Fatalf("degenerate logits: std-dev %.3e ~0 (constant output)", std)
	}

	// (4) The cacheless dense branch (glmDsaAttnSeqShared) is wired too: Forward runs, returns
	// finite logits of the right shape, and is likewise context-dependent.
	fa := lean.Forward(promptA).Logits
	fb := lean.Forward(promptB).Logits
	if len(fa) != len(promptA) {
		t.Fatalf("Forward returned %d positions, want %d", len(fa), len(promptA))
	}
	lastA, lastB := fa[len(promptA)-1], fb[len(promptB)-1]
	if len(lastA) != cfg.VocabSize {
		t.Fatalf("Forward last-token logits len=%d, want vocab=%d", len(lastA), cfg.VocabSize)
	}
	for i, v := range lastA {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("cacheless Forward produced non-finite logit at %d: %v", i, v)
		}
	}
	if cos := realDimsCosine(lastA, lastB); cos > 0.9999 {
		t.Fatalf("cacheless context-blind: two different prompts gave cosine %.6f (~1)", cos)
	}
}
