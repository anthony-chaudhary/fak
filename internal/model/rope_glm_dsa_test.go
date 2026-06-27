package model

import (
	"math"
	"testing"
)

// TestGLMDsaInvFreqUsesQKRopeDenominator pins the GLM-DSA RoPE frequency denominator to
// qk_rope_head_dim, not cfg.HeadDim. On the real GLM-5.2 GGUF, cfg.HeadDim is set from
// attention.key_length=576 (the MLA latent key dim) while the rotary width is qk_rope_head_dim=64.
// Before the fix, invFreq used denom=cfg.HeadDim=576, making the high-index frequencies ~2780x too
// large — the long-range positional signal wrapped multiple turns as position grew and the decode
// collapsed into a repeating attractor (the witnessed "apel" repetition). The canonical DeepSeek-V3
// MLA precompute_freqs_cis uses dim = qk_rope_head_dim.
//
// This is a millisecond CPU check (no GPU, no weights): the consumed inv_freq entries (j in
// [0, qkRope/2)) must equal 1/theta^(2j/qkRope), and must DIFFER from the buggy denom=HeadDim form.
func TestGLMDsaInvFreqUsesQKRopeDenominator(t *testing.T) {
	const (
		theta  = 10000.0
		headD  = 576 // attention.key_length on the real GLM-5.2 GGUF (the MLA latent key dim)
		qkRope = 64  // the actual rotary width
	)
	cfg := Config{
		ModelType:     "glm_moe_dsa",
		Architectures: []string{"GlmMoeDsaForCausalLM"},
		HeadDim:       headD,
		QKRopeHeadDim: qkRope,
		RopeTheta:     theta,
		// leave PartialRotaryFactor=0 (full) — GLM-DSA does not use partial rotary.
	}
	if !cfg.isGLMMoeDsa() {
		t.Fatal("cfg is not glm_moe_dsa")
	}

	inv := invFreq(cfg, 0)
	half := qkRope / 2 // glmDsaApplyInterleavedRoPE consumes inv[0..half)
	if len(inv) < half {
		t.Fatalf("invFreq returned %d entries, need at least %d (qkRope/2)", len(inv), half)
	}

	// The consumed entries must use denom = qkRope (64), NOT HeadDim (576).
	for j := 0; j < half; j++ {
		want := 1.0 / math.Pow(theta, float64(2*j)/float64(qkRope))
		if rel := math.Abs(inv[j]-want) / want; rel > 1e-9 {
			t.Fatalf("inv[%d] = %.6e, want %.6e (denom must be qkRope=%d, not HeadDim=%d) — rel err %.2e",
				j, inv[j], want, qkRope, headD, rel)
		}
	}

	// And it must be FAR from the buggy HeadDim-denominator form at the high index (the tell): the
	// buggy value is ~2780x larger, so the ratio buggy/correct must be well above 1.
	buggy31 := 1.0 / math.Pow(theta, float64(2*(half-1))/float64(headD))
	if ratio := buggy31 / inv[half-1]; ratio < 100 {
		t.Fatalf("inv[%d]=%.6e is only %.1fx below the BUGGY denom=HeadDim value %.6e — the qkRope denom is not active (expect ~2780x)",
			half-1, inv[half-1], ratio, buggy31)
	}
	t.Logf("GLM-DSA inv_freq: inv[%d]=%.6e (qkRope=64 denom), buggy HeadDim=576 would give %.6e (~%.0fx larger)",
		half-1, inv[half-1], buggy31, buggy31/inv[half-1])
}
