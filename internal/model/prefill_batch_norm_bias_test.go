package model

import (
	"math"
	"testing"
)

// TestPrefillBatchedMatchesSerialWithNormBias extends the batched-prefill contract
// (TestPrefillBatchedMatchesSerial) onto the LayerNorm-with-BIAS axis.
//
// Why the existing contract test could not catch this: it runs on loadFixture's SmolLM2
// export, which is RMSNorm — structurally bias-free. Every norm bias is ABSENT there, so
// a lane that drops the bias is bit-identical to one that applies it and the assertion
// passes vacuously on that axis. But StableLM, GPT-NeoX, Falcon, MPT and biased Cohere all
// carry learned input_layernorm.bias / post_attention_layernorm.bias / model.norm.bias, and
// prefillBatched hard-passed nil at all three norm sites (rmsnormCfg, arch.go:460, is
// literally the nil-bias wrapper) while the per-token path it is contracted to match
// bit-for-bit passes them (blockStep via attentionNorms/mlpNorms, weights.go:104; finalNorm,
// weights.go:529). The result was a silently different hidden state — and, because the
// pre-attention bias feeds q/k/v, a silently different KV cache.
//
// All three sites propagate to the logits, so the logit assertion is what catches any one of
// them; the cache assertion that follows LOCALIZES a regression to the pre-attention norm
// (input_layernorm.bias feeds q/k/v, so dropping it moves the cached K/V, while
// model.norm.bias moves only the logits).
func TestPrefillBatchedMatchesSerialWithNormBias(t *testing.T) {
	cfg := llamaArchConfig()
	cfg.LayerNorm = true // mean-subtracting LayerNorm — the norm kind that carries a bias
	H := cfg.HiddenSize

	extra := map[string][]int{"model.norm.bias": {H}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		extra[p+"input_layernorm.bias"] = []int{H}
		extra[p+"post_attention_layernorm.bias"] = []int{H}
	}
	m := newSyntheticExtra(cfg, extra)

	// Premise guard: this config must actually REACH the batched lane, or the test proves
	// nothing about production. prefillF32 routes to prefillBatched exactly when
	// q8PrefillNeedsTokenLoop is false (dynamic_precision.go:113), and a PreNorm family with
	// no MoE/ALiBi/output-gate lands there — so this is a live wrong-result path.
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("premise broken: this config no longer routes to prefillBatched, so the test is vacuous")
	}
	// Non-vacuity guard: an all-zero bias is indistinguishable from a dropped one. If the
	// synthetic filler ever stops producing non-zero extras this test must fail loudly
	// rather than quietly stop witnessing the bug.
	for name := range extra {
		nonZero := false
		for _, v := range m.tensor(name) {
			if v != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			t.Fatalf("%s is all zero: a dropped bias would be undetectable, test is vacuous", name)
		}
	}

	prompt := make([]int, 12)
	for i := range prompt {
		prompt[i] = (i*1299709 + 17) % cfg.VocabSize
	}

	// per-token reference (blockStep + finalNorm, both bias-aware)
	ref := m.NewSession()
	var refXf []float32
	for _, id := range prompt {
		refXf = ref.tokenHidden(id, ref.Cache.Len())
	}
	refLogits := ref.head(refXf)

	// batched
	bat := m.NewSession()
	batLogits := bat.head(bat.prefillBatched(prompt))

	if len(refLogits) != len(batLogits) {
		t.Fatalf("logit len %d != %d", len(refLogits), len(batLogits))
	}
	for i := range refLogits {
		if math.Float32bits(refLogits[i]) != math.Float32bits(batLogits[i]) {
			t.Fatalf("logit %d: per-token %v != batched %v (norm bias dropped in the batched lane)",
				i, refLogits[i], batLogits[i])
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K":    {ref.Cache.K[l], bat.Cache.K[l]},
			"Kraw": {ref.Cache.Kraw[l], bat.Cache.Kraw[l]},
			"V":    {ref.Cache.V[l], bat.Cache.V[l]},
		} {
			a, b := pair[0], pair[1]
			if len(a) != len(b) {
				t.Fatalf("layer %d %s len %d != %d", l, name, len(a), len(b))
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("layer %d %s[%d]: per-token %v != batched %v (pre-attn norm bias dropped)",
						l, name, i, a[i], b[i])
				}
			}
		}
	}
}
