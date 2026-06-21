package model

import (
	"math"
	"testing"
)

// TestPrefillBatchedMatchesSerial proves the batched prefill is BIT-IDENTICAL to the
// per-token tokenHidden loop — same last-token logits AND same full KV-cache state
// (K, Kraw, V, pos) in every layer. This is the contract that lets Session.Prefill run
// batched while R2/R3/R14 (which assert exact fak-vs-fak cache identity) stay green: a
// batched prefill that builds a byte-for-byte different cache than the proven path would
// silently break eviction/reuse correctness even if the logits looked fine.
func TestPrefillBatchedMatchesSerial(t *testing.T) {
	m, _ := loadFixture(t)
	vocab := m.Cfg.VocabSize
	prompt := make([]int, 40)
	for i := range prompt {
		prompt[i] = (i*1299709 + 17) % vocab
	}

	// per-token reference cache + logits
	ref := m.NewSession()
	var refXf []float32
	for _, id := range prompt {
		refXf = ref.tokenHidden(id, ref.Cache.Len())
	}
	refLogits := ref.head(refXf)

	// batched
	bat := m.NewSession()
	batXf := bat.prefillBatched(prompt)
	batLogits := bat.head(batXf)

	if len(refLogits) != len(batLogits) {
		t.Fatalf("logit len %d != %d", len(refLogits), len(batLogits))
	}
	for i := range refLogits {
		if math.Float32bits(refLogits[i]) != math.Float32bits(batLogits[i]) {
			t.Fatalf("logit %d: per-token %v != batched %v (NOT bit-identical)", i, refLogits[i], batLogits[i])
		}
	}
	if bat.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len %d != %d", bat.Cache.Len(), ref.Cache.Len())
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
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
					t.Fatalf("layer %d %s[%d]: per-token %v != batched %v (cache not bit-identical)",
						l, name, i, a[i], b[i])
				}
			}
		}
	}
	for i := range ref.Cache.pos {
		if ref.Cache.pos[i] != bat.Cache.pos[i] {
			t.Fatalf("pos[%d] %d != %d", i, ref.Cache.pos[i], bat.Cache.pos[i])
		}
	}
}
