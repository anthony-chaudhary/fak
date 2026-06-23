package spec

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// TestSplicePrefillShareIsBitExactAndGated witnesses the #534 KVCache.Clone splice
// end-to-end: a compatible-family consumer (different ModelID, same Family +
// PrefixDigest) forks the provider's prefix cache, and continuing from that spliced
// prefix is BIT-EXACT (full logit-vector equality) to a consumer that prefilled the
// prefix directly — the proof that the cross-model share loses nothing. The negative
// cases (different digest, different family, nil cache) never splice.
func TestSplicePrefillShareIsBitExactAndGated(t *testing.T) {
	m := model.NewSynthetic(cfg(48, 3, 3, 1, 16, 96))
	prefix := bytesToIDs([]byte("the prefix is shared across a declared-compatible family"))

	// A provider session prefills the prefix; its cache is the share source.
	providerSession := m.NewSession()
	providerSession.Prefill(prefix)
	providerCache := providerSession.Cache

	provider := polymodel.Model{ID: "provider", Family: "qwen", PrefixDigest: "sha-AAA"}

	// (a) A compatible consumer (different ID, same family+digest) splices losslessly.
	//     The spliced prefix, used as the consumer's prefix, yields a continuation
	//     bit-exact to a consumer that prefilled the prefix directly — the end-to-end
	//     proof that the cross-model Clone splice loses nothing.
	consumer := polymodel.Model{ID: "consumer", Family: "qwen", PrefixDigest: "sha-AAA"}
	spliced, ok := SplicePrefillShare(provider, consumer, providerCache)
	if !ok {
		t.Fatal("a share-compatible consumer must splice")
	}
	if spliced == providerCache {
		t.Fatal("splice must return a CLONE, not the provider's own cache pointer")
	}
	if spliced.Len() != providerCache.Len() {
		t.Fatalf("splice length %d, want %d (the provider's prefix length)", spliced.Len(), providerCache.Len())
	}
	splicedSession := m.SessionFromPrefix(spliced)
	reference := m.NewSession()
	reference.Prefill(prefix)
	assertContinuationsMatch(t, "cross-model-splice", splicedSession, reference, 42, 6)

	// (b) Same family, DIFFERENT PrefixDigest (the KV would differ) → no splice.
	fork := polymodel.Model{ID: "fork", Family: "qwen", PrefixDigest: "sha-BBB"}
	if c, ok := SplicePrefillShare(provider, fork, providerCache); ok || c != nil {
		t.Fatalf("a different PrefixDigest must NOT splice: clone=%v ok=%v", c, ok)
	}
	// (c) Different family (different tokenizer) → no splice.
	alien := polymodel.Model{ID: "alien", Family: "llama", PrefixDigest: "sha-AAA"}
	if c, ok := SplicePrefillShare(provider, alien, providerCache); ok || c != nil {
		t.Fatalf("a different Family must NOT splice: clone=%v ok=%v", c, ok)
	}
	// (d) A bare consumer (no declared shareable band) → no splice.
	bare := polymodel.Model{ID: "bare", Family: "qwen"}
	if c, ok := SplicePrefillShare(provider, bare, providerCache); ok || c != nil {
		t.Fatalf("an empty PrefixDigest must NOT splice: clone=%v ok=%v", c, ok)
	}
	// (e) A nil provider cache → no splice (the caller falls back to a full prefill).
	if c, ok := SplicePrefillShare(provider, consumer, nil); ok || c != nil {
		t.Fatalf("a nil provider cache must NOT splice: clone=%v ok=%v", c, ok)
	}
}

// TestPrefillSharePolicyForAgreesWithCanShare witnesses the bridge half of #534: the
// cachemeta policy's Allowed verdict is exactly polymodel.CanShare, and an Allowed
// policy carries the family/digest audit trail. The rule lives in ONE place
// (polymodel.CanShare); cachemeta consumes its verdict here.
func TestPrefillSharePolicyForAgreesWithCanShare(t *testing.T) {
	base := polymodel.Model{ID: "base", Family: "qwen", PrefixDigest: "sha-AAA"}
	cases := []struct {
		name string
		a, b polymodel.Model
		want bool
	}{
		{"twin shares", base, polymodel.Model{ID: "twin", Family: "qwen", PrefixDigest: "sha-AAA"}, true},
		{"self always shares", base, base, true},
		{"different digest does not share", base, polymodel.Model{ID: "fork", Family: "qwen", PrefixDigest: "sha-BBB"}, false},
		{"different family does not share", base, polymodel.Model{ID: "alien", Family: "llama", PrefixDigest: "sha-AAA"}, false},
		{"empty digest never shares", base, polymodel.Model{ID: "bare", Family: "qwen"}, false},
	}
	for _, c := range cases {
		p := PrefillSharePolicyFor(c.a, c.b)
		if p.Allowed != c.want {
			t.Fatalf("%s: policy Allowed=%v want %v", c.name, p.Allowed, c.want)
		}
		if p.Allowed != polymodel.CanShare(c.a, c.b) {
			t.Fatalf("%s: policy disagrees with polymodel.CanShare", c.name)
		}
		if c.want && (p.Family == "" || p.PrefixDigest == "") {
			t.Fatalf("%s: an Allowed policy must carry the family/digest audit trail", c.name)
		}
	}
}
