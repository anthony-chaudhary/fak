package model

import (
	"testing"
)

// TestCachedDecodeMatchesPrefill is the rung-2 internal-consistency witness: the
// kernel-owned KV cache must make incremental decode produce the SAME last-token
// logits as the (rung-1-verified) full forward pass. If the cache, the per-token
// RoPE position, or the causal accumulation were wrong, this diverges.
func TestCachedDecodeMatchesPrefill(t *testing.T) {
	m, doc := loadFixture(t)
	for _, p := range doc.Prompts {
		full := m.Forward(p.Ids)
		sess := m.NewSession()
		cached := sess.Prefill(p.Ids)
		d, _ := maxAbsDiff(cached, full.Logits[len(p.Ids)-1])
		ga, fa := argmax(cached), argmax(full.Logits[len(p.Ids)-1])
		t.Logf("prompt %d cached-vs-prefill logits max|Δ|=%.3e argmax cached=%d prefill=%d",
			p.Index, d, ga, fa)
		if ga != fa {
			t.Errorf("prompt %d cached decode argmax %d != prefill %d", p.Index, ga, fa)
		}
		// BIT-IDENTICAL, not merely close: Forward (no-cache matRows) and Prefill (batched
		// matMulBatch + KV cache) both route every reduction through the same fdot/dot in
		// the same order, so the cached-decode logits must equal the full-prefill logits
		// to the last bit. Self-witnesses the doc's "R2 max|Δ|=0" claim (was a loose 1e-3).
		if d != 0 {
			t.Errorf("prompt %d cached vs prefill not bit-identical: max abs diff %.3e (expected 0)", p.Index, d)
		}
		if sess.Cache.Len() != len(p.Ids) {
			t.Errorf("prompt %d cache len %d != %d", p.Index, sess.Cache.Len(), len(p.Ids))
		}
	}
}

// TestGreedyMatchesHFOracle is the rung-2 EXTERNAL witness and the headline "the
// model runs inside the kernel" proof: greedy generation driven entirely by the
// in-kernel forward pass + kernel-owned KV cache must reproduce HF's greedy
// continuation token-for-token. HF authored the target ids, not us.
func TestGreedyMatchesHFOracle(t *testing.T) {
	m, doc := loadFixture(t)
	for _, p := range doc.Prompts {
		want := p.GreedyIds
		got := m.NewSession().Generate(p.Ids, len(want))
		t.Logf("prompt %d greedy go=%v", p.Index, got)
		t.Logf("prompt %d greedy hf=%v", p.Index, want)
		if len(got) != len(want) {
			// HF may stop at EOS; compare the common prefix
			n := len(got)
			if len(want) < n {
				n = len(want)
			}
			got, want = got[:n], want[:n]
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("prompt %d greedy token %d: go=%d hf=%d (full go=%v)",
					p.Index, i, got[i], want[i], got)
				break
			}
		}
	}
}
