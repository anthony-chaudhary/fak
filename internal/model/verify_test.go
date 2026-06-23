package model

import (
	"math"
	"testing"
)

// cfgV is a small plain-PreNorm GQA synthetic config (the regime VerifyForward batches),
// matching the shape internal/spec and cmd/polymodelbench use.
func cfgV(hidden, layers, nHeads, nKV, headDim, inter int) Config {
	return Config{
		HiddenSize:        hidden,
		NumLayers:         layers,
		NumHeads:          nHeads,
		NumKVHeads:        nKV,
		HeadDim:           headDim,
		IntermediateSize:  inter,
		VocabSize:         256,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

func argmaxV(v []float32) int {
	bi, bv := 0, float32(0)
	for i, x := range v {
		if i == 0 || x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// TestVerifyForwardChainMatchesSerial proves the single-pass batched verify (the chain
// case: nil pos, nil allow) is BIT-IDENTICAL to P sequential Session.Step calls — same
// per-position logits AND the same full KV-cache state (K/Kraw/V/pos in every layer). This
// is the losslessness contract that lets internal/spec.SpeculativeGreedy swap its sequential
// verify loop for one VerifyForward call: a batched verify that built a byte-different cache
// would silently corrupt the promote/squash offsets, so byte-equality is the gate.
func TestVerifyForwardChainMatchesSerial(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	drafts := []int{13, 17, 4, 99, 200, 5, 42} // arbitrary, incl. repeats

	ref := m.NewSession()
	ref.Prefill(prompt)
	refLogits := make([][]float32, len(drafts))
	for j, d := range drafts {
		refLogits[j] = ref.Step(d)
	}

	ver := m.NewSession()
	ver.Prefill(prompt)
	got := ver.VerifyForward(drafts, nil, nil)

	if len(got) != len(refLogits) {
		t.Fatalf("VerifyForward returned %d logit vecs, want %d", len(got), len(refLogits))
	}
	for j := range refLogits {
		a, b := refLogits[j], got[j]
		if len(a) != len(b) {
			t.Fatalf("pos %d logit width %d != %d", j, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("pos %d logit[%d]: serial %v != verify %v (NOT bit-identical)", j, i, a[i], b[i])
			}
		}
	}

	if ver.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len: verify %d != serial %d", ver.Cache.Len(), ref.Cache.Len())
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K":    {ref.Cache.K[l], ver.Cache.K[l]},
			"Kraw": {ref.Cache.Kraw[l], ver.Cache.Kraw[l]},
			"V":    {ref.Cache.V[l], ver.Cache.V[l]},
		} {
			a, b := pair[0], pair[1]
			if len(a) != len(b) {
				t.Fatalf("layer %d %s len %d != %d", l, name, len(a), len(b))
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("layer %d %s[%d]: serial %v != verify %v (cache not bit-identical)", l, name, i, a[i], b[i])
				}
			}
		}
	}
	for i := range ref.Cache.pos {
		if ref.Cache.pos[i] != ver.Cache.pos[i] {
			t.Fatalf("pos[%d]: serial %d != verify %d", i, ref.Cache.pos[i], ver.Cache.pos[i])
		}
	}
}

// TestVerifyForwardEmpty is the trivial edge: no candidates ⇒ nil, no cache mutation.
func TestVerifyForwardEmpty(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	s := m.NewSession()
	s.Prefill([]int{1, 2, 3})
	before := s.Cache.Len()
	if got := s.VerifyForward(nil, nil, nil); got != nil {
		t.Fatalf("empty VerifyForward = %v, want nil", got)
	}
	if s.Cache.Len() != before {
		t.Fatalf("empty VerifyForward mutated cache: %d != %d", s.Cache.Len(), before)
	}
}

// TestVerifyForwardTreeMaskIsolatesBranches proves the tree-attention mask is correct: a
// node verified with the ancestor mask gets EXACTLY the greedy context (its argmax matches
// a sequential Step down its branch), and a sibling branch never contaminates it (adding a
// sibling leaves the node's logits bit-identical — the mask excludes non-ancestors).
//
// Tree (BFS): A=g0 (depth1), B=distractor (depth1), C=child-of-A=g1 (depth2). The accepted
// path A->C is the greedy continuation; B is a rejected sibling that must not affect A or C.
func TestVerifyForwardTreeMaskIsolatesBranches(t *testing.T) {
	m := NewSynthetic(cfgV(48, 3, 4, 2, 16, 96))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Greedy reference: the true continuation down the accepted branch.
	greedy := m.NewSession()
	tl := greedy.Prefill(prompt)
	g0 := argmaxV(tl)
	g1 := argmaxV(greedy.Step(g0))
	g2 := argmaxV(greedy.Step(g1))

	base := greedy.Cache.Len() // == len(prompt); positions are 0-indexed contiguous
	// Panel (BFS): A=g0 (depth1), B=a distractor != g0 (depth1), C=g1 (depth2, child of A).
	distractor := (g0 + 7) % 256
	if distractor == g0 {
		distractor = (g0 + 13) % 256
	}
	tokens := []int{g0, distractor, g1}
	parent := []int{-1, -1, 0} // C's parent is A (panel index 0)
	// pos[i] = base + depth(i) - 1; siblings (same depth) share a position. depth walks the
	// parent chain to the root.
	pos := make([]int, len(tokens))
	for i := range pos {
		depth := 1
		for p := parent[i]; p >= 0; p = parent[p] {
			depth++
		}
		pos[i] = base + depth - 1
	}

	// Ancestor matrix within the panel: anc[q][k] iff k is q, or k is an ancestor of q.
	anc := make([][]bool, len(tokens))
	for q := range anc {
		anc[q] = make([]bool, len(tokens))
		anc[q][q] = true
		for p := parent[q]; p >= 0; p = parent[p] {
			anc[q][p] = true
		}
	}
	allow := func(q, k int) bool { return anc[q][k] }

	// Verify the whole tree in one pass on a fresh session.
	tree := m.NewSession()
	tree.Prefill(prompt)
	logits := tree.VerifyForward(tokens, pos, allow)
	if logits == nil {
		t.Fatal("tree VerifyForward returned nil for a supported regime")
	}
	// A (panel 0, token g0) predicts the token after g0 given prefix+g0 = the greedy g1.
	if got := argmaxV(logits[0]); got != g1 {
		t.Errorf("tree node A argmax = %d, want greedy g1=%d (ancestor mask gave wrong context)", got, g1)
	}
	// C (panel 2, token g1) predicts the token after g1 given prefix+g0+g1 = the greedy g2.
	if got := argmaxV(logits[2]); got != g2 {
		t.Errorf("tree node C argmax = %d, want greedy g2=%d (ancestor mask gave wrong context)", got, g2)
	}

	// Isolation: A's logits must be BIT-IDENTICAL to a solo verify of just [A] (no sibling B,
	// no niece C) — proving B's presence never reached A through the mask. The solo panel
	// attends to prefix + itself, exactly A's key set in the full tree.
	solo := m.NewSession()
	solo.Prefill(prompt)
	soloA := solo.VerifyForward([]int{g0}, []int{base}, func(q, k int) bool { return true })
	for i := range logits[0] {
		if math.Float32bits(logits[0][i]) != math.Float32bits(soloA[0][i]) {
			t.Fatalf("node A logit[%d]: tree %v != solo %v (sibling branch leaked through the mask)", i, logits[0][i], soloA[0][i])
		}
	}

	// Sanity: B (the distractor) is NOT the greedy token, so it differs from g0.
	if tokens[1] == g0 {
		t.Fatalf("distractor == g0; fix the test")
	}
}
