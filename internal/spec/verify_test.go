package spec

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// verify_test.go — witnesses for the tree verify execution (#533): the single-pass
// tree-attention verify (model.VerifyForward with the ancestor mask) driving
// polymodel.AcceptTree, lossless vs plain greedy decode.

// scriptedTreeDrafter builds a tree each round whose ACCEPTED branch is exactly the greedy
// continuation (want[pos], want[pos+1], …) and whose other children are distractors that must
// be rejected. It proves the tree verify ACCEPTS the greedy path through the ancestor mask.
type scriptedTreeDrafter struct {
	want  []int
	pos   int
	width int
}

func (d *scriptedTreeDrafter) Propose(maxDepth, width int) (tokens []int, parent []int) {
	if width < 1 {
		width = 1
	}
	// Each depth adds `width` nodes: the greedy node (panel index (depth-1)*width) plus
	// (width-1) distractor siblings, all children of the previous depth's greedy node.
	for depth := 1; depth <= maxDepth; depth++ {
		par := -1
		if depth > 1 {
			par = (depth - 2) * width
		}
		greedyTok := d.tok(d.pos + depth - 1)
		for s := 0; s < width; s++ {
			if s == 0 {
				tokens = append(tokens, greedyTok)
			} else {
				tokens = append(tokens, (greedyTok+s)%256) // guaranteed != greedyTok (s in [1,255])
			}
			parent = append(parent, par)
		}
	}
	return tokens, parent
}
func (d *scriptedTreeDrafter) tok(i int) int {
	if i >= 0 && i < len(d.want) {
		return d.want[i]
	}
	return 0
}
func (d *scriptedTreeDrafter) Commit(committed []int) { d.pos += len(committed) }

// constTreeDrafter proposes a fixed tree of constant tokens (independent of the target) — a
// generic, non-scripted proposer. Whatever it accepts or rejects, SpeculativeTree must stay
// token-identical to greedy (losslessness does not depend on drafter quality).
type constTreeDrafter struct{ tok int }

func (d *constTreeDrafter) Propose(maxDepth, width int) (tokens []int, parent []int) {
	if width < 1 {
		width = 1
	}
	for depth := 1; depth <= maxDepth; depth++ {
		par := -1
		if depth > 1 {
			par = (depth - 2) * width
		}
		for s := 0; s < width; s++ {
			tokens = append(tokens, d.tok)
			parent = append(parent, par)
		}
	}
	return tokens, parent
}
func (d *constTreeDrafter) Commit(_ []int) {}

// TestSpeculativeTreeLosslessGreedyPath: with the greedy path present, the single-pass tree
// verify accepts multi-token paths per round and the output is token-identical to greedy.
func TestSpeculativeTreeLosslessGreedyPath(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("tree speculation verifies many branches in one pass"))
	const N = 28
	want := greedyDecode(target, prompt, N)

	const maxDepth, width = 3, 3
	got, proposed, accepted := SpeculativeTree(target.NewSession(), prompt, N, maxDepth, width, &scriptedTreeDrafter{want: want, width: width})
	assertEqualTokens(t, "tree-greedy-path", got, want)
	if proposed == 0 {
		t.Fatal("VACUOUS: drafter proposed 0 tokens")
	}
	if accepted <= 0 {
		t.Fatalf("VACUOUS: 0 accepted tokens — the greedy-path branch was never accepted (proposed=%d)", proposed)
	}
	if accepted >= proposed {
		t.Fatalf("VACUOUS: accepted %d >= proposed %d — no distractor branches were rejected", accepted, proposed)
	}
	t.Logf("tree-greedy-path: proposed %d tokens across rounds, %d accepted (distractors rejected by the mask)", proposed, accepted)
}

// TestSpeculativeTreeLosslessArbitrary: with an arbitrary (non-scripted) proposer the output
// is STILL token-identical to greedy — losslessness does not depend on the drafter.
func TestSpeculativeTreeLosslessArbitrary(t *testing.T) {
	target := model.NewSynthetic(cfg(48, 3, 4, 2, 16, 96))
	prompt := bytesToIDs([]byte("lossless regardless of drafter quality"))
	const N = 20
	want := greedyDecode(target, prompt, N)

	got, proposed, _ := SpeculativeTree(target.NewSession(), prompt, N, 3, 2, &constTreeDrafter{tok: 7})
	assertEqualTokens(t, "tree-arbitrary", got, want)
	if proposed == 0 {
		t.Fatal("VACUOUS: drafter proposed 0 tokens")
	}
}

// TestVerifyTreeRewindsAndCommitsCleanly: one VerifyTree round leaves the target cache
// byte-identical to a greedy session that decoded the same committed tokens (the rewind +
// recommit contract). Proven by exact full-logit-vector equality on a continuation.
func TestVerifyTreeRewindsAndCommitsCleanly(t *testing.T) {
	target := model.NewSynthetic(cfg(48, 3, 4, 2, 16, 96))
	prompt := bytesToIDs([]byte("rewind and recommit keeps the cache clean"))

	// Greedy reference: decode g0, g1, g2 into the cache (base+3), matching the 3-token
	// commit of an accepted A->C path (g0, g1, plus the g2 correction).
	ref := target.NewSession()
	refLogits := ref.Prefill(prompt)
	g0 := argmax(refLogits)
	g1 := argmax(ref.Step(g0))
	g2 := argmax(ref.Step(g1))
	ref.Step(g2) // ref cache now base+3

	// Tree round on a fresh session: a depth-2 tree whose accepted branch is g0 -> g1, with a
	// distractor sibling at depth 1 and a distractor niece at depth 2.
	s := target.NewSession()
	tl := s.Prefill(prompt)
	base := s.Cache.Len()
	distractor := (g0 + 7) % 256
	if distractor == g0 {
		distractor = (g0 + 13) % 256
	}
	tokens := []int{g0, distractor, g1} // A=g0, B=distractor, C=g1
	parent := []int{-1, -1, 0}          // C's parent is A
	committed, tree := VerifyTree(s, tl, tokens, parent)

	// Accepted path A -> C commits [g0, g1, g2-correction].
	wantCommitted := []int{g0, g1, g2}
	if len(tree.Nodes) != 4 {
		t.Fatalf("SpecTree has %d nodes, want 4", len(tree.Nodes))
	}
	if !equalInts(committed, wantCommitted) {
		t.Fatalf("committed %v, want %v", committed, wantCommitted)
	}
	// The speculative panel was rewound: cache len == base (prefix only) right after VerifyTree.
	if s.Cache.Len() != base {
		t.Fatalf("after VerifyTree rewind: cache len %d, want base %d", s.Cache.Len(), base)
	}
	// Recommit the accepted chain (what the SpeculativeTree driver does) and prove the cache
	// is byte-exact to the greedy session that decoded the same tokens.
	s.VerifyForward(committed, nil, nil)
	if s.Cache.Len() != base+len(committed) {
		t.Fatalf("after recommit: cache len %d, want %d", s.Cache.Len(), base+len(committed))
	}
	assertContinuationsMatch(t, "rewind-recommit-bit-exact", s, ref, 55, 4)
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
