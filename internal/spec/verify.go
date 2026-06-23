package spec

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// verify.go — the TREE verify execution (rung #533), the twin of SpeculativeGreedy's chain
// verify. Where the chain verifies a LINEAR draft in one batched pass (model.VerifyForward
// with nil pos / nil allow), the tree verifies a whole speculation TREE (Medusa / EAGLE-2 /
// SpecInfer: many candidate continuations sharing a KV prefix) in one pass with a
// tree-attention mask — each node attends only to its ancestor chain, so siblings (alternative
// next tokens) never see one another. polymodel.AcceptTree then picks the accepted path and
// the committed tokens advance the target.
//
// Commit model (and why no Sink): a tree's accepted vs rejected branches are NOT one
// contiguous cache span (the case the Sink's Promote/Rollback resolve for the chain), so
// VerifyTree rewinds the WHOLE speculation with one bit-exact model.KVCache.Evict and
// recommits the accepted path as one chain VerifyForward. The result is byte-identical to a
// greedy decode of the accepted tokens: the rewind restores the prefix (no survivor re-RoPE,
// since the prefix keeps pos[i]==i and the whole speculative panel is dropped together), and
// the recommit appends the accepted chain at sequential positions. The single-pass tree
// VERIFY (the expensive forward) is the throughput half; recommitting the (short) accepted
// chain is the price of not yet having a tree-aware KV-compaction primitive that would keep
// the accepted path's KV directly — an honest, sequenced cost, not a correctness gap.

// TreeDrafter proposes a speculation tree for one round — the many-models / multi-token
// generalization of the chain Drafter. Where Drafter.Draft proposes one linear sequence,
// TreeDrafter.Propose returns a TREE of candidate continuations that the target verifies in
// a single tree-attention pass.
type TreeDrafter interface {
	// Propose returns the panel tokens (BFS order, the non-root nodes) and parent[i], the
	// panel index of token i's parent (-1 ⇒ token i is a depth-1 child of the root, i.e. the
	// current committed position). maxDepth bounds the tree depth and width bounds the
	// children per node; a proposer may return fewer. The panel order must be such that a
	// node appears AFTER its parent (BFS / top-down), so depths are non-decreasing.
	Propose(maxDepth, width int) (tokens []int, parent []int)
	// Commit advances the drafter by the REAL tokens the target committed this round (the
	// accepted path + correction), so the next Propose continues from the true context.
	Commit(committed []int)
}

// VerifyTree runs ONE single-pass tree-attention verify round and returns the REAL tokens it
// commits (the accepted path's tokens + the correction) plus the filled SpecTree. The target
// cache is left byte-identical to a greedy decode of those committed tokens (rewind +
// recommit, see the file note). committedLogits is the target's next-token logits at the
// current committed position (the root's prediction); tokens/parent describe the tree.
//
// Losslessness: the accepted path's node at depth d was verified attending only to its
// ancestor chain (the path root→it) at positions base..base+d-1 — exactly the greedy context
// at that depth — so its TargetArgmax is the greedy token, and the committed path equals the
// greedy continuation. A tree whose accepted branch IS the greedy path therefore commits
// exactly the greedy tokens (witnessed in TestVerifyTreeLossless).
func VerifyTree(target *model.Session, committedLogits []float32, tokens []int, parent []int) (committed []int, tree polymodel.SpecTree) {
	base := target.Cache.Len()
	root := polymodel.TreeNode{TargetArgmax: argmax(committedLogits)}
	P := len(tokens)
	if P == 0 {
		return []int{root.TargetArgmax}, polymodel.SpecTree{Nodes: []polymodel.TreeNode{root}}
	}

	pos := make([]int, P)
	for i := range pos {
		depth := 1
		for p := parent[i]; p >= 0; p = parent[p] {
			depth++
		}
		pos[i] = base + depth - 1
	}
	anc := make([][]bool, P)
	for q := range anc {
		anc[q] = make([]bool, P)
		anc[q][q] = true
		for p := parent[q]; p >= 0; p = parent[p] {
			anc[q][p] = true
		}
	}
	allow := func(q, k int) bool { return anc[q][k] }

	// The single-pass tree-attention verify (rung #533). Appends P speculative positions with
	// depth-based absolute positions (siblings share); those positions are dropped wholesale
	// by the rewind below, so the target cache never keeps a non-sequential (tree) layout.
	logits := target.VerifyForward(tokens, pos, allow)
	target.Cache.Evict(base, P) // bit-exact rewind to the prefix

	nodes := make([]polymodel.TreeNode, P+1)
	nodes[0] = root
	for i := 0; i < P; i++ {
		nodes[i+1] = polymodel.TreeNode{Token: tokens[i], TargetArgmax: argmax(logits[i])}
	}
	for i := 0; i < P; i++ {
		par := parent[i]
		if par < 0 {
			nodes[0].Children = append(nodes[0].Children, i+1)
		} else {
			nodes[par+1].Children = append(nodes[par+1].Children, i+1)
		}
	}
	tree = polymodel.SpecTree{Nodes: nodes}
	res := polymodel.AcceptTree(tree)

	committed = make([]int, 0, res.Advance)
	for _, idx := range res.Path {
		committed = append(committed, nodes[idx].Token)
	}
	cur := 0
	if len(res.Path) > 0 {
		cur = res.Path[len(res.Path)-1]
	}
	committed = append(committed, nodes[cur].TargetArgmax) // the correction / bonus
	return committed, tree
}

// SpeculativeTree runs n tokens of greedy TREE speculative decode (#533): each round the
// drafter proposes a token tree, the target verifies ALL of it in one tree-attention pass,
// polymodel.AcceptTree picks the accepted path, and the committed tokens advance the target
// (recommitted as one chain VerifyForward). Because the accepted path's context at every
// depth is the greedy context, the output is token-identical to plain greedy decode. Returns
// the committed token ids plus proposed/accepted counts (proposed > accepted ⇒ branches were
// rejected, so the tree-attention mask + AcceptTree actually ran). target must be a fresh,
// un-prefilled session.
func SpeculativeTree(target *model.Session, prompt []int, n, maxDepth, width int, drafter TreeDrafter) (out []int, proposed, accepted int) {
	if maxDepth < 1 {
		maxDepth = 1
	}
	if width < 1 {
		width = 1
	}
	tl := target.Prefill(prompt)
	out = make([]int, 0, n)
	for len(out) < n {
		tokens, parent := drafter.Propose(maxDepth, width)
		committed, _ := VerifyTree(target, tl, tokens, parent)
		proposed += len(tokens)
		// accepted = the tokens on the accepted path (Advance-1), excluding the correction.
		a := len(committed) - 1
		if a < 0 {
			a = 0
		}
		accepted += a

		commit := committed
		if len(out)+len(commit) > n {
			commit = commit[:n-len(out)]
		}
		if len(commit) > 0 {
			lg := target.VerifyForward(commit, nil, nil)
			tl = lg[len(lg)-1]
			out = append(out, commit...)
			drafter.Commit(commit)
		}
		if len(committed) == 0 { // no-progress guard (cannot happen: VerifyTree always returns >=1)
			break
		}
	}
	return out, proposed, accepted
}
