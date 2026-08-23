package polymodel

// specdecodetree.go — the TREE-shaped LIVE draft→verify→accept→rollback speculative-decode
// loop (#5100, follow-up to #4877). #4877 shipped the LINEAR chain loop (SpecDecode over
// AcceptGreedy); this is its Medusa / EAGLE-2 / SpecInfer generalization, the piece the
// #4877 done-condition left undriven: a drafter proposes a whole SpecTree of candidate
// continuations that SHARE a KV prefix, the target verifies the ENTIRE tree in ONE pass
// behind a tree-attention mask (model.Session.VerifyForward with depth-based positions and
// an ancestor allow mask), AcceptTree keeps the single accepted root→leaf path, and every
// rejected BRANCH is rolled back bit-exactly (model.KVCache.Evict).
//
// WHY A TREE BEATS A CHAIN. A linear K-token draft is ONE bet: the first wrong token throws
// away the whole suffix, so acceptance collapses as soon as the drafter is unsure. A tree of
// the same depth HEDGES — at each level it proposes several alternative next tokens, and the
// pass keeps descending as long as ANY sibling matched the target's own argmax. It costs
// more verify POSITIONS (a wider batch — compute the verify pass has to spare, being
// compute-bound) and buys a longer expected accepted path per bandwidth-bound round, which
// is the whole trade decode wants. The witness makes this concrete rather than asserted: a
// tree whose FIRST child at every level is deliberately wrong still accepts the full depth,
// where the same-depth LINEAR chain of exactly those first children accepts nothing at all
// (mean acceptance length k+1 vs exactly 1.0).
//
// SAME LANE-PURE SEAM AS SpecDecode. polymodel imports nothing internal (architest tier 1),
// so the engine pieces this drives cross the seam as CLOSURES the caller binds — never a
// session, never a backend:
//
//	run, _ := polymodel.SpecDecodeTree(prompt,
//	    func(committed []int) polymodel.SpecTree { return drafter.ProposeTree(committed) },
//	    func(committed []int, t polymodel.SpecTree) []int {   // ONE tree-masked forward
//	        ids, pos, allow := treePanel(t)                   // depth positions + ancestor mask
//	        rows := targetSession.VerifyForward(ids, pos, allow)
//	        argmax := make([]int, 0, len(t.Nodes))
//	        argmax = append(argmax, mathx.ArgmaxF32(targetLogits))  // node 0 (root): known
//	        for _, r := range rows { argmax = append(argmax, mathx.ArgmaxF32(r)) }
//	        return argmax
//	    },
//	    polymodel.SpecDecodeTreeConfig{MaxNewTokens: n, MaxNodes: 16,
//	        Rollback: func(evict int) { targetSession.Cache.Evict(base+kept, evict) }})
//
// LOSSLESS BY CONSTRUCTION, FOR ANY TREE. The emitted stream is TOKEN-IDENTICAL to plain
// sequential greedy decode of the target no matter what the drafter proposes — a perfect
// tree, a decoy-first tree, a fully wrong one, or no tree at all. Every committed token is
// either an accepted node, which AcceptTree only descends into when its Token EQUALS the
// parent's TargetArgmax (the token the target would itself have emitted there), or the
// correction, which IS the target's own argmax at the last accepted node. Draft quality
// therefore moves Rounds — speed — and never the output. The SpecDecodeTreeLossless witness
// proves both halves the issue names: a LINEAR-chain tree reduces EXACTLY to the shipped
// SpecDecode run (same tokens, rounds, evictions and acceptance length), and a BRANCHING
// tree stays token-identical to greedy while accepting paths a chain would have rejected.
//
// A LIVE LOOP MUST NOT HANG (the tree-specific hazard). AcceptTree descends a caller-shaped
// index graph; a malformed proposal with a self- or back-edge would spin it forever. The
// linear loop's equivalent hazard is a stalled verifier, guarded by ErrVerifierStalled.
// Here the guard is structural: normalizeSpecTree keeps only strictly FORWARD child edges,
// which is already SpecTree's documented flat-tree layout (root at index 0, every other node
// laid out after its parent), so the descend is acyclic BY CONSTRUCTION and a bad drafter
// costs acceptance, never liveness.

import "errors"

// TreeDrafter proposes a speculation tree continuing the committed history — the tree
// generalization of Drafter. It returns a SpecTree whose index 0 is the ROOT (the current
// committed position; its Token is ignored) and whose every other node is a proposed
// speculative token, with Children holding the indices of that node's alternatives. Only the
// SHAPE and the Tokens are the drafter's to fill; TargetArgmax is the verifier's output and
// is ignored on the way in. Like Drafter it may propose ANYTHING — the verify pass gates
// every node, so a wrong branch costs a rollback, never correctness. Returning a zero
// SpecTree (or nil nodes) means "no draft this round", handled as a plain single-token
// decode. The drafter owns its own context/session; the loop supplies only committed ids.
type TreeDrafter func(committed []int) SpecTree

// TreeVerifier runs the target model's SINGLE tree-verify pass over the proposed tree and
// returns the target's OWN argmax at every node, indexed by node:
//
//	index 0    — the target's next token after `committed` (the root: no draft applied);
//	index i>0  — the target's next token after committed + the root→node-i path's tokens.
//
// So len(result) MUST be len(tree.Nodes). This is what model.Session.VerifyForward yields
// under a tree-attention mask (depth-based positions so siblings share a slot, an ancestor
// allow mask so a node never attends to a sibling branch), with the already-known
// current-position logits prepended for the root; a pure test binds it to a deterministic
// oracle. A verifier returning FEWER argmaxes than nodes has its unreachable tail dropped
// rather than descended into — the loop never guesses a node's target token.
type TreeVerifier func(committed []int, tree SpecTree) []int

// SpecDecodeTreeConfig configures one SpecDecodeTree run. It mirrors SpecDecodeConfig,
// with MaxDraft (a chain LENGTH) generalized to MaxNodes (a tree BUDGET).
type SpecDecodeTreeConfig struct {
	// MaxNewTokens caps how many tokens the run emits (the decode budget). The run stops
	// as soon as this many tokens are committed, mid-round if necessary.
	MaxNewTokens int
	// MaxNodes caps the SPECULATIVE nodes verified per round (the root is not counted) —
	// the tree analogue of MaxDraft, and the real knob, since a tree's cost is its node
	// count, not its depth. An over-eager drafter has its proposal truncated to the first
	// MaxNodes speculative nodes; child edges into the dropped tail are simply not
	// descended. 0 or negative means "no cap" — the full proposal is verified.
	MaxNodes int
	// StopToken, when StopEnabled, ends the run after this token id is committed (an EOS).
	StopToken   int
	StopEnabled bool
	// Rollback, if non-nil, is called with EvictKV after each round that rejected branches,
	// so the engine can roll back the rejected branches' KV. nil for a KV-less pure loop.
	Rollback Rollback
}

// SpecDecodeTreeRun is the outcome of a SpecDecodeTree run: the emitted tokens plus the
// accounting that makes the throughput honest. It mirrors SpecDecodeRun with the draft
// counted in NODES rather than chain positions.
type SpecDecodeTreeRun struct {
	// Output is the emitted token ids — TOKEN-IDENTICAL to plain greedy decode of the
	// target for the same budget, for any tree drafter.
	Output []int
	// Rounds is the number of tree-verify passes performed (the bandwidth-bound cost unit).
	Rounds int
	// DraftedNodes is the total speculative nodes verified across all rounds (roots
	// excluded) — the tree's compute cost, and what MaxNodes bounds per round.
	DraftedNodes int
	// AcceptedNodes is the total speculative nodes that lay on an accepted path and were
	// committed (excludes the per-round correction token).
	AcceptedNodes int
	// AcceptanceProfile retains accepted/proposed counts by zero-based draft position.
	AcceptanceProfile []AcceptancePosition
	// EvictKV is the total rejected-branch KV positions rolled back. Over a whole run
	// AcceptedNodes + EvictKV == DraftedNodes: every verified node is either kept or
	// rolled back, the same conservation AcceptTree asserts per round.
	EvictKV int
	// MeanAcceptanceLength is the mean REAL tokens committed per verify pass
	// (len(Output)/Rounds). It is > 1 exactly when the tree bought throughput: a plain
	// decode (nothing ever accepted) yields 1.0, and a tree whose target path is fully
	// accepted at depth d yields d+1 — regardless of how WIDE the tree was, which is the
	// point of hedging with siblings.
	MeanAcceptanceLength float64
}

// Tree speculative-decode loop errors, the tree analogues of ErrNoVerifier /
// ErrVerifierStalled.
var (
	// ErrNoTreeVerifier is returned when SpecDecodeTree is called without a TreeVerifier —
	// there is no target to gate the branches against, so no token can be committed.
	ErrNoTreeVerifier = errors.New("polymodel: SpecDecodeTree needs a TreeVerifier (bind model.Session.VerifyForward under a tree-attention mask)")
	// ErrTreeVerifierStalled is returned when a TreeVerifier returns an empty argmax
	// vector, so a round commits no token and the loop would spin forever. A
	// contract-honoring TreeVerifier (len == len(tree.Nodes) ≥ 1) never triggers it; it is
	// the guard against a broken binding.
	ErrTreeVerifierStalled = errors.New("polymodel: TreeVerifier returned no argmax; a round cannot advance (want one per tree node)")
)

// SpecDecodeTree runs the live TREE speculative-decode loop and returns the emitted tokens
// plus acceptance accounting. Each round: the TreeDrafter proposes a candidate tree (capped
// at MaxNodes speculative nodes), the TreeVerifier returns the target's argmax at every
// node from ONE tree-masked pass, AcceptTree keeps the accepted root→leaf path, the rejected
// branches' KV is rolled back (Rollback), and the accepted path's tokens plus the target's
// correction token are committed. It repeats until MaxNewTokens tokens are emitted or the
// StopToken is committed.
//
// The output is token-identical to plain sequential greedy decode of the target for ANY
// TreeDrafter — correctness is independent of the tree's quality or shape (only Rounds,
// hence speed, depends on them). draft may be nil (every round is then a plain single-token
// decode), and a linear-chain tree reduces exactly to SpecDecode over the same tokens. It
// reports MeanAcceptanceLength = emitted/Rounds, the honest throughput the tree bought.
func SpecDecodeTree(prompt []int, draft TreeDrafter, verify TreeVerifier, cfg SpecDecodeTreeConfig) (SpecDecodeTreeRun, error) {
	var run SpecDecodeTreeRun
	var profile acceptanceProfile
	if verify == nil {
		return run, ErrNoTreeVerifier
	}
	max := cfg.MaxNewTokens
	if max <= 0 {
		return run, nil // empty budget: nothing to decode
	}
	committed := append([]int(nil), prompt...)
	out := make([]int, 0, max)

	for len(out) < max {
		var proposed SpecTree
		if draft != nil {
			proposed = draft(committed)
		}
		tree := normalizeSpecTree(proposed, cfg.MaxNodes)

		argmax := verify(committed, tree)
		if len(argmax) == 0 {
			return run, ErrTreeVerifierStalled // no token to commit → would spin forever
		}
		// A verifier that returned fewer argmaxes than nodes left the tail unverified;
		// drop those nodes rather than descend into one whose target token is unknown.
		if len(argmax) < len(tree.Nodes) {
			tree.Nodes = tree.Nodes[:len(argmax)]
		}
		for i := range tree.Nodes {
			tree.Nodes[i].TargetArgmax = argmax[i]
		}

		res := AcceptTree(tree)
		run.Rounds++
		run.DraftedNodes += len(tree.Nodes) - 1
		run.AcceptedNodes += len(res.Path)
		profile.recordCounts(treeProposalPositions(tree), len(res.Path))
		if res.EvictKV > 0 {
			run.EvictKV += res.EvictKV
			if cfg.Rollback != nil {
				cfg.Rollback(res.EvictKV)
			}
		}

		// Commit the accepted path, then the target's correction/bonus token. Each accepted
		// node's Token equals its parent's TargetArgmax, so the stream stays byte-for-byte
		// greedy. Honor MaxNewTokens and the optional StopToken mid-commit; `last` tracks
		// the deepest node actually COMMITTED, so a budget-truncated path still takes the
		// correction that belongs to the position the stream really stands at.
		stop, last := false, 0
		for _, idx := range res.Path {
			if len(out) >= max {
				break
			}
			last = idx
			if commitToken(&out, &committed, tree.Nodes[idx].Token, cfg.StopEnabled, cfg.StopToken) {
				stop = true
				break
			}
		}
		if !stop && len(out) < max {
			if commitToken(&out, &committed, tree.Nodes[last].TargetArgmax, cfg.StopEnabled, cfg.StopToken) {
				stop = true
			}
		}
		if stop {
			break
		}
	}

	run.Output = out
	run.AcceptanceProfile = profile.snapshot()
	if run.Rounds > 0 {
		run.MeanAcceptanceLength = float64(len(out)) / float64(run.Rounds)
	}
	return run, nil
}

// normalizeSpecTree copies a drafter's proposal into a tree the loop can drive safely. It
// guarantees a root exists (an empty proposal becomes a root-only tree — a plain
// single-token decode), truncates to maxNodes speculative nodes, and keeps only strictly
// FORWARD child edges (child index > parent index). The forward-edge rule is already
// SpecTree's documented layout — index 0 is the root and every other node is a proposed
// token placed after its parent — so a well-formed proposal passes through unchanged, while
// a malformed one with a self- or back-edge cannot make AcceptTree's descend cycle. Copying
// (rather than mutating the drafter's slice) also keeps TargetArgmax the VERIFIER's field:
// whatever the drafter left there is dropped, so it can never pre-answer its own gate.
func normalizeSpecTree(t SpecTree, maxNodes int) SpecTree {
	n := len(t.Nodes)
	if n == 0 {
		return SpecTree{Nodes: []TreeNode{{}}} // root only → a plain single-token decode
	}
	if maxNodes > 0 && n > maxNodes+1 {
		n = maxNodes + 1 // the root plus at most maxNodes speculative nodes
	}
	nodes := make([]TreeNode, n)
	for i := 0; i < n; i++ {
		nodes[i].Token = t.Nodes[i].Token
		for _, c := range t.Nodes[i].Children {
			if c > i && c < n {
				nodes[i].Children = append(nodes[i].Children, c)
			}
		}
	}
	return SpecTree{Nodes: nodes}
}
