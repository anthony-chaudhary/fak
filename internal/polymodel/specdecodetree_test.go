package polymodel

import (
	"errors"
	"testing"
)

// specdecodetree_test.go — the #5100 done-condition witness: the LIVE tree-shaped
// draft→verify→accept→rollback loop (SpecDecodeTree over AcceptTree) is lossless. The issue
// names two halves, and both are proven here against the SAME deterministic target oracle
// the #4877 linear witness uses (specdecode_test.go), so "the greedy stream" means the very
// same reference stream:
//
//   - a LINEAR-CHAIN tree reduces EXACTLY to the greedy stream — and, more strongly, to the
//     shipped SpecDecode run itself: identical tokens, rounds, evictions, drafted/accepted
//     counts and mean acceptance length, for every drafter quality (TestSpecDecodeTree-
//     LosslessChainReducesToLinear);
//   - a BRANCHING tree stays TOKEN-IDENTICAL to greedy, including the wide and adversarial
//     shapes, while accepting paths a chain of its own first children rejects outright
//     (TestSpecDecodeTreeLossless, TestSpecDecodeTreeLosslessBranchBeatsChain).
//
// The witness named by the issue is `go test ./internal/polymodel -run SpecDecodeTreeLossless`.
//
// Reusing the linear witness's oracle is deliberate: oracleStep IS the target, so the
// reference greedy decode (oracleGreedy), the chain verify pass (oracleVerifier) and the
// tree verify pass (oracleTreeVerifier) are all built from one function. "Lossless" then
// means the tree loop's descend/commit/rollback bookkeeping reproduces the target's own
// greedy stream exactly — with no weights, no GPU and no backend, exactly as a tier-1 leaf
// must be testable.

// oracleTreeVerifier is the target's TREE verify pass built from the oracle: the target's
// argmax at EVERY node, given that node's root→node path. This is what one masked
// model.Session.VerifyForward yields (depth positions, ancestor allow mask) with the
// current-position logits prepended for the root. It walks forward edges only — the same
// flat-tree layout normalizeSpecTree enforces — so unreachable nodes keep argmax 0 and are
// never descended into.
func oracleTreeVerifier(committed []int, tree SpecTree) []int {
	n := len(tree.Nodes)
	argmax := make([]int, n)
	if n == 0 {
		return argmax
	}
	var walk func(idx int, ctx []int)
	walk = func(idx int, ctx []int) {
		argmax[idx] = oracleStep(ctx)
		for _, c := range tree.Nodes[idx].Children {
			if c > idx && c < n {
				walk(c, append(append([]int(nil), ctx...), tree.Nodes[c].Token))
			}
		}
	}
	walk(0, committed) // the root's context is the committed prefix; its Token is ignored
	return argmax
}

// chainTree lifts a LINEAR Drafter into a TreeDrafter that proposes that same draft as a
// one-child-per-node chain. It is the reduction harness: SpecDecodeTree over chainTree(d)
// must equal SpecDecode over d, exactly. A nil Drafter yields a root-only tree — the tree
// spelling of "no draft this round".
func chainTree(d Drafter) TreeDrafter {
	return func(committed []int) SpecTree {
		var toks []int
		if d != nil {
			toks = d(committed)
		}
		nodes := make([]TreeNode, 1, len(toks)+1) // index 0 is the root
		for i, tok := range toks {
			nodes = append(nodes, TreeNode{Token: tok})
			nodes[i].Children = []int{i + 1}
		}
		return SpecTree{Nodes: nodes}
	}
}

// treeDecoy returns a token GUARANTEED different from correct. t ↦ 1+(t%(vocab-1)) is a
// single cycle over [1, vocab), so one step always escapes a collision and the loop below
// runs at most once.
func treeDecoy(correct, j int) int {
	t := 1 + ((correct + j) % (specOracleVocab - 1))
	for t == correct {
		t = 1 + (t % (specOracleVocab - 1))
	}
	return t
}

// decoyFirstTreeDrafter builds a depth-`depth` tree that hedges: at every level it proposes
// `decoys` deliberately WRONG tokens FIRST and the target's own next token LAST. Because
// AcceptTree takes the first child whose Token equals the parent's TargetArgmax, the decoys
// are all rejected and the correct sibling is accepted — so the run descends the full depth
// (mean acceptance length depth+1) while rolling back `decoys` branches per level. It is the
// adversarial shape for the tree claim: the LINEAR chain of those same first children
// (decoyChainDrafter) accepts nothing at all.
func decoyFirstTreeDrafter(depth, decoys int) TreeDrafter {
	return func(committed []int) SpecTree {
		nodes := []TreeNode{{}} // root
		ctx := append([]int(nil), committed...)
		cur := 0
		for lvl := 0; lvl < depth; lvl++ {
			correct := oracleStep(ctx)
			for j := 0; j < decoys; j++ {
				nodes = append(nodes, TreeNode{Token: treeDecoy(correct, j)})
				nodes[cur].Children = append(nodes[cur].Children, len(nodes)-1)
			}
			nodes = append(nodes, TreeNode{Token: correct})
			nodes[cur].Children = append(nodes[cur].Children, len(nodes)-1)
			cur = len(nodes) - 1
			ctx = append(ctx, correct)
		}
		return SpecTree{Nodes: nodes}
	}
}

// decoyChainDrafter is the LINEAR chain of decoyFirstTreeDrafter's first children — the
// drafter a chain-only loop would have to be to make the same first guess. Its very first
// token cannot match the target's argmax, so acceptance is 0 every round and the mean
// acceptance length is exactly 1.0.
func decoyChainDrafter(k int) Drafter {
	return func(committed []int) []int {
		ctx := append([]int(nil), committed...)
		d := make([]int, 0, k)
		for j := 0; j < k; j++ {
			t := treeDecoy(oracleStep(ctx), 0)
			d = append(d, t)
			ctx = append(ctx, t)
		}
		return d
	}
}

// TestSpecDecodeTreeLossless is the #5100 witness: across chain, branching, wide-branching,
// adversarial and no-draft tree shapes the loop's output is token-identical to sequential
// greedy, the mean acceptance length is reported, the rollback hook fires exactly the
// accounted KV positions, and the per-run KEEP/EVICT conservation holds.
func TestSpecDecodeTreeLossless(t *testing.T) {
	const n, k = 24, 4
	prompt := []int{2, 9, 5, 40, 17}
	want := oracleGreedy(prompt, n)

	cases := []struct {
		name          string
		draft         TreeDrafter
		maxNodes      int
		wantMeanAbove float64 // MeanAcceptanceLength must exceed this
		wantEvictPos  bool    // the rollback path must have run (EvictKV > 0)
	}{
		// Linear-chain trees: the four #4877 drafter qualities, re-shaped as chains.
		{"chain-perfect", chainTree(perfectDrafter(k)), k, 1.0, false},
		{"chain-partial", chainTree(partialDrafter(k, 2)), k, 1.0, true},
		{"chain-wrong", chainTree(wrongDrafter(k)), k, 0.0, true}, // acceptance exactly 1.0
		{"chain-none", chainTree(nil), k, 0.0, false},             // root-only tree
		// Branching trees: the generalization the issue is about. The correct token is a
		// LATER sibling at every level, so acceptance depends on the tree descend working.
		{"branch-decoy-first", decoyFirstTreeDrafter(k, 1), 0, 1.0, true},
		{"branch-wide", decoyFirstTreeDrafter(k, 3), 0, 1.0, true},
		{"branch-node-capped", decoyFirstTreeDrafter(k, 3), 4, 0.0, true},
		{"nil-drafter", nil, 0, 0.0, false}, // plain decode, no tree at all
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var evictSeen int
			run, err := SpecDecodeTree(prompt, tc.draft, oracleTreeVerifier, SpecDecodeTreeConfig{
				MaxNewTokens: n,
				MaxNodes:     tc.maxNodes,
				Rollback:     func(e int) { evictSeen += e },
			})
			if err != nil {
				t.Fatalf("SpecDecodeTree: %v", err)
			}
			// (1) LOSSLESS: token-identical to sequential greedy over the full budget.
			if len(run.Output) != n {
				t.Fatalf("emitted %d tokens, want %d", len(run.Output), n)
			}
			for i := 0; i < n; i++ {
				if run.Output[i] != want[i] {
					t.Fatalf("LOSSLESS VIOLATED at token %d: tree=%d greedy=%d (drafter=%s)",
						i, run.Output[i], want[i], tc.name)
				}
			}
			// (2) ACCEPTANCE-LENGTH metric emitted, and > 1 where the tree helped.
			if run.MeanAcceptanceLength <= tc.wantMeanAbove {
				t.Fatalf("mean acceptance length = %.3f, want > %.3f (drafter=%s)",
					run.MeanAcceptanceLength, tc.wantMeanAbove, tc.name)
			}
			if run.Rounds == 0 || run.MeanAcceptanceLength != float64(len(run.Output))/float64(run.Rounds) {
				t.Fatalf("mean acceptance length %.3f != emitted/rounds %d/%d",
					run.MeanAcceptanceLength, len(run.Output), run.Rounds)
			}
			// (3) The Rollback hook fired exactly the accounted KV positions, and the
			// rejected-branch path is genuinely exercised where rejections are guaranteed.
			if evictSeen != run.EvictKV {
				t.Fatalf("rollback hook saw %d KV, run accounted %d", evictSeen, run.EvictKV)
			}
			if tc.wantEvictPos && run.EvictKV == 0 {
				t.Fatalf("drafter=%s should have rolled back rejected branches, but EvictKV=0 (vacuous)", tc.name)
			}
			// (4) CONSERVATION: every verified speculative node was either kept or evicted.
			if run.AcceptedNodes+run.EvictKV != run.DraftedNodes {
				t.Fatalf("accepted %d + evicted %d != drafted %d nodes",
					run.AcceptedNodes, run.EvictKV, run.DraftedNodes)
			}
			// (5) The node cap is honored per round where one was configured.
			if tc.maxNodes > 0 && run.DraftedNodes > tc.maxNodes*run.Rounds {
				t.Fatalf("drafted %d nodes over %d rounds exceeds MaxNodes=%d/round",
					run.DraftedNodes, run.Rounds, tc.maxNodes)
			}
		})
	}
}

// TestSpecDecodeTreeLosslessChainReducesToLinear is the issue's first half stated at full
// strength: a LINEAR-CHAIN tree does not merely match the greedy stream, it reduces EXACTLY
// to the shipped #4877 SpecDecode run — same tokens, same rounds, same drafted/accepted
// counts, same evictions, same acceptance length — for every drafter quality. This is the
// loop-level counterpart of TestAcceptTree's "a chain tree reduces exactly to AcceptGreedy".
func TestSpecDecodeTreeLosslessChainReducesToLinear(t *testing.T) {
	const n, k = 24, 4
	prompt := []int{2, 9, 5, 40, 17}

	for _, tc := range []struct {
		name string
		d    Drafter
	}{
		{"perfect", perfectDrafter(k)},
		{"partial", partialDrafter(k, 2)},
		{"wrong", wrongDrafter(k)},
		{"decoy", decoyChainDrafter(k)},
		{"none", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lin, err := SpecDecode(prompt, tc.d, oracleVerifier, SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k})
			if err != nil {
				t.Fatalf("SpecDecode: %v", err)
			}
			tre, err := SpecDecodeTree(prompt, chainTree(tc.d), oracleTreeVerifier, SpecDecodeTreeConfig{MaxNewTokens: n, MaxNodes: k})
			if err != nil {
				t.Fatalf("SpecDecodeTree: %v", err)
			}
			if len(tre.Output) != len(lin.Output) {
				t.Fatalf("tree emitted %d tokens, linear emitted %d", len(tre.Output), len(lin.Output))
			}
			for i := range lin.Output {
				if tre.Output[i] != lin.Output[i] {
					t.Fatalf("CHAIN REDUCTION VIOLATED at token %d: tree=%d linear=%d (drafter=%s)",
						i, tre.Output[i], lin.Output[i], tc.name)
				}
			}
			if tre.Rounds != lin.Rounds || tre.EvictKV != lin.EvictKV {
				t.Fatalf("tree rounds/evict = %d/%d, linear = %d/%d",
					tre.Rounds, tre.EvictKV, lin.Rounds, lin.EvictKV)
			}
			if tre.DraftedNodes != lin.DraftedTokens || tre.AcceptedNodes != lin.AcceptedDrafts {
				t.Fatalf("tree drafted/accepted = %d/%d, linear = %d/%d",
					tre.DraftedNodes, tre.AcceptedNodes, lin.DraftedTokens, lin.AcceptedDrafts)
			}
			if tre.MeanAcceptanceLength != lin.MeanAcceptanceLength {
				t.Fatalf("tree mean %.6f != linear mean %.6f", tre.MeanAcceptanceLength, lin.MeanAcceptanceLength)
			}
		})
	}
}

// TestSpecDecodeTreeLosslessBranchBeatsChain is the issue's second half made non-vacuous:
// the BRANCHING tree does not just stay lossless, it accepts what a chain cannot. Both runs
// make the same first guess at every level (the decoy); the tree hedges with the correct
// sibling behind it. The chain therefore accepts NOTHING (mean exactly 1.0) while the tree
// descends the full depth (mean exactly k+1) — and both emit the identical greedy stream.
func TestSpecDecodeTreeLosslessBranchBeatsChain(t *testing.T) {
	const n, k = 20, 4 // n divisible by k+1 → the tree's rounds/mean are exact, not approximate
	prompt := []int{7, 3, 3, 1}
	want := oracleGreedy(prompt, n)

	lin, err := SpecDecode(prompt, decoyChainDrafter(k), oracleVerifier, SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	tre, err := SpecDecodeTree(prompt, decoyFirstTreeDrafter(k, 1), oracleTreeVerifier, SpecDecodeTreeConfig{MaxNewTokens: n})
	if err != nil {
		t.Fatalf("SpecDecodeTree: %v", err)
	}

	// Both are lossless — the tree's extra acceptance never costs a token.
	for i := 0; i < n; i++ {
		if tre.Output[i] != want[i] || lin.Output[i] != want[i] {
			t.Fatalf("token %d: tree=%d linear=%d greedy=%d", i, tre.Output[i], lin.Output[i], want[i])
		}
	}
	// The chain's first guess is always wrong → 0 accepted, 1 real token per round.
	if lin.AcceptedDrafts != 0 || lin.MeanAcceptanceLength != 1.0 || lin.Rounds != n {
		t.Fatalf("decoy chain should accept nothing: accepted=%d mean=%.3f rounds=%d",
			lin.AcceptedDrafts, lin.MeanAcceptanceLength, lin.Rounds)
	}
	// The tree hedges the same guess and descends the full depth → k+1 real tokens/round.
	if tre.Rounds != n/(k+1) || tre.MeanAcceptanceLength != float64(k+1) {
		t.Fatalf("branching tree rounds=%d mean=%.3f, want rounds=%d mean=%.1f",
			tre.Rounds, tre.MeanAcceptanceLength, n/(k+1), float64(k+1))
	}
	// Strictly better than the chain, at the cost of the rejected decoy branches.
	if tre.MeanAcceptanceLength <= lin.MeanAcceptanceLength {
		t.Fatalf("tree mean %.3f should beat chain mean %.3f", tre.MeanAcceptanceLength, lin.MeanAcceptanceLength)
	}
	if wantEvict := k * tre.Rounds; tre.EvictKV != wantEvict { // exactly one decoy per level
		t.Fatalf("tree evicted %d branches, want %d", tre.EvictKV, wantEvict)
	}
}

// TestSpecDecodeTreeStopToken proves the run halts once an EOS token is committed, whether
// the EOS lands on an accepted path node or on the correction, and does not emit past it.
func TestSpecDecodeTreeStopToken(t *testing.T) {
	prompt := []int{1}
	stream := oracleGreedy(prompt, 30)
	stop := stream[5]
	firstStop := 0
	for firstStop < len(stream) && stream[firstStop] != stop {
		firstStop++
	}
	run, err := SpecDecodeTree(prompt, decoyFirstTreeDrafter(4, 2), oracleTreeVerifier, SpecDecodeTreeConfig{
		MaxNewTokens: 30, StopToken: stop, StopEnabled: true,
	})
	if err != nil {
		t.Fatalf("SpecDecodeTree: %v", err)
	}
	if len(run.Output) != firstStop+1 || run.Output[len(run.Output)-1] != stop {
		t.Fatalf("run emitted %d tokens ending %v, want %d ending on stop token %d",
			len(run.Output), run.Output[len(run.Output)-1:], firstStop+1, stop)
	}
	for i := 0; i < firstStop; i++ {
		if run.Output[i] != stream[i] {
			t.Fatalf("token %d = %d, want greedy %d", i, run.Output[i], stream[i])
		}
	}
}

// TestSpecDecodeTreeMalformedEdgesCannotHang is the liveness guard: AcceptTree descends a
// caller-shaped index graph, so a proposal carrying a SELF edge would spin it forever. Here
// node 1 carries token 7, a self edge, and a verifier that answers 7 everywhere — without
// normalizeSpecTree's forward-only rule the descend 1→1→1… never terminates and this test
// hangs rather than fails. With it, the self edge is dropped, the round accepts node 1, and
// the run completes with the verifier's own tokens.
func TestSpecDecodeTreeMalformedEdgesCannotHang(t *testing.T) {
	const n = 6
	selfLoop := func(committed []int) SpecTree {
		return SpecTree{Nodes: []TreeNode{
			{Children: []int{1}},
			{Token: 7, Children: []int{1, 0, 99}}, // self edge, back edge, out-of-range edge
		}}
	}
	allSeven := func(committed []int, tree SpecTree) []int {
		a := make([]int, len(tree.Nodes))
		for i := range a {
			a[i] = 7
		}
		return a
	}
	run, err := SpecDecodeTree([]int{1}, selfLoop, allSeven, SpecDecodeTreeConfig{MaxNewTokens: n})
	if err != nil {
		t.Fatalf("SpecDecodeTree: %v", err)
	}
	if len(run.Output) != n {
		t.Fatalf("emitted %d tokens, want %d", len(run.Output), n)
	}
	for i, tok := range run.Output {
		if tok != 7 {
			t.Fatalf("token %d = %d, want the verifier's own token 7", i, tok)
		}
	}
	// Node 1 is accepted every round (its token IS the root's argmax), so nothing is
	// evicted — the dropped edges cost liveness nothing.
	if run.EvictKV != 0 || run.AcceptedNodes != run.DraftedNodes {
		t.Fatalf("accepted %d of %d nodes, evicted %d; want all accepted",
			run.AcceptedNodes, run.DraftedNodes, run.EvictKV)
	}
}

// TestSpecDecodeTreeNodeCap proves an over-eager drafter's tree is truncated to MaxNodes
// speculative nodes per round, so the verify pass never exceeds the configured budget, and
// the truncated run is still lossless.
func TestSpecDecodeTreeNodeCap(t *testing.T) {
	const n, budget = 12, 2
	prompt := []int{3}
	want := oracleGreedy(prompt, n)
	// A depth-6, 2-decoy tree proposes 18 speculative nodes; the budget allows 2.
	run, err := SpecDecodeTree(prompt, decoyFirstTreeDrafter(6, 2), oracleTreeVerifier,
		SpecDecodeTreeConfig{MaxNewTokens: n, MaxNodes: budget})
	if err != nil {
		t.Fatalf("SpecDecodeTree: %v", err)
	}
	if run.DraftedNodes > budget*run.Rounds {
		t.Fatalf("drafted %d nodes over %d rounds exceeds MaxNodes=%d/round", run.DraftedNodes, run.Rounds, budget)
	}
	for i := 0; i < n; i++ {
		if run.Output[i] != want[i] {
			t.Fatalf("capped run not lossless at %d: %d != %d", i, run.Output[i], want[i])
		}
	}
}

// TestSpecDecodeTreeErrors proves the loop's guards: a nil TreeVerifier refuses, an
// empty-argmax (contract-violating) verifier is caught rather than spinning, a short argmax
// vector truncates the unverified tail instead of descending into it, and a zero budget is
// a clean no-op.
func TestSpecDecodeTreeErrors(t *testing.T) {
	prompt := []int{1}
	if _, err := SpecDecodeTree(prompt, chainTree(perfectDrafter(2)), nil, SpecDecodeTreeConfig{MaxNewTokens: 4}); !errors.Is(err, ErrNoTreeVerifier) {
		t.Fatalf("nil verifier should be ErrNoTreeVerifier, got %v", err)
	}
	stalled := func(committed []int, tree SpecTree) []int { return nil } // violates len==len(Nodes)
	if _, err := SpecDecodeTree(prompt, nil, stalled, SpecDecodeTreeConfig{MaxNewTokens: 4}); !errors.Is(err, ErrTreeVerifierStalled) {
		t.Fatalf("empty-argmax verifier should be ErrTreeVerifierStalled, got %v", err)
	}
	// A verifier answering for the ROOT ONLY leaves every speculative node unverified: the
	// tail is dropped, so each round is a plain single-token decode — still lossless.
	const n = 8
	rootOnly := func(committed []int, tree SpecTree) []int { return []int{oracleStep(committed)} }
	run, err := SpecDecodeTree(prompt, decoyFirstTreeDrafter(4, 2), rootOnly, SpecDecodeTreeConfig{MaxNewTokens: n})
	if err != nil {
		t.Fatalf("short-argmax verifier: %v", err)
	}
	want := oracleGreedy(prompt, n)
	for i := 0; i < n; i++ {
		if run.Output[i] != want[i] {
			t.Fatalf("short-argmax run not lossless at %d: %d != %d", i, run.Output[i], want[i])
		}
	}
	if run.DraftedNodes != 0 || run.Rounds != n {
		t.Fatalf("short-argmax run should verify no speculative node: drafted=%d rounds=%d", run.DraftedNodes, run.Rounds)
	}
	// A zero budget decodes nothing at all.
	zero, err := SpecDecodeTree(prompt, chainTree(perfectDrafter(2)), oracleTreeVerifier, SpecDecodeTreeConfig{MaxNewTokens: 0})
	if err != nil || len(zero.Output) != 0 || zero.Rounds != 0 {
		t.Fatalf("zero budget should be a clean no-op, got run=%+v err=%v", zero, err)
	}
}
