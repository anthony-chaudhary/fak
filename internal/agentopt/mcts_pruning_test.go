package agentopt

import (
	"strings"
	"testing"
)

func TestMCTSBranchPruning(t *testing.T) {
	t.Run("compilation_error_prunes_immediately_and_blocks_expansion", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.2)
		node := NewMCTSNode("node_1", "apply_patch")

		eval := BranchEvalResult{
			Score:            0.0,
			CompilationError: true,
			FailureDetails:   "undefined: SymbolXYZ",
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if !pruned {
			t.Fatal("expected node to be pruned on compilation error, got false")
		}
		if !node.Pruned {
			t.Fatal("node.Pruned is false, expected true")
		}
		if !strings.Contains(node.PruneReason, "compilation error") {
			t.Fatalf("unexpected prune reason: %s", node.PruneReason)
		}
		if node.CanExpand() {
			t.Fatal("node.CanExpand() returned true on pruned node")
		}
		if pruner.CanExpand(node) {
			t.Fatal("pruner.CanExpand(node) returned true on pruned node")
		}

		child := NewMCTSNode("node_1_1", "retry_patch")
		if node.AddChild(child) {
			t.Fatal("node.AddChild unexpectedly succeeded on pruned node")
		}
		if pruner.AddChild(node, child) {
			t.Fatal("pruner.AddChild unexpectedly succeeded on pruned node")
		}
		if len(node.Children) != 0 {
			t.Fatalf("node has %d children, expected 0", len(node.Children))
		}
	})

	t.Run("syntax_violation_prunes_immediately_and_blocks_expansion", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.2)
		node := NewMCTSNode("node_2", "generate_code")

		eval := BranchEvalResult{
			Score:           0.1,
			SyntaxViolation: true,
			FailureDetails:  "unexpected token EOF",
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if !pruned {
			t.Fatal("expected node to be pruned on syntax violation, got false")
		}
		if !node.Pruned {
			t.Fatal("node.Pruned is false, expected true")
		}
		if !strings.Contains(node.PruneReason, "syntax violation") {
			t.Fatalf("unexpected prune reason: %s", node.PruneReason)
		}
		if node.CanExpand() {
			t.Fatal("node.CanExpand() returned true on pruned node")
		}
	})

	t.Run("deterministic_failure_flag_prunes_immediately", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.2)
		node := NewMCTSNode("node_3", "run_check")

		eval := BranchEvalResult{
			Score:             0.0,
			DeterministicFail: true,
			FailureDetails:    "type mismatch",
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if !pruned {
			t.Fatal("expected node to be pruned on deterministic fail, got false")
		}
		if !node.Pruned {
			t.Fatal("node.Pruned is false, expected true")
		}
		if !strings.Contains(node.PruneReason, "deterministic failure") {
			t.Fatalf("unexpected prune reason: %s", node.PruneReason)
		}
	})

	t.Run("zero_score_cutoff_prunes_immediately", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.0)
		node := NewMCTSNode("node_4", "execute_step")

		eval := BranchEvalResult{
			Score: 0.0,
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if !pruned {
			t.Fatal("expected node to be pruned on zero score cutoff, got false")
		}
		if !node.Pruned {
			t.Fatal("node.Pruned is false, expected true")
		}
		if node.PruneReason != "zero score cutoff" {
			t.Fatalf("expected prune reason 'zero score cutoff', got %s", node.PruneReason)
		}
		if node.CanExpand() {
			t.Fatal("node.CanExpand() returned true on pruned node")
		}
	})

	t.Run("cutoff_score_threshold_prunes_sub_threshold_branches", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.5)
		node := NewMCTSNode("node_5", "partial_fix")

		eval := BranchEvalResult{
			Score: 0.35,
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if !pruned {
			t.Fatal("expected node with score below cutoff threshold to be pruned")
		}
		if !node.Pruned {
			t.Fatal("node.Pruned is false, expected true")
		}
		if !strings.Contains(node.PruneReason, "score cutoff") {
			t.Fatalf("unexpected prune reason: %s", node.PruneReason)
		}
	})

	t.Run("viable_branch_retained_and_allows_expansion", func(t *testing.T) {
		pruner := NewMCTSPruner(4, 3, 0.3)
		node := NewMCTSNode("node_6", "valid_refactor")

		eval := BranchEvalResult{
			Score: 0.85,
		}

		pruned := pruner.EvaluateAndPrune(node, eval)
		if pruned {
			t.Fatal("expected viable node not to be pruned, got true")
		}
		if node.Pruned {
			t.Fatal("node.Pruned is true, expected false")
		}
		if node.PruneReason != "" {
			t.Fatalf("expected empty prune reason, got %s", node.PruneReason)
		}
		if !node.CanExpand() {
			t.Fatal("node.CanExpand() returned false for viable node")
		}
		if !pruner.CanExpand(node) {
			t.Fatal("pruner.CanExpand(node) returned false for viable node")
		}

		child := NewMCTSNode("node_6_1", "followup_step")
		if !pruner.AddChild(node, child) {
			t.Fatal("pruner.AddChild failed on viable node")
		}
		if len(node.Children) != 1 {
			t.Fatalf("expected 1 child, got %d", len(node.Children))
		}
		if child.Depth != node.Depth+1 {
			t.Fatalf("child depth = %d, expected %d", child.Depth, node.Depth+1)
		}
	})

	t.Run("bounds_tree_depth", func(t *testing.T) {
		pruner := NewMCTSPruner(2, 5, 0.1)
		root := NewMCTSNode("root", "root_step")
		root.Score = 0.9

		level1 := NewMCTSNode("lvl1", "step_1")
		level1.Score = 0.9
		if !pruner.AddChild(root, level1) {
			t.Fatal("failed to add level 1 child")
		}

		level2 := NewMCTSNode("lvl2", "step_2")
		level2.Score = 0.9
		if !pruner.AddChild(level1, level2) {
			t.Fatal("failed to add level 2 child")
		}
		if level2.Depth != 2 {
			t.Fatalf("level2 depth = %d, expected 2", level2.Depth)
		}

		// Level 2 is at MaxDepth: expansion must be blocked
		if pruner.CanExpand(level2) {
			t.Fatal("pruner.CanExpand returned true at max depth")
		}

		level3 := NewMCTSNode("lvl3", "step_3")
		if pruner.AddChild(level2, level3) {
			t.Fatal("pruner.AddChild unexpectedly succeeded beyond max depth")
		}
		if len(level2.Children) != 0 {
			t.Fatalf("level2 has %d children, expected 0", len(level2.Children))
		}

		// Direct evaluation of a node exceeding max depth triggers prune
		deepNode := NewMCTSNode("deep", "step_too_deep")
		deepNode.Depth = 3
		eval := BranchEvalResult{Score: 0.95}
		if !pruner.EvaluateAndPrune(deepNode, eval) {
			t.Fatal("expected node exceeding max depth to be pruned")
		}
		if !deepNode.Pruned || !strings.Contains(deepNode.PruneReason, "depth ceiling") {
			t.Fatalf("unexpected state for deep node: pruned=%v, reason=%s", deepNode.Pruned, deepNode.PruneReason)
		}
	})

	t.Run("bounds_fanout_children_ceiling", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 2, 0.1)
		root := NewMCTSNode("root", "root_action")

		c1 := NewMCTSNode("c1", "action_1")
		c2 := NewMCTSNode("c2", "action_2")
		c3 := NewMCTSNode("c3", "action_3")

		if !pruner.AddChild(root, c1) {
			t.Fatal("failed to add first child")
		}
		if !pruner.AddChild(root, c2) {
			t.Fatal("failed to add second child")
		}

		// Reached MaxChildren limit: cannot expand further
		if pruner.CanExpand(root) {
			t.Fatal("pruner.CanExpand returned true when max children reached")
		}
		if pruner.AddChild(root, c3) {
			t.Fatal("pruner.AddChild unexpectedly succeeded beyond max children budget")
		}
		if len(root.Children) != 2 {
			t.Fatalf("root has %d children, expected 2", len(root.Children))
		}
	})

	t.Run("prune_subtree_cascades_to_all_descendants", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 3, 0.1)
		root := NewMCTSNode("root", "start")
		c1 := NewMCTSNode("c1", "step_1")
		c11 := NewMCTSNode("c11", "step_1_1")

		pruner.AddChild(root, c1)
		pruner.AddChild(c1, c11)

		pruner.PruneSubtree(c1, "subgraph abandoned")

		if !c1.Pruned || c1.PruneReason != "subgraph abandoned" {
			t.Fatalf("c1 not pruned correctly: pruned=%v, reason=%s", c1.Pruned, c1.PruneReason)
		}
		if !c11.Pruned || c11.PruneReason != "subgraph abandoned" {
			t.Fatalf("c11 not pruned correctly: pruned=%v, reason=%s", c11.Pruned, c11.PruneReason)
		}
		if c1.CanExpand() || c11.CanExpand() {
			t.Fatal("pruned subtree nodes reported CanExpand true")
		}
	})

	t.Run("tree_stats_and_best_path_selection", func(t *testing.T) {
		pruner := NewMCTSPruner(5, 5, 0.2)
		root := NewMCTSNode("root", "start")
		root.Score = 0.5

		good1 := NewMCTSNode("good1", "good_step_1")
		good1.Score = 0.7
		pruner.AddChild(root, good1)

		bad1 := NewMCTSNode("bad1", "bad_step_1")
		pruner.AddChild(root, bad1)
		pruner.EvaluateAndPrune(bad1, BranchEvalResult{Score: 0.0})

		fail1 := NewMCTSNode("fail1", "fail_step_1")
		pruner.AddChild(root, fail1)
		pruner.EvaluateAndPrune(fail1, BranchEvalResult{CompilationError: true})

		good2 := NewMCTSNode("good2", "good_step_2")
		good2.Score = 0.9
		pruner.AddChild(good1, good2)

		stats := pruner.CollectStats(root)
		if stats.TotalNodes != 5 {
			t.Fatalf("total nodes = %d, expected 5", stats.TotalNodes)
		}
		if stats.PrunedNodes != 2 {
			t.Fatalf("pruned nodes = %d, expected 2", stats.PrunedNodes)
		}
		if stats.ActiveNodes != 3 {
			t.Fatalf("active nodes = %d, expected 3", stats.ActiveNodes)
		}
		if stats.PrunedByError != 1 {
			t.Fatalf("pruned by error = %d, expected 1", stats.PrunedByError)
		}
		if stats.PrunedByScore != 1 {
			t.Fatalf("pruned by score = %d, expected 1", stats.PrunedByScore)
		}

		bestPath := pruner.BestPath(root)
		if len(bestPath) != 3 {
			t.Fatalf("best path length = %d, expected 3", len(bestPath))
		}
		if bestPath[0].ID != "root" || bestPath[1].ID != "good1" || bestPath[2].ID != "good2" {
			t.Fatalf("unexpected best path: %s -> %s -> %s", bestPath[0].ID, bestPath[1].ID, bestPath[2].ID)
		}
	})
}
