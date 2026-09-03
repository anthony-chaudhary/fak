package polymodel

import (
	"testing"
)

func TestGenerateLinearTree(t *testing.T) {
	// Non-positive k
	zeroTree := GenerateLinearTree(0)
	if len(zeroTree.Nodes) != 1 {
		t.Fatalf("expected 1 node for k=0, got %d", len(zeroTree.Nodes))
	}
	negTree := GenerateLinearTree(-5)
	if len(negTree.Nodes) != 1 {
		t.Fatalf("expected 1 node for k=-5, got %d", len(negTree.Nodes))
	}

	// Positive k
	for _, k := range []int{1, 4, 16} {
		tree := GenerateLinearTree(k)
		if len(tree.Nodes) != k+1 {
			t.Fatalf("k=%d: expected %d nodes, got %d", k, k+1, len(tree.Nodes))
		}

		// Verify linear chain structure
		for i := 0; i < k; i++ {
			if len(tree.Nodes[i].Children) != 1 || tree.Nodes[i].Children[0] != i+1 {
				t.Errorf("k=%d: node %d children = %v, want [%d]", k, i, tree.Nodes[i].Children, i+1)
			}
		}
		if len(tree.Nodes[k].Children) != 0 {
			t.Errorf("k=%d: leaf node %d has children: %v", k, k, tree.Nodes[k].Children)
		}

		// Panel & mask validation
		panel, err := BuildTreePanel(tree)
		if err != nil {
			t.Fatalf("k=%d: BuildTreePanel failed: %v", k, err)
		}
		if err := ValidateTreeMask(panel); err != nil {
			t.Fatalf("k=%d: ValidateTreeMask failed: %v", k, err)
		}
	}
}

func TestGenerateWideShallowTree(t *testing.T) {
	// Defaults
	defTree := GenerateWideShallowTree(0, 0)
	// Default branch=4, depth=2 -> root(1) + 4 + 16 = 21 nodes
	if len(defTree.Nodes) != 21 {
		t.Fatalf("expected 21 nodes for default wide-shallow, got %d", len(defTree.Nodes))
	}

	// Custom
	tree := GenerateWideShallowTree(3, 2)
	// Root(1) + 3 + 9 = 13 nodes
	if len(tree.Nodes) != 13 {
		t.Fatalf("expected 13 nodes for branch=3, depth=2, got %d", len(tree.Nodes))
	}

	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed: %v", err)
	}
	if err := ValidateTreeMask(panel); err != nil {
		t.Fatalf("ValidateTreeMask failed: %v", err)
	}
}

func TestGenerateDeepNarrowTree(t *testing.T) {
	// Defaults
	defTree := GenerateDeepNarrowTree(0, 0)
	// Default branch=2, depth=4 -> root(1) + 2 + 4 + 8 + 16 = 31 nodes
	if len(defTree.Nodes) != 31 {
		t.Fatalf("expected 31 nodes for default deep-narrow, got %d", len(defTree.Nodes))
	}

	// Custom
	tree := GenerateDeepNarrowTree(1, 6)
	// Root(1) + 6 = 7 nodes (linear chain)
	if len(tree.Nodes) != 7 {
		t.Fatalf("expected 7 nodes for branch=1, depth=6, got %d", len(tree.Nodes))
	}

	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed: %v", err)
	}
	if err := ValidateTreeMask(panel); err != nil {
		t.Fatalf("ValidateTreeMask failed: %v", err)
	}
}

func TestGenerateTargetSizeTree(t *testing.T) {
	// targetK <= 0
	if tree := GenerateTargetSizeTree(0, "linear"); len(tree.Nodes) != 1 {
		t.Fatalf("expected 1 node for targetK=0, got %d", len(tree.Nodes))
	}

	topologies := []string{"linear", "wide", "deep", "unknown_custom"}
	targetSizes := []int{1, 2, 5, 8, 16, 32}

	for _, top := range topologies {
		for _, targetK := range targetSizes {
			tree := GenerateTargetSizeTree(targetK, top)
			candidates := len(tree.Nodes) - 1
			if candidates != targetK {
				t.Fatalf("top=%s, targetK=%d: candidate count = %d, want %d",
					top, targetK, candidates, targetK)
			}

			panel, err := BuildTreePanel(tree)
			if err != nil {
				t.Fatalf("top=%s, targetK=%d: BuildTreePanel failed: %v", top, targetK, err)
			}
			if err := ValidateTreeMask(panel); err != nil {
				t.Fatalf("top=%s, targetK=%d: ValidateTreeMask failed: %v", top, targetK, err)
			}
		}
	}
}
