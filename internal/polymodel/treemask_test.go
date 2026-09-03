package polymodel

import (
	"testing"
)

// ---------------------------------------------------------------------------
// 1. BuildTreePanel Tests
// ---------------------------------------------------------------------------

func TestBuildTreePanel_EmptyAndSingleNode(t *testing.T) {
	// Empty tree (nil nodes)
	emptyTreeNil := SpecTree{Nodes: nil}
	if _, err := BuildTreePanel(emptyTreeNil); err == nil {
		t.Fatal("expected error for nil nodes tree, got nil")
	}

	// Empty tree (0 length nodes slice)
	emptyTreeZero := SpecTree{Nodes: []TreeNode{}}
	if _, err := BuildTreePanel(emptyTreeZero); err == nil {
		t.Fatal("expected error for 0 nodes tree, got nil")
	}

	// Single node (root-only, K = 0 candidate tokens)
	singleNodeTree := SpecTree{Nodes: []TreeNode{{Token: 42}}}
	panel, err := BuildTreePanel(singleNodeTree)
	if err != nil {
		t.Fatalf("unexpected error for single node tree: %v", err)
	}
	if len(panel.Tokens) != 0 {
		t.Errorf("expected 0 candidate tokens, got %d", len(panel.Tokens))
	}
	if len(panel.NodeIndices) != 0 {
		t.Errorf("expected 0 node indices, got %d", len(panel.NodeIndices))
	}
	if len(panel.Depths) != 0 {
		t.Errorf("expected 0 depths, got %d", len(panel.Depths))
	}
	if len(panel.Positions) != 0 {
		t.Errorf("expected 0 positions, got %d", len(panel.Positions))
	}
	if len(panel.Parents) != 0 {
		t.Errorf("expected 0 parents, got %d", len(panel.Parents))
	}
	if panel.Mask.Size != 0 || len(panel.Mask.Data) != 0 {
		t.Errorf("expected empty mask (size 0, len 0), got size=%d, len=%d", panel.Mask.Size, len(panel.Mask.Data))
	}

	// ValidateTreeMask should pass vacuously on single node tree panel
	if err := ValidateTreeMask(panel); err != nil {
		t.Fatalf("ValidateTreeMask failed on single node panel: %v", err)
	}
}

func TestBuildTreePanel_LinearTree(t *testing.T) {
	// Linear tree: root(0) -> 1 -> 2 -> 3 -> 4
	k := 4
	tree := GenerateLinearTree(k)
	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed on linear tree: %v", err)
	}

	if len(panel.Tokens) != k {
		t.Fatalf("expected %d tokens, got %d", k, len(panel.Tokens))
	}
	if panel.Mask.Size != k {
		t.Fatalf("expected mask size %d, got %d", k, panel.Mask.Size)
	}

	for i := 0; i < k; i++ {
		// Node indices 1..k
		if panel.NodeIndices[i] != i+1 {
			t.Errorf("NodeIndices[%d] = %d, want %d", i, panel.NodeIndices[i], i+1)
		}
		// Depths: 1, 2, 3, 4
		if panel.Depths[i] != i+1 {
			t.Errorf("Depths[%d] = %d, want %d", i, panel.Depths[i], i+1)
		}
		// Positions: depth - 1 -> 0, 1, 2, 3
		if panel.Positions[i] != i {
			t.Errorf("Positions[%d] = %d, want %d", i, panel.Positions[i], i)
		}
		// Parents: -1 (root), 0, 1, 2
		expectedParent := i - 1
		if panel.Parents[i] != expectedParent {
			t.Errorf("Parents[%d] = %d, want %d", i, panel.Parents[i], expectedParent)
		}
	}

	// Linear causality: mask[q][key] is true iff key <= q
	for q := 0; q < k; q++ {
		for key := 0; key < k; key++ {
			want := key <= q
			if got := panel.Mask.Allow(q, key); got != want {
				t.Errorf("linear mask Allow(%d, %d) = %v, want %v", q, key, got, want)
			}
		}
	}

	// Helper index mapping methods
	for i := 0; i < k; i++ {
		nodeIdx := i + 1
		if pIdx := panel.PanelIndex(nodeIdx); pIdx != i {
			t.Errorf("PanelIndex(%d) = %d, want %d", nodeIdx, pIdx, i)
		}
		if tIdx := panel.TreeNodeIndex(i); tIdx != nodeIdx {
			t.Errorf("TreeNodeIndex(%d) = %d, want %d", i, tIdx, nodeIdx)
		}
	}
	if pIdx := panel.PanelIndex(0); pIdx != -1 {
		t.Errorf("PanelIndex(0) = %d, want -1 for root node", pIdx)
	}
	if pIdx := panel.PanelIndex(999); pIdx != -1 {
		t.Errorf("PanelIndex(999) = %d, want -1 for non-existent node", pIdx)
	}
	if tIdx := panel.TreeNodeIndex(-1); tIdx != -1 {
		t.Errorf("TreeNodeIndex(-1) = %d, want -1 for negative index", tIdx)
	}
	if tIdx := panel.TreeNodeIndex(k); tIdx != -1 {
		t.Errorf("TreeNodeIndex(%d) = %d, want -1 for out of bounds", k, tIdx)
	}
}

func TestBuildTreePanel_WideShallowTree(t *testing.T) {
	// Branch factor 3, depth 2 -> root(1) + level 1 (3) + level 2 (9) = 13 nodes, K = 12
	branchFactor := 3
	depth := 2
	tree := GenerateWideShallowTree(branchFactor, depth)
	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed on wide-shallow tree: %v", err)
	}

	expectedK := 12
	if len(panel.Tokens) != expectedK {
		t.Fatalf("expected %d tokens, got %d", expectedK, len(panel.Tokens))
	}
	if panel.Mask.Size != expectedK {
		t.Fatalf("expected mask size %d, got %d", expectedK, panel.Mask.Size)
	}

	// Level 1 nodes (first 3 candidates, panel indices 0, 1, 2)
	for i := 0; i < branchFactor; i++ {
		if panel.Parents[i] != -1 {
			t.Errorf("level 1 candidate %d parent = %d, want -1 (child of root)", i, panel.Parents[i])
		}
		if panel.Depths[i] != 1 {
			t.Errorf("level 1 candidate %d depth = %d, want 1", i, panel.Depths[i])
		}
		if panel.Positions[i] != 0 {
			t.Errorf("level 1 candidate %d position = %d, want 0", i, panel.Positions[i])
		}
	}

	// Level 1 siblings must NOT attend to each other
	for a := 0; a < branchFactor; a++ {
		for b := 0; b < branchFactor; b++ {
			if a != b && panel.Mask.Allow(a, b) {
				t.Errorf("level 1 sibling %d permitted to attend to sibling %d", a, b)
			}
		}
	}

	// Level 2 nodes (panel indices 3..11)
	for i := branchFactor; i < expectedK; i++ {
		if panel.Depths[i] != 2 {
			t.Errorf("level 2 candidate %d depth = %d, want 2", i, panel.Depths[i])
		}
		if panel.Positions[i] != 1 {
			t.Errorf("level 2 candidate %d position = %d, want 1", i, panel.Positions[i])
		}
		parent := panel.Parents[i]
		if parent < 0 || parent >= branchFactor {
			t.Errorf("level 2 candidate %d parent = %d, want [0, %d)", i, parent, branchFactor)
		}
		// Candidate must attend to its parent
		if !panel.Mask.Allow(i, parent) {
			t.Errorf("level 2 candidate %d blocked from attending to parent %d", i, parent)
		}
		// Candidate must attend to itself
		if !panel.Mask.Allow(i, i) {
			t.Errorf("candidate %d blocked from attending to itself", i)
		}
	}
}

func TestBuildTreePanel_DeepNarrowTree(t *testing.T) {
	// Branch factor 2, depth 4 -> root(1) + 2 + 4 + 8 + 16 = 31 nodes, K = 30
	branchFactor := 2
	depth := 4
	tree := GenerateDeepNarrowTree(branchFactor, depth)
	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed on deep-narrow tree: %v", err)
	}

	expectedK := 30
	if len(panel.Tokens) != expectedK {
		t.Fatalf("expected %d tokens, got %d", expectedK, len(panel.Tokens))
	}
	if panel.Mask.Size != expectedK {
		t.Fatalf("expected mask size %d, got %d", expectedK, panel.Mask.Size)
	}

	// Maximum depth must be depth (4)
	maxDepth := 0
	for _, d := range panel.Depths {
		if d > maxDepth {
			maxDepth = d
		}
	}
	if maxDepth != depth {
		t.Errorf("maxDepth = %d, want %d", maxDepth, depth)
	}

	// Verify that each candidate attends exclusively to itself and its ancestor chain
	for q := 0; q < expectedK; q++ {
		isAncestor := make(map[int]bool)
		isAncestor[q] = true
		for p := panel.Parents[q]; p >= 0; p = panel.Parents[p] {
			isAncestor[p] = true
		}

		for key := 0; key < expectedK; key++ {
			allowed := panel.Mask.Allow(q, key)
			expected := isAncestor[key]
			if allowed != expected {
				t.Errorf("deep-narrow candidate %d -> %d: got allowed=%v, want %v", q, key, allowed, expected)
			}
		}
	}
}

func TestBuildTreePanel_MalformedTrees(t *testing.T) {
	testCases := []struct {
		name string
		tree SpecTree
	}{
		{
			name: "child index out of bounds",
			tree: SpecTree{
				Nodes: []TreeNode{
					{Children: []int{5}},
					{Children: nil},
				},
			},
		},
		{
			name: "child index references root (0)",
			tree: SpecTree{
				Nodes: []TreeNode{
					{Children: []int{1}},
					{Children: []int{0}},
				},
			},
		},
		{
			name: "backward edge (non-forward layout)",
			tree: SpecTree{
				Nodes: []TreeNode{
					{Children: []int{2}},
					{Children: []int{}},
					{Children: []int{1}}, // 2 -> 1 violates topological order (child <= parent)
				},
			},
		},
		{
			name: "multiple parents for candidate node",
			tree: SpecTree{
				Nodes: []TreeNode{
					{Children: []int{1, 2}},
					{Children: []int{3}},
					{Children: []int{3}}, // node 3 claimed by both 1 and 2
					{Children: nil},
				},
			},
		},
		{
			name: "unreachable candidate node from root",
			tree: SpecTree{
				Nodes: []TreeNode{
					{Children: []int{1}},
					{Children: nil},
					{Children: nil}, // node 2 is disconnected
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildTreePanel(tc.tree)
			if err == nil {
				t.Errorf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestTreeMask_AllowBounds(t *testing.T) {
	mask := TreeMask{
		Size: 2,
		Data: []bool{
			true, false,
			true, true,
		},
	}

	// Valid in-bounds
	if !mask.Allow(0, 0) || mask.Allow(0, 1) || !mask.Allow(1, 0) || !mask.Allow(1, 1) {
		t.Error("unexpected Allow result for in-bounds coordinates")
	}

	// Out-of-bounds queries or keys
	outOfBounds := [][2]int{
		{-1, 0}, {0, -1}, {-1, -1},
		{2, 0}, {0, 2}, {2, 2},
		{10, 1}, {1, 10},
	}
	for _, pt := range outOfBounds {
		if mask.Allow(pt[0], pt[1]) {
			t.Errorf("expected false for out-of-bounds (%d, %d)", pt[0], pt[1])
		}
	}
}

// ---------------------------------------------------------------------------
// 2. ValidateTreeMask on Valid Trees
// ---------------------------------------------------------------------------

func TestValidateTreeMask_ValidTrees(t *testing.T) {
	t.Run("single node / root only", func(t *testing.T) {
		tree := SpecTree{Nodes: []TreeNode{{Token: 1}}}
		panel, err := BuildTreePanel(tree)
		if err != nil {
			t.Fatalf("BuildTreePanel failed: %v", err)
		}
		if err := ValidateTreeMask(panel); err != nil {
			t.Fatalf("ValidateTreeMask failed on valid root-only tree: %v", err)
		}
	})

	t.Run("linear trees of various sizes", func(t *testing.T) {
		for _, k := range []int{1, 2, 4, 8, 16} {
			tree := GenerateLinearTree(k)
			panel, err := BuildTreePanel(tree)
			if err != nil {
				t.Fatalf("k=%d: BuildTreePanel failed: %v", k, err)
			}
			if err := ValidateTreeMask(panel); err != nil {
				t.Fatalf("k=%d: ValidateTreeMask failed: %v", k, err)
			}
		}
	})

	t.Run("wide shallow trees", func(t *testing.T) {
		for _, b := range []int{2, 3, 5} {
			tree := GenerateWideShallowTree(b, 2)
			panel, err := BuildTreePanel(tree)
			if err != nil {
				t.Fatalf("b=%d: BuildTreePanel failed: %v", b, err)
			}
			if err := ValidateTreeMask(panel); err != nil {
				t.Fatalf("b=%d: ValidateTreeMask failed: %v", b, err)
			}
		}
	})

	t.Run("deep narrow trees", func(t *testing.T) {
		for _, d := range []int{1, 2, 4, 6} {
			tree := GenerateDeepNarrowTree(2, d)
			panel, err := BuildTreePanel(tree)
			if err != nil {
				t.Fatalf("d=%d: BuildTreePanel failed: %v", d, err)
			}
			if err := ValidateTreeMask(panel); err != nil {
				t.Fatalf("d=%d: ValidateTreeMask failed: %v", d, err)
			}
		}
	})

	t.Run("target size trees across topologies", func(t *testing.T) {
		topologies := []string{"linear", "wide", "deep"}
		sizes := []int{1, 4, 7, 16, 32}
		for _, top := range topologies {
			for _, sz := range sizes {
				tree := GenerateTargetSizeTree(sz, top)
				panel, err := BuildTreePanel(tree)
				if err != nil {
					t.Fatalf("top=%s, sz=%d: BuildTreePanel failed: %v", top, sz, err)
				}
				if err := ValidateTreeMask(panel); err != nil {
					t.Fatalf("top=%s, sz=%d: ValidateTreeMask failed: %v", top, sz, err)
				}
			}
		}
	})
}

// ---------------------------------------------------------------------------
// 3. ValidateTreeMask Negative Tests (Corrupted Masks)
// ---------------------------------------------------------------------------

func TestValidateTreeMask_CorruptedMasks(t *testing.T) {
	// Base valid tree: wide-shallow tree with branching factor 2, depth 2 (root + 2 + 4 = 7 nodes, K = 6)
	baseTree := GenerateWideShallowTree(2, 2)
	basePanel, err := BuildTreePanel(baseTree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed: %v", err)
	}
	if err := ValidateTreeMask(basePanel); err != nil {
		t.Fatalf("initial base panel must be valid, got: %v", err)
	}
	k := basePanel.Mask.Size

	t.Run("reflexivity violation", func(t *testing.T) {
		// Invariant: mask[i][i] == true for all i in [0, K)
		// Corrupt by setting Data[i*K + i] = false
		for i := 0; i < k; i++ {
			p := basePanel
			p.Mask.Data = append([]bool(nil), basePanel.Mask.Data...)
			p.Mask.Data[i*k+i] = false
			err := ValidateTreeMask(p)
			if err == nil {
				t.Fatalf("candidate %d: expected error for reflexivity violation, got nil", i)
			}
		}
	})

	t.Run("causal violation future candidate", func(t *testing.T) {
		// Invariant: mask[q][key] == false for key > q
		// Corrupt by setting Data[q*K + key] = true where key > q
		p := basePanel
		p.Mask.Data = append([]bool(nil), basePanel.Mask.Data...)
		q := 0
		key := 1 // key > q
		p.Mask.Data[q*k+key] = true

		err := ValidateTreeMask(p)
		if err == nil {
			t.Fatal("expected error for causal lower-triangular violation, got nil")
		}
	})

	t.Run("sibling leakage", func(t *testing.T) {
		// Invariant: siblings (sharing same parent) must NOT attend to each other
		// Find two candidates with the same parent
		var sibA, sibB int
		found := false
		for a := 0; a < k && !found; a++ {
			for b := a + 1; b < k; b++ {
				if basePanel.Parents[a] == basePanel.Parents[b] {
					sibA = a
					sibB = b
					found = true
					break
				}
			}
		}
		if !found {
			t.Fatal("could not find sibling pair in basePanel")
		}

		// Leakage sibB -> sibA (since sibB > sibA, sibB is permitted to attend backwards causally, but NOT to sibling!)
		p := basePanel
		p.Mask.Data = append([]bool(nil), basePanel.Mask.Data...)
		p.Mask.Data[sibB*k+sibA] = true

		err := ValidateTreeMask(p)
		if err == nil {
			t.Fatalf("expected error for sibling leakage between %d and %d, got nil", sibB, sibA)
		}
	})

	t.Run("missing ancestor", func(t *testing.T) {
		// Invariant: candidate q must attend to all ancestors along its root path
		// Find a candidate with an ancestor key (e.g. parent) and set Data[q*K + parent] = false
		var targetQ, targetP int
		found := false
		for q := 0; q < k; q++ {
			if basePanel.Parents[q] >= 0 {
				targetQ = q
				targetP = basePanel.Parents[q]
				found = true
				break
			}
		}
		if !found {
			t.Fatal("could not find candidate with parent >= 0")
		}

		p := basePanel
		p.Mask.Data = append([]bool(nil), basePanel.Mask.Data...)
		p.Mask.Data[targetQ*k+targetP] = false // Drop ancestor attention

		err := ValidateTreeMask(p)
		if err == nil {
			t.Fatalf("expected error for missing ancestor (%d dropped parent %d), got nil", targetQ, targetP)
		}
	})

	t.Run("position inconsistency", func(t *testing.T) {
		// Invariant 1: positions[i] == depths[i] - 1
		p1 := basePanel
		p1.Positions = append([]int(nil), basePanel.Positions...)
		p1.Positions[0] = 999
		if err := ValidateTreeMask(p1); err == nil {
			t.Fatal("expected error for position mismatch != depths[i]-1, got nil")
		}

		// Invariant 2: candidate depth must be >= 1
		p2 := basePanel
		p2.Depths = append([]int(nil), basePanel.Depths...)
		p2.Positions = append([]int(nil), basePanel.Positions...)
		p2.Depths[0] = 0
		p2.Positions[0] = -1
		if err := ValidateTreeMask(p2); err == nil {
			t.Fatal("expected error for candidate depth < 1, got nil")
		}

		// Invariant 3: root child must have depth 1
		p3 := basePanel
		p3.Depths = append([]int(nil), basePanel.Depths...)
		p3.Positions = append([]int(nil), basePanel.Positions...)
		p3.Depths[0] = 3
		p3.Positions[0] = 2
		if err := ValidateTreeMask(p3); err == nil {
			t.Fatal("expected error for root child depth != 1, got nil")
		}

		// Invariant 4: non-root child depth must be parent depth + 1
		var nonRootChild int
		for i := 0; i < k; i++ {
			if basePanel.Parents[i] >= 0 {
				nonRootChild = i
				break
			}
		}
		p4 := basePanel
		p4.Depths = append([]int(nil), basePanel.Depths...)
		p4.Positions = append([]int(nil), basePanel.Positions...)
		p4.Depths[nonRootChild] = p4.Depths[p4.Parents[nonRootChild]] + 5
		p4.Positions[nonRootChild] = p4.Depths[nonRootChild] - 1
		if err := ValidateTreeMask(p4); err == nil {
			t.Fatal("expected error for child depth != parent depth + 1, got nil")
		}

		// Invariant 5: parent index out of bounds
		p5 := basePanel
		p5.Parents = append([]int(nil), basePanel.Parents...)
		p5.Parents[0] = -5
		if err := ValidateTreeMask(p5); err == nil {
			t.Fatal("expected error for parent out of bounds < -1, got nil")
		}

		// Invariant 6: parent index violates topological order (parent >= child)
		p6 := basePanel
		p6.Parents = append([]int(nil), basePanel.Parents...)
		p6.Parents[0] = 0 // parent == child
		if err := ValidateTreeMask(p6); err == nil {
			t.Fatal("expected error for parent >= child index, got nil")
		}
	})

	t.Run("dimension inconsistencies", func(t *testing.T) {
		// Slice length mismatches
		pTokens := basePanel
		pTokens.Tokens = pTokens.Tokens[:len(pTokens.Tokens)-1]
		if err := ValidateTreeMask(pTokens); err == nil {
			t.Fatal("expected error for tokens length mismatch")
		}

		pIndices := basePanel
		pIndices.NodeIndices = append(pIndices.NodeIndices, 99)
		if err := ValidateTreeMask(pIndices); err == nil {
			t.Fatal("expected error for node_indices length mismatch")
		}

		pDepths := basePanel
		pDepths.Depths = pDepths.Depths[:len(pDepths.Depths)-1]
		if err := ValidateTreeMask(pDepths); err == nil {
			t.Fatal("expected error for depths length mismatch")
		}

		pPositions := basePanel
		pPositions.Positions = pPositions.Positions[:len(pPositions.Positions)-1]
		if err := ValidateTreeMask(pPositions); err == nil {
			t.Fatal("expected error for positions length mismatch")
		}

		pParents := basePanel
		pParents.Parents = pParents.Parents[:len(pParents.Parents)-1]
		if err := ValidateTreeMask(pParents); err == nil {
			t.Fatal("expected error for parents length mismatch")
		}

		pMaskData := basePanel
		pMaskData.Mask.Data = pMaskData.Mask.Data[:len(pMaskData.Mask.Data)-1]
		if err := ValidateTreeMask(pMaskData); err == nil {
			t.Fatal("expected error for mask data length mismatch != K*K")
		}

		pNegativeSize := basePanel
		pNegativeSize.Mask.Size = -1
		if err := ValidateTreeMask(pNegativeSize); err == nil {
			t.Fatal("expected error for negative mask size")
		}
	})
}

// ---------------------------------------------------------------------------
// 4. VerifyMaskAcceptanceConsistency with AcceptTree Tests
// ---------------------------------------------------------------------------

func TestVerifyMaskAcceptanceConsistency_AcceptTree(t *testing.T) {
	// Build a branching tree:
	// Root 0 (token 10, targetArgmax 101, children [1, 2])
	//   Child 1 (token 101, targetArgmax 201, children [3])  <- matches root target!
	//   Child 2 (token 102, targetArgmax 999, children [4])  <- alternative branch
	//   Child 3 (token 201, targetArgmax 301, children [5])  <- matches node 1 target!
	//   Child 4 (token 202, targetArgmax 999, children nil)
	//   Child 5 (token 999, targetArgmax 999, children nil)  <- diverges (token 999 != targetArgmax 301)
	tree := SpecTree{
		Nodes: []TreeNode{
			{Token: 10, TargetArgmax: 101, Children: []int{1, 2}},
			{Token: 101, TargetArgmax: 201, Children: []int{3}},
			{Token: 102, TargetArgmax: 999, Children: []int{4}},
			{Token: 201, TargetArgmax: 301, Children: []int{5}},
			{Token: 202, TargetArgmax: 999, Children: nil},
			{Token: 999, TargetArgmax: 999, Children: nil},
		},
	}

	panel, err := BuildTreePanel(tree)
	if err != nil {
		t.Fatalf("BuildTreePanel failed: %v", err)
	}

	result := AcceptTree(tree)

	// AcceptTree should have accepted path [1, 3]:
	// Step 1: Root(10) target is 101 -> matches child 1 (token 101).
	// Step 2: Node 1 target is 201 -> matches child 3 (token 201).
	// Step 3: Node 3 target is 301 -> child 5 has token 999 (divergence, stop).
	if len(result.Path) != 2 || result.Path[0] != 1 || result.Path[1] != 3 {
		t.Fatalf("unexpected AcceptTree path: got %v, want [1, 3]", result.Path)
	}
	if result.KeepKV != 2 {
		t.Fatalf("unexpected KeepKV: got %d, want 2", result.KeepKV)
	}
	if result.Advance != 3 {
		t.Fatalf("unexpected Advance: got %d, want 3", result.Advance)
	}
	totalCandidates := len(panel.Tokens)
	if result.KeepKV+result.EvictKV != totalCandidates {
		t.Fatalf("KV conservation failed: KeepKV(%d) + EvictKV(%d) != %d",
			result.KeepKV, result.EvictKV, totalCandidates)
	}

	// 1. Verify consistency passes on valid result
	if err := VerifyMaskAcceptanceConsistency(tree, panel, result); err != nil {
		t.Fatalf("VerifyMaskAcceptanceConsistency failed on valid path: %v", err)
	}

	// 2. Empty path acceptance (root target matches none of its children)
	t.Run("empty path acceptance", func(t *testing.T) {
		treeNoMatch := SpecTree{
			Nodes: []TreeNode{
				{Token: 10, TargetArgmax: 777, Children: []int{1, 2}},
				{Token: 101, TargetArgmax: 999, Children: nil},
				{Token: 102, TargetArgmax: 999, Children: nil},
			},
		}
		panelNoMatch, err := BuildTreePanel(treeNoMatch)
		if err != nil {
			t.Fatalf("BuildTreePanel failed: %v", err)
		}
		resultNoMatch := AcceptTree(treeNoMatch)
		if len(resultNoMatch.Path) != 0 {
			t.Fatalf("expected empty path, got %v", resultNoMatch.Path)
		}
		if err := VerifyMaskAcceptanceConsistency(treeNoMatch, panelNoMatch, resultNoMatch); err != nil {
			t.Fatalf("VerifyMaskAcceptanceConsistency failed on empty path: %v", err)
		}
	})

	// 3. Corrupted KeepKV
	t.Run("corrupted KeepKV", func(t *testing.T) {
		badResult := result
		badResult.KeepKV = 99
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for corrupted KeepKV, got nil")
		}
	})

	// 4. Corrupted Advance
	t.Run("corrupted Advance", func(t *testing.T) {
		badResult := result
		badResult.Advance = len(result.Path) // Should be len(result.Path) + 1
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for corrupted Advance, got nil")
		}
	})

	// 5. Corrupted EvictKV (KV conservation violation)
	t.Run("KV conservation violation", func(t *testing.T) {
		badResult := result
		badResult.EvictKV = 0 // KeepKV(2) + EvictKV(0) = 2 != 5
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for KV conservation violation, got nil")
		}
	})

	// 6. Discontinuous path (jumping across sibling branches: node 2 then node 3)
	t.Run("discontinuous path", func(t *testing.T) {
		badResult := result
		badResult.Path = []int{2, 3} // node 2 is sibling of node 1, node 3 is child of node 1
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for discontinuous path, got nil")
		}
	})

	// 7. Out of bounds tree node index on path
	t.Run("out of bounds node index on path", func(t *testing.T) {
		badResult := result
		badResult.Path = []int{1, 999}
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for out of bounds node index on path, got nil")
		}
	})

	// 8. First node on path not a child of root
	t.Run("first node not child of root", func(t *testing.T) {
		badResult := result
		badResult.Path = []int{3} // node 3 is child of node 1, not root
		badResult.KeepKV = 1
		badResult.Advance = 2
		badResult.EvictKV = totalCandidates - 1
		if err := VerifyMaskAcceptanceConsistency(tree, panel, badResult); err == nil {
			t.Fatal("expected error for first node not child of root, got nil")
		}
	})
}
