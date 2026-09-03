package polymodel

import (
	"errors"
	"fmt"
)

// TreeMask represents a 2D causal ancestor attention mask (K x K) for
// tree speculative verification. Size is K (the number of speculative
// candidate tokens, excluding the root), and Data holds the row-major
// K x K boolean adjacency matrix where Data[q*Size + k] is true iff
// query candidate q is permitted to attend to key candidate k.
type TreeMask struct {
	Size int    `json:"size"`
	Data []bool `json:"data"`
}

// Allow reports whether candidate token query q is permitted to attend
// to candidate token key k. In a causal tree attention mask, query q
// may attend to key k if and only if k == q (reflexivity) or k is an
// ancestor of q along the candidate continuation path in the tree.
// Returns false for any out-of-bounds indices.
func (m TreeMask) Allow(q, k int) bool {
	if q < 0 || q >= m.Size || k < 0 || k >= m.Size {
		return false
	}
	idx := q*m.Size + k
	if idx < 0 || idx >= len(m.Data) {
		return false
	}
	return m.Data[idx]
}

// TreePanel holds the flattened candidate tokens and topological structures
// required for a single-pass tree verification forward step. All candidate
// slices have length K = len(tree.Nodes) - 1, excluding the root node 0.
type TreePanel struct {
	Tokens      []int    `json:"tokens"`       // Candidate tokens proposed by the tree (excluding root node 0).
	NodeIndices []int    `json:"node_indices"` // Mapping from panel index (0..K-1) to original SpecTree node index.
	Depths      []int    `json:"depths"`       // Tree depth of each candidate node (root is depth 0; candidates have depth >= 1).
	Positions   []int    `json:"positions"`    // Relative verification position offset for RoPE/embeddings (depth - 1).
	Parents     []int    `json:"parents"`      // Parent index within the panel (-1 if candidate's parent is root node 0).
	Mask        TreeMask `json:"mask"`         // 2D causal ancestor attention mask (K x K).
}

// PanelIndex returns the panel index (0..K-1) corresponding to the given
// SpecTree node index. Returns -1 if the node index is not present in the panel.
func (p TreePanel) PanelIndex(nodeIdx int) int {
	for i, idx := range p.NodeIndices {
		if idx == nodeIdx {
			return i
		}
	}
	return -1
}

// TreeNodeIndex returns the original SpecTree node index corresponding to the
// given panel index (0..K-1). Returns -1 if panelIdx is out of bounds.
func (p TreePanel) TreeNodeIndex(panelIdx int) int {
	if panelIdx < 0 || panelIdx >= len(p.NodeIndices) {
		return -1
	}
	return p.NodeIndices[panelIdx]
}

// BuildTreePanel constructs the flattened candidate TreePanel, depths, relative
// positions (depth - 1), parent pointers, and 2D causal ancestor TreeMask from a
// SpecTree.
//
// SpecTree layout invariants:
//   - Node 0 is the committed root position (excluded from the candidate panel).
//   - Candidate nodes are indexed 1..len(tree.Nodes)-1 in topological forward order
//     where child index > parent index.
//   - Each candidate node must be reachable from the root and have exactly one parent.
func BuildTreePanel(tree SpecTree) (TreePanel, error) {
	n := len(tree.Nodes)
	if n == 0 {
		return TreePanel{}, errors.New("polymodel: empty tree has no root node")
	}

	// Root-only tree: zero speculative candidate tokens (K = 0).
	if n == 1 {
		return TreePanel{
			Tokens:      []int{},
			NodeIndices: []int{},
			Depths:      []int{},
			Positions:   []int{},
			Parents:     []int{},
			Mask:        TreeMask{Size: 0, Data: []bool{}},
		}, nil
	}

	// Map each tree node to its unique parent index in tree.Nodes.
	// -1 indicates root node; -2 indicates unvisited/unreachable.
	treeParent := make([]int, n)
	for i := range treeParent {
		treeParent[i] = -2
	}
	treeParent[0] = -1 // Root node 0 has no parent in the tree

	for pIdx, node := range tree.Nodes {
		for _, cIdx := range node.Children {
			if cIdx <= 0 || cIdx >= n {
				return TreePanel{}, fmt.Errorf("polymodel: child index %d out of bounds [1, %d)", cIdx, n)
			}
			if cIdx <= pIdx {
				return TreePanel{}, fmt.Errorf("polymodel: non-forward child edge %d -> %d violates causal layout", pIdx, cIdx)
			}
			if treeParent[cIdx] != -2 {
				return TreePanel{}, fmt.Errorf("polymodel: node %d has multiple parents (%d and %d)", cIdx, treeParent[cIdx], pIdx)
			}
			treeParent[cIdx] = pIdx
		}
	}

	// Verify that every candidate node 1..n-1 is reachable from the root.
	for i := 1; i < n; i++ {
		if treeParent[i] == -2 {
			return TreePanel{}, fmt.Errorf("polymodel: candidate node %d is unreachable from root", i)
		}
	}

	k := n - 1
	tokens := make([]int, k)
	nodeIndices := make([]int, k)
	depths := make([]int, k)
	positions := make([]int, k)
	parents := make([]int, k)

	for i := 0; i < k; i++ {
		nodeIdx := i + 1
		tokens[i] = tree.Nodes[nodeIdx].Token
		nodeIndices[i] = nodeIdx

		p := treeParent[nodeIdx]
		if p == 0 {
			// Direct child of root node 0
			parents[i] = -1
			depths[i] = 1
		} else {
			// Child of an earlier candidate node (panel index p - 1)
			pPanel := p - 1
			parents[i] = pPanel
			depths[i] = depths[pPanel] + 1
		}
		positions[i] = depths[i] - 1
	}

	// Construct 2D causal ancestor attention mask (K x K).
	maskData := make([]bool, k*k)
	for q := 0; q < k; q++ {
		// Reflexivity: candidate q always attends to itself
		maskData[q*k+q] = true

		// Ancestry: candidate q attends to all ancestors along its root path
		for p := parents[q]; p >= 0; p = parents[p] {
			maskData[q*k+p] = true
		}
	}

	return TreePanel{
		Tokens:      tokens,
		NodeIndices: nodeIndices,
		Depths:      depths,
		Positions:   positions,
		Parents:     parents,
		Mask: TreeMask{
			Size: k,
			Data: maskData,
		},
	}, nil
}
