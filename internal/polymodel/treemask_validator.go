package polymodel

import "fmt"

// ValidateTreeMask verifies that a TreePanel satisfies all structural invariants:
//   - Dimension consistency: all slice lengths match Mask.Size and Data has size K*K.
//   - Reflexivity: mask[i][i] == true for all candidates 0 <= i < K.
//   - Lower-triangular causality: mask[q][k] == false for any key k > query q.
//   - Exact ancestry: mask[q][k] == true iff k == q or k is on the ancestor path of q.
//   - Sibling/cousin isolation: mask[a][b] == false for non-ancestor peers (siblings, cousins).
//   - Position consistency: positions[i] == depths[i] - 1 and parent/child depth increments.
func ValidateTreeMask(panel TreePanel) error {
	k := panel.Mask.Size
	if k < 0 {
		return fmt.Errorf("polymodel: negative mask size %d", k)
	}

	// 1. Dimension consistency checks
	if len(panel.Tokens) != k {
		return fmt.Errorf("polymodel: tokens length %d != mask size %d", len(panel.Tokens), k)
	}
	if len(panel.NodeIndices) != k {
		return fmt.Errorf("polymodel: node indices length %d != mask size %d", len(panel.NodeIndices), k)
	}
	if len(panel.Depths) != k {
		return fmt.Errorf("polymodel: depths length %d != mask size %d", len(panel.Depths), k)
	}
	if len(panel.Positions) != k {
		return fmt.Errorf("polymodel: positions length %d != mask size %d", len(panel.Positions), k)
	}
	if len(panel.Parents) != k {
		return fmt.Errorf("polymodel: parents length %d != mask size %d", len(panel.Parents), k)
	}
	if len(panel.Mask.Data) != k*k {
		return fmt.Errorf("polymodel: mask data length %d != %d (K*K)", len(panel.Mask.Data), k*k)
	}

	// For empty candidate set (K = 0), all constraints hold vacuously.
	if k == 0 {
		return nil
	}

	// 2. Position consistency checks
	for i := 0; i < k; i++ {
		if panel.Positions[i] != panel.Depths[i]-1 {
			return fmt.Errorf("polymodel: position mismatch at index %d: position=%d, depth=%d (want depth-1)",
				i, panel.Positions[i], panel.Depths[i])
		}
		if panel.Depths[i] < 1 {
			return fmt.Errorf("polymodel: invalid depth %d at index %d (candidate depths must be >= 1)",
				panel.Depths[i], i)
		}
		p := panel.Parents[i]
		if p < -1 || p >= k {
			return fmt.Errorf("polymodel: parent index %d at candidate %d out of bounds [-1, %d)", p, i, k)
		}
		if p >= i {
			return fmt.Errorf("polymodel: parent index %d at candidate %d violates topological order (parent must precede child)", p, i)
		}
		if p == -1 {
			if panel.Depths[i] != 1 {
				return fmt.Errorf("polymodel: root child at index %d has depth %d (want 1)", i, panel.Depths[i])
			}
		} else {
			if panel.Depths[i] != panel.Depths[p]+1 {
				return fmt.Errorf("polymodel: depth increment mismatch at index %d: depth=%d, parent depth=%d (want parent depth + 1)",
					i, panel.Depths[i], panel.Depths[p])
			}
		}
	}

	// 3. Reflexivity check: mask[i][i] == true
	for i := 0; i < k; i++ {
		if !panel.Mask.Allow(i, i) {
			return fmt.Errorf("polymodel: reflexivity violated: mask[%d][%d] is false", i, i)
		}
	}

	// 4. Lower-triangular causality check: mask[q][key] == false for key > q
	for q := 0; q < k; q++ {
		for key := q + 1; key < k; key++ {
			if panel.Mask.Allow(q, key) {
				return fmt.Errorf("polymodel: lower-triangular causality violated: mask[%d][%d] is true for key > query", q, key)
			}
		}
	}

	// 5. Exact ancestry check: mask[q][key] == true iff key is on ancestor path of q (or self)
	for q := 0; q < k; q++ {
		isAncestor := make([]bool, k)
		isAncestor[q] = true // self is allowed by reflexivity
		for p := panel.Parents[q]; p >= 0; p = panel.Parents[p] {
			isAncestor[p] = true
		}

		for key := 0; key < k; key++ {
			allowed := panel.Mask.Allow(q, key)
			if allowed && !isAncestor[key] {
				return fmt.Errorf("polymodel: isolation violated: query %d permitted to attend to non-ancestor %d", q, key)
			}
			if !allowed && isAncestor[key] {
				return fmt.Errorf("polymodel: ancestry violated: query %d blocked from attending to ancestor %d", q, key)
			}
		}
	}

	// 6. Sibling/cousin isolation check: explicitly verify non-ancestor peer nodes
	for a := 0; a < k; a++ {
		for b := a + 1; b < k; b++ {
			if panel.Parents[a] == panel.Parents[b] {
				// a and b are siblings sharing the same parent
				if panel.Mask.Allow(a, b) {
					return fmt.Errorf("polymodel: sibling isolation violated: candidate %d attends to sibling %d", a, b)
				}
				if panel.Mask.Allow(b, a) {
					return fmt.Errorf("polymodel: sibling isolation violated: candidate %d attends to sibling %d", b, a)
				}
			}
		}
	}

	return nil
}

// VerifyMaskAcceptanceConsistency verifies that the nodes along result.Path
// (from AcceptTree) correspond to a valid, continuous attention subpath in the
// TreePanel's mask, and that KV conservation holds.
func VerifyMaskAcceptanceConsistency(tree SpecTree, panel TreePanel, result TreeResult) error {
	// Ensure the panel itself is structurally valid first.
	if err := ValidateTreeMask(panel); err != nil {
		return fmt.Errorf("polymodel: invalid panel in acceptance consistency check: %w", err)
	}

	// 1. Verify result bookkeeping matches result.Path
	if result.KeepKV != len(result.Path) {
		return fmt.Errorf("polymodel: result KeepKV (%d) does not match len(Path) (%d)",
			result.KeepKV, len(result.Path))
	}
	if result.Advance != len(result.Path)+1 {
		return fmt.Errorf("polymodel: result Advance (%d) does not match len(Path) + 1 (%d)",
			result.Advance, len(result.Path)+1)
	}

	// Speculative KV conservation: KeepKV + EvictKV == total speculative candidates
	totalSpec := len(panel.Tokens)
	if len(tree.Nodes) > 1 {
		if result.KeepKV+result.EvictKV != totalSpec {
			return fmt.Errorf("polymodel: KV conservation violated: KeepKV (%d) + EvictKV (%d) != %d speculative candidates",
				result.KeepKV, result.EvictKV, totalSpec)
		}
	}

	// If no speculative nodes were accepted, the path is empty (only root advance).
	if len(result.Path) == 0 {
		return nil
	}

	// 2. Map tree node indices on result.Path to panel indices and verify tokens
	panelPath := make([]int, len(result.Path))
	for step, treeNodeIdx := range result.Path {
		if treeNodeIdx <= 0 || treeNodeIdx >= len(tree.Nodes) {
			return fmt.Errorf("polymodel: result path step %d references out-of-bounds tree node %d", step, treeNodeIdx)
		}
		pIdx := panel.PanelIndex(treeNodeIdx)
		if pIdx < 0 {
			return fmt.Errorf("polymodel: tree node %d on result path not found in panel", treeNodeIdx)
		}
		if panel.Tokens[pIdx] != tree.Nodes[treeNodeIdx].Token {
			return fmt.Errorf("polymodel: token mismatch at path step %d: panel token %d != tree token %d",
				step, panel.Tokens[pIdx], tree.Nodes[treeNodeIdx].Token)
		}
		panelPath[step] = pIdx
	}

	// 3. Verify path continuity in the panel
	// First step must be a child of root (parent == -1 in panel).
	if panel.Parents[panelPath[0]] != -1 {
		return fmt.Errorf("polymodel: first path candidate %d is not a child of root (parent=%d)",
			panelPath[0], panel.Parents[panelPath[0]])
	}
	// Subsequent steps must be direct children of the preceding path candidate.
	for step := 1; step < len(panelPath); step++ {
		expectedParent := panelPath[step-1]
		actualParent := panel.Parents[panelPath[step]]
		if actualParent != expectedParent {
			return fmt.Errorf("polymodel: path discontinuity at step %d: candidate %d parent is %d, want previous path candidate %d",
				step, panelPath[step], actualParent, expectedParent)
		}
	}

	// 4. Verify attention subpath in the mask: each step on the accepted path must
	// attend to all preceding steps on the path, and must NOT attend to non-ancestor siblings.
	for j := 0; j < len(panelPath); j++ {
		curr := panelPath[j]
		for i := 0; i <= j; i++ {
			prev := panelPath[i]
			if !panel.Mask.Allow(curr, prev) {
				return fmt.Errorf("polymodel: mask blocked path candidate %d from attending to path ancestor %d",
					curr, prev)
			}
		}
	}

	return nil
}
