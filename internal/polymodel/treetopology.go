package polymodel

import "strings"

// GenerateLinearTree constructs a linear chain SpecTree of size k (k speculative candidate nodes).
// Node 0 is the root committed position. Speculative candidate nodes are indexed 1..k with
// strictly forward edges 0 -> 1 -> 2 -> ... -> k.
// If k <= 0, returns a root-only SpecTree.
func GenerateLinearTree(k int) SpecTree {
	if k <= 0 {
		return SpecTree{Nodes: []TreeNode{{}}}
	}

	nodes := make([]TreeNode, k+1)
	for i := 0; i < k; i++ {
		nodes[i].Children = []int{i + 1}
		if i > 0 {
			nodes[i].Token = 100 + i
		}
	}
	nodes[k].Token = 100 + k
	return SpecTree{Nodes: nodes}
}

// GenerateWideShallowTree constructs a SpecTree with uniform branchFactor up to depth.
// Node 0 is the root. Typically branchFactor is wide (e.g. 4 or 8) and depth is shallow
// (e.g. 1 or 2). If branchFactor <= 0, it defaults to 4. If depth <= 0, it defaults to 2.
func GenerateWideShallowTree(branchFactor, depth int) SpecTree {
	if branchFactor <= 0 {
		branchFactor = 4
	}
	if depth <= 0 {
		depth = 2
	}

	nodes := []TreeNode{{}} // Root node 0
	currLevel := []int{0}
	tokenCounter := 100

	for d := 0; d < depth; d++ {
		var nextLevel []int
		for _, parentIdx := range currLevel {
			for b := 0; b < branchFactor; b++ {
				childIdx := len(nodes)
				tokenCounter++
				nodes = append(nodes, TreeNode{Token: tokenCounter})
				nodes[parentIdx].Children = append(nodes[parentIdx].Children, childIdx)
				nextLevel = append(nextLevel, childIdx)
			}
		}
		currLevel = nextLevel
	}

	return SpecTree{Nodes: nodes}
}

// GenerateDeepNarrowTree constructs a SpecTree with a narrow branchFactor across deeper depth.
// Node 0 is the root. Typically branchFactor is 1 or 2 and depth is larger (e.g. 4, 8).
// If branchFactor <= 0, it defaults to 2. If depth <= 0, it defaults to 4.
func GenerateDeepNarrowTree(branchFactor, depth int) SpecTree {
	if branchFactor <= 0 {
		branchFactor = 2
	}
	if depth <= 0 {
		depth = 4
	}

	nodes := []TreeNode{{}} // Root node 0
	currLevel := []int{0}
	tokenCounter := 100

	for d := 0; d < depth; d++ {
		var nextLevel []int
		for _, parentIdx := range currLevel {
			for b := 0; b < branchFactor; b++ {
				childIdx := len(nodes)
				tokenCounter++
				nodes = append(nodes, TreeNode{Token: tokenCounter})
				nodes[parentIdx].Children = append(nodes[parentIdx].Children, childIdx)
				nextLevel = append(nextLevel, childIdx)
			}
		}
		currLevel = nextLevel
	}

	return SpecTree{Nodes: nodes}
}

// GenerateTargetSizeTree constructs a SpecTree targeting an exact candidate count targetK
// (excluding root 0) under the specified topology ("linear", "wide"/"wide_shallow",
// "deep"/"deep_narrow").
//
// If targetK <= 0, returns a root-only SpecTree.
func GenerateTargetSizeTree(targetK int, topology string) SpecTree {
	if targetK <= 0 {
		return SpecTree{Nodes: []TreeNode{{}}}
	}

	top := strings.ToLower(strings.TrimSpace(topology))
	switch {
	case top == "linear" || top == "chain":
		return GenerateLinearTree(targetK)

	case strings.Contains(top, "deep") || strings.Contains(top, "narrow"):
		// Narrow branching tree (branch factor <= 2) producing exactly targetK candidate nodes.
		nodes := []TreeNode{{}} // Root at 0
		queue := []int{0}
		tokenCounter := 100

		for len(nodes)-1 < targetK && len(queue) > 0 {
			parentIdx := queue[0]
			queue = queue[1:]

			for b := 0; b < 2 && len(nodes)-1 < targetK; b++ {
				childIdx := len(nodes)
				tokenCounter++
				nodes = append(nodes, TreeNode{Token: tokenCounter})
				nodes[parentIdx].Children = append(nodes[parentIdx].Children, childIdx)
				queue = append(queue, childIdx)
			}
		}
		return SpecTree{Nodes: nodes}

	case strings.Contains(top, "wide") || strings.Contains(top, "shallow"):
		fallthrough
	default:
		// Wide shallow tree (depth <= 2) producing exactly targetK candidate nodes.
		nodes := []TreeNode{{}} // Root at 0
		b := targetK
		if targetK > 3 {
			b = (targetK + 1) / 2
		}

		// Level 1: root's children (indices 1..b)
		for i := 1; i <= b; i++ {
			nodes = append(nodes, TreeNode{Token: 100 + i})
			nodes[0].Children = append(nodes[0].Children, i)
		}

		// Level 2: distribute remaining (targetK - b) nodes among level-1 parents round-robin
		rem := targetK - b
		for i := 0; i < rem; i++ {
			parentIdx := 1 + (i % b)
			childIdx := len(nodes)
			nodes = append(nodes, TreeNode{Token: 100 + childIdx})
			nodes[parentIdx].Children = append(nodes[parentIdx].Children, childIdx)
		}
		return SpecTree{Nodes: nodes}
	}
}
