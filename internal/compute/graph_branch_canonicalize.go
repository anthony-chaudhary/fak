package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// BranchCanonicalizePassName identifies the structured-branch canonicalization pass.
const BranchCanonicalizePassName GraphPassName = "branch-canonicalize"

// BranchCanonicalizeReceipt records observable metrics for branch canonicalization.
type BranchCanonicalizeReceipt struct {
	ConvertedSelects      int    `json:"converted_selects"`
	RetainedBranches      int    `json:"retained_branches"`
	SubstitutedConditions int    `json:"substituted_conditions"`
	FinalGraphDigest      string `json:"final_graph_digest,omitempty"`
}

// BranchCanonicalizeGraphPass canonicalizes pure structured branches to selects,
// propagates branch condition facts into regions, and preserves trapping branches.
type BranchCanonicalizeGraphPass struct{}

func (BranchCanonicalizeGraphPass) Name() GraphPassName { return BranchCanonicalizePassName }

func (p BranchCanonicalizeGraphPass) Apply(input Graph) (Graph, error) {
	result, _, err := p.ApplyWithReceipt(input)
	return result, err
}

// ApplyWithReceipt performs branch canonicalization and returns a deterministic receipt.
func (p BranchCanonicalizeGraphPass) ApplyWithReceipt(input Graph) (Graph, BranchCanonicalizeReceipt, error) {
	var receipt BranchCanonicalizeReceipt
	if err := input.Validate(); err != nil {
		return Graph{}, receipt, fmt.Errorf("compute graph branch canonicalize: invalid input: %w", err)
	}

	g := cloneGraph(input)
	var newNodes []GraphNode

	for _, node := range g.Nodes {
		if node.Op != GraphOpIf {
			newNodes = append(newNodes, node)
			continue
		}

		condID := node.Inputs[0]
		thenRegion := cloneSingleGraphRegion(node.Regions[0])
		elseRegion := cloneSingleGraphRegion(node.Regions[1])

		// Step 1: Propagate branch facts into then-branch (condition == 1.0).
		if regionUsesInput(thenRegion, condID) {
			factNode := GraphNode{
				ID:    NodeID(fmt.Sprintf("%s$then_true", node.ID)),
				Op:    GraphOpConstant,
				Value: 1.0,
			}
			thenRegion.Nodes = append([]GraphNode{factNode}, thenRegion.Nodes...)
			rewriteRegionInput(thenRegion, condID, factNode.ID)
			receipt.SubstitutedConditions++
		}

		// Step 2: Propagate branch facts into else-branch (condition == 0.0).
		if regionUsesInput(elseRegion, condID) {
			factNode := GraphNode{
				ID:    NodeID(fmt.Sprintf("%s$else_false", node.ID)),
				Op:    GraphOpConstant,
				Value: 0.0,
			}
			elseRegion.Nodes = append([]GraphNode{factNode}, elseRegion.Nodes...)
			rewriteRegionInput(elseRegion, condID, factNode.ID)
			receipt.SubstitutedConditions++
		}

		node.Regions[0] = thenRegion
		node.Regions[1] = elseRegion

		// Step 3: Check if both regions are pure (side-effect-free and non-trapping).
		// A trapping operation like division must remain structured.
		if isRegionPure(thenRegion) && isRegionPure(elseRegion) {
			newNodes = append(newNodes, thenRegion.Nodes...)
			newNodes = append(newNodes, elseRegion.Nodes...)
			selectNode := GraphNode{
				ID:      node.ID,
				Op:      GraphOpSelect,
				Inputs:  []NodeID{condID, thenRegion.Outputs[0], elseRegion.Outputs[0]},
				Regions: nil,
			}
			newNodes = append(newNodes, selectNode)
			receipt.ConvertedSelects++
		} else {
			newNodes = append(newNodes, node)
			receipt.RetainedBranches++
		}
	}

	candidate := Graph{
		Nodes:   newNodes,
		Outputs: cloneNodeIDs(g.Outputs),
	}

	ordered, err := stableTopologicalOrder(candidate)
	if err != nil {
		return Graph{}, receipt, fmt.Errorf("compute graph branch canonicalize: topological sort: %w", err)
	}
	candidate.Nodes = ordered

	if err := candidate.Validate(); err != nil {
		return Graph{}, receipt, fmt.Errorf("compute graph branch canonicalize: validate result: %w", err)
	}

	stableIR, err := candidate.StableIR()
	if err != nil {
		return Graph{}, receipt, err
	}
	digest := sha256.Sum256(stableIR)
	receipt.FinalGraphDigest = hex.EncodeToString(digest[:])

	return candidate, receipt, nil
}

func regionUsesInput(region GraphRegion, input NodeID) bool {
	for _, out := range region.Outputs {
		if out == input {
			return true
		}
	}
	for _, node := range region.Nodes {
		for _, in := range node.Inputs {
			if in == input {
				return true
			}
		}
		for _, sub := range node.Regions {
			if regionUsesInput(sub, input) {
				return true
			}
		}
	}
	return false
}

func rewriteRegionInput(region GraphRegion, oldID, newID NodeID) {
	for i, out := range region.Outputs {
		if out == oldID {
			region.Outputs[i] = newID
		}
	}
	for i := range region.Nodes {
		node := &region.Nodes[i]
		for j, in := range node.Inputs {
			if in == oldID {
				node.Inputs[j] = newID
			}
		}
		for _, sub := range node.Regions {
			rewriteRegionInput(sub, oldID, newID)
		}
	}
}

func isRegionPure(region GraphRegion) bool {
	for _, node := range region.Nodes {
		switch node.Op {
		case GraphOpConstant, GraphOpInput, GraphOpIdentity, GraphOpAdd, GraphOpMultiply, GraphOpSelect:
			// Pure and non-trapping.
		case GraphOpIf:
			for _, sub := range node.Regions {
				if !isRegionPure(sub) {
					return false
				}
			}
		default:
			// GraphOpDivide or any unknown/trapping operation.
			return false
		}
	}
	return true
}

func cloneSingleGraphRegion(r GraphRegion) GraphRegion {
	return cloneGraphRegions([]GraphRegion{r})[0]
}
