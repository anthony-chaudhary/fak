package polymodel

// AcceptancePosition reports speculative acceptance at one zero-based draft
// position. Rate is nil when no proposal reached the position.
type AcceptancePosition struct {
	Position int      `json:"position"`
	Proposed int      `json:"proposed"`
	Accepted int      `json:"accepted"`
	Rate     *float64 `json:"rate,omitempty"`
}

type acceptanceProfile struct {
	proposed []int
	accepted []int
}

func (p *acceptanceProfile) record(proposed, accepted int) {
	if proposed < 0 {
		proposed = 0
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > proposed {
		accepted = proposed
	}
	if proposed > len(p.proposed) {
		p.proposed = append(p.proposed, make([]int, proposed-len(p.proposed))...)
		p.accepted = append(p.accepted, make([]int, proposed-len(p.accepted))...)
	}
	for position := 0; position < proposed; position++ {
		p.proposed[position]++
		if position < accepted {
			p.accepted[position]++
		}
	}
}

func (p acceptanceProfile) snapshot() []AcceptancePosition {
	result := make([]AcceptancePosition, len(p.proposed))
	for position, proposed := range p.proposed {
		row := AcceptancePosition{Position: position, Proposed: proposed, Accepted: p.accepted[position]}
		if proposed > 0 {
			rate := float64(row.Accepted) / float64(proposed)
			row.Rate = &rate
		}
		result[position] = row
	}
	return result
}

func (p *acceptanceProfile) recordCounts(proposedByPosition []int, accepted int) {
	if len(proposedByPosition) > len(p.proposed) {
		p.proposed = append(p.proposed, make([]int, len(proposedByPosition)-len(p.proposed))...)
		p.accepted = append(p.accepted, make([]int, len(proposedByPosition)-len(p.accepted))...)
	}
	for position, proposed := range proposedByPosition {
		if proposed < 0 {
			proposed = 0
		}
		p.proposed[position] += proposed
		if position < accepted {
			p.accepted[position]++
		}
	}
}

func treeProposalPositions(tree SpecTree) []int {
	if len(tree.Nodes) <= 1 {
		return nil
	}
	depth := make([]int, len(tree.Nodes))
	counts := []int{}
	for parent, node := range tree.Nodes {
		for _, child := range node.Children {
			if child <= parent || child >= len(tree.Nodes) {
				continue
			}
			depth[child] = depth[parent] + 1
			position := depth[child] - 1
			for position >= len(counts) {
				counts = append(counts, 0)
			}
			counts[position]++
		}
	}
	return counts
}
