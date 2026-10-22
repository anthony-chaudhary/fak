package polymodel

// AcceptancePosition reports speculative acceptance at one zero-based draft
// position. Rate is the marginal Accepted/Proposed and is nil when no proposal
// reached the position. Reached counts rounds whose drafts before this position
// were all accepted, so ConditionalRate = Accepted/Reached is the per-position
// acceptance probability a geometric yield model multiplies; it is nil when
// Reached is zero.
type AcceptancePosition struct {
	Position        int      `json:"position"`
	Proposed        int      `json:"proposed"`
	Accepted        int      `json:"accepted"`
	Rate            *float64 `json:"rate,omitempty"`
	Reached         int      `json:"reached"`
	ConditionalRate *float64 `json:"conditional_rate,omitempty"`
}

type acceptanceProfile struct {
	proposed []int
	accepted []int
	reached  []int
}

func (p *acceptanceProfile) grow(positions int) {
	if positions > len(p.proposed) {
		p.proposed = append(p.proposed, make([]int, positions-len(p.proposed))...)
		p.accepted = append(p.accepted, make([]int, positions-len(p.accepted))...)
		p.reached = append(p.reached, make([]int, positions-len(p.reached))...)
	}
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
	p.grow(proposed)
	for position := 0; position < proposed; position++ {
		p.proposed[position]++
		if position <= accepted {
			p.reached[position]++
		}
		if position < accepted {
			p.accepted[position]++
		}
	}
}

func (p acceptanceProfile) snapshot() []AcceptancePosition {
	result := make([]AcceptancePosition, len(p.proposed))
	for position, proposed := range p.proposed {
		row := AcceptancePosition{Position: position, Proposed: proposed, Accepted: p.accepted[position], Reached: p.reached[position]}
		if proposed > 0 {
			rate := float64(row.Accepted) / float64(proposed)
			row.Rate = &rate
		}
		if row.Reached > 0 {
			rate := float64(row.Accepted) / float64(row.Reached)
			row.ConditionalRate = &rate
		}
		result[position] = row
	}
	return result
}

// recordCounts folds one tree round: proposedByPosition counts nodes per depth,
// while Reached and Accepted count rounds, so a wide level reaches once.
func (p *acceptanceProfile) recordCounts(proposedByPosition []int, accepted int) {
	p.grow(len(proposedByPosition))
	for position, proposed := range proposedByPosition {
		if proposed < 0 {
			proposed = 0
		}
		p.proposed[position] += proposed
		if proposed > 0 && position <= accepted {
			p.reached[position]++
		}
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
