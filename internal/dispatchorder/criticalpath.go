package dispatchorder

import (
	"fmt"
	"sort"
	"strings"
)

// CriticalPathDisposition explains why an issue is or is not admitted by the
// opt-in critical-path picker.
type CriticalPathDisposition string

const (
	CriticalPathAdmit         CriticalPathDisposition = "admit"
	CriticalPathBlockedPrereq CriticalPathDisposition = "blocked_prerequisite"
	CriticalPathMissingPrereq CriticalPathDisposition = "missing_prerequisite"
	CriticalPathCycle         CriticalPathDisposition = "prerequisite_cycle"
	CriticalPathTreeCollision CriticalPathDisposition = "tree_collision"
	CriticalPathSeatLimit     CriticalPathDisposition = "seat_limit"
)

// CriticalPathScore is the auditable score used by CriticalPathPick.
type CriticalPathScore struct {
	IssueID          string                  `json:"issue_id"`
	Ready            bool                    `json:"ready"`
	DownstreamUnlock int                     `json:"downstream_unlock"`
	CriticalDepth    int                     `json:"critical_depth"`
	Disposition      CriticalPathDisposition `json:"disposition"`
	Reason           string                  `json:"reason,omitempty"`
}

// CriticalPathResult is one refill decision. Callers pass newly completed IDs
// back into the next call; the function never owns leases or intent claims.
type CriticalPathResult struct {
	Seats    int                 `json:"seats"`
	Admitted []string            `json:"admitted"`
	Scores   []CriticalPathScore `json:"scores"`
	Refill   bool                `json:"refill"`
}

// CriticalPathPick selects ready, tree-disjoint work that unlocks the greatest
// amount of downstream work. It is deliberately opt-in so legacy ordering does
// not change before its dispatch adapter selects this profile.
func CriticalPathPick(candidates []Candidate, completed []string, seats int) CriticalPathResult {
	if seats < 0 {
		seats = 0
	}
	byID := make(map[string]Candidate, len(candidates))
	for _, c := range candidates {
		if strings.TrimSpace(c.ID) != "" {
			byID[c.ID] = c
		}
	}
	done := make(map[string]bool, len(completed))
	for _, id := range completed {
		done[id] = true
	}
	cycle := criticalPathCycles(byID)
	children := make(map[string][]string)
	for _, c := range byID {
		for _, p := range c.BlockedBy {
			children[p] = append(children[p], c.ID)
		}
	}

	type scored struct {
		c Candidate
		s CriticalPathScore
	}
	rows := make([]scored, 0, len(byID))
	for _, c := range byID {
		if done[c.ID] {
			continue
		}
		s := CriticalPathScore{IssueID: c.ID, Ready: true, Disposition: CriticalPathSeatLimit}
		if cycle[c.ID] {
			s.Ready = false
			s.Disposition = CriticalPathCycle
			s.Reason = "issue participates in prerequisite cycle"
		} else {
			for _, p := range c.BlockedBy {
				if _, ok := byID[p]; !ok && !done[p] {
					s.Ready = false
					s.Disposition = CriticalPathMissingPrereq
					s.Reason = fmt.Sprintf("missing prerequisite %s", p)
					break
				}
				if !done[p] {
					s.Ready = false
					s.Disposition = CriticalPathBlockedPrereq
					s.Reason = fmt.Sprintf("waiting for prerequisite %s", p)
					break
				}
			}
		}
		s.DownstreamUnlock = criticalDescendants(c.ID, children, map[string]bool{})
		s.CriticalDepth = criticalDepth(c.ID, children, map[string]bool{})
		rows = append(rows, scored{c, s})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].s.Ready != rows[j].s.Ready {
			return rows[i].s.Ready
		}
		if rows[i].s.CriticalDepth != rows[j].s.CriticalDepth {
			return rows[i].s.CriticalDepth > rows[j].s.CriticalDepth
		}
		if rows[i].s.DownstreamUnlock != rows[j].s.DownstreamUnlock {
			return rows[i].s.DownstreamUnlock > rows[j].s.DownstreamUnlock
		}
		return rows[i].c.ID < rows[j].c.ID
	})

	selected := []Candidate{}
	for i := range rows {
		if !rows[i].s.Ready {
			continue
		}
		if len(selected) >= seats {
			rows[i].s.Disposition = CriticalPathSeatLimit
			rows[i].s.Reason = "ready but no seat available"
			continue
		}
		collision := false
		for _, prior := range selected {
			if TreesOverlap(rows[i].c.Tree, prior.Tree) {
				collision = true
				break
			}
		}
		if collision {
			rows[i].s.Disposition = CriticalPathTreeCollision
			rows[i].s.Reason = "ready but overlaps an admitted tree"
			continue
		}
		rows[i].s.Disposition = CriticalPathAdmit
		rows[i].s.Reason = ""
		selected = append(selected, rows[i].c)
	}
	out := CriticalPathResult{Seats: seats, Refill: len(completed) > 0}
	for _, c := range selected {
		out.Admitted = append(out.Admitted, c.ID)
	}
	for _, r := range rows {
		out.Scores = append(out.Scores, r.s)
	}
	return out
}

func criticalDescendants(id string, children map[string][]string, seen map[string]bool) int {
	total := 0
	for _, child := range children[id] {
		if seen[child] {
			continue
		}
		seen[child] = true
		total++
		total += criticalDescendants(child, children, seen)
	}
	return total
}
func criticalDepth(id string, children map[string][]string, seen map[string]bool) int {
	if seen[id] {
		return 0
	}
	seen[id] = true
	best := 0
	for _, child := range children[id] {
		if d := 1 + criticalDepth(child, children, seen); d > best {
			best = d
		}
	}
	delete(seen, id)
	return best
}
func criticalPathCycles(byID map[string]Candidate) map[string]bool {
	state := map[string]uint8{}
	stack := []string{}
	cycles := map[string]bool{}
	var visit func(string)
	visit = func(id string) {
		if state[id] == 2 {
			return
		}
		if state[id] == 1 {
			for i := len(stack) - 1; i >= 0; i-- {
				cycles[stack[i]] = true
				if stack[i] == id {
					break
				}
			}
			return
		}
		state[id] = 1
		stack = append(stack, id)
		for _, p := range byID[id].BlockedBy {
			if _, ok := byID[p]; ok {
				visit(p)
			}
		}
		stack = stack[:len(stack)-1]
		state[id] = 2
	}
	for id := range byID {
		visit(id)
	}
	return cycles
}
