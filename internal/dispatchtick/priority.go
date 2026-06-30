package dispatchtick

import (
	"sort"
	"strings"
)

// Priority weights mirror the label-taxonomy weights in tools/issue_triage.py
// (its PRIORITY map: priority/P0=1000, priority/P1=400, priority/P2=150, with an
// unlabeled default of 60) and cmd/fak/tui.go's tuiPriorityWeights, so the
// dispatch picker, the triage scorer, and the issues TUI never disagree about how
// heavy a priority label is. The `p0-p1` named view in .github/issue-views.json
// orders by the same P0>P1 ranking. Keep these surfaces in lockstep if the
// taxonomy changes.
const (
	PriorityWeightP0      = 1000
	PriorityWeightP1      = 400
	PriorityWeightP2      = 150
	PriorityWeightDefault = 60
)

var priorityLabelWeight = map[string]int{
	"priority/P0": PriorityWeightP0,
	"priority/P1": PriorityWeightP1,
	"priority/P2": PriorityWeightP2,
}

// PriorityWeight is the dispatch-priority weight an issue earns from its labels:
// the HEAVIEST priority/P* label it carries, or PriorityWeightDefault when it
// carries none. An issue is never lighter than the default, so an unlabeled
// backlog can never sink below a labeled one by accident.
func PriorityWeight(labels []string) int {
	weight := PriorityWeightDefault
	for _, label := range labels {
		if w, ok := priorityLabelWeight[strings.TrimSpace(label)]; ok && w > weight {
			weight = w
		}
	}
	return weight
}

// laneIssueWeight resolves an issue's weight from a number->weight map, falling
// back to PriorityWeightDefault when the issue carries no priority/* label (so a
// map that stores only labeled issues stays small while every issue still ranks).
func laneIssueWeight(priority map[int]int, number int) int {
	if w, ok := priority[number]; ok {
		return w
	}
	return PriorityWeightDefault
}

// LaneCandidate is one orderable issue in a lane: its number and the priority
// weight earned from its labels.
type LaneCandidate struct {
	Number int
	Weight int
}

// OrderLaneCandidates returns the candidate issue numbers in dispatch order:
// highest priority weight first (an old priority/P1 outranks newer unlabeled
// noise), then by number — oldest-first (ascending) by default, newest-first
// when preferNewest. The number tiebreaker preserves the pre-priority ordering
// exactly, so when every candidate shares a weight (e.g. none carries a
// priority/* label) the result is byte-for-byte the old by-number order. A
// stable sort keeps equal-weight, equal-number inputs in their input order.
func OrderLaneCandidates(cands []LaneCandidate, preferNewest bool) []int {
	ordered := append([]LaneCandidate(nil), cands...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Weight != ordered[j].Weight {
			return ordered[i].Weight > ordered[j].Weight
		}
		if preferNewest {
			return ordered[i].Number > ordered[j].Number
		}
		return ordered[i].Number < ordered[j].Number
	})
	out := make([]int, len(ordered))
	for i, c := range ordered {
		out[i] = c.Number
	}
	return out
}
