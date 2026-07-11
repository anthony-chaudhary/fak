package dispatchtick

import "strings"

type ProjectIssueFields struct {
	Issue    int    `json:"issue"`
	Priority string `json:"priority,omitempty"`
	Status   string `json:"status,omitempty"`
}

func ProjectPriorityWeight(priority string) int {
	switch strings.ToUpper(strings.TrimSpace(priority)) {
	case "P0", "URGENT":
		return PriorityWeightP0
	case "P1", "HIGH":
		return PriorityWeightP1
	case "P2", "MEDIUM":
		return PriorityWeightP2
	default:
		return PriorityWeightDefault
	}
}

func ProjectStatusDispatchable(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "done", "closed", "blocked", "icebox":
		return false
	default:
		return true
	}
}

// MergeProjectFields applies board authority when present and leaves label-derived
// weights unchanged when the board has no row, so rollout is additive/fail-open.
func MergeProjectFields(priority map[int]int, issues []int, fields map[int]ProjectIssueFields) (map[int]int, []int) {
	outP := make(map[int]int, len(priority)+len(fields))
	for n, w := range priority {
		outP[n] = w
	}
	outI := make([]int, 0, len(issues))
	for _, n := range issues {
		f, ok := fields[n]
		if ok && !ProjectStatusDispatchable(f.Status) {
			continue
		}
		if ok && strings.TrimSpace(f.Priority) != "" {
			outP[n] = ProjectPriorityWeight(f.Priority)
		}
		outI = append(outI, n)
	}
	return outP, outI
}
