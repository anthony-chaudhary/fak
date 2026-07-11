package dispatchtick

import "strings"

const RefuseObjectiveNoScorer = "OBJECTIVE_SCORER_MISSING"

type ObjectiveContract struct {
	Objective string `json:"objective,omitempty"`
	Scorers   string `json:"scorers,omitempty"`
	Attached  bool   `json:"attached"`
	Refusal   string `json:"refusal,omitempty"`
}

// ParseObjectiveContract derives the child objective and its independently
// checkable scorer contract from the issue contract. Working spine says what
// trajectory to pursue; Witness says how progress is measured without trusting
// worker narration. Legacy issues with neither section remain uncontracted.
func ParseObjectiveContract(body string) ObjectiveContract {
	objective := markdownSection(body, "working spine")
	scorers := markdownSection(body, "witness")
	out := ObjectiveContract{Objective: objective, Scorers: scorers}
	if objective == "" {
		return out
	}
	if scorers == "" {
		out.Refusal = RefuseObjectiveNoScorer
		return out
	}
	out.Attached = true
	return out
}

func markdownSection(body, wanted string) string {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	start := -1
	for i, line := range lines {
		t := strings.TrimSpace(line)
		if !strings.HasPrefix(t, "## ") {
			continue
		}
		name := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(t, "## ")))
		if start >= 0 {
			return strings.TrimSpace(strings.Join(lines[start:i], "\n"))
		}
		if name == wanted {
			start = i + 1
		}
	}
	if start >= 0 {
		return strings.TrimSpace(strings.Join(lines[start:], "\n"))
	}
	return ""
}

func (c ObjectiveContract) PromptBlock() string {
	if !c.Attached {
		return ""
	}
	return "objective contract (kernel-authored from the issue; do not replace with self-reported progress):\n" +
		"objective:\n" + c.Objective + "\n\n" +
		"attached scorers / witnessed progress:\n" + c.Scorers
}
