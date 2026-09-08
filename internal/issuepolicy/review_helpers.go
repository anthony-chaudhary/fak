package issuepolicy

import (
	"sort"
	"strings"
)

func containsPrivateBoundary(c Candidate) bool {
	fields := []string{
		c.Title, c.ParentRef, c.CurrentState, c.WhyNow, c.WorkingSpine,
		c.PriorityContext, c.InScope, c.OutOfScope, c.DoneCondition, c.Witness, c.AcceptanceGate,
	}
	fields = append(fields, c.Paths...)
	fields = append(fields, c.BoundaryNotes...)
	text := strings.ToLower(strings.Join(fields, "\n"))
	for _, needle := range []string{
		"fak-private",
		"slack control",
		"slack-control",
		"gpu-server reservation",
		"operator-only evidence",
		"credential dump",
		"secret key",
		"api key",
		"private token",
	} {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func compact(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

type reasonSet map[string]bool

func (s reasonSet) add(reason string) { s[reason] = true }
func (s reasonSet) has(reason string) bool {
	return s[reason]
}
func (s reasonSet) list() []string {
	out := make([]string, 0, len(s))
	for reason := range s {
		out = append(out, reason)
	}
	sort.Strings(out)
	return out
}

func removeReasons(list []string, toRemove ...string) []string {
	removeSet := make(map[string]bool, len(toRemove))
	for _, r := range toRemove {
		removeSet[r] = true
	}
	out := make([]string, 0, len(list))
	for _, r := range list {
		if !removeSet[r] {
			out = append(out, r)
		}
	}
	return out
}

func removeStrings(list []string, toRemove ...string) []string {
	removeSet := make(map[string]bool, len(toRemove))
	for _, s := range toRemove {
		removeSet[s] = true
	}
	out := make([]string, 0, len(list))
	for _, s := range list {
		if !removeSet[s] {
			out = append(out, s)
		}
	}
	return out
}

func hasProblemFrameAndDescriptions(c Candidate, reasons reasonSet) bool {
	if reasons.has(ReasonProblemFrameIncomplete) {
		return false
	}
	if reasons.has(ReasonPrivateBoundary) ||
		reasons.has(ReasonLiveUnarmored) ||
		reasons.has(ReasonNoiseIncomplete) ||
		reasons.has(ReasonAgentIncomplete) ||
		reasons.has(ReasonNotDispatchLeaf) ||
		reasons.has(ReasonOversizedSteps) ||
		reasons.has(ReasonUnexpandedTemplate) {
		return false
	}
	hasDesc := strings.TrimSpace(c.Title) != "" && (strings.TrimSpace(c.CurrentState) != "" ||
		strings.TrimSpace(c.InScope) != "" ||
		strings.TrimSpace(c.DoneCondition) != "" ||
		strings.TrimSpace(c.WhyNow) != "" ||
		strings.TrimSpace(c.BetterBecause) != "" ||
		strings.TrimSpace(c.Body) != "")
	return hasDesc
}
