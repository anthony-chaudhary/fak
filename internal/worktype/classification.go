package worktype

import "strings"

const ClassificationSchema = "fak-worktype/1"

type Classification struct {
	Schema      string   `json:"schema"`
	SessionID   string   `json:"session_id"`
	TraceID     string   `json:"trace_id,omitempty"`
	PatternID   string   `json:"pattern_id"`
	Subpatterns []string `json:"subpatterns,omitempty"`
	Evidence    []string `json:"evidence"`
	Provenance  string   `json:"provenance"`
}

// ClassifyDispatchPrompt maps only explicit issue-title verbs to the existing catalog.
// It abstains when multiple classes match or no explicit title signal is present.
func ClassifyDispatchPrompt(sessionID, prompt string) Classification {
	out := Classification{Schema: ClassificationSchema, SessionID: sessionID, TraceID: sessionID, PatternID: "unknown", Evidence: []string{"no_unique_explicit_issue_signal"}, Provenance: "deterministic_rule"}
	lower := strings.ToLower(prompt)
	rules := []struct {
		id, signal string
		subs       []string
	}{{"wp.issue-to-patch", "fix(", []string{"sp.reproduce-fix-witness"}}, {"wp.spec-to-feature", "feat(", []string{"sp.read-edit-verify"}}, {"wp.behavior-preserving-restructure", "refactor(", []string{"sp.inspect-edit-verify"}}, {"wp.interface-migration", "migrate(", []string{"sp.plan-fanout-reconcile"}}, {"wp.comprehension-report", "docs(", []string{"sp.search-fanout"}}}
	var matches []struct {
		id, signal string
		subs       []string
	}
	for _, r := range rules {
		if strings.Contains(lower, r.signal) {
			matches = append(matches, r)
		}
	}
	if len(matches) == 1 {
		m := matches[0]
		out.PatternID = m.id
		out.Subpatterns = m.subs
		out.Evidence = []string{"issue_title_verb:" + strings.TrimSuffix(m.signal, "(")}
	}
	return out
}
