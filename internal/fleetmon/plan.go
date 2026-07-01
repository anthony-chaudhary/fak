package fleetmon

import (
	"encoding/json"
	"sort"
	"strings"
)

// RunPlanSchema tags a run-plan file so a reader can version the format.
const RunPlanSchema = "fak-fleet-runplan/1"

// RunPlan is the issue -> worker mapping a fleet launch produces and the
// monitor/fold/replace verbs consume. It is the join key between a launched
// worker (session id, account bucket, transcript namespace) and the issue it was
// dispatched to work. The plan is optional input: without one the monitor still
// classifies discovered workers, but a plan lets it (a) name the expected set so
// a missing worker is visible, and (b) carry the issue URL/area forward into a
// replacement prompt.
type RunPlan struct {
	Schema  string       `json:"schema,omitempty"`
	RunID   string       `json:"run_id,omitempty"`
	Created string       `json:"created,omitempty"`
	Workers []PlanWorker `json:"workers"`
}

// PlanWorker is one dispatched worker's identity and the issue it owns.
type PlanWorker struct {
	Issue          int    `json:"issue"`
	IssueURL       string `json:"issue_url,omitempty"`
	Session        string `json:"session"`
	Account        string `json:"account,omitempty"`         // account/config bucket
	Area           string `json:"area,omitempty"`            // expected area/subsystem
	Namespace      string `json:"namespace,omitempty"`       // transcript project dir (~/.claude/projects/<ns>)
	TranscriptPath string `json:"transcript_path,omitempty"` // resolved JSONL path, when known
	PID            int    `json:"pid,omitempty"`             // worker root PID, when known
	// ReplacementOf, when set, marks this worker as a replacement session for a
	// prior (superseded) session id — so the fold can link the two to one issue.
	ReplacementOf string `json:"replacement_of,omitempty"`
}

// ParseRunPlan reads a run plan from JSON. It tolerates two shapes: the full
// {"schema":...,"workers":[...]} object, or a bare array of PlanWorker rows (the
// minimal hand-authored form). A row missing both an issue and a session is
// dropped. An input that is neither shape returns an error.
func ParseRunPlan(data []byte) (RunPlan, error) {
	trimmed := strings.TrimSpace(string(data))
	if trimmed == "" {
		return RunPlan{}, errEmptyPlan
	}
	var plan RunPlan
	if trimmed[0] == '{' {
		if err := json.Unmarshal(data, &plan); err != nil {
			return RunPlan{}, err
		}
	} else {
		var rows []PlanWorker
		if err := json.Unmarshal(data, &rows); err != nil {
			return RunPlan{}, err
		}
		plan.Workers = rows
	}
	kept := plan.Workers[:0]
	for _, w := range plan.Workers {
		if w.Issue == 0 && strings.TrimSpace(w.Session) == "" {
			continue
		}
		kept = append(kept, w)
	}
	plan.Workers = kept
	return plan, nil
}

// errEmptyPlan is returned by ParseRunPlan for empty input.
var errEmptyPlan = &planError{"run plan is empty"}

type planError struct{ msg string }

func (e *planError) Error() string { return "fleet run plan: " + e.msg }

// WorkerBySession returns the plan worker with the given session id, if any.
func (p RunPlan) WorkerBySession(session string) (PlanWorker, bool) {
	for _, w := range p.Workers {
		if w.Session == session {
			return w, true
		}
	}
	return PlanWorker{}, false
}

// SessionsForIssue returns every session id the plan carries for one issue, in
// plan order — the original plus any replacements. It is the join the fold uses
// to link a superseded session to the issue's live replacement.
func (p RunPlan) SessionsForIssue(issue int) []string {
	var out []string
	for _, w := range p.Workers {
		if w.Issue == issue && strings.TrimSpace(w.Session) != "" {
			out = append(out, w.Session)
		}
	}
	return out
}

// Issues returns the sorted, de-duplicated set of issue numbers in the plan.
func (p RunPlan) Issues() []int {
	seen := map[int]bool{}
	var out []int
	for _, w := range p.Workers {
		if w.Issue != 0 && !seen[w.Issue] {
			seen[w.Issue] = true
			out = append(out, w.Issue)
		}
	}
	sort.Ints(out)
	return out
}
