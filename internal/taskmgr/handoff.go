package taskmgr

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	// SchemaHandoff is the input contract for a task completion handoff.
	SchemaHandoff = "fak.task-handoff.v1"
	// SchemaHandoffReview is the output contract for the pure handoff gate.
	SchemaHandoffReview = "fak.task-handoff-review.v1"
)

var handoffMarkerRE = regexp.MustCompile(`<!--\s*fak-task-handoff-key:\s*([^>\s]+)\s*-->`)
var handoffKeyRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$`)

// Handoff is the machine-readable record a finishing agent hands to the next
// loop. It turns "remember to follow up" into typed state: the task's claimed
// completion, the independent witness beside it, where the item currently
// stands, and either one or two concrete next steps or an explicit reason that
// no follow-up is reasonable.
type Handoff struct {
	Schema              string            `json:"schema"`
	Task                HandoffTask       `json:"task"`
	CurrentState        string            `json:"current_state"`
	Summary             string            `json:"summary,omitempty"`
	CompletedBy         string            `json:"completed_by,omitempty"`
	Labels              map[string]string `json:"labels,omitempty"`
	NextSteps           []HandoffNextStep `json:"next_steps,omitempty"`
	NoNextStepReason    string            `json:"no_next_step_reason,omitempty"`
	CompletionEvidence  []EvidenceRef     `json:"completion_evidence,omitempty"`
	CompletionTimestamp int64             `json:"completion_unix_nano,omitempty"`
}

// HandoffTask is the compact task slice needed by the handoff gate. It mirrors
// TaskSnapshot's identity/state/witness fields without requiring a full runtime
// snapshot in hand-authored fixtures.
type HandoffTask struct {
	TaskID  string         `json:"task_id"`
	Title   string         `json:"title,omitempty"`
	State   State          `json:"state"`
	Witness *WitnessRecord `json:"witness,omitempty"`
}

// HandoffNextStep is one concrete follow-up the next agent can pick up. The CLI
// can sync each entry to one stable GitHub issue.
type HandoffNextStep struct {
	Key          string        `json:"key"`
	Title        string        `json:"title"`
	Body         string        `json:"body"`
	Reason       string        `json:"reason"`
	Priority     string        `json:"priority,omitempty"`
	Labels       []string      `json:"labels,omitempty"`
	EvidenceRefs []EvidenceRef `json:"evidence_refs,omitempty"`
}

// HandoffReview is the pure verdict. OK means the handoff has enough witnessed
// completion evidence and next-step state for an automated loop to act on it.
type HandoffReview struct {
	Schema       string   `json:"schema"`
	OK           bool     `json:"ok"`
	Verdict      string   `json:"verdict"`
	Reasons      []string `json:"reasons,omitempty"`
	TaskID       string   `json:"task_id,omitempty"`
	NextStepKeys []string `json:"next_step_keys,omitempty"`
	IssueCount   int      `json:"issue_count"`
}

// HandoffIssue is the subset of a GitHub issue needed to dedupe handoff-created
// follow-ups by marker.
type HandoffIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url,omitempty"`
}

// HandoffIssuePlanRow is a create/update decision for one next step.
type HandoffIssuePlanRow struct {
	Action       string   `json:"action"`
	Key          string   `json:"key"`
	Number       *int     `json:"number,omitempty"`
	State        string   `json:"state,omitempty"`
	Title        string   `json:"title"`
	Body         string   `json:"-"`
	Labels       []string `json:"labels,omitempty"`
	Reason       string   `json:"reason"`
	Priority     string   `json:"priority,omitempty"`
	EvidenceRefs []string `json:"evidence_refs,omitempty"`
}

// ReviewHandoff grades h without side effects. The gate deliberately requires
// VerifiedDone for StateDone handoffs: a task's own "done" string is not proof
// that it should fan out follow-up work.
func ReviewHandoff(h Handoff) HandoffReview {
	var reasons []string
	if h.Schema != SchemaHandoff {
		reasons = append(reasons, "BAD_SCHEMA")
	}
	taskID := strings.TrimSpace(h.Task.TaskID)
	if taskID == "" {
		reasons = append(reasons, "MISSING_TASK_ID")
	}
	if h.Task.State != StateDone {
		reasons = append(reasons, "TASK_NOT_DONE")
	}
	if h.Task.Witness == nil {
		reasons = append(reasons, "MISSING_COMPLETION_WITNESS")
	} else if h.Task.Witness.VerifiedState != VerifiedDone {
		reasons = append(reasons, "COMPLETION_NOT_VERIFIED")
	}
	if strings.TrimSpace(h.CurrentState) == "" {
		reasons = append(reasons, "MISSING_CURRENT_STATE")
	}

	nextCount := len(h.NextSteps)
	noNextReason := strings.TrimSpace(h.NoNextStepReason)
	switch {
	case nextCount == 0 && noNextReason == "":
		reasons = append(reasons, "MISSING_NEXT_STEP_OR_NOT_APPLICABLE_REASON")
	case nextCount > 0 && noNextReason != "":
		reasons = append(reasons, "NEXT_STEP_AND_NOT_APPLICABLE_BOTH_SET")
	case nextCount > 2:
		reasons = append(reasons, "TOO_MANY_NEXT_STEPS")
	}

	seen := map[string]bool{}
	keys := make([]string, 0, len(h.NextSteps))
	for i, step := range h.NextSteps {
		prefix := "NEXT_STEP_" + strconv.Itoa(i+1) + "_"
		key := strings.TrimSpace(step.Key)
		if key == "" {
			reasons = append(reasons, prefix+"MISSING_KEY")
		} else if !handoffKeyRE.MatchString(key) {
			reasons = append(reasons, prefix+"BAD_KEY")
		} else if seen[key] {
			reasons = append(reasons, prefix+"DUPLICATE_KEY")
		} else {
			seen[key] = true
			keys = append(keys, key)
		}
		if strings.TrimSpace(step.Title) == "" {
			reasons = append(reasons, prefix+"MISSING_TITLE")
		}
		if strings.TrimSpace(step.Body) == "" {
			reasons = append(reasons, prefix+"MISSING_BODY")
		}
		if strings.TrimSpace(step.Reason) == "" {
			reasons = append(reasons, prefix+"MISSING_REASON")
		}
	}

	review := HandoffReview{
		Schema:       SchemaHandoffReview,
		OK:           len(reasons) == 0,
		TaskID:       taskID,
		Reasons:      reasons,
		NextStepKeys: keys,
		IssueCount:   nextCount,
	}
	switch {
	case !review.OK:
		review.Verdict = "refused"
	case nextCount == 0:
		review.Verdict = "not_applicable"
	default:
		review.Verdict = "ready"
	}
	return review
}

// HandoffMarkerKey extracts the stable marker key from an issue body.
func HandoffMarkerKey(body string) string {
	m := handoffMarkerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// BuildHandoffIssuePlan decides create vs update for every next step.
func BuildHandoffIssuePlan(h Handoff, existing []HandoffIssue) []HandoffIssuePlanRow {
	byKey := map[string]HandoffIssue{}
	for _, issue := range existing {
		if key := HandoffMarkerKey(issue.Body); key != "" {
			byKey[key] = issue
		}
	}
	rows := make([]HandoffIssuePlanRow, 0, len(h.NextSteps))
	for _, step := range h.NextSteps {
		row := HandoffIssuePlanRow{
			Action:       "create",
			Key:          strings.TrimSpace(step.Key),
			Title:        strings.TrimSpace(step.Title),
			Body:         HandoffIssueBody(h, step),
			Labels:       compactStrings(step.Labels),
			Reason:       strings.TrimSpace(step.Reason),
			Priority:     strings.TrimSpace(step.Priority),
			EvidenceRefs: evidenceRefStrings(step.EvidenceRefs),
		}
		if found, ok := byKey[row.Key]; ok {
			row.Action = "update"
			n := found.Number
			row.Number = &n
			row.State = found.State
		}
		rows = append(rows, row)
	}
	return rows
}

// HandoffIssueBody renders the dedupe marker plus enough state for a future
// agent to understand where the item stands before picking it up.
func HandoffIssueBody(h Handoff, step HandoffNextStep) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-task-handoff-key: %s -->\n", strings.TrimSpace(step.Key))
	fmt.Fprintf(&b, "# %s\n\n", strings.TrimSpace(step.Title))
	fmt.Fprintf(&b, "This follow-up was pushed by a verified task handoff.\n\n")
	fmt.Fprintf(&b, "- Handoff schema: `%s`\n", SchemaHandoff)
	fmt.Fprintf(&b, "- Task: `%s`", strings.TrimSpace(h.Task.TaskID))
	if title := strings.TrimSpace(h.Task.Title); title != "" {
		fmt.Fprintf(&b, " - %s", title)
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Claimed state: `%s`\n", h.Task.State)
	if h.Task.Witness != nil {
		fmt.Fprintf(&b, "- Completion witness: `%s`", h.Task.Witness.VerifiedState)
		if src := strings.TrimSpace(h.Task.Witness.Source); src != "" {
			fmt.Fprintf(&b, " via `%s`", src)
		}
		if sha := strings.TrimSpace(h.Task.Witness.SHA); sha != "" {
			fmt.Fprintf(&b, " (`%s`)", sha)
		}
		fmt.Fprintln(&b)
	}
	fmt.Fprintf(&b, "- Current state: %s\n", oneLine(h.CurrentState))
	if p := strings.TrimSpace(step.Priority); p != "" {
		fmt.Fprintf(&b, "- Priority: `%s`\n", p)
	}
	fmt.Fprintf(&b, "- Why this is next: %s\n", oneLine(step.Reason))
	if refs := evidenceRefStrings(append(append([]EvidenceRef{}, h.CompletionEvidence...), step.EvidenceRefs...)); len(refs) > 0 {
		fmt.Fprintln(&b, "- Evidence:")
		for _, ref := range refs {
			fmt.Fprintf(&b, "  - `%s`\n", ref)
		}
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Next step")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, strings.TrimSpace(step.Body))
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "Managed by `fak task handoff`; re-running the same handoff updates this issue in place.")
	return b.String()
}

func compactStrings(in []string) []string {
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

func evidenceRefStrings(refs []EvidenceRef) []string {
	out := make([]string, 0, len(refs))
	for _, ref := range refs {
		kind := strings.TrimSpace(ref.Kind)
		value := strings.TrimSpace(ref.Ref)
		note := strings.TrimSpace(ref.Note)
		if kind == "" && value == "" && note == "" {
			continue
		}
		var b strings.Builder
		if kind != "" {
			b.WriteString(kind)
		}
		if value != "" {
			if b.Len() > 0 {
				b.WriteString(":")
			}
			b.WriteString(value)
		}
		if note != "" {
			if b.Len() > 0 {
				b.WriteString(" ")
			}
			b.WriteString("(")
			b.WriteString(note)
			b.WriteString(")")
		}
		out = append(out, b.String())
	}
	return out
}

func oneLine(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.Join(strings.Fields(s), " ")
}
