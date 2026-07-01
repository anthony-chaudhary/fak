package taskmgr

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
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
	Key                     string        `json:"key"`
	Title                   string        `json:"title"`
	Body                    string        `json:"body"`
	Reason                  string        `json:"reason"`
	Generation              string        `json:"generation,omitempty"`
	PromotionEvidence       []string      `json:"promotion_evidence,omitempty"`
	DemotionEvidence        []string      `json:"demotion_evidence,omitempty"`
	InvalidatingAssumptions []string      `json:"invalidating_assumptions,omitempty"`
	GenerationNonGoals      []string      `json:"generation_non_goals,omitempty"`
	WorkingSpine            string        `json:"working_spine,omitempty"`
	PriorityContext         string        `json:"priority_context,omitempty"`
	WorkUnit                string        `json:"work_unit,omitempty"`
	ExpectedSteps           int           `json:"expected_steps,omitempty"`
	Assumptions             []string      `json:"assumptions,omitempty"`
	ConfusionRisks          []string      `json:"confusion_risks,omitempty"`
	Coordination            []string      `json:"coordination,omitempty"`
	Trigger                 string        `json:"trigger,omitempty"`
	BatchPolicy             string        `json:"batch_policy,omitempty"`
	InScope                 string        `json:"in_scope,omitempty"`
	OutOfScope              string        `json:"out_of_scope,omitempty"`
	DoneCondition           string        `json:"done_condition,omitempty"`
	Witness                 string        `json:"witness,omitempty"`
	AcceptanceGate          string        `json:"acceptance_gate,omitempty"`
	Lane                    string        `json:"lane,omitempty"`
	Paths                   []string      `json:"paths,omitempty"`
	Priority                string        `json:"priority,omitempty"`
	Labels                  []string      `json:"labels,omitempty"`
	BoundaryNotes           []string      `json:"boundary_notes,omitempty"`
	ClosureBinding          string        `json:"closure_binding,omitempty"`
	EvidenceRefs            []EvidenceRef `json:"evidence_refs,omitempty"`
}

// HandoffReview is the pure verdict. OK means the handoff has enough witnessed
// completion evidence and next-step state for an automated loop to act on it.
type HandoffReview struct {
	Schema       string                 `json:"schema"`
	OK           bool                   `json:"ok"`
	Verdict      string                 `json:"verdict"`
	Reasons      []string               `json:"reasons,omitempty"`
	TaskID       string                 `json:"task_id,omitempty"`
	NextStepKeys []string               `json:"next_step_keys,omitempty"`
	IssueCount   int                    `json:"issue_count"`
	IssueReviews []issuecontract.Review `json:"issue_reviews,omitempty"`
}

// HandoffReviewOptions turns on the stricter GitHub-issue contract for callers
// that are about to plan or sync follow-up issues. The default ReviewHandoff path
// stays the basic task-completion gate for existing non-issue users.
//
// Closure binding: StrictScope plus the typed next-step fields on HandoffNextStep
// (InScope, OutOfScope, DoneCondition, Witness, AcceptanceGate, Lane, Paths,
// BoundaryNotes, ClosureBinding, ...), the ReviewHandoffWithOptions gate below that
// refuses a vague next step via issuecontract.ReviewCandidate before live issue
// sync, and HandoffIssueBody's stable-section rendering together satisfy #1460's
// ask in full, covered by handoff_test.go's TestReviewHandoffStrictScopeRejectsVagueNextStep,
// TestReviewHandoffStrictScopeAcceptsDispatchableNextStep, and
// TestHandoffIssueBodyIncludesStrictScopeSections, with cmd/fak/taskmgr.go's live
// sync path already wiring StrictScope: true. The work shipped citing #1639 and a
// generic worktree-sync subject, never #1460 itself; history on origin/main cannot
// be rewritten, so this comment restates the closure binding explicitly for the
// grep-based referee.
type HandoffReviewOptions struct {
	StrictScope   bool
	Live          bool
	DedupeChecked bool
	DedupeCap     int
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
	return ReviewHandoffWithOptions(h, HandoffReviewOptions{})
}

// ReviewHandoffWithOptions grades h with optional strict review of every
// next-step issue candidate. StrictScope is the guard used by the CLI before it
// can create GitHub follow-up issues.
func ReviewHandoffWithOptions(h Handoff, opt HandoffReviewOptions) HandoffReview {
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
		if strings.TrimSpace(step.Generation) != "" && normalizeHandoffGeneration(step.Generation) == "" {
			reasons = append(reasons, prefix+"BAD_GENERATION")
		}
	}

	var issueReviews []issuecontract.Review
	if opt.StrictScope && nextCount > 0 {
		issueReviews = make([]issuecontract.Review, 0, nextCount)
		for i, step := range h.NextSteps {
			prefix := "NEXT_STEP_" + strconv.Itoa(i+1) + "_"
			ir := issuecontract.ReviewCandidate(handoffIssueCandidate(h, step), issuecontract.Options{
				Live:          opt.Live,
				DedupeChecked: opt.DedupeChecked,
				DedupeCap:     opt.DedupeCap,
			})
			issueReviews = append(issueReviews, ir)
			for _, reason := range ir.Reasons {
				reasons = append(reasons, prefix+reason)
			}
		}
	}

	review := HandoffReview{
		Schema:       SchemaHandoffReview,
		OK:           len(reasons) == 0,
		TaskID:       taskID,
		Reasons:      reasons,
		NextStepKeys: keys,
		IssueCount:   nextCount,
		IssueReviews: issueReviews,
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

func handoffIssueCandidate(h Handoff, step HandoffNextStep) issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:          issuecontract.Schema,
		Key:             strings.TrimSpace(step.Key),
		Title:           strings.TrimSpace(step.Title),
		ParentRef:       strings.TrimSpace(h.Task.TaskID),
		CurrentState:    strings.TrimSpace(h.CurrentState),
		WhyNow:          strings.TrimSpace(step.Reason),
		WorkingSpine:    strings.TrimSpace(step.WorkingSpine),
		PriorityContext: strings.TrimSpace(step.PriorityContext),
		WorkUnit:        firstNonEmpty(step.WorkUnit, "leaf"),
		ExpectedSteps:   step.ExpectedSteps,
		Assumptions:     compactStrings(step.Assumptions),
		ConfusionRisks:  compactStrings(step.ConfusionRisks),
		Coordination:    compactStrings(step.Coordination),
		Trigger:         firstNonEmpty(step.Trigger, "Verified task handoff proposed this next step."),
		BatchPolicy:     firstNonEmpty(step.BatchPolicy, "At most two follow-up issues per handoff; update by marker key on rerun."),
		InScope:         firstNonEmpty(step.InScope, step.Body),
		OutOfScope:      strings.TrimSpace(step.OutOfScope),
		DoneCondition:   strings.TrimSpace(step.DoneCondition),
		Witness:         strings.TrimSpace(step.Witness),
		AcceptanceGate:  strings.TrimSpace(step.AcceptanceGate),
		Lane:            strings.TrimSpace(step.Lane),
		Paths:           compactStrings(step.Paths),
		Labels:          compactStrings(append(append([]string{}, step.Labels...), handoffGenerationLabels(step.Generation)...)),
		Priority:        strings.TrimSpace(step.Priority),
		BoundaryNotes:   compactStrings(step.BoundaryNotes),
		ClosureBinding:  strings.TrimSpace(step.ClosureBinding),
	}
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
			Labels:       compactStrings(append(append([]string{}, step.Labels...), handoffGenerationLabels(step.Generation)...)),
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
	if gen := normalizeHandoffGeneration(step.Generation); gen != "" {
		fmt.Fprintf(&b, "- Generation: `%s`\n", gen)
	}
	fmt.Fprintf(&b, "- Why this is next: %s\n", oneLine(step.Reason))
	if refs := evidenceRefStrings(append(append([]EvidenceRef{}, h.CompletionEvidence...), step.EvidenceRefs...)); len(refs) > 0 {
		fmt.Fprintln(&b, "- Evidence:")
		for _, ref := range refs {
			fmt.Fprintf(&b, "  - `%s`\n", ref)
		}
	}
	fmt.Fprintln(&b)
	writeSection(&b, "Parent context", strings.TrimSpace(h.Task.TaskID), "This handoff did not name the parent task.")
	writeSection(&b, "Current state", strings.TrimSpace(h.CurrentState), "Not specified by this handoff.")
	writeSection(&b, "Why this is next", strings.TrimSpace(step.Reason), "Not specified by this handoff.")
	if gen := normalizeHandoffGeneration(step.Generation); gen != "" {
		writeSection(&b, "Generation intent", handoffGenerationIntent(gen), "Not specified by this handoff.")
		writeListSection(&b, "Promotion evidence", step.PromotionEvidence, "None named.")
		writeListSection(&b, "Demotion or retirement evidence", step.DemotionEvidence, "None named.")
		writeListSection(&b, "Invalidating assumptions", step.InvalidatingAssumptions, "None named.")
		writeListSection(&b, "Generation non-goals", step.GenerationNonGoals, "None named.")
	}
	writeSection(&b, "Working spine", strings.TrimSpace(step.WorkingSpine), "Not specified by this handoff.")
	writeSection(&b, "Priority context", strings.TrimSpace(step.PriorityContext), "Not specified by this handoff.")
	writeSection(&b, "Work unit", firstNonEmpty(step.WorkUnit, "leaf"), "leaf")
	if step.ExpectedSteps > 0 {
		writeSection(&b, "Expected steps", strconv.Itoa(step.ExpectedSteps), "Not specified by this handoff.")
	} else {
		writeSection(&b, "Expected steps", "", "Not specified by this handoff.")
	}
	writeListSection(&b, "Assumptions", step.Assumptions, "None named.")
	writeListSection(&b, "Confusion risks", step.ConfusionRisks, "None named.")
	writeListSection(&b, "Coordination notes", step.Coordination, "No special coordination beyond the lane lease.")
	writeSection(&b, "Trigger", firstNonEmpty(step.Trigger, "Verified task handoff proposed this next step."), "Verified task handoff proposed this next step.")
	writeSection(&b, "Batch policy", firstNonEmpty(step.BatchPolicy, "At most two follow-up issues per handoff; update by marker key on rerun."), "At most two follow-up issues per handoff; update by marker key on rerun.")
	writeSection(&b, "In scope", firstNonEmpty(step.InScope, step.Body), "Not specified by this handoff.")
	writeSection(&b, "Out of scope", strings.TrimSpace(step.OutOfScope), "Not specified by this handoff.")
	writeSection(&b, "Done condition", strings.TrimSpace(step.DoneCondition), "Not specified by this handoff.")
	writeSection(&b, "Witness", strings.TrimSpace(step.Witness), "Not specified by this handoff.")
	writeSection(&b, "Acceptance gate", strings.TrimSpace(step.AcceptanceGate), "Not specified by this handoff.")
	writeSection(&b, "Lane", strings.TrimSpace(step.Lane), "Not specified by this handoff.")
	writeListSection(&b, "Path hints", step.Paths, "Not specified by this handoff.")
	writeListSection(&b, "Boundary notes", step.BoundaryNotes, "Public issue only; no private evidence named.")
	writeSection(&b, "Closure binding", strings.TrimSpace(step.ClosureBinding), "Resolving commit cites this issue and carries the matching fak trailer.")
	fmt.Fprintln(&b, "Managed by `fak task handoff`; re-running the same handoff updates this issue in place.")
	return b.String()
}

func writeSection(b *strings.Builder, title, value, fallback string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	value = strings.TrimSpace(value)
	if value == "" {
		value = fallback
	}
	fmt.Fprintln(b, value)
	fmt.Fprintln(b)
}

func writeListSection(b *strings.Builder, title string, values []string, fallback string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	values = compactStrings(values)
	if len(values) == 0 {
		fmt.Fprintln(b, fallback)
		fmt.Fprintln(b)
		return
	}
	for _, value := range values {
		fmt.Fprintf(b, "- `%s`\n", value)
	}
	fmt.Fprintln(b)
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

func normalizeHandoffGeneration(g string) string {
	s := strings.ToLower(strings.TrimSpace(g))
	s = strings.TrimPrefix(s, "gen/")
	switch s {
	case "now", "next", "second-next", "future":
		return "gen/" + s
	default:
		return ""
	}
}

func handoffGenerationLabels(g string) []string {
	gen := normalizeHandoffGeneration(g)
	if gen == "" {
		return nil
	}
	return []string{"generation", gen}
}

func handoffGenerationIntent(gen string) string {
	var intent string
	switch normalizeHandoffGeneration(gen) {
	case "gen/now":
		intent = "now - immediate trunk-safe product/operator work; do not wait for a future architecture bet."
	case "gen/next":
		intent = "next - near-term foundation that should become agent-runnable after gates, handoffs, and operator visibility exist."
	case "gen/second-next":
		intent = "second-next - architectural option that needs simulation, compatibility policy, or cross-generation dependency management."
	case "gen/future":
		intent = "future - research or long-horizon option that should stay visible without pretending it is on the current release train."
	default:
		return ""
	}
	return intent + " Generation is orthogonal to priority, shared trunk, and runtime feature gates."
}

func oneLine(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.Join(strings.Fields(s), " ")
}

func firstNonEmpty(vals ...string) string {
	for _, val := range vals {
		if s := strings.TrimSpace(val); s != "" {
			return s
		}
	}
	return ""
}
