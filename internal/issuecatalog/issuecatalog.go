package issuecatalog

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Schema is the stable schema tag stamped on the machine-readable result.
const Schema = "fak.issue-catalog.v1"

// MarkerName is the HTML-comment marker key name stamped into each issue body so
// a later run finds the issue it already opened instead of duplicating it.
const MarkerName = "fak-issue-catalog-key"

var markerRE = regexp.MustCompile(`<!--\s*` + MarkerName + `:\s*([^>\s]+)\s*-->`)

// Row is one catalog entry: an issuecontract.Candidate's spine-first fields plus
// the producer metadata the contract does not model (milestone, lens, priority).
// The JSON tags match the recon-agent gap-row schema exactly, and unknown fields
// (e.g. expected_steps) decode away harmlessly.
type Row struct {
	Key            string   `json:"key"`
	Title          string   `json:"title"`
	ParentRef      string   `json:"parent_ref"`
	CurrentState   string   `json:"current_state"`
	WhyNow         string   `json:"why_now"`
	WorkingSpine   string   `json:"working_spine"`
	WorkUnit       string   `json:"work_unit"`
	ExpectedSteps  int      `json:"expected_steps"`
	Assumptions    []string `json:"assumptions"`
	ConfusionRisks []string `json:"confusion_risks"`
	Coordination   []string `json:"coordination"`
	Trigger        string   `json:"trigger"`
	BatchPolicy    string   `json:"batch_policy"`
	InScope        string   `json:"in_scope"`
	OutOfScope     string   `json:"out_of_scope"`
	DoneCondition  string   `json:"done_condition"`
	Witness        string   `json:"witness"`
	AcceptanceGate string   `json:"acceptance_gate"`
	ClosureBinding string   `json:"closure_binding"`
	Lane           string   `json:"lane"`
	Paths          []string `json:"paths"`
	Labels         []string `json:"labels"`
	Milestone      string   `json:"milestone"`
	Lens           string   `json:"lens"`
	Priority       string   `json:"priority"`
}

// PlanRow is one create/update decision for a single Row.
type PlanRow struct {
	Action    string               `json:"action"`
	Key       string               `json:"key"`
	Number    *int                 `json:"number"`
	State     string               `json:"state"`
	Title     string               `json:"title"`
	Body      string               `json:"-"`
	Milestone string               `json:"milestone,omitempty"`
	Labels    []string             `json:"labels,omitempty"`
	Lane      string               `json:"lane,omitempty"`
	Paths     []string             `json:"paths,omitempty"`
	Review    issuecontract.Review `json:"review,omitempty"`
}

// SkippedRow records a catalog entry that does not meet the issue contract and so
// is not synced as a dispatchable public issue.
type SkippedRow struct {
	Key             string               `json:"key"`
	Title           string               `json:"title"`
	Reason          string               `json:"reason"`
	Dispatchability string               `json:"dispatchability"`
	MissingFields   []string             `json:"missing_fields,omitempty"`
	Review          issuecontract.Review `json:"review,omitempty"`
}

// SyncRow is one gh create/edit outcome on a --live run.
type SyncRow struct {
	Key    string `json:"key"`
	Action string `json:"action"`
	OK     bool   `json:"ok"`
	Stdout string `json:"stdout"`
	Stderr string `json:"stderr"`
}

// Result is the machine-readable plan/result fold.
type Result struct {
	Schema  string       `json:"schema"`
	Mode    string       `json:"mode"`
	Catalog string       `json:"catalog"`
	Total   int          `json:"total"`
	Planned []PlanRow    `json:"planned"`
	Synced  []SyncRow    `json:"synced"`
	Skipped []SkippedRow `json:"skipped,omitempty"`
}

// Issue is the subset of a `gh issue list --json ...` row this tool reads.
type Issue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url"`
}

// Options carries the producer context the issue body alone cannot prove.
type Options struct {
	Live          bool
	DedupeChecked bool
	DedupeCap     int
}

// LoadCatalog parses a catalog JSON file (a JSON array of Row) from disk.
func LoadCatalog(path string) ([]Row, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParseCatalog(b)
}

// ParseCatalog parses catalog bytes (a JSON array of Row).
func ParseCatalog(b []byte) ([]Row, error) {
	var rows []Row
	if err := json.Unmarshal(b, &rows); err != nil {
		return nil, fmt.Errorf("catalog must be a JSON array of rows: %w", err)
	}
	return rows, nil
}

// Candidate projects a Row onto the shared issue-candidate contract.
func (r Row) Candidate() issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:         issuecontract.Schema,
		Key:            r.Key,
		Title:          r.Title,
		ParentRef:      r.ParentRef,
		CurrentState:   r.CurrentState,
		WhyNow:         r.WhyNow,
		WorkingSpine:   r.WorkingSpine,
		WorkUnit:       r.WorkUnit,
		ExpectedSteps:  r.ExpectedSteps,
		Assumptions:    append([]string(nil), r.Assumptions...),
		ConfusionRisks: append([]string(nil), r.ConfusionRisks...),
		Coordination:   append([]string(nil), r.Coordination...),
		Trigger:        r.Trigger,
		BatchPolicy:    r.BatchPolicy,
		InScope:        r.InScope,
		OutOfScope:     r.OutOfScope,
		DoneCondition:  r.DoneCondition,
		Witness:        r.Witness,
		AcceptanceGate: r.AcceptanceGate,
		Lane:           r.Lane,
		Paths:          append([]string(nil), r.Paths...),
		Labels:         append([]string(nil), r.Labels...),
		Priority:       r.Priority,
		ClosureBinding: r.ClosureBinding,
	}
}

// Review grades one Row against the shared machine-created issue contract.
func Review(r Row, opt Options) issuecontract.Review {
	return issuecontract.ReviewCandidate(r.Candidate(), issuecontract.Options{
		Live:          opt.Live,
		DedupeChecked: opt.DedupeChecked,
		DedupeCap:     opt.DedupeCap,
	})
}

// IssueBody renders the stable, marker-stamped issue body for a row. The section
// headings match the issuecontract draft grammar exactly, so a later
// ReviewIssueDraft re-audit of the open issue reads back as dispatchable.
func IssueBody(r Row) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- %s: %s -->\n", MarkerName, r.Key)
	fmt.Fprintf(&b, "# %s\n\n", strings.TrimSpace(r.Title))
	if r.Lens != "" {
		fmt.Fprintf(&b, "- Lens: `%s`\n", r.Lens)
	}
	if r.Priority != "" {
		fmt.Fprintf(&b, "- Priority: `%s`\n", r.Priority)
	}
	if r.Milestone != "" {
		fmt.Fprintf(&b, "- Milestone: `%s`\n", r.Milestone)
	}
	if len(r.Labels) > 0 {
		fmt.Fprintf(&b, "- Labels: `%s`\n", strings.Join(r.Labels, "`, `"))
	}
	b.WriteString("\n")
	section(&b, "Parent context", r.ParentRef)
	section(&b, "Current state", r.CurrentState)
	section(&b, "Why this is next", r.WhyNow)
	section(&b, "Working spine", r.WorkingSpine)
	section(&b, "Work unit", firstNonEmpty(r.WorkUnit, "leaf"))
	if r.ExpectedSteps > 0 {
		section(&b, "Expected steps", strconv.Itoa(r.ExpectedSteps))
	} else {
		section(&b, "Expected steps", "")
	}
	listSection(&b, "Assumptions", r.Assumptions, "None named.")
	listSection(&b, "Confusion risks", r.ConfusionRisks, "None named.")
	listSection(&b, "Coordination notes", r.Coordination, "No special coordination beyond the lane lease.")
	section(&b, "Trigger", r.Trigger)
	section(&b, "Batch policy", r.BatchPolicy)
	section(&b, "In scope", r.InScope)
	section(&b, "Out of scope", r.OutOfScope)
	section(&b, "Done condition", r.DoneCondition)
	section(&b, "Witness", r.Witness)
	section(&b, "Acceptance gate", r.AcceptanceGate)
	if r.Lane != "" {
		section(&b, "Lane", "`"+r.Lane+"`")
	}
	pathSection(&b, r.Paths)
	section(&b, "Closure binding", r.ClosureBinding)
	fmt.Fprintln(&b, "---")
	fmt.Fprintln(&b, "Managed by `fak issue-catalog` (the performance-enablement catalog). Re-running the")
	fmt.Fprintln(&b, "helper updates this issue in place by marker key instead of opening duplicates.")
	return b.String()
}

func section(b *strings.Builder, title, body string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	body = strings.TrimSpace(body)
	if body == "" {
		body = "Not specified."
	}
	fmt.Fprintln(b, body)
	fmt.Fprintln(b)
}

func listSection(b *strings.Builder, title string, values []string, fallback string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	wrote := false
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			fmt.Fprintf(b, "- %s\n", value)
			wrote = true
		}
	}
	if !wrote {
		fmt.Fprintln(b, fallback)
	}
	fmt.Fprintln(b)
}

func firstNonEmpty(vals ...string) string {
	for _, val := range vals {
		if s := strings.TrimSpace(val); s != "" {
			return s
		}
	}
	return ""
}

func pathSection(b *strings.Builder, paths []string) {
	fmt.Fprintln(b, "## Path hints")
	fmt.Fprintln(b)
	if len(paths) == 0 {
		fmt.Fprintln(b, "Not specified.")
		fmt.Fprintln(b)
		return
	}
	for _, p := range paths {
		if p = strings.TrimSpace(p); p != "" {
			fmt.Fprintf(b, "- `%s`\n", p)
		}
	}
	fmt.Fprintln(b)
}

// MarkerKey extracts the stable key from an issue body's marker, or "" when absent.
func MarkerKey(body string) string {
	m := markerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func existingByKey(issues []Issue) map[string]Issue {
	out := map[string]Issue{}
	for _, issue := range issues {
		if key := MarkerKey(issue.Body); key != "" {
			out[key] = issue
		}
	}
	return out
}

// BuildPlan reviews each row against the issue contract, then diffs the dispatchable
// rows against the existing issues (matched by marker key) to decide create vs
// update. Non-dispatchable rows are returned as skipped triage rows.
func BuildPlan(rows []Row, existing []Issue, opt Options) ([]PlanRow, []SkippedRow) {
	byKey := existingByKey(existing)
	seen := map[string]bool{}
	plan := make([]PlanRow, 0, len(rows))
	skipped := []SkippedRow{}
	for _, row := range rows {
		if row.Key != "" && seen[row.Key] {
			skipped = append(skipped, SkippedRow{
				Key:    row.Key,
				Title:  row.Title,
				Reason: "DUPLICATE_KEY_IN_CATALOG",
			})
			continue
		}
		seen[row.Key] = true
		review := Review(row, opt)
		if !review.OK {
			skipped = append(skipped, SkippedRow{
				Key:             row.Key,
				Title:           row.Title,
				Reason:          strings.Join(review.Reasons, ","),
				Dispatchability: review.Dispatchability,
				MissingFields:   review.MissingFields,
				Review:          review,
			})
			continue
		}
		pr := PlanRow{
			Action:    "create",
			Key:       row.Key,
			Title:     row.Title,
			Body:      IssueBody(row),
			Milestone: row.Milestone,
			Labels:    append([]string(nil), row.Labels...),
			Lane:      row.Lane,
			Paths:     append([]string(nil), row.Paths...),
			Review:    review,
		}
		if found, ok := byKey[row.Key]; ok {
			pr.Action = "update"
			n := found.Number
			pr.Number = &n
			pr.State = found.State
		}
		plan = append(plan, pr)
	}
	return plan, skipped
}

// Runner runs a `gh` subprocess and returns stdout, stderr, and an ok flag. It is
// injectable so Sync is testable without a real gh.
type Runner func(args []string) (stdout, stderr string, ok bool)

func defaultRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

// FetchExistingIssues queries `gh issue list` for the existing issues to classify
// create vs update. repo "" uses the current repo.
func FetchExistingIssues(repo string, limit int) ([]Issue, error) {
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit),
		"--json", "number,title,body,state,url"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := defaultRunner(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []Issue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, err
	}
	return issues, nil
}

// Sync creates or edits each planned issue via gh. The body is written to a temp
// file and passed with --body-file so multi-KB bodies and special characters never
// hit an argv length or quoting limit. Milestone + labels are applied on both
// create and edit (a token without triage scope silently drops them — by design,
// the body still carries the same metadata). runner defaults to the real gh CLI.
func Sync(plan []PlanRow, repo string, runner Runner) []SyncRow {
	run := runner
	if run == nil {
		run = defaultRunner
	}
	results := make([]SyncRow, 0, len(plan))
	for _, row := range plan {
		bodyFile, cleanup, err := writeBodyFile(row.Body)
		if err != nil {
			results = append(results, SyncRow{Key: row.Key, Action: row.Action, OK: false,
				Stderr: "write body file: " + err.Error()})
			continue
		}
		var args []string
		if row.Action == "update" && row.Number != nil {
			args = []string{"issue", "edit", strconv.Itoa(*row.Number),
				"--title", row.Title, "--body-file", bodyFile}
			for _, label := range row.Labels {
				args = append(args, "--add-label", label)
			}
			if row.Milestone != "" {
				args = append(args, "--milestone", row.Milestone)
			}
		} else {
			args = []string{"issue", "create", "--title", row.Title, "--body-file", bodyFile}
			for _, label := range row.Labels {
				args = append(args, "--label", label)
			}
			if row.Milestone != "" {
				args = append(args, "--milestone", row.Milestone)
			}
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := run(args)
		cleanup()
		results = append(results, SyncRow{
			Key:    row.Key,
			Action: row.Action,
			OK:     ok,
			Stdout: strings.TrimSpace(stdout),
			Stderr: strings.TrimSpace(stderr),
		})
	}
	return results
}

func writeBodyFile(body string) (string, func(), error) {
	f, err := os.CreateTemp("", "fak-issue-catalog-*.md")
	if err != nil {
		return "", func() {}, err
	}
	name := f.Name()
	cleanup := func() { _ = os.Remove(name) }
	if _, err := f.WriteString(body); err != nil {
		_ = f.Close()
		cleanup()
		return "", func() {}, err
	}
	if err := f.Close(); err != nil {
		cleanup()
		return "", func() {}, err
	}
	return name, cleanup, nil
}

// Render produces the human-readable summary of a plan/result.
func Render(r Result) string {
	creates, updates := 0, 0
	for _, row := range r.Planned {
		if row.Action == "update" {
			updates++
		} else {
			creates++
		}
	}
	lines := []string{
		fmt.Sprintf("issue-catalog: %s  %d row(s) -> %d create, %d update, %d skipped",
			r.Mode, r.Total, creates, updates, len(r.Skipped)),
		fmt.Sprintf("  catalog: %s", r.Catalog),
	}
	byMilestone := map[string]int{}
	for _, row := range r.Planned {
		m := row.Milestone
		if m == "" {
			m = "(no milestone)"
		}
		byMilestone[m]++
	}
	keys := make([]string, 0, len(byMilestone))
	for k := range byMilestone {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		lines = append(lines, fmt.Sprintf("  %3d  %s", byMilestone[k], k))
	}
	if len(r.Skipped) > 0 {
		lines = append(lines, fmt.Sprintf("  skipped (contract): %d", len(r.Skipped)))
		for i, row := range r.Skipped {
			if i >= 10 {
				lines = append(lines, fmt.Sprintf("    ... and %d more", len(r.Skipped)-10))
				break
			}
			lines = append(lines, fmt.Sprintf("    %s: %s", row.Key, row.Reason))
		}
	}
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create/update issues with gh")
	}
	if len(r.Synced) > 0 {
		okN := 0
		for _, s := range r.Synced {
			if s.OK {
				okN++
			}
		}
		lines = append(lines, fmt.Sprintf("  synced: %d/%d ok", okN, len(r.Synced)))
	}
	return strings.Join(lines, "\n")
}
