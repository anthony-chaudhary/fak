package dogfoodissues

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

// Schema is the stable schema tag stamped on the machine-readable result.
const Schema = "fak.dogfood-action-issues.v1"

var markerRE = regexp.MustCompile(`<!--\s*fak-dogfood-action-key:\s*([^>\s]+)\s*-->`)

// ActionItem is one scorecard ACTION item extracted from a dogfood report.json.
type ActionItem struct {
	Key            string
	Title          string
	SourceProbe    string
	ScoreName      string
	Score          string
	Grade          string
	DebtName       string
	DebtCount      int
	EvidencePath   string
	NextAction     string
	Finding        string
	ParentRef      string
	CurrentState   string
	WhyNow         string
	WorkingSpine   string
	InScope        string
	OutOfScope     string
	DoneCondition  string
	Witness        string
	AcceptanceGate string
	Lane           string
	Paths          []string
	Labels         []string
	BoundaryNotes  []string
	ClosureBinding string
}

// PlanRow is one create/update decision for a single ActionItem.
type PlanRow struct {
	Action       string               `json:"action"`
	Key          string               `json:"key"`
	Number       *int                 `json:"number"`
	State        string               `json:"state"`
	Title        string               `json:"title"`
	Body         string               `json:"-"`
	Score        string               `json:"score"`
	Grade        string               `json:"grade"`
	DebtCount    int                  `json:"debt_count"`
	EvidencePath string               `json:"evidence_path"`
	NextAction   string               `json:"next_action"`
	Lane         string               `json:"lane,omitempty"`
	Paths        []string             `json:"paths,omitempty"`
	Labels       []string             `json:"labels,omitempty"`
	Review       issuecontract.Review `json:"review,omitempty"`
}

// SkippedRow records a scorecard ACTION item that remains visible in the
// dogfood report, but is not scoped enough to create/update a dispatchable
// public GitHub issue.
type SkippedRow struct {
	Key             string               `json:"key"`
	Title           string               `json:"title"`
	Reason          string               `json:"reason"`
	Dispatchability string               `json:"dispatchability"`
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
	Report  string       `json:"report"`
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

// BuildOptions carries the producer context needed to turn scorecard ACTION
// rows into dispatchable issues. The legacy BuildPlan path is left unreviewed
// for tests and older callers; effectful callers should use BuildPlanWithOptions.
type BuildOptions struct {
	Live          bool
	DedupeChecked bool
	DedupeCap     int
}

func toInt(v any, def int) int {
	switch n := v.(type) {
	case bool:
		return def
	case float64: // JSON numbers decode to float64
		return int(n)
	case int:
		return n
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return def
		}
		return int(i)
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(n))
		if err != nil {
			return def
		}
		return i
	default:
		return def
	}
}

func toStr(v any, def string) string {
	if v == nil {
		return def
	}
	switch s := v.(type) {
	case string:
		return s
	case float64:
		// Mirror Python str(float): drop a trailing ".0" so 71.5 stays "71.5"
		// and 12.0 becomes "12".
		return strconv.FormatFloat(s, 'f', -1, 64)
	case bool:
		if s {
			return "True"
		}
		return "False"
	default:
		return fmt.Sprintf("%v", s)
	}
}

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return nil
}

// LoadReport parses a dogfood report.json file into a generic object fold.
func LoadReport(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var data any
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}
	m, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("report must be a JSON object")
	}
	return m, nil
}

// ExtractActionItems folds the report's probes into the scorecard ACTION items
// that warrant a tracked backlog issue. reportPath is recorded as the evidence
// path on each item.
func ExtractActionItems(report map[string]any, reportPath string) []ActionItem {
	evidence := reportPath
	var items []ActionItem
	probes, _ := report["probes"].([]any)
	for _, p := range probes {
		probe := asMap(p)
		if probe == nil {
			continue
		}
		key := toStr(probe["key"], "")
		payload := asMap(probe["payload"])
		if payload == nil {
			continue
		}
		schema := toStr(payload["schema"], "")

		if key == "code-slop-scorecard" || schema == "fleet-code-slop-scorecard/1" {
			corpus := asMap(payload["corpus"])
			if corpus == nil {
				corpus = map[string]any{}
			}
			debt := toInt(corpus["slop_debt"], toInt(payload["slop_debt"], 0))
			verdict := toStr(payload["verdict"], "")
			if verdict != "ACTION" && debt <= 0 {
				continue
			}
			finding := toStr(payload["finding"], "code_slop")
			items = append(items, ActionItem{
				Key:          "recent-feature-dogfood/code-slop-scorecard/" + finding,
				Title:        "dogfood ACTION: code-slop scorecard debt",
				SourceProbe:  key,
				ScoreName:    "slop_score",
				Score:        toStr(corpus["score"], toStr(payload["score"], "?")),
				Grade:        toStr(corpus["grade"], toStr(payload["grade"], "?")),
				DebtName:     "slop_debt",
				DebtCount:    debt,
				EvidencePath: evidence,
				NextAction: toStr(payload["next_action"],
					"Retire code-slop debt worst-first, then rerun the dogfood packet."),
				Finding: finding,
			})
			continue
		}

		if key == "dogfood-coverage-scorecard" || schema == "dogfood-coverage/1" {
			hardDebt := toInt(payload["dogfood_debt"], 0)
			worstFirst, _ := payload["worst_first"].([]any)
			softDebt := 0
			for _, x := range worstFirst {
				if strings.TrimSpace(toStr(x, "")) != "" {
					softDebt++
				}
			}
			grade := toStr(payload["grade"], "")
			if hardDebt <= 0 && softDebt <= 0 && (grade == "" || grade == "A") {
				continue
			}
			debt := softDebt
			if hardDebt > 0 {
				debt = hardDebt
			}
			action := "Raise dogfood coverage to grade A, then rerun the dogfood packet."
			if len(worstFirst) > 0 {
				n := len(worstFirst)
				if n > 5 {
					n = 5
				}
				parts := make([]string, 0, n)
				for _, x := range worstFirst[:n] {
					parts = append(parts, toStr(x, ""))
				}
				action = "Address dogfood coverage gaps worst-first: " + strings.Join(parts, ", ")
			}
			gradeOut := grade
			if gradeOut == "" {
				gradeOut = "?"
			}
			items = append(items, ActionItem{
				Key:          "recent-feature-dogfood/dogfood-coverage-scorecard/dogfood_coverage",
				Title:        "dogfood ACTION: dogfood coverage gap",
				SourceProbe:  key,
				ScoreName:    "coverage",
				Score:        toStr(payload["coverage"], "?"),
				Grade:        gradeOut,
				DebtName:     "dogfood_debt_or_gaps",
				DebtCount:    debt,
				EvidencePath: evidence,
				NextAction:   action,
				Finding:      "dogfood_coverage",
			})
		}
	}
	return items
}

// IssueBody renders the stable, marker-stamped issue body for an item.
func IssueBody(item ActionItem) string {
	c := actionCandidate(item)
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- fak-dogfood-action-key: %s -->\n", item.Key)
	fmt.Fprintln(&b, "# Dogfood scorecard ACTION")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- Stable key: `%s`\n", item.Key)
	fmt.Fprintf(&b, "- Source probe: `%s`\n", item.SourceProbe)
	fmt.Fprintf(&b, "- Finding: `%s`\n", item.Finding)
	fmt.Fprintf(&b, "- %s: `%s`\n", item.ScoreName, item.Score)
	fmt.Fprintf(&b, "- grade: `%s`\n", item.Grade)
	fmt.Fprintf(&b, "- %s: `%d`\n", item.DebtName, item.DebtCount)
	fmt.Fprintf(&b, "- Evidence path: `%s`\n", item.EvidencePath)
	if item.Lane != "" {
		fmt.Fprintf(&b, "- Lane: `%s`\n", item.Lane)
	}
	if len(item.Paths) > 0 {
		fmt.Fprintln(&b, "- Path hints:")
		for _, p := range item.Paths {
			fmt.Fprintf(&b, "  - `%s`\n", p)
		}
	}
	fmt.Fprintln(&b)
	issueSection(&b, "Current state", c.CurrentState)
	issueSection(&b, "Why this is next", c.WhyNow)
	issueSection(&b, "Working spine", item.WorkingSpine)
	issueSection(&b, "Priority context", c.PriorityContext)
	issueSection(&b, "In scope", item.InScope)
	issueSection(&b, "Out of scope", item.OutOfScope)
	issueSection(&b, "Done condition", item.DoneCondition)
	issueSection(&b, "Witness", item.Witness)
	issueSection(&b, "Acceptance gate", item.AcceptanceGate)
	issueListSection(&b, "Boundary notes", item.BoundaryNotes)
	issueSection(&b, "Closure binding", c.ClosureBinding)
	fmt.Fprintln(&b, "Suggested next action:")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, firstNonEmpty(item.NextAction, "Triage this scorecard ACTION row into a scoped, witness-backed leaf before dispatch."))
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "This issue is managed by `fak dogfood-issues`. Re-running the helper updates this issue in place instead of opening duplicates.")
	return b.String()
}

// ReviewActionItem grades one ACTION item against the shared machine-created
// issue contract.
func ReviewActionItem(item ActionItem, opt BuildOptions) issuecontract.Review {
	return issuecontract.ReviewCandidate(actionCandidate(item), issuecontract.Options{
		Live:          opt.Live,
		DedupeChecked: opt.DedupeChecked,
		DedupeCap:     opt.DedupeCap,
	})
}

func actionCandidate(item ActionItem) issuecontract.Candidate {
	scoreState := fmt.Sprintf("Source probe `%s` reported finding `%s`, grade `%s`, and %s `%d`.",
		item.SourceProbe, item.Finding, item.Grade, item.DebtName, item.DebtCount)
	return issuecontract.Candidate{
		Schema:          issuecontract.Schema,
		Key:             item.Key,
		Title:           item.Title,
		ParentRef:       firstNonEmpty(item.ParentRef, "fak dogfood-issues"),
		CurrentState:    firstNonEmpty(item.CurrentState, scoreState),
		WhyNow:          firstNonEmpty(item.WhyNow, "The recent-feature dogfood report emitted an ACTION/debt row for this scorecard."),
		WorkingSpine:    item.WorkingSpine,
		PriorityContext: dogfoodPriorityContext(item),
		InScope:         item.InScope,
		OutOfScope:      item.OutOfScope,
		DoneCondition:   item.DoneCondition,
		Witness:         item.Witness,
		AcceptanceGate:  item.AcceptanceGate,
		Lane:            item.Lane,
		Paths:           append([]string(nil), item.Paths...),
		Labels:          append([]string(nil), item.Labels...),
		BoundaryNotes:   append([]string(nil), item.BoundaryNotes...),
		ClosureBinding:  firstNonEmpty(item.ClosureBinding, "Resolving commit must cite `#N` and carry a matching `(fak <leaf>)` trailer."),
	}
}

func dogfoodPriorityContext(item ActionItem) string {
	spine := firstNonEmpty(item.WorkingSpine, "Retire the scorecard ACTION row on the smallest witnessed path.")
	current := fmt.Sprintf("Scorecard `%s` reports `%s` with %s `%d`.", item.SourceProbe, item.Finding, item.DebtName, item.DebtCount)
	return strings.Join([]string{
		"Working path: " + spine,
		"Current blocker: " + current,
		"Unblocks: resolving this ACTION row keeps recent-feature dogfood from hiding a real breakage.",
		"Not polish: address the named probe row before broad dogfood expansion or optimization.",
	}, "\n")
}

func issueSection(b *strings.Builder, title, body string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	body = strings.TrimSpace(body)
	if body == "" {
		body = "Not specified."
	}
	fmt.Fprintln(b, body)
	fmt.Fprintln(b)
}

func issueListSection(b *strings.Builder, title string, items []string) {
	fmt.Fprintf(b, "## %s\n\n", title)
	if len(items) == 0 {
		fmt.Fprintln(b, "No boundary notes supplied.")
		fmt.Fprintln(b)
		return
	}
	for _, item := range items {
		if s := strings.TrimSpace(item); s != "" {
			fmt.Fprintf(b, "- %s\n", s)
		}
	}
	fmt.Fprintln(b)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if s := strings.TrimSpace(value); s != "" {
			return s
		}
	}
	return ""
}

// MarkerKey extracts the stable key from an issue body's HTML-comment marker,
// or "" when absent.
func MarkerKey(body string) string {
	m := markerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// existingByKey indexes existing issues by their marker key.
func existingByKey(issues []Issue) map[string]Issue {
	out := map[string]Issue{}
	for _, issue := range issues {
		key := MarkerKey(issue.Body)
		if key != "" {
			out[key] = issue
		}
	}
	return out
}

// BuildPlan decides create vs update for each item against the existing issues
// (matched by marker key).
func BuildPlan(items []ActionItem, existing []Issue) []PlanRow {
	byKey := existingByKey(existing)
	plan := make([]PlanRow, 0, len(items))
	for _, item := range items {
		found, ok := byKey[item.Key]
		row := planRow(item)
		if ok {
			row.Action = "update"
			n := found.Number
			row.Number = &n
			row.State = found.State
		}
		plan = append(plan, row)
	}
	return plan
}

// BuildPlanWithOptions is BuildPlan plus the shared issue-candidate contract.
// Non-OK candidates are returned as skipped rows instead of being synced as vague
// public issues.
func BuildPlanWithOptions(items []ActionItem, existing []Issue, opt BuildOptions) ([]PlanRow, []SkippedRow) {
	byKey := existingByKey(existing)
	plan := make([]PlanRow, 0, len(items))
	skipped := []SkippedRow{}
	for _, item := range items {
		review := ReviewActionItem(item, opt)
		if !review.OK {
			skipped = append(skipped, SkippedRow{
				Key:             item.Key,
				Title:           item.Title,
				Reason:          strings.Join(review.Reasons, ","),
				Dispatchability: review.Dispatchability,
				Review:          review,
			})
			continue
		}
		row := planRow(item)
		row.Review = review
		if found, ok := byKey[item.Key]; ok {
			row.Action = "update"
			n := found.Number
			row.Number = &n
			row.State = found.State
		}
		plan = append(plan, row)
	}
	return plan, skipped
}

func planRow(item ActionItem) PlanRow {
	return PlanRow{
		Action:       "create",
		Key:          item.Key,
		Title:        item.Title,
		Body:         IssueBody(item),
		Score:        item.Score,
		Grade:        item.Grade,
		DebtCount:    item.DebtCount,
		EvidencePath: item.EvidencePath,
		NextAction:   item.NextAction,
		Lane:         item.Lane,
		Paths:        append([]string(nil), item.Paths...),
		Labels:       append([]string(nil), item.Labels...),
	}
}

// Runner runs a `gh` subprocess and returns its stdout, stderr, and an ok flag
// (true when the process exited 0). It is injectable so Sync is testable without
// a real gh.
type Runner func(args []string) (stdout, stderr string, ok bool)

// defaultRunner shells out to the real `gh` CLI.
func defaultRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
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

// Sync creates or edits each planned issue via gh. The body is passed inline with
// --body so no temp file is needed. runner defaults to the real gh CLI when nil.
func Sync(plan []PlanRow, repo string, labels []string, runner Runner) []SyncRow {
	run := runner
	if run == nil {
		run = defaultRunner
	}
	results := make([]SyncRow, 0, len(plan))
	for _, row := range plan {
		var args []string
		if row.Action == "update" {
			num := ""
			if row.Number != nil {
				num = strconv.Itoa(*row.Number)
			}
			args = []string{"issue", "edit", num, "--title", row.Title, "--body", row.Body}
		} else {
			args = []string{"issue", "create", "--title", row.Title, "--body", row.Body}
			for _, label := range mergeDogfoodIssueLabels(row.Labels, labels) {
				args = append(args, "--label", label)
			}
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := run(args)
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

func mergeDogfoodIssueLabels(base, extra []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, group := range [][]string{base, extra} {
		for _, label := range group {
			label = strings.TrimSpace(label)
			if label == "" || seen[label] {
				continue
			}
			seen[label] = true
			out = append(out, label)
		}
	}
	return out
}

// Render produces the human-readable summary of a plan/result.
func Render(r Result) string {
	lines := []string{
		fmt.Sprintf("dogfood-action-issues: %s  %d item(s)", r.Mode, len(r.Planned)),
		fmt.Sprintf("  report: %s", r.Report),
	}
	if len(r.Skipped) > 0 {
		lines = append(lines, fmt.Sprintf("  skipped-contract: %d item(s)", len(r.Skipped)))
		for _, row := range r.Skipped {
			lines = append(lines, fmt.Sprintf("    key=%s: %s", row.Key, row.Reason))
		}
	}
	if len(r.Planned) == 0 {
		lines = append(lines, "  no dispatchable scorecard ACTION items found")
		return strings.Join(lines, "\n")
	}
	for _, row := range r.Planned {
		target := "new issue"
		if row.Number != nil {
			target = "#" + strconv.Itoa(*row.Number)
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s: %s (grade=%s debt=%d)",
			row.Action, target, row.Title, row.Grade, row.DebtCount))
	}
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create/update issues with gh")
	}
	return strings.Join(lines, "\n")
}
