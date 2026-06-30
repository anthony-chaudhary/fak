package maturity

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// IssueSchema is the machine-readable envelope for the maturity backlog ->
// GitHub issue bridge. It turns the ranked `fak maturity next` backlog into the
// work surface the issue-dispatch loop already consumes.
const IssueSchema = "fak-maturity-issues/1"

var maturityIssueMarkerRE = regexp.MustCompile(`<!--\s*fak-maturity-work-key:\s*([^>\s]+)\s*-->`)

// IssueItem is one maturity next-work item rendered as a dedupable GitHub issue.
type IssueItem struct {
	Key      string   `json:"key"`
	Lane     string   `json:"lane"`
	FromRung string   `json:"from_rung"`
	Gap      string   `json:"gap"`
	Title    string   `json:"title"`
	Witness  string   `json:"witness"`
	Skip     bool     `json:"skip"`
	Labels   []string `json:"labels,omitempty"`
	Body     string   `json:"-"`
}

// IssueSkippedRow records a maturity item that remains visible in `fak maturity
// next`, but is not safe to auto-route into public GitHub issues.
type IssueSkippedRow struct {
	Key    string `json:"key"`
	Lane   string `json:"lane"`
	Reason string `json:"reason"`
	Title  string `json:"title"`
}

// IssueProjection is the public-issue projection of the maturity backlog.
type IssueProjection struct {
	Items   []IssueItem       `json:"items"`
	Skipped []IssueSkippedRow `json:"skipped,omitempty"`
}

// ExistingIssue is the subset of a gh issue row this bridge needs.
type ExistingIssue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	State  string `json:"state"`
	URL    string `json:"url"`
}

// IssuePlanRow is one create/update decision for a maturity issue.
type IssuePlanRow struct {
	Action  string `json:"action"`
	Key     string `json:"key"`
	Number  *int   `json:"number,omitempty"`
	State   string `json:"state,omitempty"`
	Lane    string `json:"lane"`
	Title   string `json:"title"`
	Body    string `json:"-"`
	Gap     string `json:"gap"`
	Witness string `json:"witness"`
	Skip    bool   `json:"skip"`
}

// IssueSyncRow is one gh create/edit outcome from a --live run.
type IssueSyncRow struct {
	Key    string `json:"key"`
	Action string `json:"action"`
	OK     bool   `json:"ok"`
	Stdout string `json:"stdout"`
	Stderr string `json:"stderr"`
}

// IssueResult is the CLI result for `fak maturity route`.
type IssueResult struct {
	Schema    string            `json:"schema"`
	Mode      string            `json:"mode"`
	Workspace string            `json:"workspace"`
	Maturity  map[string]any    `json:"maturity"`
	Planned   []IssuePlanRow    `json:"planned"`
	Synced    []IssueSyncRow    `json:"synced"`
	Skipped   []IssueSkippedRow `json:"skipped,omitempty"`
}

// IssueItems projects the ranked maturity backlog into at most limit issue-shaped
// work items. limit <= 0 means all current backlog items.
func IssueItems(p ScorecardPayload, limit int, labels []string) []IssueItem {
	return ProjectIssueItems(p, limit, labels).Items
}

// ProjectIssueItems projects the ranked maturity backlog into public GitHub issue
// work while preserving private-boundary skips for operator visibility. The limit
// counts routed public items, not skipped private-boundary rows.
func ProjectIssueItems(p ScorecardPayload, limit int, labels []string) IssueProjection {
	out := IssueProjection{Items: make([]IssueItem, 0, len(p.Backlog))}
	for _, w := range p.Backlog {
		if !PublicIssueRouteableLane(w.Lane) {
			out.Skipped = append(out.Skipped, IssueSkippedRow{
				Key:    issueKey(w),
				Lane:   w.Lane,
				Reason: "private-boundary lane; keep live GPU-server control work in fak-private, not public GitHub issues",
				Title:  issueTitle(w),
			})
			continue
		}
		item := IssueItem{
			Key:      issueKey(w),
			Lane:     w.Lane,
			FromRung: w.FromRung.String(),
			Gap:      gapName(w.Gap),
			Title:    issueTitle(w),
			Witness:  w.Witness,
			Skip:     w.Skip,
			Labels:   append([]string(nil), labels...),
		}
		item.Body = IssueBody(item)
		out.Items = append(out.Items, item)
		if limit > 0 && len(out.Items) >= limit {
			break
		}
	}
	return out
}

// PublicIssueRouteableLane reports whether a maturity lane can be turned into a
// public GitHub issue by default. The scorecard can still measure private-boundary
// lanes; only the public issue feeder filters them.
func PublicIssueRouteableLane(lane string) bool {
	s := strings.ToLower(strings.TrimSpace(lane))
	if s == "" {
		return false
	}
	for _, privateNeedle := range []string{"dgx", "slackgc"} {
		if strings.Contains(s, privateNeedle) {
			return false
		}
	}
	return true
}

func issueKey(w NextWork) string {
	return "maturity/" + cleanKeyPart(w.Lane) + "/" + cleanKeyPart(gapName(w.Gap))
}

func cleanKeyPart(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return "unknown"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

func gapName(g Rung) string {
	if g == rungBenchmark {
		return "benchmark"
	}
	return g.String()
}

func issueTitle(w NextWork) string {
	switch w.Gap {
	case RungPrototyped:
		return fmt.Sprintf("maturity(%s): prototype the declared capability", w.Lane)
	case RungTested:
		return fmt.Sprintf("maturity(%s): add tests for the capability", w.Lane)
	case RungDogfooded:
		return fmt.Sprintf("maturity(%s): dogfood the capability in fak", w.Lane)
	case RungDefault:
		return fmt.Sprintf("maturity(%s): promote the capability to a documented default", w.Lane)
	case rungBenchmark:
		return fmt.Sprintf("maturity(%s): benchmark the default surface", w.Lane)
	default:
		return fmt.Sprintf("maturity(%s): advance the capability one rung", w.Lane)
	}
}

// IssueBody renders a stable marker-stamped issue body. The title carries the
// maturity(<lane>) scope so the existing issue router sends the issue back to
// the capability lane.
func IssueBody(item IssueItem) string {
	return fmt.Sprintf("<!-- fak-maturity-work-key: %s -->\n", item.Key) +
		"# Maturity next work\n\n" +
		fmt.Sprintf("- Stable key: `%s`\n", item.Key) +
		fmt.Sprintf("- Lane: `%s`\n", item.Lane) +
		fmt.Sprintf("- Current rung: `%s`\n", item.FromRung) +
		fmt.Sprintf("- Gap: `%s`\n", item.Gap) +
		fmt.Sprintf("- Ladder-skip: `%t`\n", item.Skip) +
		"- Source: `fak maturity next`\n\n" +
		"Suggested next action:\n\n" +
		fmt.Sprintf("%s\n\n", item.Title) +
		"Done condition / witness:\n\n" +
		fmt.Sprintf("%s\n\n", item.Witness) +
		"Regenerate the backlog with `go run ./cmd/fak maturity next`; this issue is " +
		"managed by `fak maturity route`, which updates the same stable key instead " +
		"of opening duplicates.\n"
}

// MarkerKey extracts the maturity work key from a GitHub issue body.
func MarkerKey(body string) string {
	m := maturityIssueMarkerRE.FindStringSubmatch(body)
	if m == nil {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// BuildIssuePlan decides whether each maturity item creates a new issue or updates
// the issue that already carries the stable marker key.
func BuildIssuePlan(items []IssueItem, existing []ExistingIssue) []IssuePlanRow {
	byKey := map[string]ExistingIssue{}
	for _, issue := range existing {
		if key := MarkerKey(issue.Body); key != "" {
			byKey[key] = issue
		}
	}
	plan := make([]IssuePlanRow, 0, len(items))
	for _, item := range items {
		row := IssuePlanRow{
			Action:  "create",
			Key:     item.Key,
			Lane:    item.Lane,
			Title:   item.Title,
			Body:    item.Body,
			Gap:     item.Gap,
			Witness: item.Witness,
			Skip:    item.Skip,
		}
		if found, ok := byKey[item.Key]; ok {
			row.Action = "update"
			row.State = found.State
			n := found.Number
			row.Number = &n
		}
		plan = append(plan, row)
	}
	return plan
}

// IssueRunner is injectable so tests never shell out to gh.
type IssueRunner func(args []string) (stdout, stderr string, ok bool)

func defaultIssueRunner(args []string) (string, string, bool) {
	cmd := exec.Command("gh", args...)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

// FetchExistingIssues queries gh for all issues that may carry maturity markers.
func FetchExistingIssues(repo string, limit int) ([]ExistingIssue, error) {
	if limit <= 0 {
		limit = 300
	}
	args := []string{"issue", "list", "--state", "all", "--limit", strconv.Itoa(limit), "--json", "number,title,body,state,url"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	stdout, stderr, ok := defaultIssueRunner(args)
	if !ok {
		return nil, fmt.Errorf("gh %s: %s", strings.Join(args, " "), strings.TrimSpace(stderr))
	}
	if strings.TrimSpace(stdout) == "" {
		return nil, nil
	}
	var issues []ExistingIssue
	if err := json.Unmarshal([]byte(stdout), &issues); err != nil {
		return nil, err
	}
	return issues, nil
}

// SyncIssuePlan applies a planned create/update set via gh. Labels are applied
// only to newly-created issues, matching gh's issue edit/create split.
func SyncIssuePlan(plan []IssuePlanRow, repo string, labels []string, runner IssueRunner) []IssueSyncRow {
	run := runner
	if run == nil {
		run = defaultIssueRunner
	}
	rows := make([]IssueSyncRow, 0, len(plan))
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
			for _, label := range labels {
				if strings.TrimSpace(label) != "" {
					args = append(args, "--label", strings.TrimSpace(label))
				}
			}
		}
		if repo != "" {
			args = append(args, "--repo", repo)
		}
		stdout, stderr, ok := run(args)
		rows = append(rows, IssueSyncRow{
			Key:    row.Key,
			Action: row.Action,
			OK:     ok,
			Stdout: strings.TrimSpace(stdout),
			Stderr: strings.TrimSpace(stderr),
		})
	}
	return rows
}

// RenderIssueResult is the human dry-run/live card for `fak maturity route`.
func RenderIssueResult(r IssueResult) string {
	lines := []string{
		fmt.Sprintf("maturity-route: %s  %d item(s)", r.Mode, len(r.Planned)),
		fmt.Sprintf("  workspace: %s", r.Workspace),
	}
	if len(r.Skipped) > 0 {
		lines = append(lines, fmt.Sprintf("  skipped-private: %d item(s)", len(r.Skipped)))
		for _, row := range r.Skipped {
			lines = append(lines, fmt.Sprintf("    lane=%s key=%s: %s", row.Lane, row.Key, row.Reason))
		}
	}
	if len(r.Planned) == 0 {
		lines = append(lines, "  no maturity backlog items to route")
		return strings.Join(lines, "\n")
	}
	for _, row := range r.Planned {
		target := "new issue"
		if row.Number != nil {
			target = "#" + strconv.Itoa(*row.Number)
		}
		skip := ""
		if row.Skip {
			skip = " skip"
		}
		lines = append(lines, fmt.Sprintf("  [%s] %s lane=%s gap=%s%s: %s",
			row.Action, target, row.Lane, row.Gap, skip, row.Title))
	}
	if r.Mode == "dry-run" {
		lines = append(lines, "  dry-run: pass --live to create/update GitHub issues")
	}
	return strings.Join(lines, "\n")
}
