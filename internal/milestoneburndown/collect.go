package milestoneburndown

// Live runner for the burndown dimension: it reads the repo's OPEN GitHub
// milestones (their due_on + open/closed counts) and, per milestone, the count of
// issues closed within the trailing velocity window. Kept separate from the pure
// fold so milestoneburndown.go stays unit-testable without a process or a network;
// the JSON decoders below are pure and are exercised directly by fixtures.

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/epicprogress"
)

// Runner is the injectable `gh` seam, an alias of epicprogress.Runner so the
// burndown collector and the epic resolver wire one runner type. DefaultRunner
// shells the real `gh` with a bounded timeout.
type Runner = epicprogress.Runner

// DefaultRunner is epicprogress.DefaultRunner — the real `gh` CLI.
var DefaultRunner = epicprogress.DefaultRunner

// rawMilestone mirrors the subset of the GitHub milestone object we fold. due_on is
// nullable (absent/"" == no due date); open_issues/closed_issues are the milestone's
// own live counters.
type rawMilestone struct {
	Number  int    `json:"number"`
	Title   string `json:"title"`
	HTMLURL string `json:"html_url"`
	DueOn   string `json:"due_on"`
	State   string `json:"state"`
	Open    int    `json:"open_issues"`
	Closed  int    `json:"closed_issues"`
}

// rawIssue mirrors the one field velocity needs: when the issue was closed.
type rawIssue struct {
	ClosedAt string `json:"closed_at"`
}

// Collect reads the live open milestones and folds them into a Portfolio as of
// `now`. A nil runner uses the real `gh`. `repo` is "owner/name" or "" to use the
// current checkout's gh context. Velocity is measured over the trailing windowDays;
// a per-milestone velocity read that fails degrades only that row (ClosedInWindow
// = -1), never the whole portfolio. Only a failure to read the milestone LIST makes
// the portfolio unmeasured.
func Collect(repo string, runner Runner, windowDays int, now time.Time) Portfolio {
	if runner == nil {
		runner = DefaultRunner
	}
	if windowDays <= 0 {
		windowDays = DefaultWindowDays
	}
	ms, err := collectMilestones(repo, runner, windowDays, now)
	if err != "" {
		return Interpret(nil, windowDays, now, err)
	}
	return Interpret(ms, windowDays, now, "")
}

// collectMilestones is Collect's impure half: it returns the []Milestone (velocity
// filled) or a non-empty read-error string when the milestone list is unreadable.
func collectMilestones(repo string, runner Runner, windowDays int, now time.Time) ([]Milestone, string) {
	stdout, stderr, ok := runner([]string{"api", milestonesPath(repo)})
	if !ok {
		return nil, ghErr("list milestones", stderr, stdout)
	}
	raw, err := decodeMilestones([]byte(stdout))
	if err != nil {
		return nil, "decode milestones: " + err.Error()
	}
	since := now.Add(-time.Duration(windowDays) * 24 * time.Hour).UTC()
	out := make([]Milestone, 0, len(raw))
	for _, rm := range raw {
		m := Milestone{
			Number:         rm.Number,
			Title:          rm.Title,
			URL:            rm.HTMLURL,
			DueOn:          rm.DueOn,
			Open:           rm.Open,
			Closed:         rm.Closed,
			WindowDays:     windowDays,
			ClosedInWindow: -1, // default: unmeasured until the velocity read succeeds
		}
		if n, velOK := collectVelocity(repo, runner, rm.Number, since); velOK {
			m.ClosedInWindow = n
		}
		out = append(out, m)
	}
	return out, ""
}

// collectVelocity counts issues in milestone `number` closed at or after `since`.
// The `since` query prunes the payload to issues updated in the window (a superset
// of those closed in it); we filter by closed_at so the count is exact.
func collectVelocity(repo string, runner Runner, number int, since time.Time) (int, bool) {
	stdout, _, ok := runner([]string{"api", velocityPath(repo, number, since)})
	if !ok {
		return 0, false
	}
	n, err := countClosedSince([]byte(stdout), since)
	if err != nil {
		return 0, false
	}
	return n, true
}

// milestonesPath is the gh api path for the open milestone list. gh fills the
// {owner}/{repo} placeholders from the current checkout when repo is "".
func milestonesPath(repo string) string {
	return repoBase(repo) + "/milestones?state=open&per_page=100&sort=due_on&direction=asc"
}

// velocityPath is the gh api path for a milestone's recently-closed issues.
func velocityPath(repo string, number int, since time.Time) string {
	return fmt.Sprintf("%s/issues?milestone=%d&state=closed&since=%s&per_page=100&sort=updated&direction=desc",
		repoBase(repo), number, since.Format(time.RFC3339))
}

func repoBase(repo string) string {
	repo = strings.TrimSpace(repo)
	if repo == "" {
		return "repos/{owner}/{repo}"
	}
	return "repos/" + repo
}

// decodeMilestones parses the gh milestones array. Pure and fixture-tested.
func decodeMilestones(b []byte) ([]rawMilestone, error) {
	b = trimJSON(b)
	if len(b) == 0 {
		return nil, nil
	}
	var out []rawMilestone
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// countClosedSince parses a gh issues array and counts entries whose closed_at is at
// or after `since`. An entry with an empty/unparseable closed_at is skipped (it
// cannot be witnessed as closed in-window). Pure and fixture-tested.
func countClosedSince(b []byte, since time.Time) (int, error) {
	b = trimJSON(b)
	if len(b) == 0 {
		return 0, nil
	}
	var issues []rawIssue
	if err := json.Unmarshal(b, &issues); err != nil {
		return 0, err
	}
	n := 0
	for _, is := range issues {
		t, err := time.Parse(time.RFC3339, strings.TrimSpace(is.ClosedAt))
		if err != nil {
			continue
		}
		if !t.UTC().Before(since.UTC()) {
			n++
		}
	}
	return n, nil
}

func trimJSON(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}

func ghErr(what, stderr, stdout string) string {
	msg := strings.TrimSpace(stderr)
	if msg == "" {
		msg = strings.TrimSpace(stdout)
	}
	if msg == "" {
		msg = "gh returned no output"
	}
	if len(msg) > 200 {
		msg = msg[:200]
	}
	return what + ": " + msg
}
