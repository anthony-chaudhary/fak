package main

// scorecardpane_qadogfood.go — issue #1982: the `fak scorecard qa-dogfood` verb, the
// compact control-pane card for QA dogfood issue health. It reads the standing set of
// QA-dogfood-spine issues from the tracker (live `gh issue list --label qa-dogfood`,
// or a cached JSON array via --existing-json for hermetic/offline runs), decodes each
// issue body's markdown sections into a scorecardpane.QADogfoodIssue, and folds them
// into the panel (open / stale / closure-witness / percent-with-root-point-fields).
//
// The gather is impure (shells gh, parses issue bodies); the fold is the pure
// internal/scorecardpane.FoldQADogfoodPanel. buildQADogfoodIssues below is the pure
// seam between them so the whole card is witnessable offline from a JSON fixture.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
)

// qaDogfoodIssueRow is the subset of a `gh issue list --json ...` row the QA dogfood
// health panel needs: state + body (for the closure-witness and root-point sections)
// + updatedAt (for staleness).
type qaDogfoodIssueRow struct {
	Number    int    `json:"number"`
	State     string `json:"state"`
	Body      string `json:"body"`
	UpdatedAt string `json:"updatedAt"`
}

// runScorecardQADogfood implements `fak scorecard qa-dogfood`. Exit 0 on a folded
// panel, 2 on bad flags/input, 1 on a gh/encode failure.
func runScorecardQADogfood(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak scorecard qa-dogfood", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "scorecard")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	existingJSON := fs.String("existing-json", "", "read a cached `gh issue list --json number,state,body,updatedAt` array instead of shelling gh (hermetic/offline)")
	label := fs.String("label", "qa-dogfood", "gh label to scope the QA dogfood issue set")
	repo := fs.String("repo", "", "gh --repo override (default: the current repo)")
	limit := fs.Int("limit", 500, "gh issue list scan limit")
	staleHours := fs.Float64("stale-hours", scorecardpane.DefaultQADogfoodStaleHorizon.Hours(), "hours an open issue may sit untouched before it counts stale")
	if !parseFlags(fs, argv) {
		return 2
	}

	raw, code := loadQADogfoodRows(stderr, *existingJSON, *label, *repo, *limit)
	if code != 0 {
		return code
	}
	var rows []qaDogfoodIssueRow
	if err := json.Unmarshal(raw, &rows); err != nil {
		fmt.Fprintf(stderr, "fak scorecard qa-dogfood: parse issue list: %v\n", err)
		return 2
	}

	horizon := time.Duration(*staleHours * float64(time.Hour))
	issues := buildQADogfoodIssues(rows, time.Now(), horizon)
	panel := scorecardpane.FoldQADogfoodPanel(issues)

	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, panel); err != nil {
			fmt.Fprintf(stderr, "fak scorecard qa-dogfood: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintln(stdout, scorecardpane.RenderQADogfoodPanel(panel))
	return 0
}

// loadQADogfoodRows returns the raw gh JSON array — from the cached file when
// --existing-json is set (offline), otherwise from a live read-only `gh issue list`.
func loadQADogfoodRows(stderr io.Writer, existingJSON, label, repo string, limit int) ([]byte, int) {
	if existingJSON != "" {
		b, err := os.ReadFile(existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak scorecard qa-dogfood: read --existing-json: %v\n", err)
			return nil, 2
		}
		return b, 0
	}
	b, err := fetchQADogfoodIssues(label, repo, limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak scorecard qa-dogfood: %v\n", err)
		return nil, 1
	}
	return b, 0
}

// fetchQADogfoodIssues shells to gh for the label-scoped QA dogfood set, read-only
// (state=all so closed issues still count toward closure-witness coverage).
func fetchQADogfoodIssues(label, repo string, limit int) ([]byte, error) {
	if limit <= 0 {
		limit = 500
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	args := []string{"issue", "list", "--state", "all", "--limit", fmt.Sprint(limit),
		"--json", "number,state,body,updatedAt"}
	if label != "" {
		args = append(args, "--label", label)
	}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	cmd := exec.CommandContext(ctx, "gh", args...)
	configureDispatchHelperCommand(cmd)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("gh issue list failed: %w (%s)", err, strings.TrimSpace(string(b)))
	}
	return b, nil
}

// buildQADogfoodIssues is the pure seam from decoded gh rows to the fold's input: it
// reads state, derives staleness from updatedAt vs the horizon, and extracts the
// closure-witness and root-point markdown sections from each issue body. Kept pure
// (no I/O, injected clock) so the whole panel is witnessable from a JSON fixture.
func buildQADogfoodIssues(rows []qaDogfoodIssueRow, now time.Time, horizon time.Duration) []scorecardpane.QADogfoodIssue {
	out := make([]scorecardpane.QADogfoodIssue, 0, len(rows))
	for _, r := range rows {
		open := strings.EqualFold(strings.TrimSpace(r.State), "open")
		updated, _ := time.Parse(time.RFC3339, strings.TrimSpace(r.UpdatedAt))
		out = append(out, scorecardpane.QADogfoodIssue{
			Number:          r.Number,
			Open:            open,
			Stale:           scorecardpane.QADogfoodStale(open, updated, now, horizon),
			ClosureWitness:  qaDogfoodSection(r.Body, "Witness"),
			RootPointChange: qaDogfoodSection(r.Body, "Root-point change"),
			DoneCondition:   qaDogfoodSection(r.Body, "Done condition"),
		})
	}
	return out
}

// qaDogfoodSection returns the trimmed text under a `## <heading>` markdown section of
// an issue body, up to the next `## ` heading (or end of body); "" when the heading is
// absent. Matches the QA-dogfood-spine issue template (## Root-point change / ## Done
// condition / ## Witness), the shape #1982 itself carries.
func qaDogfoodSection(body, heading string) string {
	want := "## " + heading
	lines := strings.Split(body, "\n")
	var collecting bool
	var out []string
	for _, ln := range lines {
		trimmed := strings.TrimSpace(ln)
		if strings.HasPrefix(trimmed, "## ") {
			if collecting {
				break // next section — stop
			}
			if strings.EqualFold(trimmed, want) {
				collecting = true
			}
			continue
		}
		if collecting {
			out = append(out, ln)
		}
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
