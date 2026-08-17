package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	defaultProgressWindow     = 10 * time.Minute
	defaultProgressBaseline   = 20 * time.Minute
	defaultProgressStallAfter = 3
)

var progressNow = time.Now
var progressCommand = func(dir, name string, args ...string) ([]byte, error) {
	c := exec.Command(name, args...)
	c.Dir = dir
	return c.Output()
}

type progressReport struct {
	Verdict           string  `json:"verdict"`
	WindowMinutes     int     `json:"window_minutes"`
	Commits           int     `json:"commits"`
	CommitsPer10M     float64 `json:"commits_per_10m"`
	IssueClosesPer10M float64 `json:"issue_closes_per_10m"`
	Baseline          struct {
		WindowMinutes     int     `json:"window_minutes"`
		Commits           int     `json:"commits"`
		CommitsPer10M     float64 `json:"commits_per_10m"`
		IssuesClosed      int     `json:"issues_closed"`
		IssueClosesPer10M float64 `json:"issue_closes_per_10m"`
	} `json:"baseline"`
	StallAfterWindows int `json:"stall_after_windows"`
	WIP               struct {
		Files     int `json:"files"`
		Staged    int `json:"staged"`
		Unstaged  int `json:"unstaged"`
		Untracked int `json:"untracked"`
		Conflicts int `json:"conflicts"`
	} `json:"wip"`
	GitHub struct {
		Available      bool   `json:"available"`
		RecentlyClosed int    `json:"recently_closed"`
		OpenTotal      int    `json:"open_total"`
		Error          string `json:"error,omitempty"`
	} `json:"github"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

func cmdProgress(args []string) {
	os.Exit(runProgress(os.Stdout, os.Stderr, args))
}
func runProgress(out, errOut io.Writer, args []string) int {
	fs := flag.NewFlagSet("progress", flag.ContinueOnError)
	fs.SetOutput(errOut)
	window := fs.Duration("window", defaultProgressWindow, "current lookback window")
	baseline := fs.Duration("baseline", defaultProgressBaseline, "comparison period immediately before the current window")
	stallAfter := fs.Int("stall-after", defaultProgressStallAfter, "consecutive zero-delivery windows required for STALLED")
	jsonOut := fs.Bool("json", false, "emit JSON")
	dir := fs.String("dir", ".", "repository directory")
	fs.Usage = func() {
		fmt.Fprintln(errOut, "Usage: fak progress [--window 10m] [--baseline 20m] [--stall-after 3] [--json] [--dir PATH]")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *window <= 0 || *baseline <= 0 || *stallAfter < 2 {
		fmt.Fprintln(errOut, "progress: --window and --baseline must be positive; --stall-after must be at least 2")
		return 2
	}
	r, err := collectProgress(*dir, *window, *baseline, *stallAfter, progressNow())
	if err != nil {
		fmt.Fprintf(errOut, "progress: %v\n", err)
		return 1
	}
	if *jsonOut {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if enc.Encode(r) != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintf(out, "%s — %s\n", r.Verdict, r.Reason)
	fmt.Fprintf(out, "delivered: commits=%d (%.2f/10m) issues_closed=%d (%.2f/10m) window=%dm\n", r.Commits, r.CommitsPer10M, r.GitHub.RecentlyClosed, r.IssueClosesPer10M, r.WindowMinutes)
	fmt.Fprintf(out, "baseline: commits=%d (%.2f/10m) issues_closed=%d (%.2f/10m) window=%dm; stall_after=%d windows\n", r.Baseline.Commits, r.Baseline.CommitsPer10M, r.Baseline.IssuesClosed, r.Baseline.IssueClosesPer10M, r.Baseline.WindowMinutes, r.StallAfterWindows)
	fmt.Fprintf(out, "local WIP: files=%d staged=%d unstaged=%d untracked=%d conflicts=%d\n", r.WIP.Files, r.WIP.Staged, r.WIP.Unstaged, r.WIP.Untracked, r.WIP.Conflicts)
	if r.GitHub.Available {
		fmt.Fprintf(out, "GitHub: open=%d recently_closed=%d\n", r.GitHub.OpenTotal, r.GitHub.RecentlyClosed)
	} else {
		fmt.Fprintf(out, "GitHub: unavailable (%s)\n", r.GitHub.Error)
	}
	fmt.Fprintf(out, "next: %s\n", r.NextAction)
	return 0
}

func collectProgress(dir string, window, baseline time.Duration, stallAfter int, now time.Time) (progressReport, error) {
	var r progressReport
	r.WindowMinutes = int(window.Round(time.Minute) / time.Minute)
	r.Baseline.WindowMinutes = int(baseline.Round(time.Minute) / time.Minute)
	r.StallAfterWindows = stallAfter
	currentStart := now.Add(-window)
	baselineStart := currentStart.Add(-baseline)
	since := currentStart.UTC().Format(time.RFC3339)
	b, err := progressCommand(dir, "git", "rev-list", "--count", "--since="+since, "HEAD")
	if err != nil {
		return r, fmt.Errorf("read recent commits: %w", err)
	}
	r.Commits, err = strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return r, fmt.Errorf("parse recent commits: %w", err)
	}
	r.CommitsPer10M = progressRate(r.Commits, window)
	b, err = progressCommand(dir, "git", "rev-list", "--count", "--since="+baselineStart.UTC().Format(time.RFC3339), "--until="+since, "HEAD")
	if err != nil {
		return r, fmt.Errorf("read baseline commits: %w", err)
	}
	r.Baseline.Commits, err = strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return r, fmt.Errorf("parse baseline commits: %w", err)
	}
	r.Baseline.CommitsPer10M = progressRate(r.Baseline.Commits, baseline)
	b, err = progressCommand(dir, "git", "status", "--porcelain=v1", "-z")
	if err != nil {
		return r, fmt.Errorf("read local WIP: %w", err)
	}
	parseProgressWIP(b, &r)
	closed, e1 := progressCommand(dir, "gh", "issue", "list", "--state", "closed", "--search", "closed:>="+since, "--limit", "10000", "--json", "number")
	baselineClosed, e2 := progressCommand(dir, "gh", "issue", "list", "--state", "closed", "--search", "closed:"+baselineStart.UTC().Format(time.RFC3339)+".."+since, "--limit", "10000", "--json", "number")
	open, e3 := progressCommand(dir, "gh", "issue", "list", "--state", "open", "--limit", "10000", "--json", "number")
	if e1 == nil && e2 == nil && e3 == nil {
		var a, historical, c []struct {
			Number int `json:"number"`
		}
		if json.Unmarshal(closed, &a) == nil && json.Unmarshal(baselineClosed, &historical) == nil && json.Unmarshal(open, &c) == nil {
			r.GitHub.Available = true
			r.GitHub.RecentlyClosed = len(a)
			r.GitHub.OpenTotal = len(c)
			r.Baseline.IssuesClosed = len(historical)
			r.IssueClosesPer10M = progressRate(r.GitHub.RecentlyClosed, window)
			r.Baseline.IssueClosesPer10M = progressRate(r.Baseline.IssuesClosed, baseline)
		} else {
			r.GitHub.Error = "invalid gh JSON"
		}
	} else {
		r.GitHub.Error = "gh query failed"
	}
	delivered := r.Commits > 0 || (r.GitHub.Available && r.GitHub.RecentlyClosed > 0)
	baselineDelivered := r.Baseline.Commits > 0 || (r.GitHub.Available && r.Baseline.IssuesClosed > 0)
	stallEvidence := baseline >= window*time.Duration(stallAfter-1)
	if delivered {
		r.Verdict = "PROGRESS"
		r.Reason = "delivered-work evidence exists in the lookback window"
		r.NextAction = "keep shipping; recheck after the next window"
	} else if !baselineDelivered && stallEvidence {
		r.Verdict = "STALLED"
		r.Reason = fmt.Sprintf("no delivered-work evidence across at least %d consecutive windows", stallAfter)
		r.NextAction = "inspect the active bottleneck and replan before dispatching more work"
	} else if r.WIP.Files > 0 {
		r.Verdict = "WIP_ONLY"
		r.Reason = "local changes exist but no commit or issue closure proves delivery"
		r.NextAction = "finish, verify, and commit the coherent leaf"
	} else {
		r.Verdict = "QUIET"
		r.Reason = "the current window is quiet, but the baseline does not prove a sustained stall"
		r.NextAction = "select the next dispatchable issue"
	}
	if !r.GitHub.Available {
		r.Reason += "; GitHub evidence is unavailable"
	}
	return r, nil
}

func parseProgressWIP(status []byte, r *progressReport) {
	rows := strings.Split(string(status), "\x00")
	for i := 0; i < len(rows); i++ {
		row := rows[i]
		if len(row) < 3 {
			continue
		}
		r.WIP.Files++
		x, y := row[0], row[1]
		if x == '?' && y == '?' {
			r.WIP.Untracked++
			continue
		}
		if x == 'U' || y == 'U' || x == 'A' && y == 'A' || x == 'D' && y == 'D' {
			r.WIP.Conflicts++
		}
		if x != ' ' && x != '?' {
			r.WIP.Staged++
		}
		if y != ' ' && y != '?' {
			r.WIP.Unstaged++
		}
		if x == 'R' || x == 'C' {
			i++
		}
	}
}

func progressRate(count int, window time.Duration) float64 {
	return float64(count) * float64(defaultProgressWindow) / float64(window)
}
