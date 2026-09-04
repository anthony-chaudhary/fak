package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
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
		Files                 int   `json:"files"`
		Staged                int   `json:"staged"`
		Unstaged              int   `json:"unstaged"`
		Untracked             int   `json:"untracked"`
		Conflicts             int   `json:"conflicts"`
		Additions             int64 `json:"additions"`
		Deletions             int64 `json:"deletions"`
		BinaryFiles           int   `json:"binary_files"`
		UntrackedBytes        int64 `json:"untracked_bytes"`
		OldestExistingMinutes int64 `json:"oldest_existing_minutes"`
		AgeFilesObserved      int   `json:"age_files_observed"`
		AgeFilesUnavailable   int   `json:"age_files_unavailable"`
	} `json:"wip"`
	CLQ                  float64      `json:"clq"`
	WIPHalfLifeMinutes   float64      `json:"wip_halflife_minutes"`
	DrainVelocityPerHour float64      `json:"drain_velocity_per_hour"`
	Flow                 progressFlow `json:"flow"`
	GitHub               struct {
		Available      bool   `json:"available"`
		RecentlyClosed int    `json:"recently_closed"`
		OpenTotal      int    `json:"open_total"`
		Error          string `json:"error,omitempty"`
	} `json:"github"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

type progressFlow struct {
	Available             bool                      `json:"available"`
	ObservedMinutes       int                       `json:"observed_minutes,omitempty"`
	Opening               progressInventorySnapshot `json:"opening"`
	Closing               progressInventorySnapshot `json:"closing"`
	WIPFilesDelta         int                       `json:"wip_files_delta,omitempty"`
	WIPLinesDelta         int64                     `json:"wip_lines_delta,omitempty"`
	UntrackedBytesDelta   int64                     `json:"untracked_bytes_delta,omitempty"`
	OldestWIPMinutesDelta int64                     `json:"oldest_wip_minutes_delta,omitempty"`
	OpenIssuesDelta       int                       `json:"open_issues_delta,omitempty"`
	Reason                string                    `json:"reason,omitempty"`
}

type progressInventorySnapshot struct {
	ObservedAt       time.Time `json:"observed_at"`
	WIPFiles         int       `json:"wip_files"`
	WIPLines         int64     `json:"wip_lines"`
	UntrackedBytes   int64     `json:"untracked_bytes"`
	OldestWIPMinutes int64     `json:"oldest_wip_minutes"`
	OpenIssues       int       `json:"open_issues"`
	GitHubAvailable  bool      `json:"github_available"`
}

type progressInventoryHistory struct {
	Schema    string                      `json:"schema"`
	Snapshots []progressInventorySnapshot `json:"snapshots"`
}

var progressInventory = observeProgressInventory

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
	*dir = pathutil.ExpandTilde(*dir)
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
	fmt.Fprintf(out, "local WIP: files=%d staged=%d unstaged=%d untracked=%d conflicts=%d additions=%d deletions=%d binary=%d untracked_bytes=%d oldest_existing=%dm age_observed=%d/%d\n", r.WIP.Files, r.WIP.Staged, r.WIP.Unstaged, r.WIP.Untracked, r.WIP.Conflicts, r.WIP.Additions, r.WIP.Deletions, r.WIP.BinaryFiles, r.WIP.UntrackedBytes, r.WIP.OldestExistingMinutes, r.WIP.AgeFilesObserved, r.WIP.AgeFilesObserved+r.WIP.AgeFilesUnavailable)
	fmt.Fprintf(out, "metrics: clq=%.2f wip_halflife=%.1fm drain_velocity=%.2f/h\n", r.CLQ, r.WIPHalfLifeMinutes, r.DrainVelocityPerHour)
	if r.Flow.Available {
		fmt.Fprintf(out, "flow: observed=%dm wip_files=%+d wip_lines=%+d untracked_bytes=%+d oldest_wip=%+dm open_issues=%+d\n", r.Flow.ObservedMinutes, r.Flow.WIPFilesDelta, r.Flow.WIPLinesDelta, r.Flow.UntrackedBytesDelta, r.Flow.OldestWIPMinutesDelta, r.Flow.OpenIssuesDelta)
	} else {
		fmt.Fprintf(out, "flow: unavailable (%s)\n", r.Flow.Reason)
	}
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
	b, err = progressCommand(dir, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return r, fmt.Errorf("read local WIP: %w", err)
	}
	wipPaths := parseProgressWIP(b, &r)
	if err := collectProgressWIPDetails(dir, wipPaths, now, &r); err != nil {
		return r, err
	}
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
	current := progressInventorySnapshot{ObservedAt: now.UTC(), WIPFiles: r.WIP.Files, WIPLines: r.WIP.Additions + r.WIP.Deletions, UntrackedBytes: r.WIP.UntrackedBytes, OldestWIPMinutes: r.WIP.OldestExistingMinutes, OpenIssues: r.GitHub.OpenTotal, GitHubAvailable: r.GitHub.Available}
	flow, flowErr := progressInventory(dir, window, current)
	if flowErr != nil {
		flow = progressFlow{Closing: current, Reason: "inventory state unavailable: " + flowErr.Error()}
	}
	r.Flow = flow
	r.CLQ = computeCLQ(r.WIP.Conflicts, r.WIP.Untracked, r.WIP.Additions+r.WIP.Deletions)
	if r.WIPHalfLifeMinutes == 0 && r.WIP.OldestExistingMinutes > 0 {
		r.WIPHalfLifeMinutes = math.Round((float64(r.WIP.OldestExistingMinutes)/2.0)*100) / 100
	}
	r.DrainVelocityPerHour = computeDrainVelocity(r.CommitsPer10M)
	delivered := r.Commits > 0 || (r.GitHub.Available && r.GitHub.RecentlyClosed > 0)
	baselineDelivered := r.Baseline.Commits > 0 || (r.GitHub.Available && r.Baseline.IssuesClosed > 0)
	stallEvidence := baseline >= window*time.Duration(stallAfter-1)
	switch {
	case !flow.Available:
		r.Verdict, r.Reason, r.NextAction = "UNKNOWN", flow.Reason, "capture another inventory sample after the window; do not infer convergence from activity"
	case flow.WIPFilesDelta > 0 || flow.WIPLinesDelta > 0 || flow.UntrackedBytesDelta > 0 || flow.OpenIssuesDelta > 0:
		r.Verdict, r.Reason, r.NextAction = "DIVERGING", "unfinished-work inventory grew across the observation window", "finish or retire inventory before launching more work"
	case flow.WIPFilesDelta < 0 || flow.WIPLinesDelta < 0 || flow.UntrackedBytesDelta < 0 || flow.OpenIssuesDelta < 0:
		r.Verdict, r.Reason, r.NextAction = "CONVERGING", "unfinished-work inventory shrank without a countervailing growth signal", "keep retiring inventory and recheck after the next window"
	case flow.OldestWIPMinutesDelta > 0 || delivered || (!baselineDelivered && stallEvidence):
		r.Verdict, r.Reason, r.NextAction = "FLOW_STALLED", "inventory did not shrink across the observation window", "finish or retire one existing unit before counting more activity as progress"
	default:
		r.Verdict, r.Reason, r.NextAction = "FLOW_STALLED", "no inventory reduction was observed", "finish or retire one existing unit and recheck after the next window"
	}
	return r, nil
}

func observeProgressInventory(dir string, window time.Duration, current progressInventorySnapshot) (progressFlow, error) {
	path, err := progressInventoryPath(dir)
	if err != nil {
		return progressFlow{Closing: current}, err
	}
	history, err := readProgressInventoryHistory(path)
	if err != nil {
		return progressFlow{Closing: current}, err
	}
	opening, ok := selectProgressOpening(history.Snapshots, current.ObservedAt.Add(-window), window/2)
	history.Schema = "fak-progress-inventory/1"
	history.Snapshots = append(history.Snapshots, current)
	cutoff := current.ObservedAt.Add(-7 * 24 * time.Hour)
	kept := history.Snapshots[:0]
	for _, x := range history.Snapshots {
		if !x.ObservedAt.Before(cutoff) {
			kept = append(kept, x)
		}
	}
	history.Snapshots = kept
	if err := writeProgressInventoryHistory(path, history); err != nil {
		return progressFlow{Closing: current}, err
	}
	if !ok {
		return progressFlow{Closing: current, Reason: "no inventory sample near the opening boundary"}, nil
	}
	f := progressFlow{Available: true, Opening: opening, Closing: current, ObservedMinutes: int(current.ObservedAt.Sub(opening.ObservedAt).Round(time.Minute) / time.Minute), WIPFilesDelta: current.WIPFiles - opening.WIPFiles, WIPLinesDelta: current.WIPLines - opening.WIPLines, UntrackedBytesDelta: current.UntrackedBytes - opening.UntrackedBytes, OldestWIPMinutesDelta: current.OldestWIPMinutes - opening.OldestWIPMinutes}
	if !opening.GitHubAvailable || !current.GitHubAvailable {
		f.Available = false
		f.Reason = "GitHub inventory unavailable at one or both boundaries"
		return f, nil
	}
	f.OpenIssuesDelta = current.OpenIssues - opening.OpenIssues
	return f, nil
}

func progressInventoryPath(dir string) (string, error) {
	root, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	gp := filepath.Join(root, ".git")
	info, err := os.Stat(gp)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return filepath.Join(gp, "fak-progress-inventory.json"), nil
	}
	data, err := os.ReadFile(gp)
	if err != nil {
		return "", err
	}
	line := strings.TrimSpace(string(data))
	const prefix = "gitdir:"
	if !strings.HasPrefix(strings.ToLower(line), prefix) {
		return "", errors.New(".git file does not declare gitdir")
	}
	gd := strings.TrimSpace(line[len(prefix):])
	if !filepath.IsAbs(gd) {
		gd = filepath.Join(root, gd)
	}
	return filepath.Join(filepath.Clean(gd), "fak-progress-inventory.json"), nil
}
func readProgressInventoryHistory(path string) (progressInventoryHistory, error) {
	var h progressInventoryHistory
	b, e := os.ReadFile(path)
	if errors.Is(e, os.ErrNotExist) {
		return h, nil
	}
	if e != nil {
		return h, e
	}
	if e = json.Unmarshal(b, &h); e != nil {
		return h, fmt.Errorf("decode inventory: %w", e)
	}
	if h.Schema != "" && h.Schema != "fak-progress-inventory/1" {
		return h, fmt.Errorf("unsupported inventory schema %q", h.Schema)
	}
	return h, nil
}
func writeProgressInventoryHistory(path string, h progressInventoryHistory) error {
	b, e := json.MarshalIndent(h, "", "  ")
	if e != nil {
		return e
	}
	b = append(b, '\n')
	tmp, e := os.CreateTemp(filepath.Dir(path), ".fak-progress-*.tmp")
	if e != nil {
		return e
	}
	tp := tmp.Name()
	defer os.Remove(tp)
	if _, e = tmp.Write(b); e == nil {
		e = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if e != nil {
		return e
	}
	if e = os.Rename(tp, path); e != nil {
		if re := os.Remove(path); re != nil && !errors.Is(re, os.ErrNotExist) {
			return e
		}
		return os.Rename(tp, path)
	}
	return nil
}
func selectProgressOpening(xs []progressInventorySnapshot, boundary time.Time, tolerance time.Duration) (progressInventorySnapshot, bool) {
	ys := append([]progressInventorySnapshot(nil), xs...)
	sort.Slice(ys, func(i, j int) bool { return ys[i].ObservedAt.Before(ys[j].ObservedAt) })
	var best progressInventorySnapshot
	distance := time.Duration(1<<63 - 1)
	for _, x := range ys {
		d := x.ObservedAt.Sub(boundary)
		if d < 0 {
			d = -d
		}
		if d <= tolerance && d < distance {
			best = x
			distance = d
		}
	}
	return best, !best.ObservedAt.IsZero()
}

func parseProgressWIP(status []byte, r *progressReport) []progressWIPPath {
	rows := strings.Split(string(status), "\x00")
	paths := make([]progressWIPPath, 0, len(rows))
	for i := 0; i < len(rows); i++ {
		row := rows[i]
		if len(row) < 3 {
			continue
		}
		r.WIP.Files++
		x, y := row[0], row[1]
		path := row[3:]
		untracked := x == '?' && y == '?'
		paths = append(paths, progressWIPPath{path: path, untracked: untracked})
		if untracked {
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
	return paths
}

type progressWIPPath struct {
	path      string
	untracked bool
}

func collectProgressWIPDetails(dir string, paths []progressWIPPath, now time.Time, r *progressReport) error {
	numstat, err := progressCommand(dir, "git", "diff", "--numstat", "-z", "HEAD", "--")
	if err != nil {
		return fmt.Errorf("read WIP magnitude: %w", err)
	}
	parseProgressNumstat(numstat, r)
	root, err := filepath.Abs(dir)
	if err != nil {
		return fmt.Errorf("resolve repository root: %w", err)
	}
	var observedAges []float64
	for _, candidate := range paths {
		full := filepath.Join(root, filepath.FromSlash(candidate.path))
		rel, relErr := filepath.Rel(root, full)
		if relErr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			r.WIP.AgeFilesUnavailable++
			continue
		}
		info, statErr := os.Lstat(full)
		if statErr != nil {
			r.WIP.AgeFilesUnavailable++
			continue
		}
		r.WIP.AgeFilesObserved++
		age := now.Sub(info.ModTime())
		if age > 0 {
			minutes := int64(age / time.Minute)
			if minutes > r.WIP.OldestExistingMinutes {
				r.WIP.OldestExistingMinutes = minutes
			}
			observedAges = append(observedAges, float64(age)/float64(time.Minute))
		}
		if candidate.untracked && info.Mode().IsRegular() {
			r.WIP.UntrackedBytes += info.Size()
		}
	}
	r.WIPHalfLifeMinutes = computeWIPHalfLife(observedAges, r.WIP.OldestExistingMinutes)
	return nil
}

func parseProgressNumstat(data []byte, r *progressReport) {
	for _, row := range strings.Split(string(data), "\x00") {
		fields := strings.Split(row, "\t")
		if len(fields) < 3 {
			continue
		}
		if fields[0] == "-" || fields[1] == "-" {
			r.WIP.BinaryFiles++
			continue
		}
		additions, addErr := strconv.ParseInt(fields[0], 10, 64)
		deletions, delErr := strconv.ParseInt(fields[1], 10, 64)
		if addErr == nil && delErr == nil {
			r.WIP.Additions += additions
			r.WIP.Deletions += deletions
		}
	}
}

func progressRate(count int, window time.Duration) float64 {
	return float64(count) * float64(defaultProgressWindow) / float64(window)
}

func computeCLQ(conflicts, untracked int, diffLines int64) float64 {
	if conflicts > 0 {
		return 0.0
	}
	score := 1.0
	if untracked > 0 {
		untrackedPenalty := float64(untracked/10) * 0.05
		if untrackedPenalty > 0.30 {
			untrackedPenalty = 0.30
		}
		score -= untrackedPenalty
	}
	if diffLines > 500 {
		score -= 0.20
	}
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}
	return math.Round(score*100) / 100
}

func computeWIPHalfLife(observedAges []float64, oldestMinutes int64) float64 {
	if len(observedAges) > 0 {
		sorted := make([]float64, len(observedAges))
		copy(sorted, observedAges)
		sort.Float64s(sorted)
		n := len(sorted)
		var median float64
		if n%2 == 1 {
			median = sorted[n/2]
		} else {
			median = (sorted[n/2-1] + sorted[n/2]) / 2.0
		}
		return math.Round(median*100) / 100
	}
	if oldestMinutes > 0 {
		return math.Round((float64(oldestMinutes)/2.0)*100) / 100
	}
	return 0.0
}

func computeDrainVelocity(commitsPer10M float64) float64 {
	return math.Round(commitsPer10M*6.0*100) / 100
}
