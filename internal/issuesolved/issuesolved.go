// Package issuesolved collects and reports issues solved across public and private repositories
// within a specified time window. It supports querying GitHub via gh or falling back to git history.
package issuesolved

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
)

// Schema identifies the versioned JSON payload for solved issues reports.
const Schema = "fak-issues-solved/1"

// Options configures report generation across repositories.
type Options struct {
	Hours          float64
	Since          time.Time
	PublicRepo     string
	PrivateRepo    string
	PublicDir      string
	PrivateDir     string
	IncludePrivate bool
	Source         string // "auto", "github", "git"
	Now            time.Time
	GhExecutor     func(ctx context.Context, args ...string) ([]byte, error)
	GitExecutor    func(ctx context.Context, dir string, args ...string) ([]byte, error)

	// ExcludePrivate explicitly forces exclusion of the private repository.
	ExcludePrivate bool
}

// DefaultOptions returns the standard options with recommended defaults.
func DefaultOptions() Options {
	return Options{
		Hours:          24,
		PublicRepo:     "anthony-chaudhary/fak",
		PrivateRepo:    "anthony-chaudhary/fak-private",
		PublicDir:      ".",
		PrivateDir:     "../fak-private",
		IncludePrivate: true,
		Source:         "auto",
	}
}

// Report is the structured report of solved issues and commit activity.
type Report struct {
	Schema          string       `json:"schema"`
	GeneratedAt     time.Time    `json:"generated_at"`
	Since           time.Time    `json:"since"`
	Until           time.Time    `json:"until"`
	Hours           float64      `json:"hours"`
	TotalSolved     int          `json:"total_solved"`
	TotalClosed     int          `json:"total_closed"`
	TotalNotPlanned int          `json:"total_not_planned"`
	TotalCommits    int          `json:"total_commits"`
	Repos           []RepoReport `json:"repos"`
}

// RepoReport summarizes solved issues and commits for a single repository.
type RepoReport struct {
	Repo        string      `json:"repo"`
	Path        string      `json:"path"`
	Solved      int         `json:"solved"`
	Closed      int         `json:"closed"`
	NotPlanned  int         `json:"not_planned"`
	CommitCount int         `json:"commit_count"`
	Issues      []IssueItem `json:"issues"`
}

// IssueItem details a single closed issue.
type IssueItem struct {
	Number      int       `json:"number"`
	Title       string    `json:"title"`
	ClosedAt    time.Time `json:"closed_at"`
	StateReason string    `json:"state_reason"`
	URL         string    `json:"url"`
}

// defaultGhExecutor builds deadlined gh commands using internal/ghexec.
func defaultGhExecutor(ctx context.Context, args ...string) ([]byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cmd := ghexec.Command(ctx, args...)
	return cmd.Output()
}

// defaultGitExecutor runs git commands directly using os/exec.
func defaultGitExecutor(ctx context.Context, dir string, args ...string) ([]byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	return cmd.Output()
}

func resolvePrivateDir(publicDir, privateDir string) string {
	if privateDir != "" {
		return privateDir
	}
	if env := strings.TrimSpace(os.Getenv("FAK_PRIVATE_ROOT")); env != "" {
		if fi, err := os.Stat(env); err == nil && fi.IsDir() {
			return env
		}
	}
	if publicDir != "" {
		sibling := filepath.Join(publicDir, "..", "fak-private")
		if fi, err := os.Stat(sibling); err == nil && fi.IsDir() {
			return sibling
		}
	}
	if env := strings.TrimSpace(os.Getenv("FAK_PRIVATE_ROOT")); env != "" {
		return env
	}
	if fi, err := os.Stat("../fak-private"); err == nil && fi.IsDir() {
		return "../fak-private"
	}
	return "../fak-private"
}

func normalizeOptions(opts Options) Options {
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	if opts.Hours <= 0 && opts.Since.IsZero() {
		opts.Hours = 24
	}
	if opts.Hours > 0 && opts.Since.IsZero() {
		opts.Since = opts.Now.Add(-time.Duration(opts.Hours * float64(time.Hour)))
	}
	if opts.Hours <= 0 && !opts.Since.IsZero() {
		opts.Hours = opts.Now.Sub(opts.Since).Hours()
	}
	if opts.PublicRepo == "" {
		opts.PublicRepo = "anthony-chaudhary/fak"
	}
	if opts.PrivateRepo == "" {
		opts.PrivateRepo = "anthony-chaudhary/fak-private"
	}
	if opts.PublicDir == "" {
		opts.PublicDir = "."
	}
	if opts.PrivateDir == "" {
		opts.PrivateDir = resolvePrivateDir(opts.PublicDir, "")
	}
	if opts.Source == "" {
		opts.Source = "auto"
	}
	if opts.ExcludePrivate || opts.PrivateRepo == "-" || opts.PrivateRepo == "none" {
		opts.IncludePrivate = false
	}
	return opts
}

// Collect gathers issue and commit metrics across repositories according to opts.
func Collect(ctx context.Context, opts Options) (*Report, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	opts = normalizeOptions(opts)

	ghExec := opts.GhExecutor
	if ghExec == nil {
		ghExec = defaultGhExecutor
	}
	gitExec := opts.GitExecutor
	if gitExec == nil {
		gitExec = defaultGitExecutor
	}

	type targetRepo struct {
		name string
		dir  string
	}

	targets := []targetRepo{
		{name: opts.PublicRepo, dir: opts.PublicDir},
	}
	if opts.IncludePrivate && opts.PrivateRepo != "" {
		targets = append(targets, targetRepo{name: opts.PrivateRepo, dir: opts.PrivateDir})
	}

	rep := &Report{
		Schema:      Schema,
		GeneratedAt: opts.Now,
		Since:       opts.Since,
		Until:       opts.Now,
		Hours:       opts.Hours,
		Repos:       make([]RepoReport, 0, len(targets)),
	}

	for _, t := range targets {
		rr, err := collectRepo(ctx, t.name, t.dir, opts, ghExec, gitExec)
		if err != nil {
			return nil, err
		}
		rep.Repos = append(rep.Repos, rr)
		rep.TotalSolved += rr.Solved
		rep.TotalClosed += rr.Closed
		rep.TotalNotPlanned += rr.NotPlanned
		rep.TotalCommits += rr.CommitCount
	}

	return rep, nil
}

// Generate is an alias for Collect for caller convenience.
func Generate(ctx context.Context, opts Options) (*Report, error) {
	return Collect(ctx, opts)
}

func collectRepo(ctx context.Context, repo, dir string, opts Options,
	ghExec func(ctx context.Context, args ...string) ([]byte, error),
	gitExec func(ctx context.Context, dir string, args ...string) ([]byte, error),
) (RepoReport, error) {
	rr := RepoReport{
		Repo:   repo,
		Path:   dir,
		Issues: []IssueItem{},
	}

	rr.CommitCount = countCommits(ctx, dir, opts.Since, opts.Now, gitExec)

	source := strings.ToLower(strings.TrimSpace(opts.Source))
	if source == "" {
		source = "auto"
	}

	var issues []IssueItem
	var issuesErr error

	switch source {
	case "github":
		issues, issuesErr = collectIssuesFromGh(ctx, repo, opts.Since, opts.Now, ghExec)
		if issuesErr != nil {
			return rr, issuesErr
		}
	case "git":
		issues, issuesErr = collectIssuesFromGit(ctx, repo, dir, opts.Since, opts.Now, gitExec)
		if issuesErr != nil {
			return rr, issuesErr
		}
	case "auto":
		issues, issuesErr = collectIssuesFromGh(ctx, repo, opts.Since, opts.Now, ghExec)
		if issuesErr != nil {
			gitIssues, gitErr := collectIssuesFromGit(ctx, repo, dir, opts.Since, opts.Now, gitExec)
			if gitErr == nil {
				issues = gitIssues
			} else {
				issues = []IssueItem{}
			}
		}
	default:
		return rr, fmt.Errorf("unknown source: %q (supported: auto, github, git)", opts.Source)
	}

	rr.Issues = issues
	for _, it := range issues {
		rr.Closed++
		reason := strings.ToUpper(strings.TrimSpace(it.StateReason))
		if reason == "COMPLETED" {
			rr.Solved++
		} else if reason == "NOT_PLANNED" {
			rr.NotPlanned++
		}
	}

	return rr, nil
}

func countCommits(ctx context.Context, dir string, since, until time.Time,
	execFn func(ctx context.Context, dir string, args ...string) ([]byte, error),
) int {
	sinceStr := since.UTC().Format(time.RFC3339)
	untilStr := until.UTC().Format(time.RFC3339)

	args := []string{
		"rev-list",
		"--count",
		fmt.Sprintf("--since=%s", sinceStr),
		fmt.Sprintf("--until=%s", untilStr),
		"HEAD",
	}
	out, err := execFn(ctx, dir, args...)
	if err != nil {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		return 0
	}
	return n
}

type ghIssueWire struct {
	Number      int    `json:"number"`
	Title       string `json:"title"`
	ClosedAt    string `json:"closedAt"`
	StateReason string `json:"stateReason"`
	URL         string `json:"url"`
}

func parseTime(s string) (time.Time, error) {
	formats := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05Z07:00",
		"2006-01-02T15:04:05Z",
		"2006-01-02 15:04:05",
		"2006-01-02",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("unable to parse time: %s", s)
}

func collectIssuesFromGh(ctx context.Context, repo string, since, until time.Time,
	execFn func(ctx context.Context, args ...string) ([]byte, error),
) ([]IssueItem, error) {
	args := []string{
		"issue", "list",
		"--repo", repo,
		"--state", "closed",
		"--limit", "300",
		"--json", "number,title,closedAt,stateReason,url",
	}
	out, err := execFn(ctx, args...)
	if err != nil {
		return nil, fmt.Errorf("gh issue list: %w", err)
	}

	var rows []ghIssueWire
	if err := json.Unmarshal(out, &rows); err != nil {
		return nil, fmt.Errorf("decode gh issue list: %w", err)
	}

	var items []IssueItem
	for _, r := range rows {
		if r.ClosedAt == "" {
			continue
		}
		closedAt, err := parseTime(r.ClosedAt)
		if err != nil {
			continue
		}
		if closedAt.Before(since) || closedAt.After(until) {
			continue
		}
		reason := strings.ToUpper(strings.TrimSpace(r.StateReason))
		if reason == "" {
			reason = "COMPLETED"
		}
		url := r.URL
		if url == "" {
			url = fmt.Sprintf("https://github.com/%s/issues/%d", repo, r.Number)
		}
		items = append(items, IssueItem{
			Number:      r.Number,
			Title:       r.Title,
			ClosedAt:    closedAt,
			StateReason: reason,
			URL:         url,
		})
	}

	sort.Slice(items, func(i, j int) bool {
		if items[i].ClosedAt.Equal(items[j].ClosedAt) {
			return items[i].Number > items[j].Number
		}
		return items[i].ClosedAt.After(items[j].ClosedAt)
	})

	return items, nil
}

var (
	subjectIssueRE = regexp.MustCompile(`(?:^|[^\w-])#(\d+)\b`)
	privateIssueRE = regexp.MustCompile(`(?i)\b(?:anthony-chaudhary/)?fak-private#(\d+)\b`)
	bodyTrailerRE  = regexp.MustCompile(`(?i)\b(?:fixes|closes|resolves)\s+(?:(?:anthony-chaudhary/)?fak-private#|#)?(\d+)\b`)
)

func collectIssuesFromGit(ctx context.Context, repo, dir string, since, until time.Time,
	execFn func(ctx context.Context, dir string, args ...string) ([]byte, error),
) ([]IssueItem, error) {
	sinceStr := since.UTC().Format(time.RFC3339)
	untilStr := until.UTC().Format(time.RFC3339)

	args := []string{
		"log",
		fmt.Sprintf("--since=%s", sinceStr),
		fmt.Sprintf("--until=%s", untilStr),
		"--format=%H\x1f%aI\x1f%s\x1f%b\x1e",
	}

	out, err := execFn(ctx, dir, args...)
	if err != nil {
		errStr := err.Error()
		if strings.Contains(errStr, "no such file or directory") || strings.Contains(errStr, "cannot find the file") {
			return nil, nil
		}
		return nil, fmt.Errorf("git log: %w", err)
	}

	isPrivateRepo := strings.Contains(strings.ToLower(repo), "private")
	records := strings.Split(string(out), "\x1e")
	seen := make(map[int]bool)
	var items []IssueItem

	for _, rec := range records {
		rec = strings.TrimSpace(rec)
		if rec == "" {
			continue
		}
		parts := strings.Split(rec, "\x1f")
		if len(parts) < 4 {
			continue
		}
		dateStr := strings.TrimSpace(parts[1])
		subject := strings.TrimSpace(parts[2])
		body := strings.TrimSpace(parts[3])

		commitTime, err := parseTime(dateStr)
		if err != nil {
			commitTime = until
		}

		issueNums := extractIssueNumbers(subject, body, isPrivateRepo)
		for _, num := range issueNums {
			if seen[num] {
				continue
			}
			seen[num] = true
			items = append(items, IssueItem{
				Number:      num,
				Title:       subject,
				ClosedAt:    commitTime,
				StateReason: "COMPLETED",
				URL:         fmt.Sprintf("https://github.com/%s/issues/%d", repo, num),
			})
		}
	}

	sort.Slice(items, func(i, j int) bool {
		if items[i].ClosedAt.Equal(items[j].ClosedAt) {
			return items[i].Number > items[j].Number
		}
		return items[i].ClosedAt.After(items[j].ClosedAt)
	})

	return items, nil
}

func extractIssueNumbers(subject, body string, isPrivateRepo bool) []int {
	var nums []int
	addNum := func(n int) {
		for _, existing := range nums {
			if existing == n {
				return
			}
		}
		nums = append(nums, n)
	}

	if isPrivateRepo {
		for _, m := range privateIssueRE.FindAllStringSubmatch(subject, -1) {
			if len(m) > 1 {
				if n, err := strconv.Atoi(m[1]); err == nil {
					addNum(n)
				}
			}
		}
		for _, m := range subjectIssueRE.FindAllStringSubmatch(subject, -1) {
			if len(m) > 1 {
				if n, err := strconv.Atoi(m[1]); err == nil {
					addNum(n)
				}
			}
		}
		for _, m := range bodyTrailerRE.FindAllStringSubmatch(body, -1) {
			if len(m) > 1 {
				if n, err := strconv.Atoi(m[1]); err == nil {
					addNum(n)
				}
			}
		}
	} else {
		for _, m := range subjectIssueRE.FindAllStringSubmatch(subject, -1) {
			if len(m) > 1 {
				if n, err := strconv.Atoi(m[1]); err == nil {
					addNum(n)
				}
			}
		}
		for _, m := range bodyTrailerRE.FindAllStringSubmatch(body, -1) {
			if len(m) > 1 {
				fullMatch := m[0]
				if !strings.Contains(strings.ToLower(fullMatch), "private") {
					if n, err := strconv.Atoi(m[1]); err == nil {
						addNum(n)
					}
				}
			}
		}
	}

	return nums
}

// RenderSummary formats a concise <=4 line summary.
func RenderSummary(w io.Writer, r *Report) error {
	if r == nil {
		return fmt.Errorf("report is nil")
	}

	sinceUTC := r.Since.UTC()
	untilUTC := r.Until.UTC()
	var timeRangeStr string
	if sinceUTC.Format("2006-01-02") == untilUTC.Format("2006-01-02") {
		timeRangeStr = fmt.Sprintf("%s to %s UTC", sinceUTC.Format("2006-01-02 15:04"), untilUTC.Format("15:04"))
	} else {
		timeRangeStr = fmt.Sprintf("%s to %s UTC", sinceUTC.Format("2006-01-02 15:04"), untilUTC.Format("2006-01-02 15:04"))
	}

	if _, err := fmt.Fprintf(w, "Issues solved in last %.1fh (%s):\n", r.Hours, timeRangeStr); err != nil {
		return err
	}

	for _, repo := range r.Repos {
		label := repo.Repo + ":"
		if _, err := fmt.Fprintf(w, "  %-30s %d solved (%d closed, %d not planned) | %d commits\n",
			label, repo.Solved, repo.Closed, repo.NotPlanned, repo.CommitCount); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "  %-30s %d solved (%d closed, %d not planned) | %d trunk commits\n",
		"TOTAL:", r.TotalSolved, r.TotalClosed, r.TotalNotPlanned, r.TotalCommits); err != nil {
		return err
	}

	return nil
}

// RenderDetailed outputs the summary plus list of issue numbers, titles, and timestamps grouped by repo.
func RenderDetailed(w io.Writer, r *Report) error {
	if err := RenderSummary(w, r); err != nil {
		return err
	}

	for _, repo := range r.Repos {
		if len(repo.Issues) == 0 {
			continue
		}
		if _, err := fmt.Fprintf(w, "\n%s issues (%d):\n", repo.Repo, len(repo.Issues)); err != nil {
			return err
		}
		for _, it := range repo.Issues {
			timeStr := it.ClosedAt.UTC().Format("2006-01-02 15:04 UTC")
			reason := it.StateReason
			if reason == "" {
				reason = "COMPLETED"
			}
			if _, err := fmt.Fprintf(w, "  #%-6d [%s] %s (%s)\n", it.Number, reason, it.Title, timeStr); err != nil {
				return err
			}
		}
	}
	return nil
}

// RenderJSON serializes the report as indented JSON.
func RenderJSON(w io.Writer, r *Report) error {
	if r == nil {
		return fmt.Errorf("report is nil")
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}
