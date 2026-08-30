package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
)

type slackRefreshKind string

const (
	refreshRunnable    slackRefreshKind = "runnable"
	refreshInputNeeded slackRefreshKind = "input-needed"
	refreshManual      slackRefreshKind = "manual"
	refreshBridge      slackRefreshKind = "bridge"
)

type slackRefreshAction struct {
	Kind    slackRefreshKind `json:"kind"`
	Command string           `json:"command"`
	Detail  string           `json:"detail,omitempty"`
	Run     func(stdout, stderr io.Writer, dryRun bool, opts slackRefreshOptions) int
}

type slackRefreshOptions struct {
	NewsTitle       string
	NewsFile        string
	BlockersIssues  string
	BlockersLabel   string
	BlockersRepoURL string
	BacklogIssues   string
	BacklogChannel  string
}

type slackWalkRow struct {
	Surface       string           `json:"surface"`
	Ready         bool             `json:"ready"`
	Optional      bool             `json:"optional,omitempty"`
	Status        string           `json:"status"`
	Kind          slackRefreshKind `json:"kind"`
	Command       string           `json:"command"`
	Detail        string           `json:"detail,omitempty"`
	TokenSource   string           `json:"token_source,omitempty"`
	ChannelSource string           `json:"channel_source,omitempty"`
}

type slackRefreshResult struct {
	Surface   string `json:"surface"`
	Status    string `json:"status"`
	ExitCode  int    `json:"exit_code,omitempty"`
	Command   string `json:"command"`
	Detail    string `json:"detail,omitempty"`
	Stdout    string `json:"stdout,omitempty"`
	Stderr    string `json:"stderr,omitempty"`
	ErrorType string `json:"error_type,omitempty"`
}

type slackGitHubIssue struct {
	Number    int    `json:"number"`
	Title     string `json:"title"`
	URL       string `json:"url"`
	Assignees []struct {
		Login string `json:"login"`
	} `json:"assignees"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
}

type slackGitHubRunner func(args ...string) ([]byte, error)

var slackRefreshRunScoreboardPost = runScoreboardPost

func runSlackRefreshGH(args ...string) ([]byte, error) {
	cmd, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, args...)
	defer cancel()
	out, err := cmd.Output()
	if err == nil {
		return out, nil
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return nil, fmt.Errorf("gh issue list: %w: %s", err, strings.TrimSpace(string(exitErr.Stderr)))
	}
	return nil, fmt.Errorf("gh issue list: %w", err)
}

func slackRefreshActions() map[string]slackRefreshAction {
	return map[string]slackRefreshAction{
		"product": {
			Kind:    refreshRunnable,
			Command: "fak product post --status",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runProductPost(stdout, stderr, withDryRun([]string{"--status"}, dryRun))
			},
		},
		"scoreboard": {
			Kind:    refreshRunnable,
			Command: "fak scoreboard post --kpi slack-outbox-pending",
			Detail:  "built-in rollup sourced from the durable Slack outbox status fold",
			Run:     runBuiltinScoreboardRefresh,
		},
		"alerts": {
			Kind:    refreshRunnable,
			Command: "fak slack refresh --surface alerts",
			Detail:  "audit-only readback; never synthesizes an all-clear alert",
			Run: func(stdout, stderr io.Writer, _ bool, _ slackRefreshOptions) int {
				return runSlackSurfaceAudit(stdout, stderr, "alerts", false)
			},
		},
		"guard-sessions": {
			Kind:    refreshRunnable,
			Command: "fak slack refresh --surface guard-sessions",
			Detail:  "audit-only readback plus durable outbox status",
			Run: func(stdout, stderr io.Writer, _ bool, _ slackRefreshOptions) int {
				return runSlackSurfaceAudit(stdout, stderr, "guard-sessions", true)
			},
		},
		"grafana": {
			Kind:    refreshRunnable,
			Command: "fak grafana post --rollup all",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runGrafanaPost(stdout, stderr, withDryRun([]string{"--rollup", "all"}, dryRun))
			},
		},
		"blockers": {
			Kind:    refreshInputNeeded,
			Command: "fak blockers feed --issues FILE",
			Detail:  "needs the GitHub issue-list payload; without it, a live all-clear would be ambiguous",
			Run: func(stdout, stderr io.Writer, dryRun bool, opts slackRefreshOptions) int {
				args := []string{"--issues", opts.BlockersIssues, "--label", opts.BlockersLabel}
				if opts.BlockersRepoURL != "" {
					args = append(args, "--repo-url", opts.BlockersRepoURL)
				}
				return runBlockersFeed(stdout, stderr, withDryRun(args, dryRun))
			},
		},
		"cachevalue": {
			Kind:    refreshRunnable,
			Command: "fak cachevalue feed",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runCachevalueFeed(stdout, stderr, withDryRun(nil, dryRun))
			},
		},
		"bench": {
			Kind:    refreshRunnable,
			Command: "fak bench post --rollup latest",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runBenchPost(stdout, stderr, withDryRun([]string{"--rollup", "latest"}, dryRun))
			},
		},
		"dispatch": {
			Kind:    refreshManual,
			Command: "fak dispatch ...",
			Detail:  "dispatch posts as part of the witness-gated issue loop",
		},
		"dojo": {
			Kind:    refreshRunnable,
			Command: "fak dojo post --rollup trend",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runDojoPost(stdout, stderr, withDryRun([]string{"--rollup", "trend"}, dryRun))
			},
		},
		"backlog": {
			Kind:    refreshRunnable,
			Command: "gh issue list --state open --limit 100 ... then fak scoreboard post --kpi backlog-triage",
			Detail:  "bounded current GitHub issue readout",
			Run: func(stdout, stderr io.Writer, dryRun bool, opts slackRefreshOptions) int {
				issues, err := decodeSlackGitHubIssues([]byte(opts.BacklogIssues))
				if err != nil {
					fmt.Fprintf(stderr, "fak slack refresh backlog: %v\n", err)
					return 2
				}
				detail, verdict := backlogRefreshDetail(issues), "ACTION"
				if len(issues) == 0 {
					detail, verdict = "GitHub returned zero open issues", "OK"
				}
				args := []string{"--channel", opts.BacklogChannel, "--kpi", "backlog-triage", "--value", fmt.Sprint(len(issues)), "--verdict", verdict, "--detail", detail}
				return slackRefreshRunScoreboardPost(stdout, stderr, withDryRun(args, dryRun))
			},
		},
		"marketing": {
			Kind:    refreshRunnable,
			Command: "fak marketing tick",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runMarketingTick(stdout, stderr, withDryRun(nil, dryRun))
			},
		},
		"news": {
			Kind:    refreshRunnable,
			Command: "fak slack refresh --surface news [--news-title TITLE --news-file FILE]",
			Detail:  "audits current digest state by default; publication still requires curated input",
			Run: func(stdout, stderr io.Writer, dryRun bool, opts slackRefreshOptions) int {
				if strings.TrimSpace(opts.NewsTitle) == "" || strings.TrimSpace(opts.NewsFile) == "" {
					return runSlackNewsAudit(stdout, stderr)
				}
				return runNewsPost(stdout, stderr, withDryRun([]string{"--title", opts.NewsTitle, "--notes-file", opts.NewsFile}, dryRun))
			},
		},
		"node-usage": {
			Kind:    refreshRunnable,
			Command: "fak lab status --json | fak nodeusage post --fleet -",
			Run:     runNodeUsageRefresh,
		},
		"steering": {
			Kind:    refreshRunnable,
			Command: "fak steering report",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runSteering(stdout, stderr, "report", withDryRun(nil, dryRun))
			},
		},
		"chatrelay": {
			Kind:    refreshBridge,
			Command: "fak chatrelay --endpoint URL --channel ID",
			Detail:  "long-lived bridge, not a one-shot refresh feed",
		},
	}
}

func runBuiltinScoreboardRefresh(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
	ob, err := openOutbox()
	if err != nil {
		fmt.Fprintf(stderr, "fak slack refresh scoreboard: open outbox: %v\n", err)
		return 1
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak slack refresh scoreboard: fold outbox: %v\n", err)
		return 1
	}
	verdict := "OK"
	if st.Pending > 0 || st.Dead > 0 || st.Refused > 0 || st.Corrupt > 0 {
		verdict = "ACTION"
	}
	detail := fmt.Sprintf("source=fak slack outbox status; pending=%d posted=%d dead=%d refused=%d corrupt=%d oldest_pending=%s last_drain=%s",
		st.Pending, st.Posted, st.Dead, st.Refused, st.Corrupt, ageOrDash(st.OldestPendingAgeS), ageOrDash(st.LastDrainAgeS))
	args := []string{"--kpi", "slack-outbox-pending", "--value", fmt.Sprint(st.Pending), "--verdict", verdict, "--detail", detail, "--source", "fak slack outbox status"}
	return runScoreboardPost(stdout, stderr, withDryRun(args, dryRun))
}

func runSlackNewsAudit(stdout, stderr io.Writer) int {
	var news *surfaceReport
	for _, rep := range buildSurfaceReports() {
		if rep.Name == "news" {
			news = rep
			break
		}
	}
	next := "fak slack refresh --surface news --news-title TITLE --news-file FILE"
	if news == nil || !news.Ready {
		fmt.Fprintf(stdout, "news: NEVER_POSTED_OR_UNWIRED - no readable digest; next: %s\n", next)
		return 1
	}
	runAuthChecks([]*surfaceReport{news}, "")
	if news.Auth == nil || !news.Auth.OK {
		fmt.Fprintf(stdout, "news: INACCESSIBLE - authentication failed; next: verify fak slack check --auth\n")
		return 1
	}
	age, err := lastPostAge(news, "", time.Now())
	if err != nil {
		fmt.Fprintf(stdout, "news: INACCESSIBLE_OR_EMPTY - %v; next: %s\n", err, next)
		return 1
	}
	state := "FRESH"
	if age > 8*24*time.Hour {
		state = "STALE"
	}
	fmt.Fprintf(stdout, "news: %s - last digest %s ago; source links remain in the digest body; next: %s\n", state, age.Round(time.Second), next)
	if state == "STALE" {
		return 1
	}
	return 0
}

func runSlackSurfaceAudit(stdout, stderr io.Writer, surface string, includeOutbox bool) int {
	var selected *surfaceReport
	for _, rep := range buildSurfaceReports() {
		if rep.Name == surface {
			selected = rep
			break
		}
	}
	if selected == nil {
		fmt.Fprintf(stderr, "fak slack refresh %s: surface is not registered\n", surface)
		return 2
	}
	if !selected.Ready {
		fmt.Fprintf(stdout, "%s: INCOMPLETE - token or channel unresolved\n", surface)
		if includeOutbox {
			_ = runSlackOutboxStatus(stdout, stderr, []string{"--json"})
		}
		return 1
	}
	runAuthChecks([]*surfaceReport{selected}, "")
	if selected.Auth == nil || !selected.Auth.OK {
		detail := "auth probe unavailable"
		if selected.Auth != nil && selected.Auth.Err != "" {
			detail = selected.Auth.Err
		}
		fmt.Fprintf(stdout, "%s: AUTH_FAIL - %s\n", surface, detail)
		return 1
	}
	age, err := lastPostAge(selected, "", time.Now())
	if err != nil {
		fmt.Fprintf(stdout, "%s: NO_RECENT_POST - %v\n", surface, err)
	} else {
		fmt.Fprintf(stdout, "%s: OK - last post %s ago\n", surface, age.Round(time.Second))
	}
	if includeOutbox {
		if code := runSlackOutboxStatus(stdout, stderr, []string{"--json"}); code != 0 {
			return code
		}
	}
	if err != nil {
		return 1
	}
	return 0
}

func runSlackWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the surface/refresh map as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	rows := buildSlackWalkRows()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rows, "fak slack walk")
	}
	renderSlackWalk(stdout, rows)
	return 0
}

func buildSlackWalkRows() []slackWalkRow {
	actions := slackRefreshActions()
	reports := buildSurfaceReports()
	rows := make([]slackWalkRow, 0, len(reports))
	for _, rep := range reports {
		action := actions[rep.Name]
		if action.Kind == "" {
			action.Kind = refreshManual
			action.Detail = "registered surface has no one-shot refresh action"
		}
		rows = append(rows, slackWalkRow{
			Surface:       rep.Name,
			Ready:         rep.Ready,
			Optional:      rep.Optional,
			Status:        surfaceWalkStatus(rep),
			Kind:          action.Kind,
			Command:       action.Command,
			Detail:        action.Detail,
			TokenSource:   rep.TokenSource,
			ChannelSource: rep.ChannelSource,
		})
	}
	return rows
}

func surfaceWalkStatus(rep *surfaceReport) string {
	if rep.Ready {
		return "READY"
	}
	if rep.Optional {
		return "DEFERRED"
	}
	return "INCOMPLETE"
}

func renderSlackWalk(w io.Writer, rows []slackWalkRow) {
	fmt.Fprintf(w, "fak slack walk - %d surfaces\n\n", len(rows))
	fmt.Fprintf(w, "%-12s %-11s %-13s %s\n", "SURFACE", "STATUS", "KIND", "REFRESH")
	for _, row := range rows {
		cmd := row.Command
		if cmd == "" {
			cmd = "(none)"
		}
		fmt.Fprintf(w, "%-12s %-11s %-13s %s\n", row.Surface, row.Status, row.Kind, cmd)
		if row.Detail != "" {
			fmt.Fprintf(w, "  %s\n", row.Detail)
		}
	}
}

func runSlackRefresh(stdout, stderr io.Writer, argv []string) int {
	return runSlackRefreshWithGH(stdout, stderr, argv, runSlackRefreshGH)
}

func runSlackRefreshWithGH(stdout, stderr io.Writer, argv []string, gh slackGitHubRunner) int {
	fs := flag.NewFlagSet("fak slack refresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	surfaceList := fs.String("surface", "", "comma-separated surfaces to refresh (default: all registered surfaces)")
	live := fs.Bool("live", false, "post live; default is a dry-run render")
	continueOnError := fs.Bool("continue-on-error", false, "run remaining surfaces after a refresh failure")
	asJSON := fs.Bool("json", false, "emit machine-readable refresh results")
	newsTitle := fs.String("news-title", "", "title to use when refreshing the news surface")
	newsFile := fs.String("news-file", "", "notes file to use when refreshing the news surface")
	blockersIssues := fs.String("blockers-issues", "", "gh issue-list JSON file to use when refreshing blockers")
	blockersLabel := fs.String("blockers-label", "blocked", "issue label represented by --blockers-issues")
	blockersRepoURL := fs.String("blockers-repo-url", "", "repo URL used for the blockers triage link")
	backlogChannel := fs.String("backlog-channel", "", "Slack channel for the backlog scoreboard refresh")
	if !parseFlags(fs, argv) {
		return 2
	}

	reports := buildSurfaceReports()
	selected, err := selectSlackSurfaces(reports, *surfaceList)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack refresh: %v\n", err)
		return 2
	}
	opts := slackRefreshOptions{
		NewsTitle: *newsTitle, NewsFile: *newsFile, BlockersIssues: *blockersIssues,
		BlockersLabel: *blockersLabel, BlockersRepoURL: *blockersRepoURL, BacklogChannel: *backlogChannel,
	}
	cleanup := func() {}
	if needsGitHubPayload(selected, opts.BlockersIssues) {
		payload, ghErr := gh("issue", "list", "--state", "open", "--limit", "100", "--json", "number,title,url,assignees,labels")
		if ghErr != nil {
			results := githubFailureResults(selected, ghErr)
			if *asJSON {
				if code := encodeJSONOrFail(stdout, stderr, results, "fak slack refresh"); code != 0 {
					return code
				}
			} else {
				renderSlackRefresh(stdout, results, !*live)
			}
			return refreshExit(results)
		}
		var prepErr error
		opts, cleanup, prepErr = prepareGitHubRefreshPayload(payload, opts)
		if prepErr != nil {
			fmt.Fprintf(stderr, "fak slack refresh: GITHUB_PAYLOAD_INVALID: %v\n", prepErr)
			return 2
		}
		defer cleanup()
	}
	results := refreshSelectedSurfaces(selected, !*live, *continueOnError, opts)

	if *asJSON {
		if code := encodeJSONOrFail(stdout, stderr, results, "fak slack refresh"); code != 0 {
			return code
		}
		return refreshExit(results)
	}
	renderSlackRefresh(stdout, results, !*live)
	return refreshExit(results)
}

func needsGitHubPayload(reports []*surfaceReport, blockersPath string) bool {
	for _, rep := range reports {
		if rep.Name == "backlog" || (rep.Name == "blockers" && strings.TrimSpace(blockersPath) == "") {
			return true
		}
	}
	return false
}

func decodeSlackGitHubIssues(payload []byte) ([]slackGitHubIssue, error) {
	var issues []slackGitHubIssue
	if err := json.Unmarshal(payload, &issues); err != nil {
		return nil, fmt.Errorf("decode bounded issue payload: %w", err)
	}
	return issues, nil
}

func prepareGitHubRefreshPayload(payload []byte, opts slackRefreshOptions) (slackRefreshOptions, func(), error) {
	issues, err := decodeSlackGitHubIssues(payload)
	if err != nil {
		return opts, func() {}, err
	}
	canonical, err := json.Marshal(issues)
	if err != nil {
		return opts, func() {}, err
	}
	opts.BacklogIssues = string(canonical)
	if strings.TrimSpace(opts.BlockersIssues) != "" {
		return opts, func() {}, nil
	}
	f, err := os.CreateTemp("", "fak-slack-refresh-issues-*.json")
	if err != nil {
		return opts, func() {}, fmt.Errorf("stage issue payload: %w", err)
	}
	name := f.Name()
	cleanup := func() { _ = os.Remove(name) }
	if _, err := f.Write(canonical); err != nil {
		_ = f.Close()
		cleanup()
		return opts, func() {}, fmt.Errorf("stage issue payload: %w", err)
	}
	if err := f.Close(); err != nil {
		cleanup()
		return opts, func() {}, fmt.Errorf("stage issue payload: %w", err)
	}
	opts.BlockersIssues = name
	return opts, cleanup, nil
}

func githubFailureResults(reports []*surfaceReport, err error) []slackRefreshResult {
	var results []slackRefreshResult
	for _, rep := range reports {
		if rep.Name != "blockers" && rep.Name != "backlog" {
			continue
		}
		results = append(results, slackRefreshResult{Surface: rep.Name, Status: "FAIL", ExitCode: 1, Command: slackRefreshActions()[rep.Name].Command, Detail: "GitHub issue payload unavailable; no all-clear rendered", Stderr: err.Error(), ErrorType: "GITHUB_FETCH_FAILED"})
	}
	return results
}

func backlogRefreshDetail(issues []slackGitHubIssue) string {
	if len(issues) == 0 {
		return ""
	}
	const shown = 5
	parts := make([]string, 0, shown)
	for i, issue := range issues {
		if i == shown {
			break
		}
		parts = append(parts, fmt.Sprintf("#%d %s", issue.Number, issue.Title))
	}
	if len(issues) > shown {
		parts = append(parts, fmt.Sprintf("+%d more", len(issues)-shown))
	}
	return strings.Join(parts, "; ")
}

func selectSlackSurfaces(reports []*surfaceReport, surfaceList string) ([]*surfaceReport, error) {
	byName := map[string]*surfaceReport{}
	for _, r := range reports {
		byName[r.Name] = r
	}
	if strings.TrimSpace(surfaceList) == "" {
		return reports, nil
	}
	var selected []*surfaceReport
	for _, raw := range strings.Split(surfaceList, ",") {
		name := strings.ToLower(strings.TrimSpace(raw))
		if name == "" {
			continue
		}
		rep, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("unknown surface %q (known: %s)", name, strings.Join(surfaceNames(reports), ", "))
		}
		selected = append(selected, rep)
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("--surface did not name any surfaces")
	}
	return selected, nil
}

func surfaceNames(reports []*surfaceReport) []string {
	names := make([]string, 0, len(reports))
	for _, r := range reports {
		names = append(names, r.Name)
	}
	sort.Strings(names)
	return names
}

func refreshSelectedSurfaces(reports []*surfaceReport, dryRun, continueOnError bool, opts slackRefreshOptions) []slackRefreshResult {
	actions := slackRefreshActions()
	var results []slackRefreshResult
	for _, rep := range reports {
		action := actions[rep.Name]
		res := slackRefreshResult{
			Surface: rep.Name,
			Command: action.Command,
			Detail:  action.Detail,
		}
		switch {
		case action.Run == nil:
			res.Status = "SKIP"
			if res.Detail == "" {
				res.Detail = "no one-shot refresh action"
			}
		case rep.Name == "blockers" && strings.TrimSpace(opts.BlockersIssues) == "":
			res.Status = "SKIP"
			res.Detail = "blockers refresh needs --blockers-issues from a GitHub issue-list payload"
		default:
			var out, errb bytes.Buffer
			code := action.Run(&out, &errb, dryRun, opts)
			res.Stdout = strings.TrimSpace(out.String())
			res.Stderr = strings.TrimSpace(errb.String())
			res.ExitCode = code
			if code == 0 {
				if dryRun {
					res.Status = "DRY-RUN"
				} else {
					res.Status = "OK"
				}
			} else {
				res.Status = "FAIL"
			}
		}
		results = append(results, res)
		if res.Status == "FAIL" && !continueOnError {
			break
		}
	}
	return results
}

func renderSlackRefresh(w io.Writer, results []slackRefreshResult, dryRun bool) {
	mode := "live"
	if dryRun {
		mode = "dry-run"
	}
	fmt.Fprintf(w, "fak slack refresh - %s (%d surface(s))\n", mode, len(results))
	for _, res := range results {
		fmt.Fprintf(w, "\n== %s: %s ==\n", res.Surface, res.Status)
		if res.Command != "" {
			fmt.Fprintf(w, "$ %s\n", res.Command)
		}
		if res.Detail != "" {
			fmt.Fprintf(w, "%s\n", res.Detail)
		}
		if res.Stdout != "" {
			fmt.Fprintln(w, res.Stdout)
		}
		if res.Stderr != "" {
			fmt.Fprintf(w, "stderr:\n%s\n", res.Stderr)
		}
	}
}

func refreshExit(results []slackRefreshResult) int {
	for _, res := range results {
		if res.Status == "FAIL" {
			if res.ExitCode != 0 {
				return res.ExitCode
			}
			return 1
		}
	}
	return 0
}

func withDryRun(args []string, dryRun bool) []string {
	out := append([]string{}, args...)
	if dryRun {
		out = append(out, "--dry-run")
	}
	return out
}

func runNodeUsageRefresh(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
	var snap bytes.Buffer
	if code := runLab(&snap, stderr, []string{"status", "--json"}); code != 0 {
		return code
	}
	tmp, err := os.CreateTemp("", "fak-lab-status-*.json")
	if err != nil {
		fmt.Fprintf(stderr, "fak slack refresh node-usage: %v\n", err)
		return 1
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err := tmp.Write(snap.Bytes()); err != nil {
		tmp.Close()
		fmt.Fprintf(stderr, "fak slack refresh node-usage: write snapshot: %v\n", err)
		return 1
	}
	if err := tmp.Close(); err != nil {
		fmt.Fprintf(stderr, "fak slack refresh node-usage: close snapshot: %v\n", err)
		return 1
	}
	return runNodeUsagePost(stdout, stderr, withDryRun([]string{"--fleet", name}, dryRun))
}
