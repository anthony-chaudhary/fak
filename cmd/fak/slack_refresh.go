package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
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
	Surface  string `json:"surface"`
	Status   string `json:"status"`
	ExitCode int    `json:"exit_code,omitempty"`
	Command  string `json:"command"`
	Detail   string `json:"detail,omitempty"`
	Stdout   string `json:"stdout,omitempty"`
	Stderr   string `json:"stderr,omitempty"`
}

func slackRefreshActions() map[string]slackRefreshAction {
	return map[string]slackRefreshAction{
		"scoreboard": {
			Kind:    refreshInputNeeded,
			Command: "fak scoreboard post --from FILE | --kpi NAME ...",
			Detail:  "scoreboard needs an explicit scorecard payload or KPI",
		},
		"product": {
			Kind:    refreshRunnable,
			Command: "fak product post --status",
			Run: func(stdout, stderr io.Writer, dryRun bool, _ slackRefreshOptions) int {
				return runProductPost(stdout, stderr, withDryRun([]string{"--status"}, dryRun))
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
			Kind:    refreshInputNeeded,
			Command: "gh issue list ... then fak scoreboard post --channel $FAK_BACKLOG_CHANNEL --kpi backlog-triage ...",
			Detail:  "backlog needs the GitHub issue-list/readout payload from the workflow",
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
			Command: "fak news post --title TITLE --notes-file FILE",
			Detail:  "requires a human/agent-curated, source-linked digest body",
			Run: func(stdout, stderr io.Writer, dryRun bool, opts slackRefreshOptions) int {
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

func runSlackWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak slack walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the surface/refresh map as JSON")
	if err := fs.Parse(argv); err != nil {
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
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	reports := buildSurfaceReports()
	selected, err := selectSlackSurfaces(reports, *surfaceList)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack refresh: %v\n", err)
		return 2
	}
	opts := slackRefreshOptions{
		NewsTitle:       *newsTitle,
		NewsFile:        *newsFile,
		BlockersIssues:  *blockersIssues,
		BlockersLabel:   *blockersLabel,
		BlockersRepoURL: *blockersRepoURL,
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
		case rep.Name == "news" && (strings.TrimSpace(opts.NewsTitle) == "" || strings.TrimSpace(opts.NewsFile) == ""):
			res.Status = "SKIP"
			res.Detail = "news refresh needs --news-title and --news-file"
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
