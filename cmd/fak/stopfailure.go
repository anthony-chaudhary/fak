package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stopfailure"
)

func cmdStopFailure(argv []string) { os.Exit(runStopFailure(os.Stdout, os.Stderr, argv)) }

func runStopFailure(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		stopFailureUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "plan":
		return runStopFailurePlan(stdout, stderr, argv[1:])
	case "reset-stale":
		return runStopFailureResetStale(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		stopFailureUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak stopfailure: unknown subcommand %q\n", argv[0])
		stopFailureUsage(stderr)
		return 2
	}
}

func stopFailureUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak stopfailure plan [--root DIR] [--since-hours N] [--recent-hours N] [--claude-home DIR] [--namespace NS] [--limit N] [--json]")
	fmt.Fprintln(w, "       fak stopfailure reset-stale [--root DIR] [--since-hours N] [--recent-hours N] [--claude-home DIR] [--namespace NS] [--limit N] [--apply] [--json]")
}

func runStopFailurePlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak stopfailure plan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repository root containing .dos/stop-failures")
	sinceHours := fs.Int("since-hours", 24, "marker mtime lookback in hours; 0 means all history")
	recentHours := fs.Int("recent-hours", stopfailure.DefaultRecentWindowHours, "recent active marker threshold in hours")
	claudeHome := fs.String("claude-home", "", "user home containing .claude* roots; default FLEET_USER_HOME/USERPROFILE/home")
	namespace := fs.String("namespace", stopfailure.DefaultTranscriptNamespace, "Claude projects namespace used for transcript origin lookup")
	limit := fs.Int("limit", 20, "maximum rows per settlement action in output; 0 means all")
	asJSON := fs.Bool("json", false, "emit JSON")
	nowFlag := fs.String("now", "", "override current time as RFC3339 for deterministic tests")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	now, ok := parseStopFailureNow(*nowFlag, stderr)
	if !ok {
		return 2
	}
	plan, err := stopfailure.BuildPlan(stopfailure.Options{
		Root:                *root,
		Now:                 now,
		RecentWindow:        time.Duration(*recentHours) * time.Hour,
		SinceWindow:         time.Duration(*sinceHours) * time.Hour,
		Limit:               *limit,
		ClaudeHome:          *claudeHome,
		TranscriptNamespace: *namespace,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak stopfailure plan: %v\n", err)
		return 1
	}
	if *asJSON {
		return writeStopFailureJSON(stdout, stderr, plan)
	}
	printStopFailurePlan(stdout, plan)
	return 0
}

func runStopFailureResetStale(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak stopfailure reset-stale", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repository root containing .dos/stop-failures")
	sinceHours := fs.Int("since-hours", 24, "marker mtime lookback in hours; 0 means all history")
	recentHours := fs.Int("recent-hours", stopfailure.DefaultRecentWindowHours, "recent active marker threshold in hours")
	claudeHome := fs.String("claude-home", "", "user home containing .claude* roots; default FLEET_USER_HOME/USERPROFILE/home")
	namespace := fs.String("namespace", stopfailure.DefaultTranscriptNamespace, "Claude projects namespace used for transcript origin lookup")
	limit := fs.Int("limit", 0, "maximum stale markers to reset; 0 means all candidates")
	apply := fs.Bool("apply", false, "write consecutive=0 to stale markers; omitted means dry-run")
	asJSON := fs.Bool("json", false, "emit JSON")
	nowFlag := fs.String("now", "", "override current time as RFC3339 for deterministic tests")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	now, ok := parseStopFailureNow(*nowFlag, stderr)
	if !ok {
		return 2
	}
	result, err := stopfailure.ResetStale(stopfailure.Options{
		Root:                *root,
		Now:                 now,
		RecentWindow:        time.Duration(*recentHours) * time.Hour,
		SinceWindow:         time.Duration(*sinceHours) * time.Hour,
		Limit:               *limit,
		ClaudeHome:          *claudeHome,
		TranscriptNamespace: *namespace,
	}, *apply)
	if err != nil {
		fmt.Fprintf(stderr, "fak stopfailure reset-stale: %v\n", err)
		return 1
	}
	if *asJSON {
		return writeStopFailureJSON(stdout, stderr, result)
	}
	printStopFailureReset(stdout, result)
	if len(result.Errors) > 0 {
		return 1
	}
	return 0
}

func parseStopFailureNow(value string, stderr io.Writer) (time.Time, bool) {
	if strings.TrimSpace(value) == "" {
		return time.Time{}, true
	}
	t, err := time.Parse(time.RFC3339, value)
	if err != nil {
		fmt.Fprintf(stderr, "fak stopfailure: invalid --now %q: %v\n", value, err)
		return time.Time{}, false
	}
	return t.UTC(), true
}

func writeStopFailureJSON(stdout, stderr io.Writer, v any) int {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak stopfailure: marshal JSON: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(b))
	return 0
}

func printStopFailurePlan(w io.Writer, plan stopfailure.Plan) {
	recent := plan.Counts[stopfailure.ActionRecentReview]
	stale := plan.Counts[stopfailure.ActionStaleReset]
	archive := plan.Counts[stopfailure.ActionStaleMarkerOnlyArchive]
	status := "OK"
	if recent > 0 || stale > 0 || archive > 0 || plan.Malformed > 0 {
		status = "WARN"
	}
	fmt.Fprintf(w, "fak stopfailure plan: %s markers=%d ignored_old=%d recent_review=%d stale_reset=%d marker_only_archive=%d healed=%d zero=%d malformed=%d\n",
		status,
		plan.Markers,
		plan.IgnoredOld,
		recent,
		stale,
		archive,
		plan.Counts[stopfailure.ActionHealedNonzero],
		plan.Counts[stopfailure.ActionZeroTotal],
		plan.Malformed,
	)
	printStopFailureRows(w, "recent review", plan.Candidates[stopfailure.ActionRecentReview])
	printStopFailureRows(w, "stale reset", plan.Candidates[stopfailure.ActionStaleReset])
	printStopFailureRows(w, "marker-only archive", plan.Candidates[stopfailure.ActionStaleMarkerOnlyArchive])
	if stale > 0 {
		fmt.Fprintln(w, "next: fak stopfailure reset-stale --apply")
	}
}

func printStopFailureReset(w io.Writer, result stopfailure.ResetResult) {
	mode := "DRY-RUN"
	if result.Applied {
		mode = "APPLIED"
	}
	fmt.Fprintf(w, "fak stopfailure reset-stale: %s candidates=%d updated=%d errors=%d\n", mode, len(result.Candidates), len(result.Updated), len(result.Errors))
	rows := result.Candidates
	if result.Applied {
		rows = result.Updated
	}
	printStopFailureRows(w, "stale reset", rows)
	if !result.Applied && len(result.Candidates) > 0 {
		fmt.Fprintln(w, "next: rerun with --apply to set consecutive=0 on stale markers only")
	}
	for _, err := range result.Errors {
		fmt.Fprintf(w, "error: %s\n", err)
	}
}

func printStopFailureRows(w io.Writer, label string, rows []stopfailure.Marker) {
	if len(rows) == 0 {
		return
	}
	fmt.Fprintf(w, "  %s candidates:\n", label)
	for _, row := range rows {
		fmt.Fprintf(w, "    %s total=%d consecutive=%d age=%s origin=%s project=%s action=%s\n",
			row.MarkerPath,
			row.Total,
			row.Consecutive,
			formatStopFailureAge(row.AgeSeconds),
			row.Origin,
			row.TranscriptProject,
			row.SettlementAction,
		)
	}
}

func formatStopFailureAge(seconds int64) string {
	if seconds < 0 {
		seconds = 0
	}
	d := time.Duration(seconds) * time.Second
	if d >= time.Hour {
		return d.Truncate(time.Minute).String()
	}
	return d.Truncate(time.Second).String()
}
