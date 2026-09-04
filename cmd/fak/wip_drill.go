package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func runWipDrill(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip drill", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("repo", ".", "repository root")
	cFlag := fs.String("C", "", "repository root (alias for --repo)")
	jsonOut := fs.Bool("json", false, "emit the recovery drill report as JSON")
	limit := fs.Int("limit", 5, "maximum number of checkpoints to drill (0 for all)")
	session := fs.String("session", "", "specific session id to drill")

	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	targetRepo := *repo
	if targetRepo == "." && *cFlag != "" {
		targetRepo = *cFlag
	}
	abs, err := filepath.Abs(targetRepo)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip drill: %v\n", err)
		return 1
	}

	opts := wipref.DrillOptions{
		Limit:   *limit,
		Session: *session,
	}

	start := time.Now()
	report, err := wipref.RunRecoveryDrill(context.Background(), abs, opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip drill: %v\n", err)
		return 1
	}
	totalDuration := time.Since(start).Milliseconds()

	if *jsonOut {
		b, err := report.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak wip drill: json marshal: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
		if report.FailureCount > 0 {
			return 1
		}
		return 0
	}

	// Human output leading with drilled count, pass/fail status, and recovery duration
	fmt.Fprintf(stdout, "WIP RECOVERY DRILL — %d drilled: %d PASS, %d FAIL (%dms)\n",
		report.TotalDrilled, report.SuccessCount, report.FailureCount, totalDuration)
	fmt.Fprintf(stdout, "repo: %s\n", report.Repo)
	fmt.Fprintf(stdout, "main tree preserved: %t\n", report.MainTreePreserved)

	for _, r := range report.Results {
		shortCommit := r.CommitSHA
		if len(shortCommit) > 10 {
			shortCommit = shortCommit[:10]
		}
		if r.Status == "PASS" {
			fmt.Fprintf(stdout, "  PASS  %s (commit %s, %d files, %dms)\n",
				r.Ref, shortCommit, r.RestoredPathCount, r.DurationMs)
		} else {
			fmt.Fprintf(stdout, "  FAIL  %s (status %s, %dms) — %s\n",
				r.Ref, r.Status, r.DurationMs, r.Detail)
		}
	}

	if report.FailureCount > 0 {
		return 1
	}
	return 0
}
