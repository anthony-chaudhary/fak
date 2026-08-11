package main

// `fak session-audit codex` — the native Codex half of the session-analytics
// surface (#4767). The Claude lens (#2365) reads Claude project transcripts; this
// subcommand streams the native Codex rollout store (~/.codex/sessions/**/*.jsonl)
// through internal/codexlifecycle's typed analytics: task critical-path
// decomposition, closed tool-outcome vocabulary (expected negatives and control
// exits OUT of the failure count), percentiles, ranked outliers, and actionable
// findings. Reports are scrubbed — ids, classes, reasons, hashed signatures —
// never raw commands or result bodies.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexlifecycle"
)

func runSessionAuditCodex(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session-audit codex", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the corpus report as JSON")
	root := fs.String("root", "", "rollout store root (default ~/.codex/sessions, honoring CODEX_HOME)")
	cwdFilter := fs.String("cwd", "", "keep only rollouts whose session cwd matches this path")
	here := fs.Bool("here", false, "shorthand: --cwd <current working directory>")
	freshMins := fs.Int("fresh-mins", 120, "a rollout younger than this is treated as possibly live; seven-day audit: fak session-audit codex --json --here --fresh-mins 10080 --top 30 --max 500")
	top := fs.Int("top", 10, "ranked critical-path outliers to keep")
	max := fs.Int("max", 0, "cap scanned rollouts, newest first (0 = all)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak session-audit codex [--json] [--root DIR] [--cwd DIR|--here] [--fresh-mins N] [--top N] [--max N]")
		return 2
	}
	dir := *root
	if dir == "" {
		base := os.Getenv("CODEX_HOME")
		if base == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				fmt.Fprintf(stderr, "fak session-audit codex: no home dir and no --root: %v\n", err)
				return 1
			}
			base = filepath.Join(home, ".codex")
		}
		dir = filepath.Join(base, "sessions")
	}
	cwd := *cwdFilter
	if *here {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak session-audit codex: --here: %v\n", err)
			return 1
		}
		cwd = wd
	}
	rep, err := codexlifecycle.ScanAnalyticsCorpus(dir, codexlifecycle.ScanOptions{
		CWD:         cwd,
		FreshWithin: time.Duration(*freshMins) * time.Minute,
		Limit:       *max,
	}, *top)
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit codex: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak session-audit codex: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	writeCodexCorpusText(stdout, rep)
	return 0
}

func writeCodexCorpusText(w io.Writer, c codexlifecycle.AnalyticsCorpus) {
	fmt.Fprintf(w, "codex corpus — root=%s sessions=%d unreadable=%d\n", c.Root, c.Sessions, c.Unreadable)
	fmt.Fprintf(w, "tasks=%d completed=%d tool_calls=%d\n", c.Tasks, c.Completed, c.ToolCalls)
	r := c.FreshHeadlessResume
	fmt.Fprintf(w, "fresh_headless_resume started=%d useful_work=%d completed=%d crashed=%d superseded=%d\n",
		r.Started, r.UsefulWorkReached, r.Completed, r.Crashed, r.Superseded)
	if len(r.FailureReasons) > 0 {
		keys := make([]string, 0, len(r.FailureReasons))
		for reason := range r.FailureReasons {
			keys = append(keys, reason)
		}
		sort.Strings(keys)
		for _, reason := range keys {
			fmt.Fprintf(w, "  resume_failure %-42s %d\n", reason, r.FailureReasons[reason])
		}
	}
	fmt.Fprintf(w, "duration_s  n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		c.Duration.N, c.Duration.P50, c.Duration.P90, c.Duration.P95, c.Duration.P99, c.Duration.Max)
	fmt.Fprintf(w, "ttft_s      n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		c.TTFT.N, c.TTFT.P50, c.TTFT.P90, c.TTFT.P95, c.TTFT.P99, c.TTFT.Max)
	fmt.Fprintf(w, "failure_calls=%d (expected_negative=%d control_exit=%d excluded)\n",
		c.HardFailureCount(), c.Classes[codexlifecycle.ToolExpectedNegative], c.Classes[codexlifecycle.ToolControlExit])
	rows := c.Reasons
	if len(rows) > 12 {
		rows = rows[:12]
	}
	for _, r := range rows {
		fmt.Fprintf(w, "  reason %-32s %-20s %d\n", r.Reason, string(r.Class), r.Count)
	}
	for _, o := range c.TopTasks {
		fmt.Fprintf(w, "  top task %s/%s %s %.0fs idle=%.0fs top=%v\n",
			o.Session, o.TurnID, string(o.Outcome), o.DurationS, o.IdleS, o.Top)
	}
	fmt.Fprintf(w, "timeout_kills=%d sleep_polls=%d stall_gaps=%d\n", c.TimeoutKills, c.SleepPolls, c.StallGaps)
	for _, f := range c.Findings {
		fmt.Fprintf(w, "  finding %-36s %6d  %s\n", f.Reason, f.Count, f.Action)
	}
}
