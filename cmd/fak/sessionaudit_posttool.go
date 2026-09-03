package main

// `fak session-audit posttool` — the #10662 post-tool latency lens over the same
// native Codex rollout store `fak session-audit codex` reads. It folds every
// `tool_result_recorded → next_model_item` interval into PostToolSpans and
// reports gap percentiles overall, by context band, and by call ordinal, with
// the tool-execution percentiles alongside as the control that separates genuine
// tool slowness from post-tool model latency. Reports are scrubbed: opaque ids,
// tool names, closed tokens, and numbers only.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/codexlifecycle"
)

func runSessionAuditPosttool(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session-audit posttool", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the post-tool latency report as JSON")
	root := fs.String("root", "", "rollout store root (default ~/.codex/sessions, honoring CODEX_HOME)")
	cwdFilter := fs.String("cwd", "", "keep only rollouts whose session cwd matches this path")
	here := fs.Bool("here", false, "shorthand: --cwd <current working directory>")
	max := fs.Int("max", 0, "cap scanned rollouts, newest first (0 = all)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak session-audit posttool [--json] [--root DIR] [--cwd DIR|--here] [--max N]")
		return 2
	}
	dir := *root
	if dir == "" {
		base := os.Getenv("CODEX_HOME")
		if base == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				fmt.Fprintf(stderr, "fak session-audit posttool: no home dir and no --root: %v\n", err)
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
			fmt.Fprintf(stderr, "fak session-audit posttool: --here: %v\n", err)
			return 1
		}
		cwd = wd
	}
	rep, err := codexlifecycle.ScanPostToolCorpus(dir, codexlifecycle.ScanOptions{
		CWD:   cwd,
		Limit: *max,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit posttool: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak session-audit posttool: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	writePostToolText(stdout, rep)
	return 0
}

func writePostToolText(w io.Writer, r codexlifecycle.PostToolReport) {
	fmt.Fprintf(w, "posttool — root=%s sessions=%d unreadable=%d spans=%d tail_skipped=%d\n",
		r.Root, r.Sessions, r.Unreadable, r.Spans, r.TailSkipped)
	fmt.Fprintf(w, "gap_s   n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		r.Gap.N, r.Gap.P50, r.Gap.P90, r.Gap.P95, r.Gap.P99, r.Gap.Max)
	fmt.Fprintf(w, "tool_s  n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f p99=%-8.1f max=%.1f\n",
		r.ToolMS.N, r.ToolMS.P50, r.ToolMS.P90, r.ToolMS.P95, r.ToolMS.P99, r.ToolMS.Max)
	fmt.Fprintf(w, "over30s=%.1f%% stall_spans=%d compaction_in_gap=%d\n",
		100*r.Over30sShare, r.StallSpans, r.CompactionInGap)
	for _, b := range r.ByBand {
		fmt.Fprintf(w, "band %-12s n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f over30s=%-7.1f%% tool_p50=%.1f\n",
			b.Key, b.N, b.P50, b.P90, b.P95, 100*float64(b.Over30s)/float64(b.N), b.ToolP50)
	}
	for _, b := range r.ByOrdinal {
		fmt.Fprintf(w, "ordinal %-9s n=%-6d p50=%-8.1f p90=%-8.1f p95=%-8.1f over30s=%-7.1f%% tool_p50=%.1f\n",
			b.Key, b.N, b.P50, b.P90, b.P95, 100*float64(b.Over30s)/float64(b.N), b.ToolP50)
	}
}
