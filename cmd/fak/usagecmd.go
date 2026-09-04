package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// fak usage — the read side of the CLI-invocation journal internal/usagelog
// writes at process exit (see usagelog_record.go for what gets recorded and
// the known os.Exit coverage gap). Folds the recorded rows and prints how fak
// itself has been invoked: total/error counts, timing, and optionally a
// per-verb breakdown or the raw fold as JSON.
func cmdUsage(args []string) {
	fs := flag.NewFlagSet("usage", flag.ExitOnError)
	sinceStr := fs.String("since", "", "only fold rows within this long of now (Go duration, e.g. 24h, 30m)")
	byVerb := fs.Bool("by-verb", false, "print the per-verb breakdown table")
	gitOps := fs.Bool("git-ops", false, "print commit/sweep/sync latency split by terminal outcome")
	guardDisable := fs.Bool("guard-disable", false, "print scoped break-glass launcher outcome counts")
	childResource := fs.Bool("child-resource", false, "print child resource containment weekly invocation counts")
	asJSON := fs.Bool("json", false, "print the fold as JSON instead of text")
	topN := fs.Int("top", 0, "how many recent rows to include in the fold (0 = usagelog's default)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak usage [--since DUR] [--by-verb] [--git-ops] [--guard-disable] [--child-resource] [--json] [--top N]")
		fmt.Fprintln(os.Stderr, "  reads the usage journal (FAK_USAGE_LOG_PATH, else the per-user default)")
		fmt.Fprintln(os.Stderr, "  and prints how fak itself has been invoked.")
	}
	_ = fs.Parse(args)
	if *guardDisable {
		if *sinceStr != "" || *byVerb || *gitOps || *topN != 0 {
			fmt.Fprintln(os.Stderr, "fak usage: --guard-disable accepts only --json; its ledger already folds by ISO week")
			os.Exit(2)
		}
		path, pathErr := guardDisableUsageDefaultPath()
		if code := runGuardDisableUsage(os.Stdout, os.Stderr, path, pathErr, *asJSON); code != 0 {
			os.Exit(code)
		}
		return
	}
	if *childResource {
		if *sinceStr != "" || *byVerb || *gitOps || *topN != 0 {
			fmt.Fprintln(os.Stderr, "fak usage: --child-resource accepts only --json; its ledger already folds by ISO week")
			os.Exit(2)
		}
		path := guardChildResourceUsagePath()
		if code := runChildResourceUsage(os.Stdout, os.Stderr, path, *asJSON); code != 0 {
			os.Exit(code)
		}
		return
	}

	var sinceUnixNano int64
	if *sinceStr != "" {
		d, err := time.ParseDuration(*sinceStr)
		if err != nil || d < 0 {
			fmt.Fprintf(os.Stderr, "fak usage: invalid --since %q: %v\n", *sinceStr, err)
			os.Exit(2)
		}
		sinceUnixNano = time.Now().Add(-d).UnixNano()
	}

	path := usageLogPath()
	rows, err := usagelog.ReadRows(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak usage: %v\n", err)
		os.Exit(1)
	}
	fold := usagelog.FoldRows(rows, usagelog.FoldOptions{SinceUnixNano: sinceUnixNano, TopN: *topN})
	gitOpStats := fold.ByOperationOutcome

	if *asJSON {
		if *gitOps {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			if err := enc.Encode(gitOpStats); err != nil {
				fmt.Fprintf(os.Stderr, "fak usage: %v\n", err)
				os.Exit(1)
			}
			return
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(fold); err != nil {
			fmt.Fprintf(os.Stderr, "fak usage: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if fold.Total == 0 {
		fmt.Printf("fak usage: %s — no rows recorded yet\n", path)
		return
	}
	fmt.Printf("fak usage: %s\n", path)
	fmt.Printf("  total: %d   errors: %d (%.1f%%)   p50: %dms\n",
		fold.Total, fold.Errors, 100*float64(fold.Errors)/float64(fold.Total), fold.P50MS)
	if *byVerb {
		fmt.Println("  by verb:")
		for _, v := range fold.ByVerb {
			fmt.Printf("    %-28s count=%-6d errors=%-4d p50=%dms\n", v.Verb, v.Count, v.Errors, v.P50MS)
		}
	}
	if *gitOps {
		fmt.Println("  git operations by terminal outcome (process observation, not a downstream-effect witness):")
		if len(gitOpStats) == 0 {
			fmt.Println("    no commit/sweep/sync rows recorded yet")
		}
		for _, stat := range gitOpStats {
			fmt.Printf("    %-28s %-9s count=%-6d p50=%dms\n", stat.Operation, stat.Outcome, stat.Count, stat.P50MS)
		}
	}
}
