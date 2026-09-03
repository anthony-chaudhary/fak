package main

// `fak session-audit reconcile` — the #10673 started-vs-terminal accounting
// lens over the same native Codex rollout store `fak session-audit codex` and
// `fak session-audit posttool` read. It folds every rollout through the
// exactly-once reconciler and reports TWO deltas for the same corpus: the RAW
// started-minus-observed-terminal gap (the honest producer-side number — the
// audited 498 vs 288 shape) and the post-synthesis RESIDUAL (what the fold's
// Superseded/ProcessDeath synthesis could not close — a live turn stays open).
// Reports are aggregate counts, deltas, percentages, and closed-class
// breakdowns only — no paths, session ids, prompts, or message content.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexlifecycle"
)

func runSessionAuditReconcile(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session-audit reconcile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the reconciliation report as JSON")
	root := fs.String("root", "", "rollout store root (default ~/.codex/sessions, honoring CODEX_HOME)")
	cwdFilter := fs.String("cwd", "", "keep only rollouts whose session cwd matches this path")
	here := fs.Bool("here", false, "shorthand: --cwd <current working directory>")
	freshMins := fs.Int("fresh-mins", 120, "a rollout younger than this is treated as possibly live; a closed past window: fak session-audit reconcile --json --here --fresh-mins 10080 --max 500")
	max := fs.Int("max", 0, "cap scanned rollouts, newest first (0 = all)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak session-audit reconcile [--json] [--root DIR] [--cwd DIR|--here] [--fresh-mins N] [--max N]")
		return 2
	}
	dir := *root
	if dir == "" {
		base := os.Getenv("CODEX_HOME")
		if base == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				fmt.Fprintf(stderr, "fak session-audit reconcile: no home dir and no --root: %v\n", err)
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
			fmt.Fprintf(stderr, "fak session-audit reconcile: --here: %v\n", err)
			return 1
		}
		cwd = wd
	}
	rep, err := codexlifecycle.ScanReconcileCorpus(dir, codexlifecycle.ScanOptions{
		CWD:         cwd,
		FreshWithin: time.Duration(*freshMins) * time.Minute,
		Limit:       *max,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit reconcile: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak session-audit reconcile: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	writeReconcileText(stdout, rep)
	return 0
}

func writeReconcileText(w io.Writer, r codexlifecycle.ReconcileReport) {
	fmt.Fprintf(w, "reconcile — root=%s scanned=%d unreadable=%d all_starts_typed=%t\n",
		r.Root, r.Scanned, r.Unreadable, r.AllStartsTyped)
	t := r.Totals
	fmt.Fprintf(w, "events task_started=%d task_complete=%d turn_aborted=%d\n",
		t.TaskStarted, t.TaskComplete, t.TurnAborted)
	fmt.Fprintf(w, "raw_unaccounted=%d (%.1f%% of starts)\n",
		t.RawUnaccounted, t.RawUnaccountedPct)
	fmt.Fprintf(w, "outcomes complete=%d aborted=%d superseded=%d process_death=%d live=%d closed=%d\n",
		t.Complete, t.Aborted, t.Superseded, t.ProcessDeath, t.Live, t.ClosedTotal)
	fmt.Fprintf(w, "residual_unaccounted=%d (%.1f%% of starts)\n",
		t.ResidualUnaccounted, t.ResidualPct)
	fmt.Fprintf(w, "integrity orphans=%d reused=%d multiply_terminated=%d unclassified_after=%d\n",
		t.Orphans, t.Reused, t.MultiplyTerminated, t.UnclassifiedAfter)
	for _, key := range r.ProviderVersions() {
		b := r.ByProvider[key]
		fmt.Fprintf(w, "provider %s: task_started=%d task_complete=%d turn_aborted=%d raw_unaccounted=%d (%.1f%%) closed=%d residual=%d\n",
			key, b.TaskStarted, b.TaskComplete, b.TurnAborted, b.RawUnaccounted, b.RawUnaccountedPct, b.ClosedTotal, b.ResidualUnaccounted)
	}
}
