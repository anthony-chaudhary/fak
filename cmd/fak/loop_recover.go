package main

// loop_recover.go — `fak loop recover`, the operator front door to the deterministic RECOVERY
// decision (internal/looprecover). `fak loop status` folds the ledger into a per-LOOP snapshot;
// recover answers the cross-run question status does not: which dispatched runs STARTED but were
// never finished or witnessed — the orphaned and unwitnessed work that should be re-dispatched or
// re-verified rather than left silently abandoned?
//
//	fak loop recover                       # the recovery worklist from the default ledger
//	fak loop recover --stale-min 30 --json # machine-readable, 30-min orphan window
//
// The decision is PURE (internal/looprecover.Plan): same run facts + clock in, same worklist
// out. This shell does only the I/O the leaf must not: read the durable loop ledger
// (loopmgr.Load), fold the events sharing a run id into one RunFact, supply the clock and the
// stale window, call Plan, and render the worklist as an aligned table or raw JSON. The same
// leaf/shell split resume_scan.go and dispatch_order.go use.
//
// v1 is ledger-only: it presumes a started run silent past the stale window is orphaned (a
// crash, a rate limit, a timeout). It does not yet probe the worker pid for confirmed liveness —
// the leaf already accepts that stronger signal (RunFact.WorkerKnown/WorkerLive); wiring a pid
// probe is a later, more precise rung.
//
// LEDGER INTEGRITY (separation of concerns). The loop ledger is an append-only, hash-chained
// AUDIT LOG; a concurrent double-append can fork the seq chain. recover must NOT die on a fork
// (the strict loopmgr.Load refuses the whole file) and must NEVER rewrite the audit log to
// "fix" it. So it loads via the TOLERANT loopmgr.LoadPrefix — recover the longest valid prefix,
// surface the break (line/seq/reason/recovered) as a FINDING (logging), and keep planning from
// the prefix (default working behavior). This mirrors runLoopHealth (loop.go).

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
)

// runLoopRecover reads the loop ledger, classifies every dispatched run, and renders the
// orphaned-and-unwitnessed recovery worklist. Returns the process exit code (0 ok, 1 runtime
// error, 2 usage error).
func runLoopRecover(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger path")
	staleMin := fs.Int("stale-min", 45, "a started run silent this many minutes (worker liveness unknown) is presumed orphaned (-1 disables)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for the silence math (0 = current time)")
	all := fs.Bool("all", false, "also list running, complete, and failed runs (not just the recovery worklist)")
	asJSON := fs.Bool("json", false, "emit the raw Result JSON instead of the human table")
	controlPane := fs.Bool("control-pane", false, "emit the garden control-pane envelope (ok/verdict/reason) for the garden bundle")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop recover: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// Tolerant load: recover the longest valid prefix of the append-only ledger and carry
	// the integrity break (if any) as a finding. A forked seq chain (a concurrent double
	// append) no longer takes recover down — it recovers what it can and reports the break.
	events, integ, err := loopmgr.LoadPrefix(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak loop recover: %v\n", err)
		return 1
	}
	if integ.Broken {
		fmt.Fprintf(stderr, "fak loop recover: ledger integrity break at line %d (seq %d): %s (recovered %d event(s); planning from the recovered prefix — the audit log is left intact)\n",
			integ.AtLine, integ.AtSeq, integ.Reason, integ.Recovered)
	}
	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	staleSec := int64(*staleMin) * 60
	if *staleMin < 0 {
		staleSec = -1
	}

	res := looprecover.Plan(looprecover.Input{
		Runs:         foldRuns(events),
		NowUnix:      now,
		StaleSeconds: staleSec,
	})

	if *controlPane {
		return encodeJSONOrFail(stdout, stderr, recoverControlPane(res, integ), "fak loop recover")
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, res, "fak loop recover")
	}
	renderLoopRecover(stdout, *ledger, res, *all)
	return 0
}

// recoverControlPane projects a recovery Result onto the garden-bundle control-pane envelope
// (ok/verdict/reason — the only keys gardenbundle.Fold reads). A found orphan is the pass
// WORKING, so ok is always true (advisory ACTION, never a red); a forked ledger is folded into
// the reason so the garden surface sees the integrity break too. The raw Result stays available
// under --json for the orphan worklist itself.
func recoverControlPane(res looprecover.Result, integ loopmgr.Integrity) map[string]any {
	recover := res.OrphanedCount + res.UnwitnessedCount
	verdict := "OK"
	reason := fmt.Sprintf("no runs to recover (%d running, %d complete, %d failed)",
		res.RunningCount, res.CompleteCount, res.FailedCount)
	if recover > 0 {
		verdict = "ACTION"
		reason = fmt.Sprintf("%d orphaned, %d unwitnessed run(s) to recover", res.OrphanedCount, res.UnwitnessedCount)
	}
	if integ.Broken {
		reason += fmt.Sprintf("; ledger integrity break at line %d (recovered %d event(s))", integ.AtLine, integ.Recovered)
		if verdict == "OK" {
			verdict = "ACTION"
		}
	}
	return map[string]any{
		"schema":            "fak.loop-recover-control-pane.v1",
		"ok":                true,
		"verdict":           verdict,
		"reason":            reason,
		"orphaned_count":    res.OrphanedCount,
		"unwitnessed_count": res.UnwitnessedCount,
		"ledger_broken":     integ.Broken,
	}
}

// foldRuns groups the ledger events sharing a run id into one RunFact each, keeping only runs
// that actually started (or reached a terminal state) — the runs recovery reasons about. A
// loop-level event with no run id (armed/fire/admit before a worker launches) is skipped.
func foldRuns(events []loopmgr.Event) []looprecover.RunFact {
	type acc struct {
		f    looprecover.RunFact
		seen bool
	}
	byRun := map[string]*acc{}
	order := []string{}
	for _, ev := range events {
		if ev.RunID == "" {
			continue
		}
		a := byRun[ev.RunID]
		if a == nil {
			a = &acc{f: looprecover.RunFact{RunID: ev.RunID, LoopID: ev.LoopID}}
			byRun[ev.RunID] = a
			order = append(order, ev.RunID)
		}
		if sec := ev.TSUnixNano / 1e9; sec > a.f.LastEventUnix {
			a.f.LastEventUnix = sec
		}
		if ev.Summary != "" {
			a.f.Unit = ev.Summary // the most recent non-empty summary labels the run
		}
		switch ev.Kind {
		case loopmgr.EventStart:
			a.f.Started, a.seen = true, true
		case loopmgr.EventEnd:
			a.f.Ended, a.seen = true, true
		case loopmgr.EventWitness:
			a.f.Witnessed, a.seen = true, true
		}
		switch ev.Status {
		case loopmgr.StatusWitnessedDone:
			a.f.Witnessed, a.seen = true, true
		case loopmgr.StatusClaimedDone:
			a.f.Claimed, a.seen = true, true
		case loopmgr.StatusFailed:
			a.f.Failed, a.seen = true, true
		case loopmgr.StatusCanceled:
			a.f.Canceled, a.seen = true, true
		}
	}
	out := make([]looprecover.RunFact, 0, len(order))
	for _, id := range order {
		if a := byRun[id]; a.seen {
			out = append(out, a.f)
		}
	}
	return out
}

// renderLoopRecover prints the recovery worklist as an aligned table: the orphaned and
// unwitnessed runs (with --all, the rest too), then the actionable worklist summary.
func renderLoopRecover(w io.Writer, ledger string, r looprecover.Result, all bool) {
	fmt.Fprintf(w, "loop recover — %s\n", ledger)
	fmt.Fprintf(w, "  %d orphaned  %d unwitnessed  %d running  %d complete  %d failed\n\n",
		r.OrphanedCount, r.UnwitnessedCount, r.RunningCount, r.CompleteCount, r.FailedCount)

	show := func(d looprecover.Disposition) bool {
		if all {
			return true
		}
		return d == looprecover.DispOrphaned || d == looprecover.DispUnwitnessed
	}
	any := false
	for _, x := range r.Runs {
		if !show(x.Disposition) {
			continue
		}
		if !any {
			fmt.Fprintf(w, "%-12s %-11s %10s %-20s %s\n", "run", "disposition", "age", "loop", "unit")
			any = true
		}
		fmt.Fprintf(w, "%-12s %-11s %10s %-20s %s\n",
			shortRunID(x.RunID), x.Disposition, agoString(x.AgeSeconds), truncField(x.LoopID, 20), truncField(x.Unit, 48))
	}
	if !any {
		fmt.Fprintln(w, "no runs to recover — every dispatched run is complete or in progress.")
		return
	}

	if len(r.Recover) > 0 {
		fmt.Fprintf(w, "\nrecover %d run(s): re-dispatch the orphaned units and re-verify the unwitnessed ones.\n", len(r.Recover))
		fmt.Fprintln(w, "  (orphaned-by-staleness is a presumption; pass --now/--stale-min to tune, or wire a pid probe for certainty)")
	}
}

// compactDuration renders a non-negative age in seconds compactly (Ns / Nm / Nh / Nd).
// Callers handle the negative/zero sentinel themselves — it differs by surface ("0s" vs
// "unknown") — so this shared core covers only the positive bands.
func compactDuration(s int64) string {
	switch {
	case s < 90:
		return fmt.Sprintf("%ds", s)
	case s < 5400:
		return fmt.Sprintf("%dm", s/60)
	case s < 172800:
		return fmt.Sprintf("%dh", s/3600)
	default:
		return fmt.Sprintf("%dd", s/86400)
	}
}

// agoString renders a non-negative age in seconds compactly (Ns / Nm / Nh / Nd).
func agoString(s int64) string {
	if s < 0 {
		return "0s"
	}
	return compactDuration(s)
}

// shortRunID trims a run id to a readable width.
func shortRunID(id string) string { return truncField(id, 12) }

// truncField clamps s to n runes for table alignment.
func truncField(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
