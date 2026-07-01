package main

// audit_usage.go — `fak audit usage`: the cross-session usage rollup #1612 (child
// C of epic #1601) asks for. It is a thin I/O shell only: every disk read, chain
// verification, and clock read happens HERE; internal/auditusage.Fold is the pure
// fold over the results (see that package's doc comment for the sink list and the
// witnessed-vs-observed honesty fence).
//
// A missing sink is never an error — a fresh checkout, or a box that never ran the
// producing subsystem, legitimately has no journal/ledger yet. Only a PRESENT sink
// whose hash chain fails verification is a real finding (CHAIN_BROKEN), and even
// then the rollup still folds in whatever a tolerant read recovered.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/auditusage"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

func cmdAuditUsage(args []string) {
	os.Exit(runAuditUsage(os.Stdout, os.Stderr, args))
}

// runAuditUsage is the testable core of `fak audit usage`.
func runAuditUsage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("audit usage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var roots multiFlag
	fs.Var(&roots, "root", "repo/workspace root to scan for root-scoped sinks (loop ledger, cache-value ledger, gateway-usage ledger, .dispatch-runs/); repeatable (default: the current repo root)")
	sinceStr := fs.String("since", "", "only fold rows/events newer than this duration ago, e.g. 168h (default: no cutoff)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	journalPath := fs.String("journal", "", "decision journal path (default: the guard audit journal default path)")
	usageLogPathFlag := fs.String("usage-log", "", "usage log path (default: the usage log default path)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak audit usage: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	now := time.Now()
	var since time.Time
	if *sinceStr != "" {
		d, err := time.ParseDuration(*sinceStr)
		if err != nil {
			fmt.Fprintf(stderr, "fak audit usage: --since: %v\n", err)
			return 2
		}
		since = now.Add(-d)
	}

	if len(roots) == 0 {
		roots = multiFlag{repoRoot()}
	}

	in := auditusage.Input{Now: now, Since: since}
	in.DecisionJournal = readDecisionJournalInput(firstNonEmpty(*journalPath, guardDefaultAuditPath()))
	in.UsageLog = readUsageLogInput(firstNonEmpty(*usageLogPathFlag, usagelog.DefaultPath()))
	in.GatewayUsage = mergeGatewayUsageInputs(roots)
	in.CacheValue = mergeCacheValueInputs(roots)
	in.LoopLedger = readLoopLedgerInput(defaultLoopLedger())
	in.DispatchRuns = mergeDispatchRunsInputs(roots)

	report := auditusage.Fold(in)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak audit usage")
	}
	fmt.Fprint(stdout, renderAuditUsage(report))
	return 0
}

func readDecisionJournalInput(path string) auditusage.DecisionJournalInput {
	in := auditusage.DecisionJournalInput{Path: path}
	if _, err := os.Stat(path); err != nil {
		return in
	}
	in.Present = true
	rows, err := journal.ReadRows(path)
	if err != nil {
		in.VerifyErr = err
		return in
	}
	in.Rows = rows
	if _, verr := journal.Verify(path); verr != nil {
		in.VerifyErr = verr
	}
	return in
}

func readUsageLogInput(path string) auditusage.UsageLogInput {
	in := auditusage.UsageLogInput{Path: path}
	if _, err := os.Stat(path); err != nil {
		return in
	}
	in.Present = true
	rows, err := usagelog.ReadRows(path)
	if err != nil {
		in.VerifyErr = err
		return in
	}
	in.Rows = rows
	if _, verr := usagelog.Verify(path); verr != nil {
		in.VerifyErr = verr
	}
	return in
}

func gatewayUsagePathForRoot(root string) string {
	return filepath.Join(root, filepath.FromSlash(gatewayusageledger.DefaultLedgerRel))
}

func mergeGatewayUsageInputs(roots []string) auditusage.GatewayUsageInput {
	var out auditusage.GatewayUsageInput
	for _, root := range roots {
		path := gatewayUsagePathForRoot(root)
		if _, err := os.Stat(path); err != nil {
			continue
		}
		out.Present = true
		if out.Path == "" {
			out.Path = path
		}
		out.Rows = append(out.Rows, gatewayusageledger.ReadLedgerFile(path)...)
	}
	return out
}

func cacheValuePathForRoot(root string) string {
	return filepath.Join(root, filepath.FromSlash(cachevalueledger.DefaultLedgerRel))
}

func mergeCacheValueInputs(roots []string) auditusage.CacheValueInput {
	var out auditusage.CacheValueInput
	for _, root := range roots {
		path := cacheValuePathForRoot(root)
		if _, err := os.Stat(path); err != nil {
			continue
		}
		out.Present = true
		if out.Path == "" {
			out.Path = path
		}
		out.Rows = append(out.Rows, cachevalueledger.ReadLedgerFile(path)...)
	}
	return out
}

func dispatchRunsDirForRoot(root string) string {
	return filepath.Join(root, ".dispatch-runs")
}

func mergeDispatchRunsInputs(roots []string) auditusage.DispatchRunsInput {
	var out auditusage.DispatchRunsInput
	for _, root := range roots {
		dir := dispatchRunsDirForRoot(root)
		workers, err := dispatchaudit.ScanDir(dir)
		if err != nil {
			continue
		}
		out.Present = true
		if out.RunsDir == "" {
			out.RunsDir = dir
		}
		out.Workers = append(out.Workers, workers...)
	}
	return out
}

func readLoopLedgerInput(path string) auditusage.LoopLedgerInput {
	in := auditusage.LoopLedgerInput{Path: path}
	if _, err := os.Stat(path); err != nil {
		return in
	}
	in.Present = true
	events, integrity, err := loopmgr.LoadPrefix(path)
	if err != nil {
		integrity.Broken = true
		integrity.Reason = err.Error()
	}
	in.Events = events
	in.Integrity = integrity
	return in
}

func renderAuditUsage(rep auditusage.Report) string {
	var b []byte
	add := func(format string, args ...any) {
		b = append(b, []byte(fmt.Sprintf(format, args...))...)
	}
	add("fak audit usage — cross-session rollup\n\n")
	add("sinks:\n")
	for _, s := range rep.Sinks {
		add("  %-20s present=%-5v chain=%-10s rows=%-6d %s\n", s.Kind, s.Present, s.Chain, s.RowCount, s.BrokenReason)
	}
	if len(rep.Findings) > 0 {
		add("\nfindings:\n")
		for _, f := range rep.Findings {
			add("  %s sink=%s path=%s: %s\n", f.Kind, f.Sink, f.Path, f.Reason)
		}
	}
	add("\nguard (%s):    total=%d\n", rep.Guard.Basis, rep.Guard.Total)
	add("loop (%s):     loops=%d fires=%d admitted=%d refused=%d witnessed=%d\n",
		rep.Loop.Basis, rep.Loop.Loops, rep.Loop.Fires, rep.Loop.Admitted, rep.Loop.Refused, rep.Loop.Witnessed)
	add("dispatch (%s): workers=%d findings=%d\n", rep.Dispatch.Basis, rep.Dispatch.Workers, rep.Dispatch.Findings)
	add("cache (%s):    sessions=%d\n", rep.Cache.Basis, rep.Cache.Sessions)
	add("gateway (%s):  sessions=%d\n", rep.Gateway.Basis, rep.Gateway.Sessions)
	add("usage (%s):    total=%d errors=%d\n", rep.Usage.Basis, rep.Usage.Total, rep.Usage.Errors)
	return string(b)
}
