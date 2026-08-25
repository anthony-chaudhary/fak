package main

// fak program -- the ongoing-program report: one read-only fold over the project's
// never-"done" PROGRAMS (kernel-optimization, cache-optimization, and
// human-operator-effectiveness; the classes internal/worktype marks as ongoing). It
// is the sibling of `fak milestone`:
// where the milestone roadmap measures DISCRETE epics by completion %, this measures
// the ongoing programs by a FRONTIER + a TREND, because an ongoing program has no 100%.
// The two together give the operator the right lens for each kind of work.
//
//	fak program report                     # fold + render the snapshot
//	fak program report --json              # the machine-readable envelope
//	fak program report --check             # advisory gate (exit 1 only if unmeasured)
//	fak program report --append-history    # trend a dated row into the ledger

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

func cmdProgram(argv []string) {
	dispatchSubcommands("program", "report | selfcheck", argv,
		subcommand{"report", runProgramReport},
		subcommand{"selfcheck", runProgramSelfcheck},
	)
}

// runProgramSelfcheck runs the deterministic source-level page-vs-act proof for
// the program report — no key, no network, no fixtures.
func runProgramSelfcheck(stdout, stderr io.Writer, argv []string) int {
	return runReportSelfcheck(stdout, stderr, argv, "program", programreport.TriageSelfcheck,
		"SELFCHECK OK -- decenter-the-human at the program gate: an incomplete report with a "+
			"runnable rerun routes to the fleet; one that names authority still pages.")
}

// runProgramReport collects the two ongoing-program frontier signals, folds them,
// attaches the per-tick trend vs the durable ledger, optionally appends the tick, and
// renders/JSON/gates -- the program twin of runMilestoneReport.
func runProgramReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak program report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	cacheLedger := fs.String("cache-ledger", "", "Track-1 cache-value ledger path (default: <root>/docs/nightrun/cache-value.jsonl)")
	windowDays := fs.Int("window-days", programreport.DefaultWindowDays, "trailing window (days) the kernel-opt activity signal counts ships over")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the programs dimension failed to measure")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger (docs/programs/history.jsonl)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+programreport.DefaultLedgerRel+")")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak program report: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	now := time.Now().UTC()
	snapDate := *date
	if snapDate == "" {
		snapDate = now.Format("2006-01-02")
	}

	cacheLedgerPath := *cacheLedger
	if cacheLedgerPath == "" {
		cacheLedgerPath = filepath.Join(root, filepath.FromSlash("docs/nightrun/cache-value.jsonl"))
	}

	programs := programreport.Collect(root, cacheLedgerPath, *windowDays)
	report := programreport.Fold(programs, programreport.FoldOpts{
		Workspace:   root,
		Commit:      programreport.HeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        snapDate,
	})

	// Attach the per-tick trend vs the last ledger row (read-only), and -- only under
	// --append-history -- durably append this tick so the trend accrues.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(programreport.DefaultLedgerRel))
	}
	row := programreport.RowFromReport(report)
	prior := readProgramLedgerRows(ledgerPath)
	report = report.WithTrend(programreport.TrendVsLast(row, prior))
	if code := appendReportHistory(stdout, stderr, *appendHistory, !*asJSON && !*check, root, ledgerPath,
		"program report", "program", row, programreport.AppendLedgerLine); code != 0 {
		return code
	}

	if *check {
		// Decenter the human at the source: under FAK_PROGRAM_TRIAGE_GATE=enforce an
		// incomplete report whose NextAction is a runnable rerun routes to the fleet
		// instead of paging. Default ("", "warn") is the unchanged advisory gate.
		return checkAndEmitReportGate(stdout, *asJSON, report, func(report programreport.Report) (int, string) {
			return programreport.CheckGateTriaged(report, trendreport.TriageEnforced(os.Getenv("FAK_PROGRAM_TRIAGE_GATE")))
		}, programreport.Report.WithGate)
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, report)
	} else {
		fmt.Fprintln(stdout, programreport.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}

// readProgramLedgerRows reads the durable ledger if present (absent ledger -> no
// prior rows; the first tick establishes the series).
func readProgramLedgerRows(path string) []programreport.LedgerRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return programreport.ParseLedger(string(raw))
}
