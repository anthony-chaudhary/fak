package main

// fak cadence -- the consolidated regular-cadence report: one read-only fold
// over the four dimensions an operator tracks on a cadence -- SCORES (the
// scorecard control pane), MATURITY (the feature lifecycle ladder), WORK-DONE
// (git commits + `(fak ` ships over a trailing window), and RELEASES
// (the release-status fold) -- into one
// schema/ok/verdict/finding/reason/next_action envelope. With --append-history it
// also appends a dated row to the durable JSONL ledger
// (docs/cadence/history.jsonl) so the cadence is trended across weeks, not just a
// point-in-time step summary. --check is the advisory gate (non-zero only when a
// dimension failed to MEASURE; the scorecard ratchet owns debt regressions).

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

func cmdCadence(argv []string) { os.Exit(runCadence(os.Stdout, os.Stderr, argv)) }

func runCadence(stdout, stderr io.Writer, argv []string) int {
	if code, handled := runReportSelfcheckRequest(stdout, stderr, argv, "cadence", cadencereport.TriageSelfcheck,
		"SELFCHECK OK -- decenter-the-human at the cadence gate: an incomplete report with a "+
			"runnable rerun routes to the fleet; one that names authority still pages."); handled {
		return code
	}
	fs := flag.NewFlagSet("cadence", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if a dimension failed to measure")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger (docs/cadence/history.jsonl)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+cadencereport.DefaultLedgerRel+")")
	window := fs.Int("window", cadencereport.DefaultWindowDays, "trailing window (days) the work-done dimension counts over")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	timeout := fs.Int("timeout", 300, "per-sub-tool timeout seconds")
	scoresFrom := fs.String("scores-from", "", "read a scorecard_control_pane.py JSON payload (file path, or '-' for stdin) for the SCORES dimension instead of re-running the ~4-minute pane")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak cadence: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *window <= 0 {
		fmt.Fprintf(stderr, "fak cadence: --window must be positive, got %d\n", *window)
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
	commit := cadencereport.HeadCommit(root)

	var scores cadencereport.Scores
	var maturity cadencereport.Maturity
	var work cadencereport.Work
	var releases cadencereport.Releases
	if *scoresFrom != "" {
		// Use the captured pane payload for SCORES; work + releases still run live.
		s := cadencereport.InterpretScoresFromFile(*scoresFrom, os.Stdin)
		scores, work, releases = cadencereport.CollectWithScores(root, "", s, time.Duration(*timeout)*time.Second, *window)
	} else {
		scores, work, releases = cadencereport.Collect(root, "", time.Duration(*timeout)*time.Second, *window)
	}
	maturity = cadencereport.CollectMaturity(root)
	report := cadencereport.FoldWithMaturity(scores, maturity, work, releases, cadencereport.FoldOpts{
		Workspace:   root,
		Commit:      commit,
		GeneratedAt: now.Format(time.RFC3339),
		Date:        snapDate,
	})

	// Attach the per-tick trend vs the last ledger row (read-only), and -- only
	// under --append-history -- durably append this tick so the trend accrues.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(cadencereport.DefaultLedgerRel))
	}
	row := cadencereport.RowFromReport(report)
	prior := readLedgerFile(ledgerPath, cadencereport.ParseLedger)
	row = cadencereport.ProjectStanding(row, prior)
	trend := cadencereport.TrendVsLast(row, prior)
	report.Trend = &trend
	if code := appendReportHistory(stdout, stderr, *appendHistory, !*asJSON && !*check, root, ledgerPath,
		"cadence", "cadence", row, trendreport.AppendLedgerLine); code != 0 {
		return code
	}

	if *check {
		// Decenter the human at the source: under FAK_CADENCE_TRIAGE_GATE=enforce an
		// incomplete report whose NextAction is a runnable rerun routes to the fleet
		// instead of paging. Default ("", "warn") is the unchanged advisory gate.
		code, message := cadencereport.CheckGateTriaged(report, trendreport.TriageEnforced(os.Getenv("FAK_CADENCE_TRIAGE_GATE")))
		if *asJSON {
			emitCadenceJSON(stdout, report.WithGate(code, message))
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}

	if *asJSON {
		emitCadenceJSON(stdout, report)
	} else {
		fmt.Fprintln(stdout, cadencereport.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}

func emitCadenceJSON(w io.Writer, r cadencereport.Report) {
	_ = writeIndentedJSONNoEscape(w, r)
}
