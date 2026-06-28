package main

// fak cadence -- the consolidated regular-cadence report: one read-only fold
// over the three dimensions an operator tracks on a cadence -- SCORES (the
// scorecard control pane), WORK-DONE (git commits + `(fak ` ships over a
// trailing window), and RELEASES (the release-status fold) -- into one
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
)

func cmdCadence(argv []string) { os.Exit(runCadence(os.Stdout, os.Stderr, argv)) }

func runCadence(stdout, stderr io.Writer, argv []string) int {
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
	if err := fs.Parse(argv); err != nil {
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
	var work cadencereport.Work
	var releases cadencereport.Releases
	if *scoresFrom != "" {
		// Use the captured pane payload for SCORES; work + releases still run live.
		s := cadencereport.InterpretScoresFromFile(*scoresFrom, os.Stdin)
		scores, work, releases = cadencereport.CollectWithScores(root, "", s, time.Duration(*timeout)*time.Second, *window)
	} else {
		scores, work, releases = cadencereport.Collect(root, "", time.Duration(*timeout)*time.Second, *window)
	}
	report := cadencereport.Fold(scores, work, releases, cadencereport.FoldOpts{
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
	prior := readLedgerRows(ledgerPath)
	trend := cadencereport.TrendVsLast(row, prior)
	report.Trend = &trend
	if *appendHistory {
		if err := appendLedgerRow(ledgerPath, row); err != nil {
			fmt.Fprintf(stderr, "fak cadence: append ledger: %v\n", err)
			return 1
		}
		if !*asJSON && !*check {
			rel, _ := filepath.Rel(root, ledgerPath)
			if rel == "" {
				rel = ledgerPath
			}
			fmt.Fprintf(stdout, "appended cadence row -> %s\n", filepath.ToSlash(rel))
		}
	}

	if *check {
		code, message := cadencereport.CheckGate(report)
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

// readLedgerRows reads the durable ledger if present (absent ledger -> no prior
// rows, the first tick establishes the series).
func readLedgerRows(path string) []cadencereport.LedgerRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return cadencereport.ParseLedger(string(raw))
}

// appendLedgerRow appends one JSONL row to the ledger, creating the parent
// directory on first write.
func appendLedgerRow(path string, row cadencereport.LedgerRow) error {
	line, err := cadencereport.AppendLedgerLine(row)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}
