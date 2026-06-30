package main

// fak program -- the ongoing-program report: one read-only fold over the project's
// never-"done" optimization PROGRAMS (kernel-optimization + cache-optimization, the
// classes internal/worktype marks as ongoing). It is the sibling of `fak milestone`:
// where the milestone roadmap measures DISCRETE epics by completion %, this measures
// the ongoing programs by a FRONTIER + a TREND, because an optimization program has no
// 100%. The two together give the operator the right lens for each kind of work.
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
)

func cmdProgram(argv []string) {
	dispatchSubcommands("program", "report", argv,
		subcommand{"report", runProgramReport},
	)
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
	if err := fs.Parse(argv); err != nil {
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
	if *appendHistory {
		if err := appendProgramLedgerRow(ledgerPath, row); err != nil {
			fmt.Fprintf(stderr, "fak program report: append ledger: %v\n", err)
			return 1
		}
		if !*asJSON && !*check {
			rel, _ := filepath.Rel(root, ledgerPath)
			if rel == "" {
				rel = ledgerPath
			}
			fmt.Fprintf(stdout, "appended program row -> %s\n", filepath.ToSlash(rel))
		}
	}

	if *check {
		code, message := programreport.CheckGate(report)
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, report.WithGate(code, message))
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
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

// appendProgramLedgerRow appends one JSONL row to the ledger, creating the parent
// directory on first write.
func appendProgramLedgerRow(path string, row programreport.LedgerRow) error {
	line, err := programreport.AppendLedgerLine(row)
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
