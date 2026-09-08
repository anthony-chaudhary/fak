package main

// fak milestone burndown -- the SCHEDULE dimension of the milestone tracker: a
// read-only fold over every open GitHub milestone's due date and recent closure
// velocity into one schedule verdict per milestone (OVERDUE / AT_RISK / NO_DUE_DATE
// / ON_TRACK / DONE) plus a single at-risk-debt integer. Where `fak milestone
// report` tracks the maturity CLIMB and epic ROADMAP (are we building the right
// thing, how far along), burndown tracks whether the dated milestones will LAND on
// time. With --append-history it appends a dated row to the durable JSONL ledger
// (docs/milestones/burndown.jsonl) so the at-risk debt is trended across weeks;
// --check is the advisory gate (non-zero only when the milestone list could not be
// MEASURED -- an OVERDUE milestone is a measured fact, not an incomplete report, so
// it passes). It rides the same weekly cadence loop as the milestone report.
//
//	fak milestone burndown                    # fold + render the schedule snapshot
//	fak milestone burndown --json             # the machine-readable envelope
//	fak milestone burndown --check            # advisory gate (exit 1 only if unmeasured)
//	fak milestone burndown --append-history   # trend a dated row into the ledger

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/milestoneburndown"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

// runMilestoneBurndown collects the live milestones, folds the schedule verdicts,
// attaches the per-tick trend vs the durable ledger, optionally appends the tick,
// and renders/JSON/gates -- the schedule twin of runMilestoneReport.
func runMilestoneBurndown(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("milestone burndown", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "owner/name for the `gh` milestone queries (default: the current checkout's gh context)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if the milestone list failed to measure")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger (docs/milestones/burndown.jsonl)")
	window := fs.Int("window", milestoneburndown.DefaultWindowDays, "trailing window (days) recent closure velocity is measured over")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+milestoneburndown.DefaultLedgerRel+")")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	cohortBaseline := fs.String("cohort-baseline", "", "path to milestone burndown baseline JSON file (default: docs/milestones/burndown-baseline.json if present)")
	baselineDate := fs.String("baseline-date", "", "snapshot date YYYY-MM-DD to anchor baseline cohort from history ledger")
	baselineOpen := fs.Int("baseline-open", 0, "explicit baseline open issue count to anchor cohort drain tracking")
	baselineClosed := fs.Int("baseline-closed", 0, "explicit baseline closed issue count to anchor cohort drain tracking")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak milestone burndown: unexpected argument %q\n", fs.Arg(0))
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

	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(milestoneburndown.DefaultLedgerRel))
	}
	prior := readLedgerFile(ledgerPath, milestoneburndown.ParseLedger)

	// Collect live (a nil runner shells the real `gh`); a per-milestone velocity read
	// that fails degrades only that row, never the whole portfolio.
	portfolio := milestoneburndown.Collect(*repo, nil, *window, now)

	// Resolve baseline cohort (explicit -> file -> ledger history).
	var base milestoneburndown.BurndownBaseline
	var baseResolved bool
	if *baselineOpen > 0 {
		base = milestoneburndown.BurndownBaseline{
			Schema:       milestoneburndown.BaselineSchema,
			Date:         *baselineDate,
			OpenTotal:    *baselineOpen,
			ClosedTotal:  *baselineClosed,
			BaselineOpen: *baselineOpen,
		}
		baseResolved = true
	}
	if !baseResolved {
		baseFile := *cohortBaseline
		if baseFile == "" {
			defaultPath := filepath.Join(root, filepath.FromSlash(milestoneburndown.DefaultBaselineRel))
			if fi, err := os.Stat(defaultPath); err == nil && !fi.IsDir() {
				baseFile = defaultPath
			}
		} else if !filepath.IsAbs(baseFile) {
			baseFile = filepath.Join(root, baseFile)
		}
		if baseFile != "" {
			if data, err := os.ReadFile(baseFile); err == nil {
				if parsed, err := milestoneburndown.ParseBaseline(data); err == nil {
					base = parsed
					baseResolved = true
				}
			}
		}
	}
	if !baseResolved {
		if fromLedger, ok := milestoneburndown.BaselineFromLedger(prior, *baselineDate); ok {
			base = fromLedger
			baseResolved = true
		}
	}

	if baseResolved {
		cp := milestoneburndown.ComputeCohortProgress(base, portfolio)
		if cp.BaselineOpen > 0 {
			portfolio = portfolio.WithCohort(cp)
		}
	}

	report := milestoneburndown.Fold(portfolio, milestoneburndown.FoldOpts{
		Workspace:   root,
		Commit:      milestonereport.HeadCommit(root),
		GeneratedAt: now.Format(time.RFC3339),
		Date:        snapDate,
	})

	// Attach the per-tick trend vs the last ledger row (read-only), and -- only under
	// --append-history -- durably append this tick so the trend accrues.
	row := milestoneburndown.RowFromReport(report)
	report = report.WithTrend(milestoneburndown.TrendVsLast(row, prior))
	if code := appendReportHistory(stdout, stderr, *appendHistory, !*asJSON && !*check, root, ledgerPath,
		"milestone burndown", "burndown", row, trendreport.AppendLedgerLine); code != 0 {
		return code
	}

	if *check {
		g := milestoneburndown.CheckGate(report)
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, report.WithGate(g))
		} else {
			fmt.Fprintln(stdout, g.Message)
		}
		return g.Exit
	}

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, report)
	} else {
		fmt.Fprint(stdout, milestoneburndown.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}
