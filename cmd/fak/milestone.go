package main

// fak milestone -- the milestone-based tracking report: one read-only fold over
// the project's two milestone signals -- the maturity CLIMB (the model x backend
// grid's M0-M7 distribution, witnessed from covmatrix.Grid()) and the epic ROADMAP
// (per-tracked-epic child completion read live from `gh`) -- into one
// schema/ok/verdict/finding/reason/next_action envelope. With --append-history it
// appends a dated row to the durable JSONL ledger (docs/milestones/history.jsonl) so
// the climb + roadmap are trended across weeks. --check is the advisory gate
// (non-zero only when a dimension failed to MEASURE, i.e. the `gh` read failed for
// every tracked epic). `fak milestone post` renders the report as a Slack card and
// posts it to the #milestones channel via the shared scoreboard transport.
//
//	fak milestone report                     # fold + render the snapshot
//	fak milestone report --json              # the machine-readable envelope
//	fak milestone report --check             # advisory gate (exit 1 only if unmeasured)
//	fak milestone report --append-history    # trend a dated row into the ledger
//	fak milestone post --dry-run             # render the exact card; do not post
//	fak milestone post                       # post the card to #milestones

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/milestonepost"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
)

func cmdMilestone(argv []string) {
	dispatchSubcommands("milestone", "report | post", argv,
		subcommand{"report", runMilestoneReport},
		subcommand{"post", runMilestonePost},
	)
}

// runMilestoneReport folds the two dimensions, attaches the per-tick trend vs the
// durable ledger, optionally appends the tick, and renders/JSON/gates -- the milestone
// twin of runCadence.
func runMilestoneReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("milestone report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "owner/name for the `gh` roadmap queries (default: the current checkout's gh context)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only if a dimension failed to measure")
	appendHistory := fs.Bool("append-history", false, "append a dated row to the durable ledger (docs/milestones/history.jsonl)")
	ledger := fs.String("ledger", "", "ledger path override (default: <root>/"+milestonereport.DefaultLedgerRel+")")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak milestone report: unexpected argument %q\n", fs.Arg(0))
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
	commit := milestonereport.HeadCommit(root)

	maturity, epics := milestonereport.Collect(*repo, nil)
	report := milestonereport.Fold(maturity, epics, milestonereport.FoldOpts{
		Workspace:   root,
		Commit:      commit,
		GeneratedAt: now.Format(time.RFC3339),
		Date:        snapDate,
	})

	// Attach the per-tick trend vs the last ledger row (read-only), and -- only under
	// --append-history -- durably append this tick so the trend accrues.
	ledgerPath := *ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, filepath.FromSlash(milestonereport.DefaultLedgerRel))
	}
	row := milestonereport.RowFromReport(report)
	prior := readMilestoneLedgerRows(ledgerPath)
	report = report.WithTrend(milestonereport.TrendVsLast(row, prior))
	if *appendHistory {
		if err := appendMilestoneLedgerRow(ledgerPath, row); err != nil {
			fmt.Fprintf(stderr, "fak milestone report: append ledger: %v\n", err)
			return 1
		}
		if !*asJSON && !*check {
			rel, _ := filepath.Rel(root, ledgerPath)
			if rel == "" {
				rel = ledgerPath
			}
			fmt.Fprintf(stdout, "appended milestone row -> %s\n", filepath.ToSlash(rel))
		}
	}

	if *check {
		code, message := milestonereport.CheckGate(report)
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
		fmt.Fprintln(stdout, milestonereport.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
}

// runMilestonePost folds the report live (or from a pre-rolled --report-json payload),
// renders the card, and posts it to the #milestones channel via the shared scoreboard
// transport -- the milestone twin of runCachevalueFeed.
func runMilestonePost(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak milestone post", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "owner/name for the `gh` roadmap queries (default: the current checkout's gh context)")
	reportJSON := fs.String("report-json", "", "fold a pre-rolled milestonereport.Report JSON from this file (- for stdin) instead of collecting live")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	channel := fs.String("channel", "", "override target channel id (default: $FAK_MILESTONE_CHANNEL / .env.slack.local / #milestones)")
	token := fs.String("token", "", "override bot token (default: $FAK_MILESTONE_TOKEN, then the scoreboard token)")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	var report milestonereport.Report
	if *reportJSON != "" {
		r, err := loadMilestoneReport(*reportJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak milestone post: %v\n", err)
			return 2
		}
		report = r
	} else {
		root := *workspace
		if root == "" {
			root = repoRoot()
		} else if abs, err := filepath.Abs(root); err == nil {
			root = abs
		}
		now := time.Now().UTC()
		maturity, epics := milestonereport.Collect(*repo, nil)
		report = milestonereport.Fold(maturity, epics, milestonereport.FoldOpts{
			Workspace:   root,
			Commit:      milestonereport.HeadCommit(root),
			GeneratedAt: now.Format(time.RFC3339),
			Date:        now.Format("2006-01-02"),
		})
		ledgerPath := filepath.Join(root, filepath.FromSlash(milestonereport.DefaultLedgerRel))
		report = report.WithTrend(milestonereport.TrendVsLast(milestonereport.RowFromReport(report), readMilestoneLedgerRows(ledgerPath)))
	}

	card := milestonepost.Fold(report)
	if s := resolveMilestoneSource(*source); s != "" {
		card.Source = s
	}
	return slackPostTail(stdout, stderr, slackPostSpec{
		card:           card,
		channel:        *channel,
		token:          *token,
		dryRun:         *dryRun,
		label:          "fak milestone post",
		chanEnv:        "FAK_MILESTONE_CHANNEL",
		resolveChannel: milestonepost.ResolveChannel,
		resolveToken:   milestonepost.ResolveToken,
	})
}

// loadMilestoneReport reads a pre-rolled report payload from a file (or stdin for "-").
func loadMilestoneReport(path string) (milestonereport.Report, error) {
	var report milestonereport.Report
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return report, err
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, fmt.Errorf("parse --report-json payload: %w", err)
	}
	return report, nil
}

// resolveMilestoneSource picks the post source: the flag, else the shared
// defaultSource ($FAK_SCOREBOARD_SOURCE or hostname).
func resolveMilestoneSource(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	return defaultSource()
}

// readMilestoneLedgerRows reads the durable ledger if present (absent ledger -> no
// prior rows, the first tick establishes the series).
func readMilestoneLedgerRows(path string) []milestonereport.LedgerRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return milestonereport.ParseLedger(string(raw))
}

// appendMilestoneLedgerRow appends one JSONL row to the ledger, creating the parent
// directory on first write.
func appendMilestoneLedgerRow(path string, row milestonereport.LedgerRow) error {
	line, err := milestonereport.AppendLedgerLine(row)
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
