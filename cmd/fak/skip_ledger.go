package main

// skip_ledger.go is the thin I/O shell for `fak dispatch skip-ledger`: it
// reads candidate facts (the same shape `fak dispatch order` accepts), runs
// the same dispatchorder.Plan decision, folds the result through
// internal/skipledger's pure Record, and persists one JSONL row per
// candidate to a durable local ledger -- so a later pass can audit why rate
// was lost this tick, not just what was picked (#1776). It never spawns a
// worker or mutates GitHub; the only side effect is appending to the local
// ledger file.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/skipledger"
)

const (
	skipLedgerRunsDir = ".dispatch-runs"
	skipLedgerLogName = "skip-ledger.jsonl"
)

func runDispatchSkipLedger(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch skip-ledger", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "read candidate units from this JSON file (default: stdin)")
	workspace := fs.String("workspace", ".", "workspace root the ledger is persisted under")
	cooldownMin := fs.Int("cooldown-min", 120, "skip a freshest unit attempted within this many minutes (-1 disables)")
	nowUnix := fs.Int64("now", 0, "the clock as unix seconds for cooldown math and the row timestamp (0 = current time)")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	if err := fs.Parse(argv); err != nil {
		return 2 // flag already printed the error
	}

	raw, code := readDispatchInput(stderr, *in)
	if code != 0 {
		return code
	}
	cands, err := parseCandidates(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch skip-ledger: parse candidates: %v\n", err)
		return 1
	}

	now := *nowUnix
	if now == 0 {
		now = time.Now().Unix()
	}
	cooldownSec := int64(*cooldownMin) * 60
	if *cooldownMin < 0 {
		cooldownSec = -1
	}

	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates:      cands,
		NowUnix:         now,
		CooldownSeconds: cooldownSec,
	})
	rep := skipledger.Record(res, now)

	runsDir := filepath.Join(*workspace, skipLedgerRunsDir)
	if err := skipLedgerAppend(runsDir, rep); err != nil {
		fmt.Fprintf(stderr, "fak dispatch skip-ledger: persist ledger: %v\n", err)
		return 1
	}

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak dispatch skip-ledger")
	}
	renderSkipLedger(stdout, rep, filepath.Join(runsDir, skipLedgerLogName))
	return 0
}

// skipLedgerAppend persists one JSON line per row to the durable ledger
// file, creating the runs dir if needed. Append-only, matching the
// dispatch-progress ledger's own persistence shape.
func skipLedgerAppend(runsDir string, rep skipledger.Report) error {
	if len(rep.Rows) == 0 {
		return nil
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(runsDir, skipLedgerLogName), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, row := range rep.Rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

// renderSkipLedger prints the tick's rows as an aligned, scannable table,
// then the persisted ledger path.
func renderSkipLedger(w io.Writer, rep skipledger.Report, ledgerPath string) {
	fmt.Fprintf(w, "skip ledger -- %d selected, %d skipped\n\n", rep.SelectedCount, rep.SkippedCount)
	fmt.Fprintf(w, "%-10s %-4s %-16s %-22s %s\n", "issue", "lane", "disposition", "reason", "category")
	for _, row := range rep.Rows {
		lane := row.Lane
		if lane == "" {
			lane = "-"
		}
		category := string(row.Category)
		if category == "" {
			category = "-"
		}
		fmt.Fprintf(w, "%-10s %-4s %-16s %-22s %s\n", row.Issue, lane, row.Disposition, row.Reason, category)
	}
	fmt.Fprintf(w, "\npersisted %d row(s) to %s\n", len(rep.Rows), ledgerPath)
}
