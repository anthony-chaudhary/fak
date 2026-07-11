package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// runIssueAuditLoop is the operator face of the durable cross-audit background
// loop (#3856). It runs one deterministic planning tick over a discovery
// snapshot and the durable receipt ledger + scheduler cursor, then reports the
// typed next decision (ADVANCING/STALLED/DARK/WAIT). It defaults to --dry-run:
// it plans and reports without leasing, auditing, or mutating the ledger. The
// live per-subject audit execution composes the shipped identity/router/spine
// seams under the supervised cadence; this command is the plan + status surface.
func runIssueAuditLoop(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue audit-loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	snapshotPath := fs.String("snapshot", "", "JSON array of discovery subjects {issue_number,marker_key,risk,eligible,ineligible_reason}")
	ledgerPath := fs.String("ledger", firstNonEmpty(os.Getenv("FAK_CROSSAUDIT_LEDGER"), ".fak/crossaudit/receipts.jsonl"), "at-most-once receipt ledger path")
	cursorPath := fs.String("cursor", firstNonEmpty(os.Getenv("FAK_CROSSAUDIT_CURSOR"), ".fak/crossaudit/loop-cursor.json"), "durable scheduler cursor path")
	scanCap := fs.Int("scan-cap", modelroute.IssueAuditLoopScanCapMax, "max issues scanned per planning run (<= 500)")
	batchCap := fs.Int("batch-cap", 8, "max subjects planned/audited per tick")
	replay := fs.String("replay", "", "comma-separated issue numbers to force re-audit even if settled")
	asJSON := fs.Bool("json", false, "emit the full typed JSON tick report")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issue audit-loop: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*snapshotPath) == "" {
		fmt.Fprintln(stderr, "fak issue audit-loop: --snapshot FILE is required (the bounded discovery snapshot)")
		return 2
	}
	if *batchCap <= 0 {
		fmt.Fprintln(stderr, "fak issue audit-loop: --batch-cap must be positive")
		return 2
	}
	subjects, err := loadIssueAuditLoopSnapshot(*snapshotPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit-loop: %v\n", err)
		return 2
	}
	replayIssues, err := parseIssueAuditLoopReplay(*replay)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit-loop: %v\n", err)
		return 2
	}

	cfg := modelroute.IssueAuditLoopConfig{
		LedgerPath:   *ledgerPath,
		CursorPath:   *cursorPath,
		ScanCap:      *scanCap,
		BatchCap:     *batchCap,
		DryRun:       true,
		ReplayIssues: replayIssues,
		Discoverer: modelroute.IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]modelroute.IssueAuditLoopSubject, error) {
			return subjects, nil
		}),
	}
	report, err := modelroute.RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue audit-loop: %v\n", err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak issue audit-loop: encode report: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderIssueAuditLoopReport(report))
	}
	return 0
}

func loadIssueAuditLoopSnapshot(path string) ([]modelroute.IssueAuditLoopSubject, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read --snapshot: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var subjects []modelroute.IssueAuditLoopSubject
	if err := dec.Decode(&subjects); err != nil {
		return nil, fmt.Errorf("parse --snapshot as a discovery-subject array: %w", err)
	}
	return subjects, nil
}

func parseIssueAuditLoopReplay(raw string) ([]int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var out []int
	for _, tok := range strings.Split(raw, ",") {
		tok = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(tok), "#"))
		if tok == "" {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(tok, "%d", &n); err != nil || n <= 0 {
			return nil, fmt.Errorf("--replay %q must be positive issue numbers", raw)
		}
		out = append(out, n)
	}
	return out, nil
}

func renderIssueAuditLoopReport(r modelroute.IssueAuditLoopReport) string {
	var b strings.Builder
	mode := "live"
	if r.DryRun {
		mode = "dry-run"
	}
	fmt.Fprintf(&b, "issue audit-loop %s (%s): discovered=%d eligible=%d settled=%d planned=%d ledger_rows=%d\n",
		r.State, mode, r.Discovered, r.Eligible, r.AlreadySettled, r.Planned, r.LedgerRows)
	fmt.Fprintf(&b, "  audited=%d pass=%d refute=%d inconclusive=%d unavailable=%d lease_conflicts=%d\n",
		r.Audited, r.Passed, r.Refuted, r.Inconclusive, r.Unavailable, r.LeaseConflicts)
	fmt.Fprintf(&b, "  deferred=%d dead_lettered=%d cap_deferred=%d\n", r.Deferred, r.DeadLettered, r.CapDeferred)
	if len(r.PlannedSubjects) > 0 {
		fmt.Fprintf(&b, "  planned_subjects: %s\n", intList(r.PlannedSubjects))
	}
	if len(r.DarkSubjects) > 0 {
		fmt.Fprintf(&b, "  dark_subjects: %s\n", intList(r.DarkSubjects))
	}
	if len(r.DeadLetterQueue) > 0 {
		fmt.Fprintf(&b, "  dead_letter_queue: %s\n", intList(r.DeadLetterQueue))
	}
	for _, note := range r.Notes {
		fmt.Fprintf(&b, "  note: %s\n", note)
	}
	return b.String()
}
