package devcmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestRunIssueAuditLoopDryRunPlansSnapshot(t *testing.T) {
	dir := t.TempDir()
	snapshot := filepath.Join(dir, "snapshot.json")
	subjects := []modelroute.IssueAuditLoopSubject{
		{IssueNumber: 100, MarkerKey: "k100", Eligible: true},
		{IssueNumber: 101, MarkerKey: "k101", Eligible: true},
		{IssueNumber: 102, Eligible: false, IneligibleReason: "not a closed leaf"},
	}
	b, err := json.Marshal(subjects)
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	if err := os.WriteFile(snapshot, b, 0o600); err != nil {
		t.Fatalf("write snapshot: %v", err)
	}
	ledger := filepath.Join(dir, "receipts.jsonl")
	cursor := filepath.Join(dir, "cursor.json")

	var stdout, stderr bytes.Buffer
	code := runIssueAuditLoop(&stdout, &stderr, []string{
		"--snapshot", snapshot, "--ledger", ledger, "--cursor", cursor, "--json",
	})
	if code != 0 {
		t.Fatalf("exit %d, stderr=%s", code, stderr.String())
	}
	var report modelroute.IssueAuditLoopReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\nstdout=%s", err, stdout.String())
	}
	if !report.DryRun {
		t.Fatalf("expected dry-run report, got %+v", report)
	}
	if report.Eligible != 2 || report.Planned != 2 {
		t.Fatalf("report eligible=%d planned=%d, want 2/2", report.Eligible, report.Planned)
	}
	if len(report.DarkSubjects) != 1 || report.DarkSubjects[0] != 102 {
		t.Fatalf("dark subjects = %v, want [102]", report.DarkSubjects)
	}
	if report.State != modelroute.IssueAuditLoopWait {
		t.Fatalf("state = %s, want WAIT (plannable work, dry-run)", report.State)
	}
	// A dry-run must not create the ledger or cursor.
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("dry-run created the ledger (err=%v)", err)
	}
	if _, err := os.Stat(cursor); !os.IsNotExist(err) {
		t.Fatalf("dry-run created the cursor (err=%v)", err)
	}
}

func TestRunIssueAuditLoopStatusReadsDurableStateWithoutSnapshot(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "receipts.jsonl")
	cursor := filepath.Join(dir, "cursor.json")

	// A fresh install (no ledger, no cursor) is a valid empty status, not an error.
	var stdout, stderr bytes.Buffer
	code := runIssueAuditLoop(&stdout, &stderr, []string{
		"--status", "--ledger", ledger, "--cursor", cursor, "--json",
	})
	if code != 0 {
		t.Fatalf("status exit %d, stderr=%s", code, stderr.String())
	}
	var status issueAuditLoopStatus
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v\nstdout=%s", err, stdout.String())
	}
	if status.LedgerPresent || status.LedgerRows != 0 || len(status.SettledIssues) != 0 {
		t.Fatalf("empty status = %+v, want no ledger / zero rows / no settled", status)
	}
}

func TestRunIssueAuditLoopRequiresSnapshot(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runIssueAuditLoop(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("missing --snapshot exit %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "--snapshot") {
		t.Fatalf("stderr did not name --snapshot: %s", stderr.String())
	}
}

// TestRunIssueAuditLoopResumeContinuityOverFixtureLedger is the operator-surface
// witness the issue (#3856) names: "a captured dry-run and two resumed ticks over
// a fixture ledger showing cursor and receipt continuity". The existing CLI
// witnesses only exercise an EMPTY ledger; this one seeds a real fixture ledger +
// cursor (one live producer tick over a captured, Verify()-valid cross-audit
// receipt), then drives the operator commands across a simulated restart:
//
//   - a captured `--dry-run` plan reports the seeded subject as already-settled
//     (the cursor is honored, the subject is NOT re-planned), and
//   - two resumed read-only `--status` reads return byte-identical durable state
//     (ledger rows + unique audits + settled issues), proving the at-most-once
//     receipt ledger and the scheduler cursor survive a restart with no drift.
func TestRunIssueAuditLoopResumeContinuityOverFixtureLedger(t *testing.T) {
	// A real, captured cross-audit receipt (terminal REFUTE for issue #2729),
	// reused from the shipped ledger fixtures so this witness rides the actual
	// Verify()-gated append path rather than a hand-rolled row.
	fixture := filepath.Join("..", "..", "internal", "modelroute", "testdata", "crossaudit_receipt_v1_3847.json")
	b, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatalf("read receipt fixture: %v", err)
	}
	var receipt modelroute.IssueAuditReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		t.Fatalf("decode receipt fixture: %v", err)
	}
	const seededIssue = 2729

	dir := t.TempDir()
	ledger := filepath.Join(dir, "receipts.jsonl")
	cursor := filepath.Join(dir, "cursor.json")

	// Seed the durable ledger + cursor with ONE real live producer tick: this is
	// the same code path the loop uses in production (lease -> audit -> append
	// terminal receipt -> settle cursor), so the fixture state is authentic.
	seed, err := modelroute.RunIssueAuditLoopTick(context.Background(), modelroute.IssueAuditLoopConfig{
		LedgerPath: ledger, CursorPath: cursor, BatchCap: 4,
		Discoverer: modelroute.IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]modelroute.IssueAuditLoopSubject, error) {
			return []modelroute.IssueAuditLoopSubject{{IssueNumber: seededIssue, MarkerKey: "seed", Eligible: true}}, nil
		}),
		Auditor: modelroute.IssueAuditLoopAuditorFunc(func(context.Context, modelroute.IssueAuditLoopSubject) (modelroute.IssueAuditReceipt, error) {
			return receipt, nil
		}),
	})
	if err != nil {
		t.Fatalf("seed tick: %v", err)
	}
	if seed.State != modelroute.IssueAuditLoopAdvancing || seed.Audited != 1 || seed.LedgerRows != 1 {
		t.Fatalf("seed tick = state %s audited %d ledger_rows %d, want ADVANCING/1/1", seed.State, seed.Audited, seed.LedgerRows)
	}

	// Captured dry-run over the fixture ledger: the seeded subject is honored as
	// already-settled and is NOT re-planned (cursor continuity into planning).
	snapshot := filepath.Join(dir, "snapshot.json")
	subjects, err := json.Marshal([]modelroute.IssueAuditLoopSubject{{IssueNumber: seededIssue, MarkerKey: "seed", Eligible: true}})
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	if err := os.WriteFile(snapshot, subjects, 0o600); err != nil {
		t.Fatalf("write snapshot: %v", err)
	}
	var dryOut, dryErr bytes.Buffer
	if code := runIssueAuditLoop(&dryOut, &dryErr, []string{
		"--snapshot", snapshot, "--ledger", ledger, "--cursor", cursor, "--json",
	}); code != 0 {
		t.Fatalf("dry-run exit %d, stderr=%s", code, dryErr.String())
	}
	var plan modelroute.IssueAuditLoopReport
	if err := json.Unmarshal(dryOut.Bytes(), &plan); err != nil {
		t.Fatalf("decode dry-run report: %v\nstdout=%s", err, dryOut.String())
	}
	if !plan.DryRun || plan.AlreadySettled != 1 || plan.Planned != 0 {
		t.Fatalf("dry-run over fixture ledger = dry=%t already_settled=%d planned=%d, want dry/1/0", plan.DryRun, plan.AlreadySettled, plan.Planned)
	}
	if plan.State != modelroute.IssueAuditLoopWait {
		t.Fatalf("dry-run state = %s, want WAIT (caught up, nothing re-planned)", plan.State)
	}

	// Two resumed read-only status reads: the durable state is identical across a
	// simulated restart (receipt + cursor continuity, no drift).
	readStatus := func(label string) (issueAuditLoopStatus, string) {
		var out, errBuf bytes.Buffer
		if code := runIssueAuditLoop(&out, &errBuf, []string{"--status", "--ledger", ledger, "--cursor", cursor, "--json"}); code != 0 {
			t.Fatalf("%s status exit %d, stderr=%s", label, code, errBuf.String())
		}
		var st issueAuditLoopStatus
		if err := json.Unmarshal(out.Bytes(), &st); err != nil {
			t.Fatalf("%s decode status: %v\nstdout=%s", label, err, out.String())
		}
		return st, out.String()
	}
	first, firstRaw := readStatus("resume-1")
	second, secondRaw := readStatus("resume-2")

	if !first.LedgerPresent || first.LedgerRows != 1 || first.UniqueAudits != 1 {
		t.Fatalf("resume-1 status = present=%t rows=%d unique=%d, want true/1/1", first.LedgerPresent, first.LedgerRows, first.UniqueAudits)
	}
	if len(first.SettledIssues) != 1 || first.SettledIssues[0] != seededIssue {
		t.Fatalf("resume-1 settled issues = %v, want [%d]", first.SettledIssues, seededIssue)
	}
	if firstRaw != secondRaw {
		t.Fatalf("status drifted across a resume:\nread1=%s\nread2=%s", firstRaw, secondRaw)
	}
	if first.LedgerRows != second.LedgerRows || first.UniqueAudits != second.UniqueAudits {
		t.Fatalf("ledger continuity broke: read1=%+v read2=%+v", first, second)
	}
}
