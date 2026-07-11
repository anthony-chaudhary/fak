package main

import (
	"bytes"
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
