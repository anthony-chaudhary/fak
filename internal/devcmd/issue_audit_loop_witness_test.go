package devcmd

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// updateAuditLoopCLIWitness regenerates the committed operator-CLI witness from
// live `fak-dev issue audit-loop` output rather than hand-authoring its bytes:
//
//	go test ./internal/devcmd -run TestRunIssueAuditLoopCLIWitnessGolden -update-audit-loop-cli-witness
var updateAuditLoopCLIWitness = flag.Bool("update-audit-loop-cli-witness", false, "regenerate the fak-dev issue audit-loop operator CLI witness golden")

// TestRunIssueAuditLoopCLIWitnessGolden pins the operator front-door for #3856:
// a captured `fak-dev issue audit-loop` dry-run over a fixture discovery snapshot
// (plans the batch, names the DARK subject, and writes nothing) beside a
// read-only `--status` fold of a fresh (empty) durable ledger + cursor. The
// deep cursor/receipt continuity across resumed ticks is captured by the
// modelroute witness (TestIssueAuditLoopWitnessGolden); this pins the exact
// bytes an operator sees from the command surface, so a regression in the
// front-door JSON is a visible diff, not a silent drift.
func TestRunIssueAuditLoopCLIWitnessGolden(t *testing.T) {
	const (
		snapshotPath = "testdata/audit_loop_snapshot.json"
		goldenPath   = "testdata/audit_loop_cli_witness.txt"
	)
	dir := t.TempDir()
	ledger := filepath.Join(dir, "receipts.jsonl")
	cursor := filepath.Join(dir, "cursor.json")

	var b bytes.Buffer
	b.WriteString("# fak-dev issue audit-loop — captured operator CLI witness (#3856)\n")
	b.WriteString("#\n")
	b.WriteString("# Regenerate: go test ./internal/devcmd -run TestRunIssueAuditLoopCLIWitnessGolden -update-audit-loop-cli-witness\n")
	b.WriteString("#\n")
	b.WriteString("# The default operator tick is a dry-run: it plans the bounded batch, names the\n")
	b.WriteString("# ineligible (DARK) subject, and writes neither ledger nor cursor. --status folds\n")
	b.WriteString("# the durable state read-only; a fresh install is a valid empty status, not an error.\n\n")

	// Default operator tick: dry-run plan over the fixture snapshot.
	var stdout, stderr bytes.Buffer
	if rc := runIssueAuditLoop(&stdout, &stderr, []string{
		"--snapshot", snapshotPath, "--ledger", ledger, "--cursor", cursor, "--json",
	}); rc != 0 {
		t.Fatalf("dry-run rc = %d, stderr=%s", rc, stderr.String())
	}
	b.WriteString("$ fak-dev issue audit-loop --snapshot audit_loop_snapshot.json --json\n")
	b.Write(stdout.Bytes())
	b.WriteString("\n")

	// A dry-run must not create the ledger or cursor.
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("dry-run created the ledger (err=%v)", err)
	}
	if _, err := os.Stat(cursor); !os.IsNotExist(err) {
		t.Fatalf("dry-run created the cursor (err=%v)", err)
	}

	// Read-only status over the fresh (empty) durable state.
	stdout.Reset()
	stderr.Reset()
	if rc := runIssueAuditLoop(&stdout, &stderr, []string{
		"--status", "--ledger", ledger, "--cursor", cursor, "--json",
	}); rc != 0 {
		t.Fatalf("status rc = %d, stderr=%s", rc, stderr.String())
	}
	b.WriteString("$ fak-dev issue audit-loop --status --json\n")
	b.Write(stdout.Bytes())

	got := b.Bytes()

	if *updateAuditLoopCLIWitness {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote CLI witness golden %s (%d bytes)", goldenPath, len(got))
		return
	}

	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read CLI witness golden (run with -update-audit-loop-cli-witness to create): %v", err)
	}
	lf := func(b []byte) []byte { return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")) }
	if !bytes.Equal(lf(got), lf(want)) {
		t.Fatalf("audit-loop CLI witness drifted from %s.\n--- got ---\n%s\n--- want ---\n%s", goldenPath, got, want)
	}
}
