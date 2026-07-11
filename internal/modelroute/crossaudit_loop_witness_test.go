package modelroute

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// updateAuditLoopWitness regenerates the committed audit-loop witness transcript
// from the live loop rather than hand-authoring its bytes:
//
//	go test ./internal/modelroute -run TestIssueAuditLoopWitnessGolden -update-audit-loop-witness
var updateAuditLoopWitness = flag.Bool("update-audit-loop-witness", false, "regenerate the audit-loop dry-run + resumed-tick witness golden")

// TestIssueAuditLoopWitnessGolden pins the #3856 Witness as a durable,
// human-inspectable regression artifact: a captured dry-run plus two resumed
// ticks over one on-disk fixture ledger + cursor, showing cursor and receipt
// continuity survive a simulated crash/restart. It is the operator-facing
// counterpart of the assertion-heavy integration tests in
// crossaudit_loop_test.go — the same fake-clock/fake-provider scenario, captured
// as a transcript so a reviewer can read the continuity directly.
//
// Scenario (mirrors TestIssueAuditLoopAuditsEachEligibleSubjectExactlyOnce):
// four discovered subjects — #100 PASS, #101 REFUTE, #102 ineligible (DARK), and
// #103 whose provider is UNAVAILABLE on tick 1 and recovers by tick 3. The
// transcript deliberately captures only deterministic aggregates (states, counts,
// settled/dead-letter issue sets, ledger rows, unique audits, verdict tallies);
// row hashes and receipt digests embed wall-clock append time and are omitted so
// the golden is stable across runs.
func TestIssueAuditLoopWitnessGolden(t *testing.T) {
	const goldenPath = "testdata/audit_loop_witness.txt"

	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}

	auditor := newCountingAuditor()
	auditor.receipts[100] = loopTestReceipt(t, 100, CrossAuditPass)
	auditor.receipts[101] = loopTestReceipt(t, 101, CrossAuditRefute)
	auditor.errs[103] = errors.New("provider connection refused") // UNAVAILABLE on tick 1

	snapshot := []IssueAuditLoopSubject{
		eligible(100),
		eligible(101),
		{IssueNumber: 102, Eligible: false, IneligibleReason: "not a closed dispatch leaf"}, // DARK
		eligible(103),
	}
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return append([]IssueAuditLoopSubject(nil), snapshot...), nil
	})
	cfg := IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath,
		BatchCap: 10, MaxAttempts: 3, BackoffBase: time.Minute,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
	}

	var b strings.Builder
	b.WriteString("# fak issue audit-loop — captured witness (#3856)\n")
	b.WriteString("#\n")
	b.WriteString("# Regenerate: go test ./internal/modelroute -run TestIssueAuditLoopWitnessGolden -update-audit-loop-witness\n")
	b.WriteString("#\n")
	b.WriteString("# Scenario: 4 discovered subjects — #100 PASS, #101 REFUTE, #102 ineligible (DARK),\n")
	b.WriteString("# #103 provider UNAVAILABLE on tick 1, recovers by tick 3. A fake clock and fake\n")
	b.WriteString("# provider make every tick deterministic; the ledger + cursor live on disk and are\n")
	b.WriteString("# reloaded between ticks to simulate a crash/restart resume. Row hashes and receipt\n")
	b.WriteString("# digests embed wall-clock append time and are omitted so this golden stays stable.\n\n")

	// Dry-run: plan only, no lease/audit, and no ledger/cursor write.
	dryCfg := cfg
	dryCfg.DryRun = true
	dryRep, err := RunIssueAuditLoopTick(context.Background(), dryCfg)
	if err != nil {
		t.Fatalf("dry-run tick: %v", err)
	}
	ledgerAfterDry := fileExists(ledgerPath)
	cursorAfterDry := fileExists(cursorPath)
	b.WriteString("== dry-run (plan only; no lease, no audit, no ledger/cursor write) ==\n")
	fmt.Fprintf(&b, "state=%s dry_run=%t discovered=%d eligible=%d planned=%d planned_subjects=%s dark_subjects=%s\n",
		dryRep.State, dryRep.DryRun, dryRep.Discovered, dryRep.Eligible, dryRep.Planned,
		intsField(dryRep.PlannedSubjects), intsField(dryRep.DarkSubjects))
	fmt.Fprintf(&b, "side effects: ledger_written=%t cursor_written=%t (both must be false)\n\n", ledgerAfterDry, cursorAfterDry)

	// Tick 1 (live): audits #100/#101, #102 is DARK, #103 is UNAVAILABLE and retained.
	rep1, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 1: %v", err)
	}
	writeAuditLoopTickWitness(t, &b, "tick 1 (live)", rep1, cursorPath)

	// Tick 2 (resume: cursor reloaded from disk, clock unchanged). #100/#101 already
	// settled -> not re-audited; #103 still inside its retry backoff -> retained.
	rep2, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 2: %v", err)
	}
	writeAuditLoopTickWitness(t, &b, "tick 2 (resume: cursor reloaded, still in backoff)", rep2, cursorPath)

	// Tick 3 (resume: clock advanced past backoff, provider recovered).
	clock.Advance(2 * time.Minute)
	auditor.errs[103] = nil
	auditor.receipts[103] = loopTestReceipt(t, 103, CrossAuditPass)
	rep3, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 3: %v", err)
	}
	writeAuditLoopTickWitness(t, &b, "tick 3 (resume: clock advanced, provider recovered)", rep3, cursorPath)

	// Final ledger verify: three unique audits, chain intact.
	v, err := VerifyAuditReceiptLedger(ledgerPath)
	if err != nil {
		t.Fatalf("final ledger verify: %v", err)
	}
	b.WriteString("== final ledger verify ==\n")
	fmt.Fprintf(&b, "chain=OK rows=%d unique_audits=%d verdicts: pass=%d refute=%d inconclusive=%d unavailable=%d\n",
		v.Cursor.Rows, v.UniqueAudits,
		v.VerdictCounts[CrossAuditPass], v.VerdictCounts[CrossAuditRefute],
		v.VerdictCounts[CrossAuditInconclusive], v.VerdictCounts[CrossAuditUnavailable])

	got := []byte(b.String())

	// Guard the continuity the transcript claims so the golden can never be
	// regenerated green over a broken loop: dry-run leaves no files, each eligible
	// subject is audited exactly once, the UNAVAILABLE row is retained then settled,
	// and the ledger grows by exactly the terminal verdicts.
	if ledgerAfterDry || cursorAfterDry {
		t.Fatalf("dry-run left side effects: ledger=%t cursor=%t", ledgerAfterDry, cursorAfterDry)
	}
	if rep1.State != IssueAuditLoopAdvancing || rep1.Audited != 2 || rep1.Unavailable != 1 || rep1.LedgerRows != 2 {
		t.Fatalf("tick 1 continuity = %+v", rep1)
	}
	if rep2.State != IssueAuditLoopWait || rep2.Audited != 0 || rep2.AlreadySettled != 2 || rep2.LedgerRows != 2 {
		t.Fatalf("tick 2 continuity = %+v", rep2)
	}
	if rep3.State != IssueAuditLoopAdvancing || rep3.Audited != 1 || rep3.LedgerRows != 3 {
		t.Fatalf("tick 3 continuity = %+v", rep3)
	}
	if auditor.calls[102] != 0 || auditor.calls[103] != 2 {
		t.Fatalf("audit calls: #102=%d (want 0) #103=%d (want 2)", auditor.calls[102], auditor.calls[103])
	}
	if v.UniqueAudits != 3 {
		t.Fatalf("final unique audits = %d, want 3", v.UniqueAudits)
	}

	if *updateAuditLoopWitness {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote witness golden %s (%d bytes)", goldenPath, len(got))
		return
	}

	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read witness golden (run with -update-audit-loop-witness to create): %v", err)
	}
	lf := func(b []byte) []byte { return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")) }
	if !bytes.Equal(lf(got), lf(want)) {
		t.Fatalf("audit-loop witness drifted from %s.\n--- got ---\n%s\n--- want ---\n%s", goldenPath, got, want)
	}
}

// writeAuditLoopTickWitness renders one live tick's deterministic continuity: the
// typed state, the audit/verdict tallies, and the durable cursor's settled and
// dead-letter issue sets after the tick was persisted to disk.
func writeAuditLoopTickWitness(t *testing.T, b *strings.Builder, title string, rep IssueAuditLoopReport, cursorPath string) {
	t.Helper()
	cursor, err := LoadIssueAuditLoopCursor(cursorPath)
	if err != nil {
		t.Fatalf("%s: reload cursor: %v", title, err)
	}
	fmt.Fprintf(b, "== %s ==\n", title)
	fmt.Fprintf(b, "state=%s audited=%d pass=%d refute=%d inconclusive=%d unavailable=%d already_settled=%d deferred=%d dead_lettered=%d\n",
		rep.State, rep.Audited, rep.Passed, rep.Refuted, rep.Inconclusive, rep.Unavailable, rep.AlreadySettled, rep.Deferred, rep.DeadLettered)
	fmt.Fprintf(b, "dark_subjects=%s ledger_rows=%d cursor.settled=%s cursor.dead_letters=%s dead_letter_queue=%s\n\n",
		intsField(rep.DarkSubjects), rep.LedgerRows, mapKeysField(cursor.Settled), mapKeysField(cursor.DeadLetters), intsField(rep.DeadLetterQueue))
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// intsField renders a slice of issue numbers as a sorted, stable [a b c] field.
func intsField(xs []int) string {
	cp := append([]int(nil), xs...)
	sort.Ints(cp)
	return bracketInts(cp)
}

// mapKeysField renders the (unordered) keys of a cursor map as a sorted, stable
// [a b c] field so the transcript is deterministic regardless of map iteration.
func mapKeysField[V any](m map[int]V) string {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return bracketInts(keys)
}

func bracketInts(xs []int) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%d", x)
	}
	return "[" + strings.Join(parts, " ") + "]"
}
