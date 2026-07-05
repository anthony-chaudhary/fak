package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// chainHashForTest mirrors journal.chainHash (unexported): sha256 over the previous row's
// hash chained with this row's content fields in declaration order, unit-separated. The
// TestDiagnose_ChainHashMatchesJournalVerifier test PINS it to the real verifier, so a
// drift in the journal pre-image fails loudly here instead of silently weakening the
// reconstruction. The diagnostic itself does NOT depend on this — it only calls the
// exported journal.VerifyRows — but the test fixtures need to mint sound rows.
func chainHashForTest(prev string, r journal.Row) string {
	h := sha256.New()
	h.Write([]byte(prev))
	fmt.Fprintf(h, "\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s",
		r.Seq, r.TSUnixNano, r.Kind, r.Tool, r.TraceID, r.Verdict,
		r.Reason, r.By, r.Taint, r.ArgsDigest, r.ResultDigest)
	return hex.EncodeToString(h.Sum(nil))
}

// mintRow stamps seq/prevhash/hash onto a row so it links onto prev, returning the row and
// its hash (the next prev). It is the fixture-builder for a sound chain.
func mintRow(seq uint64, prevHash string, r journal.Row) (journal.Row, string) {
	r.Seq = seq
	r.TSUnixNano = int64(seq) * 1000
	r.PrevHash = prevHash
	r.Hash = chainHashForTest(prevHash, r)
	return r, r.Hash
}

func TestDiagnose_ChainHashMatchesJournalVerifier(t *testing.T) {
	// Mint a 3-row chain with our local hash, then assert the REAL journal.VerifyRows
	// accepts it. If chainHashForTest disagreed with journal's pre-image, VerifyRows would
	// reject the hashes — so a green test proves the fixture mints genuinely-sound rows.
	var rows []journal.Row
	prev := ""
	for i := uint64(1); i <= 3; i++ {
		r, h := mintRow(i, prev, journal.Row{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE"})
		rows = append(rows, r)
		prev = h
	}
	if n, err := journal.VerifyRows(rows); err != nil || n != 3 {
		t.Fatalf("fixture rows not accepted by journal.VerifyRows: n=%d err=%v", n, err)
	}
}

func soundChain(t *testing.T, n int) []journal.Row {
	t.Helper()
	var rows []journal.Row
	prev := ""
	for i := uint64(1); i <= uint64(n); i++ {
		r, h := mintRow(i, prev, journal.Row{Kind: "DECIDE", Tool: "Read", Verdict: "ALLOW", Reason: "NONE"})
		rows = append(rows, r)
		prev = h
	}
	return rows
}

func TestDiagnoseRows_SoundSingleChain(t *testing.T) {
	rows := soundChain(t, 5)
	d := diagnoseRows("x", rows)
	if d.Verdict != diagVerdictSound {
		t.Fatalf("want SOUND, got %s (%+v)", d.Verdict, d)
	}
	if !d.LinearOK || d.SessionTips != 1 {
		t.Fatalf("sound chain should be linear-ok, 1 tip: %+v", d)
	}
}

func TestDiagnoseRows_InterleavedNotTampered(t *testing.T) {
	// Two sessions share a 3-row prefix, then BRANCH: both continue from the prefix head
	// with their OWN seq counter (the real concurrent-writer signature: duplicate seq, two
	// children off one parent). Interleave them in file order.
	prefix := soundChain(t, 3)
	head := prefix[len(prefix)-1].Hash

	// Session A continues 4,5 off head.
	a4, a4h := mintRow(4, head, journal.Row{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE"})
	a5, _ := mintRow(5, a4h, journal.Row{Kind: "DECIDE", Tool: "Edit", Verdict: "ALLOW", Reason: "NONE"})
	// Session B independently continues 4,5 off the SAME head (its counter is also at 3).
	b4, b4h := mintRow(4, head, journal.Row{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "SELF_MODIFY"})
	b5, _ := mintRow(5, b4h, journal.Row{Kind: "DECIDE", Tool: "Grep", Verdict: "ALLOW", Reason: "NONE"})

	// File order interleaves the two sessions' appends.
	rows := []journal.Row{prefix[0], prefix[1], prefix[2], a4, b4, a5, b5}

	// Sanity: the linear verifier SHOULD reject this (the bug we are diagnosing).
	if _, err := journal.VerifyRows(rows); err == nil {
		t.Fatal("expected interleaved rows to fail linear VerifyRows")
	}

	d := diagnoseRows("x", rows)
	if d.Verdict != diagVerdictInterleaved {
		t.Fatalf("want INTERLEAVED, got %s (first_break=%s, %+v)", d.Verdict, d.FirstBreak, d)
	}
	if d.SessionTips != 2 {
		t.Fatalf("want 2 session tips, got %d", d.SessionTips)
	}
	if d.BranchPoints != 1 {
		t.Fatalf("want 1 branch point, got %d", d.BranchPoints)
	}
	if d.BrokenChains != 0 || d.OrphanRows != 0 {
		t.Fatalf("interleave must have 0 broken/orphan, got broken=%d orphan=%d", d.BrokenChains, d.OrphanRows)
	}
	if d.IntactChains != 2 {
		t.Fatalf("want 2 intact chains, got %d", d.IntactChains)
	}
}

func TestDiagnoseRows_TamperedEditedRow(t *testing.T) {
	rows := soundChain(t, 5)
	// Flip a content field in the middle WITHOUT recomputing the hash — a forger's edit.
	rows[2].Verdict = "DENY"
	d := diagnoseRows("x", rows)
	if d.Verdict != diagVerdictTampered {
		t.Fatalf("want TAMPERED for an edited row, got %s (%+v)", d.Verdict, d)
	}
	if d.BrokenChains == 0 {
		t.Fatalf("edited row should break its session chain: %+v", d)
	}
}

func TestDiagnoseRows_TamperedDroppedRow(t *testing.T) {
	rows := soundChain(t, 6)
	// Drop a middle row: its child now references a parent hash absent from the file.
	rows = append(rows[:3], rows[4:]...)
	d := diagnoseRows("x", rows)
	if d.Verdict != diagVerdictTampered {
		t.Fatalf("want TAMPERED for a dropped row, got %s (%+v)", d.Verdict, d)
	}
	if d.OrphanRows == 0 {
		t.Fatalf("dropped row should orphan its child: %+v", d)
	}
}

func TestDiagnoseRows_EmptyJournalIsSound(t *testing.T) {
	d := diagnoseRows("x", nil)
	if d.Verdict != diagVerdictSound || !d.LinearOK {
		t.Fatalf("empty journal should be SOUND/linear-ok: %+v", d)
	}
}

func TestRunAuditDiagnose_RenderReportsEmptyJournal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, nil)

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("empty journal should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	for _, want := range []string{
		"rows           : 0",
		"empty journal",
		"no adjudicated row",
		"child startup/auth",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestDiagnoseRows_ReportsRepeatedAllowedCallLivelock(t *testing.T) {
	var rows []journal.Row
	prev := ""
	for i := uint64(1); i <= 4; i++ {
		r, h := mintRow(i, prev, journal.Row{
			Kind:       "DECIDE",
			Tool:       "shell_command",
			Verdict:    "ALLOW",
			Reason:     "NONE",
			ArgsDigest: "sha256:repeat",
			ArgsLabel:  "command=python dispatch_status.py",
			CallSeq:    100 + i,
		})
		rows = append(rows, r)
		prev = h
	}
	d := diagnoseRows("x", rows)
	if d.Verdict != diagVerdictSound || len(d.LivelockCandidates) != 1 {
		t.Fatalf("sound repeated-allow journal should expose one livelock candidate: %+v", d)
	}
	c := d.LivelockCandidates[0]
	if c.Tool != "shell_command" || c.Verdict != "ALLOW" || c.DigestKind != "args_digest" || c.Digest != "sha256:repeat" || c.Count != 4 || c.LongestRun != 4 {
		t.Fatalf("wrong repeated allowed candidate: %+v", c)
	}
	if c.CallLabel != "command=python dispatch_status.py" {
		t.Fatalf("candidate should carry the redacted call label, got %+v", c)
	}
	if c.FirstSeq != 1 || c.LastSeq != 4 || c.FirstCallSeq != 101 || c.LastCallSeq != 104 {
		t.Fatalf("candidate should carry row/call sequence bounds, got %+v", c)
	}
}

func TestRunAuditDiagnose_RenderAndJSONReportRepeatedAllowedCallLabel(t *testing.T) {
	var rows []journal.Row
	prev := ""
	for i := uint64(1); i <= 3; i++ {
		r, h := mintRow(i, prev, journal.Row{
			Kind:       "DECIDE",
			Tool:       "shell_command",
			Verdict:    "ALLOW",
			Reason:     "NONE",
			ArgsDigest: "sha256:repeat",
			ArgsLabel:  "command=git status",
		})
		rows = append(rows, r)
		prev = h
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("sound allowed-call journal should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"loop candidates",
		"shell_command",
		"call_label=command=git status",
		"args_digest=sha256:repeat",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "sk-test-secret") {
		t.Fatalf("render leaked raw secret-shaped text:\n%s", out)
	}

	stdout.Reset()
	stderr.Reset()
	code = runAuditDiagnose(&stdout, &stderr, path, true)
	if code != 0 {
		t.Fatalf("sound allowed-call journal --json should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	var d auditDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("--json output not parseable: %v\n%s", err, stdout.String())
	}
	if len(d.LivelockCandidates) != 1 || d.LivelockCandidates[0].CallLabel != "command=git status" {
		t.Fatalf("json candidate missing call_label: %+v", d.LivelockCandidates)
	}
}

func TestRunAuditDiagnose_RenderReportsRepeatedQuarantine(t *testing.T) {
	var rows []journal.Row
	prev := ""
	for i := uint64(1); i <= 3; i++ {
		r, h := mintRow(i, prev, journal.Row{
			Kind:         "QUARANTINE",
			Tool:         "tool_result",
			Verdict:      "QUARANTINE",
			Reason:       "SECRET_EXFIL",
			ArgsDigest:   "sha256:stable-poison",
			ResultDigest: fmt.Sprintf("sha256:rewrite-%d", i),
		})
		rows = append(rows, r)
		prev = h
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("sound quarantine journal should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	for _, want := range []string{
		"loop candidates",
		"tool_result",
		"SECRET_EXFIL",
		"args_digest=sha256:stable-poison",
		"longest_run=3",
		"seq=1..3",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunAuditDiagnose_RenderReportsChildCrashClass(t *testing.T) {
	r, _ := mintRow(1, "", journal.Row{
		Kind:    "CHILD_CRASH",
		Tool:    "opencode",
		TraceID: "guard",
		Reason:  "NONZERO_EXIT",
		By:      "guard-supervisor",
	})
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, []journal.Row{r})

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("sound child-crash journal should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	for _, want := range []string{
		"floor activity : 1 decision",
		"child crash:",
		"NONZERO_EXIT",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, stdout.String())
		}
	}
}

// TestRunAuditDiagnose_InterleavedExitsZero writes an interleaved-but-intact journal to a
// temp file and confirms the command exits 0 (a fleet user's default journal is trustworthy)
// and the render names the INTERLEAVED verdict.
func TestRunAuditDiagnose_InterleavedExitsZero(t *testing.T) {
	prefix := soundChain(t, 2)
	head := prefix[len(prefix)-1].Hash
	a3, _ := mintRow(3, head, journal.Row{Kind: "DECIDE", Tool: "Bash", Verdict: "ALLOW", Reason: "NONE"})
	b3, _ := mintRow(3, head, journal.Row{Kind: "DENY", Tool: "Bash", Verdict: "DENY", Reason: "POLICY_BLOCK"})
	rows := []journal.Row{prefix[0], prefix[1], a3, b3}

	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 0 {
		t.Fatalf("interleaved-but-intact journal should exit 0, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "INTERLEAVED") {
		t.Fatalf("render should name INTERLEAVED verdict:\n%s", stdout.String())
	}
	// The friction fold should have surfaced the one POLICY_BLOCK deny.
	if !strings.Contains(stdout.String(), "POLICY_BLOCK") {
		t.Fatalf("render should fold the floor's POLICY_BLOCK deny:\n%s", stdout.String())
	}
}

func TestRunAuditDiagnose_TamperedExitsOne(t *testing.T) {
	rows := soundChain(t, 4)
	rows[1].Reason = "FORGED" // edit without rehash
	dir := t.TempDir()
	path := filepath.Join(dir, "guard-audit.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	code := runAuditDiagnose(&stdout, &stderr, path, false)
	if code != 1 {
		t.Fatalf("tampered journal should exit 1, got %d", code)
	}
	if !strings.Contains(stdout.String(), "TAMPERED") {
		t.Fatalf("render should name TAMPERED verdict:\n%s", stdout.String())
	}
}

func TestRunAuditDiagnose_JSONShape(t *testing.T) {
	rows := soundChain(t, 3)
	dir := t.TempDir()
	path := filepath.Join(dir, "g.jsonl")
	writeRowsFile(t, path, rows)

	var stdout, stderr bytes.Buffer
	if code := runAuditDiagnose(&stdout, &stderr, path, true); code != 0 {
		t.Fatalf("sound journal --json should exit 0, got %d", code)
	}
	var d auditDiagnosis
	if err := json.Unmarshal(stdout.Bytes(), &d); err != nil {
		t.Fatalf("--json output not parseable: %v\n%s", err, stdout.String())
	}
	if d.Verdict != diagVerdictSound || d.Rows != 3 {
		t.Fatalf("json payload wrong: %+v", d)
	}
}

func writeRowsFile(t *testing.T, path string, rows []journal.Row) {
	t.Helper()
	var b bytes.Buffer
	for _, r := range rows {
		raw, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(raw)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
