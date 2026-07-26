package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// writeReceiptJournal lays down a real hash-chained guard journal on disk with one
// ALLOW decision and one DENY for trace, and returns its path. The rows carry the
// kernel's genuine chain links, so `fak session receipt` reads and verifies a real
// journal, not a hand-forged one.
func writeReceiptJournal(t *testing.T, trace string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	allow := &abi.ToolCall{Tool: "search_kb", TraceID: trace, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: allow, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})
	deny := &abi.ToolCall{Tool: "send_email", TraceID: trace, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"x@y.com"}`)}}
	j.Emit(abi.Event{Kind: abi.EvDeny, Call: deny, Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "floor"}})
	if err := j.Close(); err != nil {
		t.Fatalf("close journal: %v", err)
	}
	return path
}

// TestSessionReceiptHumanVerifies is the CLI happy path: the human render labels every
// field WITNESSED/OBSERVED, carries the journal-derived denial count, and reports the
// signature verified against the journal (exit 0).
func TestSessionReceiptHumanVerifies(t *testing.T) {
	path := writeReceiptJournal(t, "trace-a")
	var stdout, stderr bytes.Buffer
	code := runSessionReceipt(&stdout, &stderr, []string{"trace-a", "--journal", path, "--turns", "3"})
	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"WITNESSED", "OBSERVED", "denials_total", "verify: OK"} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q:\n%s", want, out)
		}
	}
}

// TestSessionReceiptJSONVerified proves the JSON envelope carries the folded receipt
// and a machine-branchable verified=true, with the WITNESSED denial count re-derived
// from the journal (one DENY row).
func TestSessionReceiptJSONVerified(t *testing.T) {
	path := writeReceiptJournal(t, "trace-a")
	var stdout, stderr bytes.Buffer
	code := runSessionReceipt(&stdout, &stderr, []string{"trace-a", "--journal", path, "--json"})
	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr=%s", code, stderr.String())
	}
	var doc sessionReceiptDoc
	if err := json.Unmarshal(stdout.Bytes(), &doc); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, stdout.String())
	}
	if !doc.Verified {
		t.Fatalf("verified=false, want true (verify_error=%q)", doc.VerifyError)
	}
	if got, _ := doc.Receipt.FieldValue("denials_total"); got != "1" {
		t.Fatalf("denials_total = %q, want \"1\"", got)
	}
}

// TestSessionReceiptTamperFailsAtCLI proves the CLI surfaces a mutated journal: flip a
// chained field on disk (the receipt's authority is the journal, not the receipt), and
// verification fails with a non-zero exit — the whole point of a non-forgeable receipt.
func TestSessionReceiptTamperFailsAtCLI(t *testing.T) {
	path := writeReceiptJournal(t, "trace-a")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read journal: %v", err)
	}
	// Mutate a chained content field (the denied tool) without recomputing the stored
	// hash -> the chain no longer verifies.
	tampered := bytes.Replace(raw, []byte("send_email"), []byte("delete_all"), 1)
	if bytes.Equal(tampered, raw) {
		t.Fatal("test setup: expected to mutate the journal bytes")
	}
	if err := os.WriteFile(path, tampered, 0o644); err != nil {
		t.Fatalf("rewrite journal: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runSessionReceipt(&stdout, &stderr, []string{"trace-a", "--journal", path})
	if code == 0 {
		t.Fatalf("a tampered journal verified (exit 0):\n%s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "verify: FAIL") {
		t.Fatalf("output did not report a failed verification:\n%s", stdout.String())
	}
}

// TestSessionReceiptUsage rejects a missing trace with the usage exit code.
func TestSessionReceiptUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runSessionReceipt(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("exit %d on missing trace, want 2", code)
	}
	if code := runSessionReceipt(&stdout, &stderr, []string{"--json"}); code != 2 {
		t.Fatalf("exit %d when a flag precedes the trace, want 2", code)
	}
}
