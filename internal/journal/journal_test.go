package journal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func testDenyEvent(tool, trace, args string) abi.Event {
	return abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{
			Tool:    tool,
			TraceID: trace,
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
	}
}

func TestMemoryJournalChainsRecentAndStreams(t *testing.T) {
	j := OpenMemory()
	j.clock = func() time.Time { return time.Unix(10, 20) }
	ch, cancel := j.Subscribe()
	defer cancel()

	j.Emit(testDenyEvent("send_email", "trace-a", `{"to":"x@y.com"}`))
	var row Row
	select {
	case row = <-ch:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for journal stream row")
	}
	if row.Seq != 1 || row.TSUnixNano != time.Unix(10, 20).UnixNano() {
		t.Fatalf("row anchor = seq %d ts %d", row.Seq, row.TSUnixNano)
	}
	if row.Kind != "DENY" || row.Tool != "send_email" || row.TraceID != "trace-a" {
		t.Fatalf("row fields = %+v", row)
	}
	if row.Hash == "" || row.ArgsDigest == "" {
		t.Fatalf("row did not carry hash/digest: %+v", row)
	}
	if n, err := VerifyRows(j.Recent(0)); err != nil || n != 1 {
		t.Fatalf("VerifyRows = n=%d err=%v, want 1 nil", n, err)
	}
}

func TestFileJournalReopensAndContinuesChain(t *testing.T) {
	path := t.TempDir() + "/audit.jsonl"
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.clock = func() time.Time { return time.Unix(1, 0) }
	j.Emit(testDenyEvent("send_email", "a", `{}`))
	if err := j.Close(); err != nil {
		t.Fatalf("Close first: %v", err)
	}

	j, err = Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	j.clock = func() time.Time { return time.Unix(2, 0) }
	j.Emit(testDenyEvent("Bash", "b", `{"cmd":"x"}`))
	if err := j.Close(); err != nil {
		t.Fatalf("Close second: %v", err)
	}
	if n, err := Verify(path); err != nil || n != 2 {
		t.Fatalf("Verify reopened journal = n=%d err=%v, want 2 nil", n, err)
	}
}

func TestVerifyDetectsTampering(t *testing.T) {
	path := t.TempDir() + "/audit.jsonl"
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.Emit(testDenyEvent("send_email", "a", `{}`))
	j.Emit(testDenyEvent("Bash", "b", `{"cmd":"x"}`))
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n, err := Verify(path); err != nil || n != 2 {
		t.Fatalf("Verify before tamper = n=%d err=%v", n, err)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := strings.Replace(string(b), `"tool":"Bash"`, `"tool":"Fish"`, 1)
	if tampered == string(b) {
		t.Fatal("test failed to modify journal bytes")
	}
	if err := os.WriteFile(path, []byte(tampered), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := Verify(path); err == nil {
		t.Fatal("Verify accepted a tampered journal")
	}
}

// Enable is the programmatic default-on path fak guard uses: it must create a
// missing parent directory, register a journal that actually records, leave a
// chain that Verify accepts, and be idempotent (a second Enable — even with a
// different path — is a no-op returning the FIRST journal, so the boot/first
// enablement always wins and the ABI emitter is never double-registered).
func TestEnableIsIdempotentCreatesDirAndVerifies(t *testing.T) {
	// Save/restore the package global so this test never leaks `active` into the
	// rest of the package's tests (Enable mutates a process-global).
	activeMu.Lock()
	saved := active
	active = nil
	activeMu.Unlock()
	defer func() {
		activeMu.Lock()
		active = saved
		activeMu.Unlock()
	}()

	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deeper", "guard-audit.jsonl") // parent dirs do NOT exist yet

	j, err := Enable(path)
	if err != nil {
		t.Fatalf("Enable: %v", err)
	}
	if j == nil || Active() != j {
		t.Fatalf("Active() must be the journal Enable returned")
	}
	if j.Path() != path {
		t.Fatalf("Path() = %q, want %q", j.Path(), path)
	}

	// A second Enable with a DIFFERENT path is a no-op: the first enablement wins.
	j2, err := Enable(filepath.Join(dir, "other.jsonl"))
	if err != nil {
		t.Fatalf("second Enable: %v", err)
	}
	if j2 != j {
		t.Fatal("Enable must be idempotent: the first/boot journal wins")
	}
	if _, err := os.Stat(filepath.Join(dir, "other.jsonl")); err == nil {
		t.Fatal("idempotent Enable must NOT open the second path")
	}

	// It records a real decision and the on-disk chain verifies.
	j.Emit(testDenyEvent("Bash", "trace-x", `{"command":"rm -rf /"}`))
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n, err := Verify(path); err != nil || n != 1 {
		t.Fatalf("Verify = n=%d err=%v, want 1 nil", n, err)
	}
}

func TestNonAuditEventsAreIgnored(t *testing.T) {
	j := OpenMemory()
	j.Emit(abi.Event{Kind: abi.EvSubmit, Call: &abi.ToolCall{Tool: "read"}})
	if got := j.Recent(0); len(got) != 0 {
		t.Fatalf("non-audit event wrote rows: %+v", got)
	}
}
