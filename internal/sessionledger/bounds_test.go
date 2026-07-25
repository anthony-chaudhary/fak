package sessionledger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAppendCostIsLinearNotQuadratic is the regression test for the blowup this
// package was rewritten to fix. The old format re-marshalled the WHOLE state on
// every append, so writing n entries of size s cost O(n^2 * s) bytes of I/O and
// allocation. Appending one line makes it O(n * s).
//
// The assertion is on the FILE, because the file is what the old code rewrote.
// With n=300 and ~1 KB entries, linear is ~300 KB; quadratic is ~45 MB. The 4x
// slack absorbs JSON framing and hash fields without coming close to quadratic.
func TestAppendCostIsLinearNotQuadratic(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	const n = 300
	payload := json.RawMessage(`{"body":"` + strings.Repeat("x", 1000) + `"}`)
	for i := 0; i < n; i++ {
		if _, err := l.Append("t", "user_message", payload); err != nil {
			t.Fatal(err)
		}
	}

	st, err := os.Stat(filepath.Join(dir, LogName))
	if err != nil {
		t.Fatal(err)
	}
	linear := int64(n * (len(payload) + 200)) // payload + framing/hash headroom
	if st.Size() > 4*linear {
		t.Fatalf("file is %d bytes; linear budget is ~%d. The whole-state rewrite is back.",
			st.Size(), linear)
	}
	// And the same n appends must still be one readable chain.
	if got := l.NodeCount(); got != n {
		t.Fatalf("NodeCount = %d, want %d", got, n)
	}
}

// TestOversizedContentIsElidedNotStored pins the second half of the fix: a caller
// that hands over a full request body gets a bounded provenance stub, not the body.
func TestOversizedContentIsElidedNotStored(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	big := json.RawMessage(`{"body":"` + strings.Repeat("y", MaxContentBytes*2) + `"}`)
	e, err := l.Append("t", "user_message", big)
	if err != nil {
		t.Fatal(err)
	}
	if len(e.Content) >= len(big) {
		t.Fatalf("content not elided: %d bytes retained", len(e.Content))
	}
	var stub struct {
		Elided bool   `json:"elided"`
		Bytes  int    `json:"bytes"`
		SHA256 string `json:"sha256"`
	}
	if err := json.Unmarshal(e.Content, &stub); err != nil {
		t.Fatalf("stub is not JSON: %v", err)
	}
	if !stub.Elided || stub.Bytes != len(big) || len(stub.SHA256) != 64 {
		t.Fatalf("stub lost provenance: %+v", stub)
	}
	// The elided entry must still verify as a chain member.
	chain, err := l.Chain("t")
	if err != nil {
		t.Fatal(err)
	}
	if err := Verify(chain); err != nil {
		t.Fatalf("elided entry broke chain verification: %v", err)
	}
}

// TestReopenRebuildsHeadsAndChain proves the append-only log is a real durable
// store, not just a write sink: a fresh process sees the same heads and chain.
func TestReopenRebuildsHeadsAndChain(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"turn_begin", "user_message", "turn_complete"} {
		if _, err := l.Append("a", k, json.RawMessage(`{"v":1}`)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := l.Fork("a", "b"); err != nil {
		t.Fatal(err)
	}
	if _, err := l.Append("b", "user_message", json.RawMessage(`{"v":2}`)); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if reopened.Head("a") != l.Head("a") || reopened.Head("b") != l.Head("b") {
		t.Fatal("heads did not survive reopen")
	}
	chain, err := reopened.Chain("b")
	if err != nil {
		t.Fatal(err)
	}
	if len(chain) != 4 {
		t.Fatalf("chain length %d, want 4", len(chain))
	}
	if err := Verify(chain); err != nil {
		t.Fatalf("reopened chain does not verify: %v", err)
	}
}

// TestTornLineIsSkippedNotFatal: a process killed mid-write must not make the
// whole ledger unreadable. The old whole-file rename was atomic; an append-only
// log trades that for partial-line risk, so the reader has to be tolerant.
func TestTornLineIsSkippedNotFatal(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Append("a", "user_message", json.RawMessage(`{"v":1}`)); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, LogName)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"trace":"a","hash":"deadbe`); err != nil { // torn
		t.Fatal(err)
	}
	f.Close()

	reopened, err := Open(dir)
	if err != nil {
		t.Fatalf("torn line made the ledger unreadable: %v", err)
	}
	if reopened.Head("a") != l.Head("a") {
		t.Fatal("torn line corrupted the surviving head")
	}
}

// TestRotationBoundsTheFile: the log must not grow without limit on disk.
func TestRotationBoundsTheFile(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Write comfortably past MaxFileBytes using max-size entries.
	payload := json.RawMessage(`{"body":"` + strings.Repeat("z", MaxContentBytes-32) + `"}`)
	writes := (MaxFileBytes / len(payload)) + 64
	for i := 0; i < writes; i++ {
		if _, err := l.Append("t", "user_message", payload); err != nil {
			t.Fatal(err)
		}
	}
	st, err := os.Stat(filepath.Join(dir, LogName))
	if err != nil {
		t.Fatal(err)
	}
	if st.Size() > MaxFileBytes {
		t.Fatalf("live log is %d bytes, past the %d ceiling", st.Size(), MaxFileBytes)
	}
	if _, err := os.Stat(filepath.Join(dir, LogName+".1")); err != nil {
		t.Fatalf("expected a rotated generation: %v", err)
	}
}

// TestLegacyWholeStateFileIsRetiredNotLoaded: the pre-JSONL ledger.json reached
// 389 MB in the field. Loading it would reproduce the blowup on the first open,
// so it is moved aside and left for forensics.
func TestLegacyWholeStateFileIsRetiredNotLoaded(t *testing.T) {
	dir := t.TempDir()
	legacy := filepath.Join(dir, legacyName)
	body := fmt.Sprintf(`{"heads":{"a":"h1"},"nodes":{"h1":{"hash":"h1","kind":"user_message","content":%q}}}`,
		strings.Repeat("q", 4096))
	if err := os.WriteFile(legacy, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	l, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if l.NodeCount() != 0 || l.Head("a") != "" {
		t.Fatal("legacy whole-state file was loaded; it must be retired instead")
	}
	if _, err := os.Stat(legacy); !os.IsNotExist(err) {
		t.Fatal("legacy file still in place; it should have been renamed aside")
	}
	if _, err := os.Stat(legacy + ".legacy"); err != nil {
		t.Fatalf("legacy file was destroyed rather than preserved: %v", err)
	}
}

// TestEvictionKeepsChainUsable: past MaxNodes the ledger forgets oldest-first and
// Chain returns the surviving suffix rather than failing outright.
func TestEvictionKeepsChainUsable(t *testing.T) {
	l := Memory()
	for i := 0; i < MaxNodes+50; i++ {
		if _, err := l.Append("t", "turn_complete", nil); err != nil {
			t.Fatal(err)
		}
	}
	if got := l.NodeCount(); got > MaxNodes {
		t.Fatalf("NodeCount %d exceeds MaxNodes %d", got, MaxNodes)
	}
	chain, err := l.Chain("t")
	if err != nil {
		t.Fatalf("Chain failed after eviction: %v", err)
	}
	if len(chain) == 0 {
		t.Fatal("Chain returned nothing after eviction")
	}
}
