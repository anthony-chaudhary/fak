package usagelog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAppendChainsAndVerifies covers the headline acceptance: every Append yields a
// hash-chained row, and Verify over the resulting file passes.
func TestAppendChainsAndVerifies(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i, verb := range []string{"guard", "run", "guard"} {
		if _, err := lg.Append(Row{Verb: verb, Argc: i, ExitCode: 0, DurationMS: int64(10 * (i + 1))}); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}
	if err := lg.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	n, err := Verify(path)
	if err != nil {
		t.Fatalf("Verify clean journal: %v", err)
	}
	if n != 3 {
		t.Fatalf("Verify counted %d rows, want 3", n)
	}

	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("ReadRows returned %d rows, want 3", len(rows))
	}
	// Seq is 1-based and monotonic; the first row's PrevHash is the empty genesis.
	if rows[0].Seq != 1 || rows[0].PrevHash != "" {
		t.Errorf("genesis row: seq=%d prev=%q, want seq=1 prev=\"\"", rows[0].Seq, rows[0].PrevHash)
	}
	if rows[1].PrevHash != rows[0].Hash || rows[2].PrevHash != rows[1].Hash {
		t.Errorf("chain not linked: prev hashes do not point at predecessors")
	}
	if rows[0].Schema != SchemaV1 {
		t.Errorf("schema = %q, want %q", rows[0].Schema, SchemaV1)
	}
}

// TestVerifyBreaksOnFlippedByte covers the tamper-evidence acceptance: editing a
// committed row (here, rewriting an exit code) breaks the chain at that row.
func TestVerifyBreaksOnFlippedByte(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := lg.Append(Row{Verb: "run", ExitCode: 0}); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	_ = lg.Close()

	// Tamper: flip the second row's exit_code 0 -> 1 in place, leaving its stored
	// hash unchanged. Verify must catch the recomputed-hash mismatch.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}
	lines[1] = strings.Replace(lines[1], `"exit_code":0`, `"exit_code":1`, 1)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatalf("write tampered: %v", err)
	}

	if _, err := Verify(path); err == nil {
		t.Fatal("Verify accepted a tampered journal; want a broken-chain error")
	}
}

// TestRedactionNoRawArgvByDefault covers the honesty fence: a row built the default
// way stores only a salted digest, never the raw argv (paths/messages/tokens).
func TestRedactionNoRawArgvByDefault(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	secret := "/home/alice/secret-token-abc123"
	digest := Digest([]byte("salty"), []string{"-m", secret})
	if _, err := lg.Append(Row{Verb: "commit", Argc: 2, ArgsDigest: digest}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	_ = lg.Close()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if strings.Contains(string(raw), secret) {
		t.Fatalf("raw argv leaked to disk: journal contains %q", secret)
	}
	if !strings.Contains(string(raw), digest) {
		t.Fatal("args_digest not persisted")
	}

	// The digest commits to the args: same salt+args -> same digest; a different
	// arg -> a different digest (so a repeated command is countable, a changed one
	// is distinguishable) — without ever disclosing the bytes.
	if Digest([]byte("salty"), []string{"-m", secret}) != digest {
		t.Error("digest not stable for identical salt+args")
	}
	if Digest([]byte("salty"), []string{"-m", "different"}) == digest {
		t.Error("digest collided across different args")
	}
	if Digest([]byte("other-salt"), []string{"-m", secret}) == digest {
		t.Error("digest not salt-dependent")
	}
}

// TestFullArgsOptInDoesNotBreakChain covers that the raw-argv disclosure layer is
// excluded from the hash pre-image: a row WITH raw Args and the same row WITHOUT it
// share the same Hash, so existing redacted journals verify unchanged.
func TestFullArgsOptInDoesNotBreakChain(t *testing.T) {
	base := Row{Verb: "commit", Argc: 1, ArgsDigest: "sha256:abc", Seq: 1, TSUnixNano: 42}
	withArgs := base
	withArgs.Args = []string{"-m", "hello"}
	if got, want := chainHash("", withArgs), chainHash("", base); got != want {
		t.Fatalf("Args changed the chain hash: %s != %s (Args must be outside the pre-image)", got, want)
	}

	// And a full-args journal still Verifies end to end.
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, _ := Open(path)
	if _, err := lg.Append(Row{Verb: "commit", Argc: 1, Args: []string{"-m", "hello"}}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	_ = lg.Close()
	if _, err := Verify(path); err != nil {
		t.Fatalf("Verify full-args journal: %v", err)
	}
	rows, _ := ReadRows(path)
	if len(rows) != 1 || len(rows[0].Args) != 2 {
		t.Fatalf("raw args not persisted under opt-in: %+v", rows)
	}
}

// TestEnabledRespectsOptOut covers FAK_USAGE_LOG=off: the gate the CLI checks before
// recording reports OFF, so nothing is written.
func TestEnabledRespectsOptOut(t *testing.T) {
	t.Setenv("FAK_USAGE_LOG", "off")
	if Enabled() {
		t.Error("Enabled() = true with FAK_USAGE_LOG=off, want false")
	}
	t.Setenv("FAK_USAGE_LOG", "OFF")
	if Enabled() {
		t.Error("Enabled() = true with FAK_USAGE_LOG=OFF (case-insensitive), want false")
	}
	t.Setenv("FAK_USAGE_LOG", "")
	if !Enabled() {
		t.Error("Enabled() = false with FAK_USAGE_LOG unset/empty, want true (on by default)")
	}
	t.Setenv("FAK_USAGE_LOG", "on")
	if !Enabled() {
		t.Error("Enabled() = false with FAK_USAGE_LOG=on, want true")
	}
}

// TestReopenContinuesChain covers durability across process restart: a second Open
// recovers the chain head so the new row links onto the prior session's tail.
func TestReopenContinuesChain(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, _ := Open(path)
	first, _ := lg.Append(Row{Verb: "run"})
	_ = lg.Close()

	lg2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	second, err := lg2.Append(Row{Verb: "guard"})
	if err != nil {
		t.Fatalf("Append after reopen: %v", err)
	}
	_ = lg2.Close()

	if second.Seq != 2 {
		t.Errorf("seq after reopen = %d, want 2 (continued, not forked)", second.Seq)
	}
	if second.PrevHash != first.Hash {
		t.Errorf("reopened chain not linked: prev=%q want %q", second.PrevHash, first.Hash)
	}
	if n, err := Verify(path); err != nil || n != 2 {
		t.Errorf("Verify across restart: n=%d err=%v, want n=2 err=nil", n, err)
	}
}

// TestFoldRows covers the read fold backing `fak usage --by-verb`: by-verb counts,
// error tally, exit-code distribution, p50 duration, and the recent tail.
func TestFoldRows(t *testing.T) {
	rows := []Row{
		{Schema: SchemaV1, Verb: "guard", ExitCode: 0, DurationMS: 100, TSUnixNano: 1},
		{Schema: SchemaV1, Verb: "run", ExitCode: 0, DurationMS: 10, TSUnixNano: 2},
		{Schema: SchemaV1, Verb: "guard", ExitCode: 1, DurationMS: 300, TSUnixNano: 3},
		{Schema: SchemaV1, Verb: "guard", ExitCode: 0, DurationMS: 200, TSUnixNano: 4},
		{Schema: "other-ledger/1", Verb: "noise", ExitCode: 9, DurationMS: 9, TSUnixNano: 5}, // foreign schema: skipped
	}
	f := FoldRows(rows, FoldOptions{TopN: 2})

	if f.Total != 4 {
		t.Errorf("Total = %d, want 4 (foreign-schema row excluded)", f.Total)
	}
	if f.Errors != 1 {
		t.Errorf("Errors = %d, want 1", f.Errors)
	}
	if f.ExitCodes[0] != 3 || f.ExitCodes[1] != 1 {
		t.Errorf("ExitCodes = %v, want {0:3, 1:1}", f.ExitCodes)
	}
	if len(f.ByVerb) == 0 || f.ByVerb[0].Verb != "guard" || f.ByVerb[0].Count != 3 {
		t.Errorf("ByVerb[0] = %+v, want guard count=3 first", f.ByVerb)
	}
	if f.ByVerb[0].Errors != 1 {
		t.Errorf("guard errors = %d, want 1", f.ByVerb[0].Errors)
	}
	// guard durations {100,300,200} -> sorted {100,200,300} -> median 200.
	if f.ByVerb[0].P50MS != 200 {
		t.Errorf("guard p50 = %d, want 200", f.ByVerb[0].P50MS)
	}
	// TopN=2 keeps the last two kept rows, oldest-first (ts 3 then ts 4).
	if len(f.Recent) != 2 || f.Recent[0].TSUnixNano != 3 || f.Recent[1].TSUnixNano != 4 {
		t.Errorf("Recent = %+v, want the last two kept rows (ts 3,4)", f.Recent)
	}
}

// TestFoldRowsSinceCutoff covers the --since window.
func TestFoldRowsSinceCutoff(t *testing.T) {
	rows := []Row{
		{Schema: SchemaV1, Verb: "a", TSUnixNano: 10},
		{Schema: SchemaV1, Verb: "b", TSUnixNano: 20},
		{Schema: SchemaV1, Verb: "c", TSUnixNano: 30},
	}
	f := FoldRows(rows, FoldOptions{SinceUnixNano: 20})
	if f.Total != 2 {
		t.Errorf("Total with since=20 = %d, want 2 (ts 20 and 30)", f.Total)
	}
}

// TestReadRowsMissingFileIsEmpty covers the live-reader contract: tailing a journal
// that has not been written yet is "no rows", not an error.
func TestReadRowsMissingFileIsEmpty(t *testing.T) {
	rows, err := ReadRows(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if err != nil {
		t.Fatalf("ReadRows missing file: %v", err)
	}
	if len(rows) != 0 {
		t.Fatalf("ReadRows missing file returned %d rows, want 0", len(rows))
	}
}

// TestLoadOrCreateSaltIsStable covers that the per-user salt is created once and
// reused, so the redaction digest is stable across invocations for that user.
func TestLoadOrCreateSaltIsStable(t *testing.T) {
	p := filepath.Join(t.TempDir(), "nested", "usage.salt")
	s1, err := LoadOrCreateSalt(p)
	if err != nil {
		t.Fatalf("LoadOrCreateSalt: %v", err)
	}
	if len(s1) == 0 {
		t.Fatal("empty salt")
	}
	s2, err := LoadOrCreateSalt(p)
	if err != nil {
		t.Fatalf("LoadOrCreateSalt reuse: %v", err)
	}
	if string(s1) != string(s2) {
		t.Fatal("salt not stable across loads")
	}
}
