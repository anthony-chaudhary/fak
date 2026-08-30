package usagelog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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

func TestFoldRowsSeparatesGitOperationTerminalOutcomes(t *testing.T) {
	rows := []Row{
		{Schema: SchemaV1, Verb: "sync push", ExitCode: 0, DurationMS: 100},
		{Schema: SchemaV1, Verb: "sync push", ExitCode: 0, DurationMS: 300},
		{Schema: SchemaV1, Verb: "dev sync push", ExitCode: 0, DurationMS: 500},
		{Schema: SchemaV1, Verb: "sync push", ExitCode: 3, DurationMS: 1},
		{Schema: SchemaV1, Verb: "sync push", ExitCode: 2, DurationMS: 2},
		{Schema: SchemaV1, Verb: "sync push", ExitCode: 4, DurationMS: 4},
		{Schema: SchemaV1, Verb: "sync", ExitCode: 0, DurationMS: 1}, // legacy: suboperation unknowable
	}
	stats := FoldRows(rows, FoldOptions{}).ByOperationOutcome
	if len(stats) != 4 {
		t.Fatalf("stats = %+v, want success/refused/usage/error", stats)
	}
	want := []OperationOutcomeStat{
		{Operation: "sync push", Outcome: OutcomeSuccess, Count: 3, P50MS: 300},
		{Operation: "sync push", Outcome: OutcomeRefused, Count: 1, P50MS: 1},
		{Operation: "sync push", Outcome: OutcomeUsage, Count: 1, P50MS: 2},
		{Operation: "sync push", Outcome: OutcomeError, Count: 1, P50MS: 4},
	}
	for i := range want {
		if stats[i] != want[i] {
			t.Errorf("stats[%d] = %+v, want %+v", i, stats[i], want[i])
		}
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

// TestRecoverHeadStartsAtTheTailNotByteZero is the boundedness witness for #5626:
// head recovery must read a bounded tail, not the whole journal. It is asserted
// deterministically rather than by timing — the journal opens with a torn line far
// older than the tail window, which a scan starting at byte 0 stops on (yielding the
// genesis head), and which a scan starting in the tail window never sees. Passing it
// therefore proves the scan no longer begins at byte 0, which is exactly the property
// that made every `fak` spawn re-parse the machine's entire invocation history.
func TestRecoverHeadStartsAtTheTailNotByteZero(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	var buf bytes.Buffer
	buf.WriteString("{ this line is torn and is NOT valid json\n")
	var last Row
	for seq := 1; buf.Len() <= 2*recoverTailWindow; seq++ {
		last = Row{
			Schema: SchemaV1,
			Seq:    uint64(seq),
			Verb:   "version",
			Hash:   fmt.Sprintf("sha256:%064x", seq),
		}
		b, err := json.Marshal(last)
		if err != nil {
			t.Fatalf("marshal row %d: %v", seq, err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write journal: %v", err)
	}

	seq, hash, end, err := recoverHead(path)
	if err != nil {
		t.Fatalf("recoverHead: %v", err)
	}
	if seq != last.Seq {
		t.Errorf("recovered seq = %d, want %d (the true last row, i.e. the scan skipped the stale prefix)", seq, last.Seq)
	}
	if hash != last.Hash {
		t.Errorf("recovered hash = %q, want %q", hash, last.Hash)
	}
	if end != int64(buf.Len()) {
		t.Errorf("recovered end = %d, want %d (the live file size)", end, buf.Len())
	}
}

// TestRecoverHeadFallsBackWhenTailWindowHoldsNoRow covers the one case the tail
// window cannot serve: a final row LONGER than the window, so the window opens past
// its terminator and yields no row at all. Recovery must fall back to a full scan
// there — returning the genesis head instead would restart seq at 1 and fork the
// chain, the #2608 failure head recovery exists to prevent.
func TestRecoverHeadFallsBackWhenTailWindowHoldsNoRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	first, err := lg.Append(Row{Verb: "run", Args: []string{strings.Repeat("x", recoverTailWindow+4096)}})
	if err != nil {
		t.Fatalf("append oversized row: %v", err)
	}
	_ = lg.Close()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Size() <= recoverTailWindow {
		t.Fatalf("row is %d bytes, need > %d for this case to exercise the fallback", st.Size(), recoverTailWindow)
	}

	lg2, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	second, err := lg2.Append(Row{Verb: "guard"})
	if err != nil {
		t.Fatalf("append after reopen: %v", err)
	}
	_ = lg2.Close()

	if second.Seq != 2 {
		t.Errorf("seq after reopen = %d, want 2 (continued, not forked)", second.Seq)
	}
	if second.PrevHash != first.Hash {
		t.Errorf("chain not linked: prev=%q want %q", second.PrevHash, first.Hash)
	}
	if n, err := Verify(path); err != nil || n != 2 {
		t.Errorf("Verify after oversized-row recovery: n=%d err=%v, want n=2 err=nil", n, err)
	}
}

func TestDiagnosticSinkLazinessRedactionAndTypedJSON(t *testing.T) {
	var output bytes.Buffer
	sink, err := NewDiagnosticSink(&output, DiagnosticInfo, "token")
	if err != nil {
		t.Fatalf("NewDiagnosticSink: %v", err)
	}

	var filteredCalls int
	if err := sink.Emit(DiagnosticDebug, "kernel.filtered",
		"secret", LazyValue(func() any {
			filteredCalls++
			return "must-not-exist"
		}),
	); err != nil {
		t.Fatalf("filtered Emit: %v", err)
	}
	if filteredCalls != 0 || output.Len() != 0 {
		t.Fatalf("filtered diagnostic: calls=%d bytes=%d, want both zero", filteredCalls, output.Len())
	}

	var enabledCalls, redactedCalls int
	if err := sink.Emit(DiagnosticInfo, "kernel.step",
		"attempt", LazyValue(func() any {
			enabledCalls++
			return int64(3)
		}),
		"cached", false,
		"ratio", 0.5,
		"result", nil,
		"token", LazyValue(func() any {
			redactedCalls++
			return "super-secret"
		}),
	); err != nil {
		t.Fatalf("enabled Emit: %v", err)
	}
	if enabledCalls != 1 {
		t.Fatalf("enabled lazy calls = %d, want 1", enabledCalls)
	}
	if redactedCalls != 0 {
		t.Fatalf("redacted lazy calls = %d, want 0 (redact before evaluation)", redactedCalls)
	}

	const want = "{\"schema\":\"fak-kernel-diagnostic/1\",\"level\":\"info\",\"event\":\"kernel.step\",\"fields\":{\"attempt\":3,\"cached\":false,\"ratio\":0.5,\"result\":null,\"token\":\"[REDACTED]\"}}\n"
	if got := output.String(); got != want {
		t.Fatalf("captured diagnostic:\n got %q\nwant %q", got, want)
	}

	var decoded struct {
		Schema string         `json:"schema"`
		Level  string         `json:"level"`
		Event  string         `json:"event"`
		Fields map[string]any `json:"fields"`
	}
	dec := json.NewDecoder(strings.NewReader(output.String()))
	dec.UseNumber()
	if err := dec.Decode(&decoded); err != nil {
		t.Fatalf("decode captured diagnostic: %v", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		t.Fatalf("trailing JSON decode = %v, want EOF", err)
	}
	if decoded.Schema != DiagnosticSchemaV1 || decoded.Level != "info" || decoded.Event != "kernel.step" {
		t.Fatalf("decoded envelope = %+v", decoded)
	}
	if len(decoded.Fields) != 5 {
		t.Fatalf("decoded fields = %#v", decoded.Fields)
	}
	if n, ok := decoded.Fields["attempt"].(json.Number); !ok || n.String() != "3" {
		t.Fatalf("attempt = %#v (%T), want json.Number(3)", decoded.Fields["attempt"], decoded.Fields["attempt"])
	}
	if v, ok := decoded.Fields["cached"].(bool); !ok || v {
		t.Fatalf("cached = %#v (%T), want bool(false)", decoded.Fields["cached"], decoded.Fields["cached"])
	}
	if n, ok := decoded.Fields["ratio"].(json.Number); !ok || n.String() != "0.5" {
		t.Fatalf("ratio = %#v (%T), want json.Number(0.5)", decoded.Fields["ratio"], decoded.Fields["ratio"])
	}
	if decoded.Fields["result"] != nil {
		t.Fatalf("result = %#v, want nil", decoded.Fields["result"])
	}
	if v, ok := decoded.Fields["token"].(string); !ok || v != "[REDACTED]" {
		t.Fatalf("token = %#v (%T), want redacted string", decoded.Fields["token"], decoded.Fields["token"])
	}
}

func TestDiagnosticSinkDisabledDoesNotEvaluate(t *testing.T) {
	sink, err := NewDiagnosticSink(nil, DiagnosticDebug)
	if err != nil {
		t.Fatalf("NewDiagnosticSink: %v", err)
	}
	var calls int
	if err := sink.Emit(DiagnosticError, "kernel.disabled", "secret", LazyValue(func() any {
		calls++
		return "must-not-exist"
	})); err != nil {
		t.Fatalf("disabled Emit: %v", err)
	}
	if calls != 0 {
		t.Fatalf("disabled lazy calls = %d, want 0", calls)
	}
}

func TestDiagnosticSinkFailsClosed(t *testing.T) {
	tests := []struct {
		name  string
		event string
		level DiagnosticLevel
		kv    []any
	}{
		{name: "invalid level", event: "kernel.step", level: DiagnosticLevel(99), kv: []any{"key", "value"}},
		{name: "empty event", event: "", level: DiagnosticInfo, kv: []any{"key", "value"}},
		{name: "odd pair count", event: "kernel.step", level: DiagnosticInfo, kv: []any{"key"}},
		{name: "non-string key", event: "kernel.step", level: DiagnosticInfo, kv: []any{7, "value"}},
		{name: "empty key", event: "kernel.step", level: DiagnosticInfo, kv: []any{"", "value"}},
		{name: "duplicate key", event: "kernel.step", level: DiagnosticInfo, kv: []any{"key", 1, "key", 2}},
		{name: "unsupported direct value", event: "kernel.step", level: DiagnosticInfo, kv: []any{"key", []string{"value"}}},
		{name: "unsupported lazy value", event: "kernel.step", level: DiagnosticInfo, kv: []any{"key", LazyValue(func() any { return map[string]string{"secret": "value"} })}},
		{name: "panicking lazy value", event: "kernel.step", level: DiagnosticInfo, kv: []any{"key", LazyValue(func() any { panic("secret") })}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var output bytes.Buffer
			sink, err := NewDiagnosticSink(&output, DiagnosticDebug)
			if err != nil {
				t.Fatalf("NewDiagnosticSink: %v", err)
			}
			if err := sink.Emit(tt.level, tt.event, tt.kv...); err == nil {
				t.Fatal("Emit accepted malformed or unsupported input")
			}
			if output.Len() != 0 {
				t.Fatalf("Emit wrote %d bytes before rejecting input: %q", output.Len(), output.String())
			}
		})
	}

	if _, err := NewDiagnosticSink(io.Discard, DiagnosticLevel(99)); err == nil {
		t.Fatal("NewDiagnosticSink accepted an invalid minimum level")
	}
	if _, err := NewDiagnosticSink(io.Discard, DiagnosticInfo, ""); err == nil {
		t.Fatal("NewDiagnosticSink accepted an empty sensitive key")
	}

	var output bytes.Buffer
	sink, err := NewDiagnosticSink(&output, DiagnosticDebug)
	if err != nil {
		t.Fatalf("NewDiagnosticSink: %v", err)
	}
	var malformedCalls int
	err = sink.Emit(DiagnosticInfo, "kernel.step",
		"key", LazyValue(func() any {
			malformedCalls++
			return "secret"
		}),
		"key", "duplicate",
	)
	if err == nil {
		t.Fatal("Emit accepted duplicate keys")
	}
	if malformedCalls != 0 || output.Len() != 0 {
		t.Fatalf("malformed diagnostic: calls=%d bytes=%d, want both zero", malformedCalls, output.Len())
	}
}
