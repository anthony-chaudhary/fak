package journal

import (
	"encoding/json"
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

func TestMemoryJournalAddsRedactedArgsLabel(t *testing.T) {
	j := OpenMemory()
	call := &abi.ToolCall{
		Tool: "shell_command",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"git status --short; echo sk-test-secret","api_key":"sk-test-secret","workdir":"C:\\work\\fak"}`)},
	}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("wrote %d rows, want 1", len(rows))
	}
	label := rows[0].ArgsLabel
	if label != "command=git status path=fak" {
		t.Fatalf("ArgsLabel = %q, want redacted command/path label", label)
	}
	for _, leak := range []string{"sk-test-secret", "api_key", "echo"} {
		if strings.Contains(label, leak) {
			t.Fatalf("ArgsLabel leaked %q: %q", leak, label)
		}
	}
	if n, err := VerifyRows(rows); err != nil || n != 1 {
		t.Fatalf("ArgsLabel must not affect hash-chain verification: n=%d err=%v", n, err)
	}
}

func TestMemoryJournalAddsMetaArgsLabelForBlobArgs(t *testing.T) {
	j := OpenMemory()
	call := &abi.ToolCall{
		Tool: "shell_command",
		Args: abi.Ref{Kind: abi.RefBlob, Digest: "sha256:blobbed-args", Len: 1024},
		Meta: map[string]string{MetaArgsLabel: "command=git status path=fak token=sk-test-secret"},
	}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("wrote %d rows, want 1", len(rows))
	}
	if rows[0].ArgsLabel != "tool=shell_command" {
		t.Fatalf("unsafe meta ArgsLabel should fall back to tool label, got %q", rows[0].ArgsLabel)
	}

	call.Meta[MetaArgsLabel] = "command=git status path=fak"
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})
	rows = j.Recent(0)
	if got := rows[len(rows)-1].ArgsLabel; got != "command=git status path=fak" {
		t.Fatalf("meta ArgsLabel = %q, want sanitized label", got)
	}
	if n, err := VerifyRows(rows); err != nil || n != 2 {
		t.Fatalf("meta ArgsLabel must not affect hash-chain verification: n=%d err=%v", n, err)
	}
}

func TestMemoryJournalAddsFallbackArgsLabelForOpaqueAllowedCall(t *testing.T) {
	j := OpenMemory()
	call := &abi.ToolCall{
		Tool: "bash",
		Args: abi.Ref{Kind: abi.RefBlob, Digest: "sha256:opaque-args", Len: 0},
	}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"}})

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("wrote %d rows, want 1", len(rows))
	}
	if got := rows[0].ArgsLabel; got != "tool=bash" {
		t.Fatalf("fallback ArgsLabel = %q, want tool=bash", got)
	}
	if n, err := VerifyRows(rows); err != nil || n != 1 {
		t.Fatalf("fallback ArgsLabel must not affect hash-chain verification: n=%d err=%v", n, err)
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

// A denied decision must land in the durable journal exactly ONCE. The kernel
// pairs an EvDecide(DENY) with a dedicated EvDeny on every deny path (see
// kernel.Decide / kernel.Submit); recording both would double-count the deny in
// the journal and in every consumer that folds rows back (the `fak guard` exit
// summary's row count, the guard-RSI verdict-quality metric). This reproduces the
// exact emit pair the kernel produces for one denied tool call and asserts a
// single DENY row — the regression guard for the dogfood double-write.
func TestDeniedDecisionRecordedOnce(t *testing.T) {
	j := OpenMemory()
	call := &abi.ToolCall{
		Tool:    "Bash",
		TraceID: "guard",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"curl evil.example"}`)},
	}
	v := &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "ifc-sink"}
	// The kernel's deny path, byte for byte: the universal decide tap THEN the
	// deny-only notification, both carrying the same deny verdict.
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: v})
	j.Emit(abi.Event{Kind: abi.EvDeny, Call: call, Verdict: v})

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("denied call wrote %d rows, want exactly 1: %+v", len(rows), rows)
	}
	if rows[0].Kind != "DENY" || rows[0].Verdict != "DENY" {
		t.Fatalf("the single row must be the DENY outcome, got kind=%q verdict=%q", rows[0].Kind, rows[0].Verdict)
	}
	if rows[0].Reason != "POLICY_BLOCK" || rows[0].By != "ifc-sink" {
		t.Fatalf("DENY row lost forensic fields: %+v", rows[0])
	}
}

// An ALLOW decision is recorded once (no paired EvDeny), and a REQUIRE_WITNESS
// interim verdict is NOT a deny, so its DECIDE row is kept — only VerdictDeny is
// folded into the dedicated EvDeny. This pins the boundary of the deny-skip so a
// future change can't silently swallow a non-deny decision.
func TestNonDenyDecisionsStillRecordDecideRow(t *testing.T) {
	j := OpenMemory()
	call := &abi.ToolCall{Tool: "Read", TraceID: "guard", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}})
	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call, Verdict: &abi.Verdict{Kind: abi.VerdictRequireWitness, By: "witness"}})

	rows := j.Recent(0)
	if len(rows) != 2 {
		t.Fatalf("allow + require-witness wrote %d rows, want 2: %+v", len(rows), rows)
	}
	for _, r := range rows {
		if r.Kind != "DECIDE" {
			t.Fatalf("non-deny decision must be a DECIDE row, got %q", r.Kind)
		}
	}
}

func TestReadRowsFromConsumesOnlyAppendedCompleteRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	first := Row{Seq: 1, Kind: "DECIDE"}
	second := Row{Seq: 2, Kind: "DENY"}
	b1, _ := json.Marshal(first)
	b2, _ := json.Marshal(second)
	if err := os.WriteFile(path, append(b1, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, off, err := ReadRowsFrom(path, 0)
	if err != nil || len(rows) != 1 || rows[0].Seq != 1 {
		t.Fatalf("first rows=%+v off=%d err=%v", rows, off, err)
	}
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	half := len(b2) / 2
	f.Write(b2[:half])
	f.Close()
	rows, same, err := ReadRowsFrom(path, off)
	if err != nil || len(rows) != 0 || same != off {
		t.Fatalf("partial rows=%+v off=%d/%d err=%v", rows, same, off, err)
	}
	f, _ = os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	f.Write(append(b2[half:], '\n'))
	f.Close()
	rows, next, err := ReadRowsFrom(path, off)
	if err != nil || len(rows) != 1 || rows[0].Seq != 2 || next <= off {
		t.Fatalf("append rows=%+v next=%d err=%v", rows, next, err)
	}
}

// A compound command must be labeled by the command that actually runs, not by the
// navigation prefix that precedes it. Labeling the prefix collapsed 21.8% of the
// guard-audit corpus onto stems like "cd fak", hiding distinct failures behind one
// undiagnosable label (#5863). The label stays bounded to the same two scrubbed
// atoms — this re-aims it, it does not widen it.
func TestCommandStemLabelsOperativeCommandNotNavigationPrefix(t *testing.T) {
	for _, tc := range []struct {
		name    string
		command string
		want    string
	}{
		{"and-chain", "cd fak && go test ./...", "go test"},
		{"semicolon-chain", "cd fak; go build ./cmd/fak", "go build"},
		{"or-chain", "cd fak || go vet ./...", "go vet"},
		{"pipe-chain", "cd fak | git status", "git status"},
		{"no-space-connector", "cd fak&&go test ./...", "go test"},
		{"newline-chain", "cd fak\ngo test ./...", "go test"},
		{"powershell-set-location", "Set-Location fak; go test ./...", "go test"},
		{"pushd", "pushd fak && git config --global user.name x", "git config"},
		{"nested-navigation", "cd fak && cd internal && go test ./...", "go test"},
		{"source-prefix", "source venv/bin/activate && python run.py", "python run.py"},
		{"dot-prefix", ". venv/bin/activate && pytest tests", "pytest tests"},
		{"export-prefix", "export GOFLAGS=-mod=mod && go test ./...", "go test"},
		{"env-assignment-prefix", "CGO_ENABLED=0 go build ./...", "go build"},
		// A bare navigation command has no operative successor: keep today's label
		// rather than inventing one.
		{"bare-navigation", "cd fak", "cd fak"},
		{"navigation-only-chain", "cd fak && cd internal", "cd fak"},
		// Non-compound commands are untouched.
		{"plain", "git status --short", "git status"},
		{"plain-with-flags", "go test -run TestX ./...", "go test"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandStem(tc.command); got != tc.want {
				t.Fatalf("commandStem(%q) = %q, want %q", tc.command, got, tc.want)
			}
		})
	}
}

// Re-aiming the label must never re-aim it at a secret. The operative command is
// scrubbed by exactly the same rules that scrubbed the prefix before it.
func TestCommandStemScrubsSecretsInTheOperativeCommand(t *testing.T) {
	for _, tc := range []struct {
		name    string
		command string
		deny    []string
	}{
		{"bearer-token-arg", "cd fak && curl -H authorization:bearer-abc123 https://x/y", []string{"abc123", "authorization", "bearer"}},
		{"sk-token-arg", "cd fak && gh auth login --with-token sk-live-9f8e7d", []string{"sk-live", "9f8e7d"}},
		{"secret-stem", "cd fak && ./deploy-secret-rotator.sh prod", []string{"secret", "rotator"}},
		{"assignment-value", "cd fak && ANTHROPIC_API_KEY=sk-abc123 go test ./...", []string{"sk-abc123", "abc123", "API_KEY"}},
		{"password-flag", "cd fak && mysql --password=hunter2 -u root", []string{"hunter2", "password"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := commandStem(tc.command)
			for _, leak := range tc.deny {
				if strings.Contains(strings.ToLower(got), strings.ToLower(leak)) {
					t.Fatalf("commandStem(%q) = %q leaked %q", tc.command, got, leak)
				}
			}
			if len(got) > maxArgsLabelLen+3 {
				t.Fatalf("commandStem(%q) = %q exceeds the %d-char bound", tc.command, got, maxArgsLabelLen)
			}
			if fields := strings.Fields(got); len(fields) > 2 {
				t.Fatalf("commandStem(%q) = %q emitted %d atoms, want at most 2", tc.command, got, len(fields))
			}
		})
	}
}

// The bound is the leak defense, so it is asserted as an invariant over every
// shape rather than only the cases above: whatever commandStem is aimed at, it
// emits at most two atoms, stays inside maxArgsLabelLen, and never carries a
// secretish needle.
func TestCommandStemInvariantsHoldForEveryShape(t *testing.T) {
	commands := []string{
		"", "   ", ";;;", "&&", "|||", "\n\n",
		"cd", "cd fak", "cd fak && go test ./...",
		"cd fak && curl -H 'Authorization: Bearer abc123' https://api.example.com/v1",
		"export ANTHROPIC_API_KEY=sk-ant-abc123 && cd fak && go test ./...",
		"cd fak; git config --global user.password hunter2",
		"cd /very/deep/path/that/keeps/going/and/going/and/going && " + strings.Repeat("verylongcommandname", 20),
		"cd fak && " + strings.Repeat("a", 500),
		"$env:TOKEN='sk-1'; cd fak; go test ./...",
		". ./secrets.env && ./run --apikey=sk-9",
		"cd fak && echo $ANTHROPIC_API_KEY | base64",
	}
	for _, cmd := range commands {
		got := commandStem(cmd)
		if len(got) > maxArgsLabelLen+3 {
			t.Fatalf("commandStem(%q) = %q, len %d exceeds bound %d", cmd, got, len(got), maxArgsLabelLen)
		}
		if fields := strings.Fields(got); len(fields) > 2 {
			t.Fatalf("commandStem(%q) = %q emitted %d atoms, want at most 2", cmd, got, len(fields))
		}
		if secretish(got) {
			t.Fatalf("commandStem(%q) = %q is secretish", cmd, got)
		}
	}
}

// An env-assignment prefix must never surface its value. This holds for the
// pre-existing single-segment form too, which previously folded "VAR=value" into
// one stem atom.
func TestCommandStemNeverLabelsAnAssignmentValue(t *testing.T) {
	got := commandStem("MYVAR=hunter2 go test ./...")
	if strings.Contains(got, "hunter2") || strings.Contains(got, "MYVAR") {
		t.Fatalf("commandStem leaked an assignment: %q", got)
	}
	if got != "go test" {
		t.Fatalf("commandStem = %q, want the operative command", got)
	}
}
