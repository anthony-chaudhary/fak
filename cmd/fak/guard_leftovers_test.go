package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
)

const guardLeftoversSummary = "Shipped the fix. There are two more things worth doing: a docs pass and a soak."

func TestScanGuardLeftoversUsesRunLifetimeIssueToolInputs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session.jsonl")

	writeGuardLeftoversTranscript(t, path, nil)
	unfiled := scanGuardLeftovers(path, guardLeftoversSummary)
	if !unfiled.LeftoversUnfiled || unfiled.Narrated == 0 || unfiled.IssuesFiled != 0 {
		t.Fatalf("zero filed: want unfiled/narrated/>0/0, got %+v", unfiled)
	}
	if !unfiled.FilingKnown || unfiled.FilingSource != headlesslint.IssuesFiledFromTranscript {
		t.Fatalf("zero filed: a transcript that was read is a WITNESSED zero, got %+v", unfiled)
	}

	writeGuardLeftoversTranscript(t, path, []toolUseFixture{{
		Name:  "functions.shell_command",
		Input: map[string]any{"command": `fak issue create --title "Docs pass" --body "done condition"`},
	}})
	filed := scanGuardLeftovers(path, guardLeftoversSummary)
	if filed.LeftoversUnfiled || filed.Narrated == 0 || filed.IssuesFiled != 1 {
		t.Fatalf("one filed: want clean/narrated/>0/1, got %+v", filed)
	}
}

// TestScanGuardLeftoversAbsentTranscriptIsUnknownNotZero: with nothing to read, the
// sensor must report "cannot say" rather than the zero that would convict the run. An
// empty path and a path to a file that is not there are both unknown — and neither is
// allowed to set LeftoversUnfiled.
func TestScanGuardLeftoversAbsentTranscriptIsUnknownNotZero(t *testing.T) {
	for _, path := range []string{"", filepath.Join(t.TempDir(), "never-written.jsonl")} {
		got := scanGuardLeftovers(path, guardLeftoversSummary)
		if got.LeftoversUnfiled {
			t.Errorf("path %q: absence of evidence must not read as an unfiled verdict: %+v", path, got)
		}
		if !got.FilingUnknown || got.FilingKnown {
			t.Errorf("path %q: want an unknown filing count, got %+v", path, got)
		}
		if got.FilingSource != headlesslint.IssuesFiledNoEvidence {
			t.Errorf("path %q: source = %q, want %q", path, got.FilingSource, headlesslint.IssuesFiledNoEvidence)
		}
		if got.Narrated == 0 {
			t.Errorf("path %q: the narration itself is still detected, got %+v", path, got)
		}
	}
}

// TestGuardIssuesFiledEvidenceFromRecordsTailZeroIsUnknown: the Stop hook reads a bounded
// TAIL, so its count is a lower bound. A positive lower bound still proves a filing; a
// zero over a truncated read proves nothing and must resolve to unknown.
func TestGuardIssuesFiledEvidenceFromRecordsTailZeroIsUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardLeftoversTranscript(t, path, nil)
	recs := loadGuardLeftoversFixture(t, path)

	if ev := guardIssuesFiledEvidenceFromRecords(recs, true); ev.Known {
		t.Errorf("truncated read with no filing: want unknown, got %+v", ev)
	}
	if ev := guardIssuesFiledEvidenceFromRecords(recs, false); !ev.Known || ev.Count != 0 {
		t.Errorf("whole read with no filing: want a witnessed 0, got %+v", ev)
	}
	if ev := guardIssuesFiledEvidenceFromRecords(nil, false); ev.Known {
		t.Errorf("no records at all: want unknown, got %+v", ev)
	}
}

// TestApplyLeftoversSignalCountsEvidenceNotProse is the Stop-hook wiring: the third
// sensor beside applyHeadlessLintSignal / applyClosingSignal. The final turn ASSERTS it
// filed the follow-ups; only the transcript decides whether it did.
func TestApplyLeftoversSignalCountsEvidenceNotProse(t *testing.T) {
	dir := t.TempDir()
	const claimed = "Done. I filed both follow-ups as gh issues.\n" +
		"There are two more things worth doing: a docs pass and a soak."

	// (a) the claim with no filing behind it — a tool turn that checked status, then the
	// prose-only end_turn. The sensor records the breach.
	bare := filepath.Join(dir, "claim-only.jsonl")
	writeStopTranscriptFixture(t, bare,
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"git status"}}]}}`,
		`{"type":"assistant","message":{"role":"assistant","content":`+guardLeftoversJSONString(t, claimed)+`}}`,
	)
	sig := readGuardStopTranscript(bare)
	if sig == nil || !sig.Read {
		t.Fatalf("transcript not read: %+v", sig)
	}
	if !sig.LeftoversUnfiled || sig.LeftoversNarrated == 0 {
		t.Fatalf("a claim with no filing must record leftovers_unfiled, got %+v", sig)
	}
	if sig.LeftoversIssuesFiled != 0 || sig.LeftoversFilingUnknown {
		t.Errorf("want a witnessed 0 filings, got %+v", sig)
	}

	// (b) the SAME closing prose, but the run really did file one. Evidence clears it.
	real := filepath.Join(dir, "really-filed.jsonl")
	writeStopTranscriptFixture(t, real,
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":"gh issue create --title backoff --body done-condition"}}]}}`,
		`{"type":"assistant","message":{"role":"assistant","content":`+guardLeftoversJSONString(t, claimed)+`}}`,
	)
	sigFiled := readGuardStopTranscript(real)
	if sigFiled == nil || sigFiled.LeftoversUnfiled {
		t.Fatalf("an evidenced filing must clear the sensor, got %+v", sigFiled)
	}
	if sigFiled.LeftoversIssuesFiled != 1 || sigFiled.LeftoversNarrated == 0 {
		t.Errorf("want 1 evidenced filing plus the narration, got %+v", sigFiled)
	}
	if sigFiled.LeftoversFilingSource != headlesslint.IssuesFiledFromTranscript {
		t.Errorf("source = %q, want %q", sigFiled.LeftoversFilingSource, headlesslint.IssuesFiledFromTranscript)
	}

	// (c) a clean closer narrates nothing, so the sensor stays entirely quiet.
	quiet := filepath.Join(dir, "quiet.jsonl")
	writeStopTranscriptFixture(t, quiet,
		`{"type":"assistant","message":{"role":"assistant","content":"Implemented the parser, tests pass, pushed."}}`,
	)
	if sigQuiet := readGuardStopTranscript(quiet); sigQuiet == nil || sigQuiet.LeftoversNarrated != 0 || sigQuiet.LeftoversUnfiled {
		t.Errorf("clean closer should leave the leftovers fields zero, got %+v", sigQuiet)
	}
}

// TestGuardDirectIssueCreateToolUnderCounts pins the bias: names that are issue-shaped
// but file nothing (commenting, listing, closing) must NOT be counted. A count that runs
// high is the failure mode that matters — it hands back the same unearned trust the
// self-reported flag did.
func TestGuardDirectIssueCreateToolUnderCounts(t *testing.T) {
	for _, name := range []string{"github.create_issue", "mcp__github__create_issue", "createIssue", "gh-issue-create"} {
		if !guardDirectIssueCreateTool(name) {
			t.Errorf("%q should count as filing an issue", name)
		}
	}
	for _, name := range []string{"github.create_issue_comment", "issue_comment_create", "list_issues", "create_pull_request", "issues_close", "update_issue", "create_issue_template"} {
		if guardDirectIssueCreateTool(name) {
			t.Errorf("%q must NOT count as filing an issue", name)
		}
	}
}

// guardLeftoversJSONString renders s as a JSON string literal for a fixture line.
func guardLeftoversJSONString(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestGuardIssuesFiledCountsShellAndNestedNativeCallsNotProse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardLeftoversTranscript(t, path, []toolUseFixture{
		{Name: "functions.shell_command", Input: map[string]any{"command": "gh issue create --title one; fak.exe issue create --title two"}},
		{Name: "multi_tool_use.parallel", Input: map[string]any{"tool_uses": []any{
			map[string]any{"recipient_name": "github.create_issue", "parameters": map[string]any{"title": "three"}},
			map[string]any{"recipient_name": "functions.shell_command", "parameters": map[string]any{"command": "fak issue create --title four"}},
		}}},
		{Name: "functions.shell_command", Input: map[string]any{"justification": "Please run fak issue create later", "command": "Write-Output clean"}},
	})
	if got := guardIssuesFiledEvidenceFromRecords(loadGuardLeftoversFixture(t, path), false); !got.Known || got.Count != 4 {
		t.Fatalf("issues filed = %+v, want witnessed 4", got)
	}
}

// fak-test:runtime fast est=20ms
func TestGuardIssuesFiledCodexNativeCoverage(t *testing.T) {
	t.Parallel()
	function := func(name, arguments string) string {
		return `{"type":"response_item","payload":{"type":"function_call","name":` + guardLeftoversJSONString(t, name) + `,"arguments":` + guardLeftoversJSONString(t, arguments) + `}}`
	}
	shell := func(command string) string {
		return function("functions.exec_command", `{"cmd":`+guardLeftoversJSONString(t, command)+`}`)
	}
	const opaque = `{"type":"response_item","payload":{"type":"custom_tool_call","name":"functions.exec","input":"text(await tools.exec_command({cmd: opaqueCommand}));"}}`
	const claude = `{"type":"assistant","message":{"role":"assistant","content":[]}}`
	for _, tc := range []struct {
		name, input string
		known       bool
		count       int
	}{
		{name: "opaque custom execution", input: opaque},
		{name: "structured native invocation", input: shell(`fak-dev issue create --title "one"`), known: true, count: 1},
		{name: "explicit Go native target", input: shell(`go run ./cmd/fak-dev issue create --title two`), known: true, count: 1},
		{name: "executable path", input: shell(`C:\tools\fak-dev.exe issue create --title three`), known: true, count: 1},
		{name: "direct issue tool", input: function("github.create_issue", `{"title":"four"}`), known: true, count: 1},
		{name: "local shell array", input: `{"type":"response_item","payload":{"type":"local_shell_call","action":{"command":["bash","-lc","gh issue create --title five"]}}}`, known: true, count: 1},
		{name: "known status zero", input: shell("git status"), known: true},
		{name: "non-create command", input: shell("fak-dev issue list"), known: true},
		{name: "quoted prose", input: shell(`echo "fak-dev issue create --title fake"`), known: true},
		{name: "output only", input: `{"type":"response_item","payload":{"type":"function_call_output","output":"gh issue create: created #123"}}`, known: true},
		{name: "dynamic script", input: shell(`$command = "fak-dev issue create"; & $command`)},
		{name: "nested interpreter", input: shell(`python -c "print('gh issue create')"`)},
		{name: "unhandled execution wrapper", input: function("functions.exec", `{"code":"run(command)"}`)},
		{name: "unsupported row", input: shell("git status") + "\n" + `{"type":"future","payload":{"type":"future"}}`},
		{name: "truncated row", input: shell("git status") + "\n" + `{"type":`},
		{name: "malformed arguments", input: function("functions.exec_command", "{")},
		{name: "positive with opaque coverage", input: shell("fak issue create") + "\n" + opaque},
		{name: "native fanout is opaque", input: shell("fak-dev issue fanout --title followups")},
		{name: "shell alias is opaque", input: shell(`git -c "alias.file=!gh issue create --title x" file`)},
		{name: "Go execution wrapper", input: shell("go test -exec ./file-issue ./internal/headlesslint")},
		{name: "Go toolexec wrapper", input: shell("go build -toolexec=./file-issue ./cmd/fak")},
		{name: "Go generate runs directives", input: shell("go generate ./...")},
		{name: "Go test without wrapper", input: shell("go test ./internal/headlesslint/"), known: true},
		{name: "Claude metadata", input: claude + "\n" + `{"type":"summary"}` + "\n" + `{"type":"system"}`, known: true},
		{name: "foreign loaded record", input: claude + "\n" + `{"type":"foreign"}`},
		{name: "nonrecord JSON", input: claude + "\n1"},
		{name: "unsupported Claude block", input: `{"type":"assistant","message":{"content":[{"type":"future_tool_call"}]}}`},
		{name: "malformed Claude message", input: claude + "\n" + `{"type":"assistant","message":[]}`},
		{name: "malformed text beside tool use", input: `{"type":"assistant","message":{"content":[{"type":"text","text":17},{"type":"tool_use","name":"Bash","input":{"command":"fak issue create"}}]}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "session.jsonl")
			if err := os.WriteFile(path, []byte(tc.input+"\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			got := guardIssuesFiledEvidence(path)
			if got.Known != tc.known || got.Count != tc.count {
				t.Fatalf("evidence = %+v, want known=%v count=%d", got, tc.known, tc.count)
			}
			if !tc.known && foldGuardLeftovers(got, guardLeftoversSummary).LeftoversUnfiled {
				t.Fatal("unknown coverage must not report a witnessed zero")
			}
		})
	}
	if got := guardIssuesFiledEvidenceFromRecords([]transcript.Record{{Type: "response_item"}}, false); got.Known {
		t.Fatalf("discarded Codex payload must remain unknown: %+v", got)
	}
	if got := guardIssuesFiledEvidenceFromRecords([]transcript.Record{{Type: "summary"}}, false); !got.Known || got.Count != 0 {
		t.Fatalf("ordinary Claude metadata must retain known zero: %+v", got)
	}
	path := filepath.Join(t.TempDir(), "claude-tail.jsonl")
	writeGuardLeftoversTranscript(t, path, []toolUseFixture{{Name: "Bash", Input: map[string]any{"command": "fak-dev issue create"}}})
	if got := guardIssuesFiledEvidenceFromRecords(loadGuardLeftoversFixture(t, path), true); !got.Known || got.Count != 1 || got.Source != headlesslint.IssuesFiledFromTranscriptTail {
		t.Fatalf("Claude tail must preserve a positive lower bound: %+v", got)
	}
}

type toolUseFixture struct {
	Name  string
	Input map[string]any
}

func writeGuardLeftoversTranscript(t *testing.T, path string, uses []toolUseFixture) {
	t.Helper()
	blocks := make([]map[string]any, 0, len(uses))
	for _, use := range uses {
		blocks = append(blocks, map[string]any{"type": "tool_use", "name": use.Name, "input": use.Input})
	}
	record := map[string]any{"type": "assistant", "message": map[string]any{"role": "assistant", "content": blocks}}
	b, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func loadGuardLeftoversFixture(t *testing.T, path string) []transcript.Record {
	t.Helper()
	return transcript.LoadFile(path)
}
