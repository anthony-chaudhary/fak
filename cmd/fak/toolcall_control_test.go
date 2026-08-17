package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
)

func TestToolcallControlShadowRecordsButDoesNotBlock(t *testing.T) {
	dir := t.TempDir()
	post := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-1","tool_input":{"file_path":"a"},"tool_response":{"content":"x"}}`
	if err := toolcallControlHook(&bytes.Buffer{}, strings.NewReader(post), "post", toolcallcontrol.ModeShadow, dir); err != nil {
		t.Fatalf("post: %v", err)
	}
	pre := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-2","prompt_units":128000,"tool_input":{"file_path":"a"}}`
	var out bytes.Buffer
	if err := toolcallControlHook(&out, strings.NewReader(pre), "pre", toolcallcontrol.ModeShadow, dir); err != nil {
		t.Fatalf("pre: %v", err)
	}
	if out.Len() != 0 {
		t.Fatalf("shadow blocked execution: %s", out.String())
	}
	rows := readDecisionRows(t, filepath.Join(dir, "decisions.jsonl"))
	if len(rows) != 2 || rows[0].Outcome == nil || rows[0].Outcome.Class != toolcallcontrol.OutcomeSuccess || rows[1].Verdict == nil || rows[1].Verdict.Action != toolcallcontrol.Reuse || rows[1].Verdict.Applied || rows[1].Verdict.ReplayUnitsSaved != 128000 || rows[1].Verdict.ReplaySquaredSaved != "16384000000" {
		t.Fatalf("rows=%+v", rows)
	}
}

func TestToolcallControlEnforceBlocksFreshRepeat(t *testing.T) {
	dir := t.TempDir()
	post := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-1","tool_input":{"file_path":"a"},"tool_response":{"content":"x"}}`
	if err := toolcallControlHook(&bytes.Buffer{}, strings.NewReader(post), "post", toolcallcontrol.ModeEnforce, dir); err != nil {
		t.Fatalf("post: %v", err)
	}
	pre := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-2","tool_input":{"file_path":"a","fak_prompt_units":128000}}`
	var out bytes.Buffer
	if err := toolcallControlHook(&out, strings.NewReader(pre), "pre", toolcallcontrol.ModeEnforce, dir); err != nil {
		t.Fatalf("pre: %v", err)
	}
	var hook toolcallHookOutput
	if err := json.Unmarshal(out.Bytes(), &hook); err != nil {
		t.Fatalf("hook output: %v: %s", err, out.String())
	}
	if hook.HookSpecificOutput.PermissionDecision != "deny" || !strings.Contains(hook.HookSpecificOutput.PermissionDecisionReason, "read-1") {
		t.Fatalf("hook=%+v", hook)
	}
}

func TestToolcallControlMutationInvalidatesReuse(t *testing.T) {
	dir := t.TempDir()
	calls := []struct{ kind, payload string }{
		{"post", `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-1","tool_input":{"file_path":"a"}}`},
		{"post", `{"session_id":"s1","tool_name":"Write","tool_use_id":"write-1","tool_input":{"file_path":"a","content":"y"}}`},
	}
	for _, call := range calls {
		if err := toolcallControlHook(&bytes.Buffer{}, strings.NewReader(call.payload), call.kind, toolcallcontrol.ModeEnforce, dir); err != nil {
			t.Fatal(err)
		}
	}
	var out bytes.Buffer
	pre := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-2","tool_input":{"file_path":"a","fak_prompt_units":128000}}`
	if err := toolcallControlHook(&out, strings.NewReader(pre), "pre", toolcallcontrol.ModeEnforce, dir); err != nil {
		t.Fatal(err)
	}
	if out.Len() != 0 {
		t.Fatalf("stale read reused after mutation: %s", out.String())
	}
	rows := readDecisionRows(t, filepath.Join(dir, "decisions.jsonl"))
	if rows[len(rows)-1].Verdict == nil {
		t.Fatalf("last row has no verdict: %+v", rows[len(rows)-1])
	}
	if got := rows[len(rows)-1].Verdict.Reason; got != "novel_at_epoch" {
		t.Fatalf("reason=%s", got)
	}
}

func TestToolcallControlMalformedStateFailsOpen(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, toolcallFileStem("s1")+".json"), []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}
	pre := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-2","tool_input":{"file_path":"a","fak_prompt_units":128000}}`
	var out bytes.Buffer
	if err := toolcallControlHook(&out, strings.NewReader(pre), "pre", toolcallcontrol.ModeEnforce, dir); err == nil {
		t.Fatal("expected malformed-state diagnostic")
	}
	if out.Len() != 0 {
		t.Fatalf("malformed state blocked: %s", out.String())
	}
}

func TestToolcallReadOnlyUsesClosedBuiltinSet(t *testing.T) {
	for _, tool := range []string{"Write", "Bash", "TargetUpdate", "mcp__server__delete_record"} {
		if toolcallReadOnly(tool) {
			t.Fatalf("mutation-like tool %q classified read-only", tool)
		}
	}
	for _, tool := range []string{"Read", "Grep", "mcp__server__query"} {
		if !toolcallReadOnly(tool) {
			t.Fatalf("known read tool %q not classified read-only", tool)
		}
	}
}

func TestRunToolprocHookIOWiresShadowAndEnforce(t *testing.T) {
	for _, tc := range []struct {
		name, mode string
		wantDeny   bool
	}{
		{name: "shadow", mode: "shadow"},
		{name: "enforce", mode: "enforce", wantDeny: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			journal := filepath.Join(dir, "toolproc.jsonl")
			control := filepath.Join(dir, "control")
			post := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-1","tool_input":{"file_path":"a"},"tool_response":{"content":"x"}}`
			if rc := runToolprocHookIO(&bytes.Buffer{}, strings.NewReader(post), &bytes.Buffer{}, []string{"post", "--journal", journal, "--control-dir", control, "--control-mode", tc.mode}); rc != 0 {
				t.Fatalf("post rc=%d", rc)
			}
			pre := `{"session_id":"s1","tool_name":"Read","tool_use_id":"read-2","tool_input":{"file_path":"a","fak_prompt_units":128000}}`
			var stdout, stderr bytes.Buffer
			if rc := runToolprocHookIO(&stdout, strings.NewReader(pre), &stderr, []string{"pre", "--journal", journal, "--control-dir", control, "--control-mode", tc.mode}); rc != 0 {
				t.Fatalf("pre rc=%d stderr=%s", rc, stderr.String())
			}
			if got := strings.Contains(stdout.String(), `"permissionDecision":"deny"`); got != tc.wantDeny {
				t.Fatalf("deny=%v want=%v output=%s", got, tc.wantDeny, stdout.String())
			}
		})
	}
}

func TestToolcallOutcomeReceiptPreservesExitOutputAndHookSemantics(t *testing.T) {
	dir := t.TempDir()
	payloads := []string{
		`{"session_id":"s1","tool_name":"Grep","tool_use_id":"grep-miss","fak_outcome":{"expected_negative":true},"tool_input":{"pattern":"absent"},"tool_response":{"is_error":true,"exit_code":1,"stderr":"no matches"}}`,
		`{"session_id":"s1","tool_name":"Bash","tool_use_id":"test-fail","tool_input":{"command":"go test ./internal/widget"},"tool_response":{"is_error":true,"exit_code":1,"stderr":"FAIL widget"}}`,
	}
	for _, payload := range payloads {
		var stdout bytes.Buffer
		if err := toolcallControlHook(&stdout, strings.NewReader(payload), "post", toolcallcontrol.ModeShadow, dir); err != nil {
			t.Fatalf("post hook changed error semantics: %v", err)
		}
		if stdout.Len() != 0 {
			t.Fatalf("post hook changed stdout semantics: %s", stdout.String())
		}
	}
	rows := readDecisionRows(t, filepath.Join(dir, "decisions.jsonl"))
	if len(rows) != 2 || rows[0].Outcome == nil || rows[1].Outcome == nil {
		t.Fatalf("outcome rows=%+v", rows)
	}
	expected, failure := rows[0].Outcome, rows[1].Outcome
	if expected.Class != toolcallcontrol.OutcomeExpectedNegative || expected.Projection != toolcallcontrol.ProjectionExpectedNegative || expected.ExitCode == nil || *expected.ExitCode != 1 || !bytes.Contains(expected.Output, []byte(`"stderr":"no matches"`)) {
		t.Fatalf("expected-negative receipt lost evidence: %+v", expected)
	}
	if failure.Class != toolcallcontrol.OutcomeTestFailure || failure.Projection != toolcallcontrol.ProjectionUnexpectedFailure || failure.ExitCode == nil || *failure.ExitCode != 1 || !bytes.Contains(failure.Output, []byte(`"stderr":"FAIL widget"`)) {
		t.Fatalf("genuine failure was hidden or lost evidence: %+v", failure)
	}
}

func TestToolcallOutcomeDeclarationIsStructuralAndStrippedFromSemanticArgs(t *testing.T) {
	payload := json.RawMessage(`{"session_id":"s1","fak_outcome":{"expected_negative":true,"class":"guard_refusal"},"tool_input":{"fak_expected_negative":true,"command":"fak commit --preview"}}`)
	input := json.RawMessage(`{"fak_expected_negative":true,"command":"fak commit --preview"}`)
	declaration := toolcallOutcomeDeclaration(payload, input)
	if declaration.Invalid || !declaration.ExpectedNegative || !declaration.ExpectedNegativeSet || declaration.Class != toolcallcontrol.OutcomeGuardRefusal {
		t.Fatalf("declaration=%+v", declaration)
	}
	semantic := toolcallSemanticArgs(input)
	if bytes.Contains(semantic, []byte("fak_expected_negative")) || !bytes.Contains(semantic, []byte("command")) {
		t.Fatalf("semantic args=%s", semantic)
	}
	if got := toolcallOutcomeDeclaration(json.RawMessage(`{"fak_expected_negative":true}`), json.RawMessage(`{"fak_expected_negative":false}`)); !got.Invalid {
		t.Fatalf("conflicting declarations must be invalid: %+v", got)
	}
}

func TestToolcallOutcomeReplayCaptured24hSample(t *testing.T) {
	// The issue's frozen active-profile capture contained 81 non-success rows:
	// 78 exit-1 outcomes and 3 exit-124 outcomes. Replaying explicit intent
	// separates all 78 known probes while every genuine timeout stays visible.
	dir := t.TempDir()
	for i := 0; i < 78; i++ {
		payload := fmt.Sprintf(`{"session_id":"captured-24h","tool_name":"Grep","tool_use_id":"probe-%02d","fak_outcome":{"expected_negative":true},"tool_input":{"pattern":"absent-%02d"},"tool_response":{"is_error":true,"exit_code":1,"stderr":"no matches"}}`, i, i)
		if err := toolcallControlHook(&bytes.Buffer{}, strings.NewReader(payload), "post", toolcallcontrol.ModeShadow, dir); err != nil {
			t.Fatalf("expected probe %d: %v", i, err)
		}
	}
	for i := 0; i < 3; i++ {
		payload := fmt.Sprintf(`{"session_id":"captured-24h","tool_name":"Bash","tool_use_id":"timeout-%02d","tool_input":{"command":"fetch artifact %d"},"tool_response":{"is_error":true,"exit_code":124,"timed_out":true,"stderr":"timed out"}}`, i, i)
		if err := toolcallControlHook(&bytes.Buffer{}, strings.NewReader(payload), "post", toolcallcontrol.ModeShadow, dir); err != nil {
			t.Fatalf("timeout %d: %v", i, err)
		}
	}

	rows := readDecisionRows(t, filepath.Join(dir, "decisions.jsonl"))
	var expectedNegative, unexpected, timeout int
	for _, row := range rows {
		if row.Outcome == nil {
			t.Fatalf("replay row missing outcome: %+v", row)
		}
		switch row.Outcome.Projection {
		case toolcallcontrol.ProjectionExpectedNegative:
			expectedNegative++
		case toolcallcontrol.ProjectionUnexpectedFailure:
			unexpected++
		}
		if row.Outcome.Class == toolcallcontrol.OutcomeTimeoutInterruption {
			timeout++
		}
	}
	if len(rows) != 81 || expectedNegative != 78 || unexpected != 3 || timeout != 3 {
		t.Fatalf("replay total=%d expected_negative=%d unexpected=%d timeout=%d; want 81/78/3/3", len(rows), expectedNegative, unexpected, timeout)
	}
	// Red-noise reduction is 78/81 (96.3%); the remaining 3/81 are genuine,
	// typed failures rather than a coerced success or suppressed output.
}

func readDecisionRows(t *testing.T, path string) []toolcallTraceRow {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var rows []toolcallTraceRow
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte{'\n'}) {
		var row toolcallTraceRow
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatal(err)
		}
		rows = append(rows, row)
	}
	return rows
}
