package main

import (
	"bytes"
	"encoding/json"
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
	if len(rows) != 1 || rows[0].Verdict.Action != toolcallcontrol.Reuse || rows[0].Verdict.Applied || rows[0].Verdict.ReplayUnitsSaved != 128000 || rows[0].Verdict.ReplaySquaredSaved != "16384000000" {
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
