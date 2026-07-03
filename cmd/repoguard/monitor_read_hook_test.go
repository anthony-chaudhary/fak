package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHookDeniesReadOfLiveMonitorOutput(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	if err := os.WriteFile(journal, []byte(`{"kind":"spawn","call_id":"mon-live","session":"s1","tool":"Monitor","at_unix_ms":1}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_REPO_GUARD_TOOLPROC_JOURNAL", journal)

	rc, out, errOut := runHookString(t, `{"session_id":"s1","tool_name":"Read","cwd":"`+wsTest+`","tool_input":{"file_path":"C:/Users/u/AppData/Local/Claude/tasks/mon-live.output"}}`)
	if rc != 0 {
		t.Fatalf("hook rc = %d, want 0", rc)
	}
	var decision hookDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &decision); err != nil {
		t.Fatalf("hook stdout is not a decision JSON: %v (out=%q, err=%q)", err, out, errOut)
	}
	reason := decision.HookSpecificOutput.PermissionDecisionReason
	if decision.HookSpecificOutput.PermissionDecision != "deny" ||
		!strings.Contains(reason, "LIVE_MONITOR_OUTPUT_READ") ||
		!strings.Contains(reason, "live Monitor events are pushed") {
		t.Fatalf("decision = %+v, want live Monitor output deny", decision.HookSpecificOutput)
	}
}

func TestHookAllowsReadOfOtherSessionMonitorOutput(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	if err := os.WriteFile(journal, []byte(`{"kind":"spawn","call_id":"mon-live","session":"s1","tool":"Monitor","at_unix_ms":1}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_REPO_GUARD_TOOLPROC_JOURNAL", journal)

	rc, out, errOut := runHookString(t, `{"session_id":"s2","tool_name":"Read","cwd":"`+wsTest+`","tool_input":{"file_path":"C:/Users/u/AppData/Local/Claude/tasks/mon-live.output"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" || strings.Contains(errOut, "LIVE_MONITOR_OUTPUT_READ") {
		t.Fatalf("other-session read: rc=%d out=%q err=%q, want silent allow", rc, out, errOut)
	}
}
