package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

const wsTest = "C:/Users/u/work/fak"

func TestSelftestPasses(t *testing.T) {
	var buf bytes.Buffer
	if rc := runSelftest(&buf); rc != 0 {
		t.Errorf("runSelftest returned %d; output:\n%s", rc, buf.String())
	}
}

func runHookString(t *testing.T, payload string) (int, string, string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	rc := runHook(strings.NewReader(payload), &out, &errBuf)
	return rc, out.String(), errBuf.String()
}

func TestHookDeniesOutOfTree(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ../tools"}}`)
	if rc != 0 {
		t.Fatalf("hook rc = %d, want 0", rc)
	}
	var decision hookDecision
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &decision); err != nil {
		t.Fatalf("hook stdout is not a decision JSON: %v (out=%q)", err, out)
	}
	if decision.HookSpecificOutput.PermissionDecision != "deny" {
		t.Errorf("permissionDecision = %q, want deny", decision.HookSpecificOutput.PermissionDecision)
	}
}

func TestHookAllowsInRepo(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ./build"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("in-repo op: rc=%d out=%q, want (0, \"\")", rc, out)
	}
}

func TestHookWarnModeAllows(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "warn")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ../tools"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("warn mode: rc=%d out=%q, want (0, \"\") - no deny JSON on stdout", rc, out)
	}
}

func TestHookOffModeDisables(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "off")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"rm -rf ../tools"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Errorf("off mode: rc=%d out=%q, want (0, \"\")", rc, out)
	}
}
