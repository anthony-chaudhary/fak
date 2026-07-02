package main

// Hook-level tests for the FOREGROUND_SLEEP advisory rung (#2366): a long
// foreground sleep warns on stderr and ALLOWS (no deny JSON), a mixed call
// with a refusal-class violation still denies, and --check reports WARN with
// exit 0 for advisory-only findings.

import (
	"bytes"
	"strings"
	"testing"
)

func TestHookWarnsButAllowsLongSleep(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, errOut := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"sleep 300"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("long sleep: rc=%d out=%q, want (0, \"\") - advisory must not deny", rc, out)
	}
	if !strings.Contains(errOut, "FOREGROUND_SLEEP") || !strings.Contains(errOut, "advisory") {
		t.Fatalf("stderr = %q, want an advisory FOREGROUND_SLEEP pointer", errOut)
	}
}

func TestHookWarnsButAllowsPowerShellStartSleep(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, errOut := runHookString(t, `{"tool_name":"PowerShell","cwd":"`+wsTest+`","tool_input":{"command":"Start-Sleep -Seconds 600; Get-Job"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("Start-Sleep: rc=%d out=%q, want (0, \"\")", rc, out)
	}
	if !strings.Contains(errOut, "FOREGROUND_SLEEP") {
		t.Fatalf("stderr = %q, want the FOREGROUND_SLEEP advisory", errOut)
	}
}

func TestHookShortSleepStaysSilent(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, errOut := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"sleep 5"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" || strings.Contains(errOut, "FOREGROUND_SLEEP") {
		t.Fatalf("sleep 5: rc=%d out=%q err=%q, want silent allow", rc, out, errOut)
	}
}

func TestHookMixedViolationsStillDeny(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, _ := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"sleep 300; rm -rf ../tools"}}`)
	if rc != 0 {
		t.Fatalf("hook rc = %d, want 0", rc)
	}
	if !strings.Contains(out, `"permissionDecision"`) || !strings.Contains(out, "deny") {
		t.Fatalf("stdout = %q, want a deny decision for the out-of-tree write", out)
	}
}

func TestCheckWarnsOnAdvisoryOnly(t *testing.T) {
	var out bytes.Buffer
	rc := runCheck("sleep 300", wsTest, false, &out)
	if rc != 0 {
		t.Fatalf("runCheck(sleep 300) = %d, want 0 (advisory-only must pass)", rc)
	}
	if !strings.HasPrefix(out.String(), "WARN") || !strings.Contains(out.String(), "FOREGROUND_SLEEP") {
		t.Fatalf("check output = %q, want a WARN FOREGROUND_SLEEP line", out.String())
	}
}
