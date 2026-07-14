package main

// Hook-level tests for the FOREGROUND_NETWORK_LOOP advisory rung (#4595): a
// foreground `for … do gh … done` fan-out warns on stderr and ALLOWS (no deny
// JSON), and --check reports WARN with exit 0 for the advisory-only finding.

import (
	"bytes"
	"strings"
	"testing"
)

func TestHookWarnsButAllowsForegroundNetworkLoop(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, errOut := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"for n in 3009 3010 3011; do gh issue view $n; done"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" {
		t.Fatalf("network loop: rc=%d out=%q, want (0, \"\") - advisory must not deny", rc, out)
	}
	if !strings.Contains(errOut, "FOREGROUND_NETWORK_LOOP") || !strings.Contains(errOut, "advisory") {
		t.Fatalf("stderr = %q, want an advisory FOREGROUND_NETWORK_LOOP pointer", errOut)
	}
}

func TestHookBatchedGhCallStaysSilent(t *testing.T) {
	t.Setenv("FAK_REPO_GUARD", "")
	rc, out, errOut := runHookString(t, `{"tool_name":"Bash","cwd":"`+wsTest+`","tool_input":{"command":"gh issue list --json number,title --limit 100"}}`)
	if rc != 0 || strings.TrimSpace(out) != "" || strings.Contains(errOut, "FOREGROUND_NETWORK_LOOP") {
		t.Fatalf("batched gh: rc=%d out=%q err=%q, want silent allow", rc, out, errOut)
	}
}

func TestCheckWarnsOnNetworkLoop(t *testing.T) {
	var out bytes.Buffer
	rc := runCheck("for n in 1 2 3; do gh issue view $n; done", wsTest, false, &out)
	if rc != 0 {
		t.Fatalf("runCheck(network loop) = %d, want 0 (advisory-only must pass)", rc)
	}
	if !strings.HasPrefix(out.String(), "WARN") || !strings.Contains(out.String(), "FOREGROUND_NETWORK_LOOP") {
		t.Fatalf("check output = %q, want a WARN FOREGROUND_NETWORK_LOOP line", out.String())
	}
}
