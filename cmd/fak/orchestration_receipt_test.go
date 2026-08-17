package main

import (
	"bytes"
	"testing"
)

func TestOrchestrationPlanWritesSessionLinkedInvocationReceipt(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_THREAD_ID", "receipt-session")
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "implement the multi-step feature and ship it", "--codex-home", home, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	receipt, ok := readCodexOrchestrationInvocationReceipt(home, "receipt-session")
	if !ok {
		t.Fatal("missing invocation receipt")
	}
	if receipt.Resolved != "ultracode" || receipt.WorkClass != "grind" || receipt.MaxWorkers != 4 || receipt.TaskID == "" {
		t.Fatalf("receipt=%+v", receipt)
	}
}

func TestOrchestrationSelfcheckDoesNotClaimInvocation(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_THREAD_ID", "selfcheck-session")
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "implement the multi-step feature", "--codex-home", home, "--selfcheck", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	if _, ok := readCodexOrchestrationInvocationReceipt(home, "selfcheck-session"); ok {
		t.Fatal("selfcheck wrote invocation receipt")
	}
}
