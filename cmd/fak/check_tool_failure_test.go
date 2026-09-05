package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunCheckToolFailureLookupJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{"--json", "TOOL_PARTIAL_APPLY"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"token": "TOOL_PARTIAL_APPLY"`, `"retryable": false`, `"fix":`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("json output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunCheckToolFailureList(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{"--list"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"TOOL_HANG", "TOOL_TIMEOUT", "TOOL_SHELL_MISMATCH", "TOOL_HANG_SHELL_MISMATCH", "TOOL_PARTIAL_APPLY"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("list missing %s:\n%s", want, out.String())
		}
	}
}

func TestRunCheckToolFailureMessage(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{"--message", "Bash exited with exit status 143 while gh was still running"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "TOOL_HANG_SHELL_MISMATCH") {
		t.Fatalf("message classification output:\n%s", out.String())
	}
}

func TestRunCheckToolFailureMessagePayload(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{
		"--message", "Bash exited with exit status 143 while gh was still running",
		"--command", "gh issue view 2072 --json title",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"TOOL_HANG_SHELL_MISMATCH", "next_command:", "powershell -NoProfile -Command 'gh issue view 2072 --json title'"} {
		if !strings.Contains(got, want) {
			t.Fatalf("payload output missing %q:\n%s", want, got)
		}
	}
}

func TestRunCheckToolFailureMessagePayloadJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{
		"--json", "--message", "tool hung: no output for 120s",
	})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"code": "TOOL_HANG"`, `"next_command":`, `"evidence":`, `"cause":`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("json payload missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunCheckToolFailureUnknown(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCheckToolFailure(&out, &errb, []string{"FILE_ADMISSION"})
	if code != 3 {
		t.Fatalf("exit=%d, want 3; stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "unknown tool-failure token") {
		t.Fatalf("stderr missing unknown-token message: %s", errb.String())
	}
}

func TestRunCheckToolFailureExitCodeZeroIgnoresErrorSignature(t *testing.T) {
	var out, errb bytes.Buffer
	// Message has substring "command not found", but exit-code is 0 -> should NOT classify as failure
	code := runCheckToolFailure(&out, &errb, []string{
		"--message", "git log: fixed command not found in readme",
		"--exit-code", "0",
	})
	if code != 3 {
		t.Fatalf("expected code 3 (did not match tool failure on exit 0), got %d", code)
	}
}
