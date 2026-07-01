package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestRunResumeResolveNotFound drives the CLI end-to-end against an empty home: no
// ~/.claude* dir holds the session, so the resolver returns NOT_FOUND, exit 1, and an
// empty stdout (nothing to pin). This exercises the roster-build + Resolve wiring
// without a live fleet.
func TestRunResumeResolveNotFound(t *testing.T) {
	home := t.TempDir()
	var out, errb bytes.Buffer
	code := runResumeResolve(&out, &errb, []string{"--home", home, "no-such-session"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (NOT_FOUND); stderr=%q", code, errb.String())
	}
	if strings.TrimSpace(out.String()) != "" {
		t.Fatalf("stdout = %q, want empty on not-found (nothing to pin)", out.String())
	}
	if !strings.Contains(errb.String(), "no ~/.claude* account holds this session") {
		t.Fatalf("stderr = %q, want the not-found reason", errb.String())
	}
}

// TestRunResumeResolveJSONNotFound: --json still exits 1 but emits the record.
func TestRunResumeResolveJSONNotFound(t *testing.T) {
	home := t.TempDir()
	var out, errb bytes.Buffer
	code := runResumeResolve(&out, &errb, []string{"--home", home, "--json", "ghost"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if !strings.Contains(out.String(), `"action": "NOT_FOUND"`) {
		t.Fatalf("stdout = %q, want NOT_FOUND record", out.String())
	}
}

// TestRunResumeResolveUsage: missing session id is a usage error (exit 2).
func TestRunResumeResolveUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runResumeResolve(&out, &errb, []string{}); code != 2 {
		t.Fatalf("no-arg exit = %d, want 2", code)
	}
	if code := runResumeResolve(&out, &errb, []string{"a", "b"}); code != 2 {
		t.Fatalf("two-arg exit = %d, want 2", code)
	}
}
