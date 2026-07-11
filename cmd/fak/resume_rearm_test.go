package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResumeRearmWritesValidatedMarkerAndLists(t *testing.T) {
	d := t.TempDir()
	var out, err bytes.Buffer
	if c := runResumeRearm(&out, &err, []string{"--registry-dir", d, "--reason", "false cap", "abc12345"}); c != 0 {
		t.Fatalf("code=%d err=%s", c, err.String())
	}
	b, _ := os.ReadFile(filepath.Join(d, "resume_ledger.jsonl"))
	if !strings.Contains(string(b), `"phase":"rearm"`) || !strings.Contains(string(b), `"manual_override":true`) {
		t.Fatalf("row=%s", b)
	}
	out.Reset()
	if c := runResumeRearm(&out, &err, []string{"--registry-dir", d, "--list"}); c != 0 || !strings.Contains(out.String(), "abc12345\tfalse cap") {
		t.Fatalf("list=%q code=%d", out.String(), c)
	}
}
func TestResumeRearmRejectsBadSession(t *testing.T) {
	var out, err bytes.Buffer
	if c := runResumeRearm(&out, &err, []string{"bad space"}); c != 2 {
		t.Fatalf("code=%d", c)
	}
}
