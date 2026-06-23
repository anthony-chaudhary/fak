package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunCodelintJSONEmptyIsArray pins the array contract: a clean run with --json
// must emit `[]`, not `null`, so a machine consumer (`jq '.[]'`, typed unmarshal)
// never gets a type surprise on the common path.
func TestRunCodelintJSONEmptyIsArray(t *testing.T) {
	dir := t.TempDir()
	ok := filepath.Join(dir, "ok.go")
	if err := os.WriteFile(ok, []byte("package x\n\nfunc F() int { return 1 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runCodelint(&out, &errb, []string{"--json", ok}); code != 0 {
		t.Fatalf("clean file: exit %d, want 0 (stderr=%s)", code, errb.String())
	}
	if got := strings.TrimSpace(out.String()); got != "[]" {
		t.Fatalf("--json on a clean run must emit [], got %q", got)
	}
}

// TestRunCodelintExitsOnError: a hard parse error exits 1 and names the finding.
func TestRunCodelintExitsOnError(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.go")
	if err := os.WriteFile(bad, []byte("package x\nfunc ("), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runCodelint(&out, &errb, []string{bad}); code != 1 {
		t.Fatalf("broken file: exit %d, want 1", code)
	}
	if !strings.Contains(out.String(), "GO_PARSE") {
		t.Errorf("want a GO_PARSE finding in the output, got %q", out.String())
	}
}

// TestRunCodelintListAndUsage: --list reports the language menu (exit 0); no target
// and no --list is a usage error (exit 2).
func TestRunCodelintListAndUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCodelint(&out, &errb, []string{"--list"}); code != 0 {
		t.Fatalf("--list: exit %d, want 0", code)
	}
	if !strings.Contains(out.String(), "go") || !strings.Contains(out.String(), "python") {
		t.Errorf("--list should name the languages, got %q", out.String())
	}
	var o2, e2 bytes.Buffer
	if code := runCodelint(&o2, &e2, nil); code != 2 {
		t.Errorf("no target and no --list: exit %d, want 2", code)
	}
}
