package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, nil)
	if code != 2 {
		t.Fatalf("code = %d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), "usage: toktdiag <model.gguf>") {
		t.Fatalf("stderr missing usage:\n%s", stderr.String())
	}
}

func TestRunOpenError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{filepath.Join(t.TempDir(), "missing.gguf")})
	if code != 1 {
		t.Fatalf("code = %d, want 1", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), "open:") {
		t.Fatalf("stderr missing open error:\n%s", stderr.String())
	}
}
