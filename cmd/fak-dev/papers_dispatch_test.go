package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestPapersDispatch(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(&out, &errb, []string{"papers"}); code != 2 || !strings.Contains(errb.String(), "usage: fak-dev papers") {
		t.Fatalf("papers without subcommand: code=%d", code)
	}
	out.Reset()
	idx := filepath.Join(t.TempDir(), "index.jsonl")
	if code := run(&out, &errb, []string{"papers", "latest", "--index", idx, "--json"}); code != 0 || strings.TrimSpace(out.String()) != "[]" {
		t.Fatalf("papers latest on empty index: code=%d out=%q", code, out.String())
	}
}
