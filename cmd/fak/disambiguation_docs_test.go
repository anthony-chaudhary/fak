package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDisambiguationDocsWritesAndChecksDeterministically(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"docs", "--output-dir", dir, "--json"}); code != 0 {
		t.Fatalf("write code=%d stderr=%s", code, stderr.String())
	}
	var first disambiguationDocsReport
	if err := json.Unmarshal(stdout.Bytes(), &first); err != nil {
		t.Fatal(err)
	}
	if len(first.Files) != 2 {
		t.Fatalf("files=%d, want 2", len(first.Files))
	}
	for _, file := range first.Files {
		if !file.Changed {
			t.Errorf("initial %s not changed", file.Path)
		}
		if _, err := os.Stat(filepath.FromSlash(file.Path)); err != nil {
			t.Errorf("stat %s: %v", file.Path, err)
		}
	}

	stdout.Reset()
	stderr.Reset()
	if code := runDisambiguation(&stdout, &stderr, []string{"docs", "--output-dir", dir, "--check", "--json"}); code != 0 {
		t.Fatalf("check code=%d stderr=%s", code, stderr.String())
	}
	var checked disambiguationDocsReport
	if err := json.Unmarshal(stdout.Bytes(), &checked); err != nil {
		t.Fatal(err)
	}
	for _, file := range checked.Files {
		if file.Changed {
			t.Errorf("checked %s changed", file.Path)
		}
	}
}

func TestDisambiguationDocsCheckRejectsStalePage(t *testing.T) {
	dir := t.TempDir()
	if code := runDisambiguation(&bytes.Buffer{}, &bytes.Buffer{}, []string{"docs", "--output-dir", dir}); code != 0 {
		t.Fatalf("write code=%d", code)
	}
	if err := os.WriteFile(filepath.Join(dir, "canonical-terms.md"), []byte("stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	if code := runDisambiguation(&bytes.Buffer{}, &stderr, []string{"docs", "--output-dir", dir, "--check"}); code != 1 {
		t.Fatalf("check code=%d, want 1; stderr=%s", code, stderr.String())
	}
}
