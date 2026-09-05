package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTestCorpus(t *testing.T, path string) {
	t.Helper()
	cp := corpus{
		Sources: []string{"test-source"},
		Results: []resultCase{
			{Name: "clean", Tool: "read", Payload: "safe context payload"},
		},
		Calls: []callCase{
			{Tool: "read", Args: "{}"},
		},
	}
	raw, err := json.Marshal(cp)
	if err != nil {
		t.Fatalf("marshal test corpus: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write test corpus: %v", err)
	}
}

func TestRunOutputWriteFailure(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeTestCorpus(t, corpusPath)

	// An existing directory is an invalid destination for file writing.
	invalidOut := t.TempDir()

	var stdout, stderr bytes.Buffer
	err := run(&stdout, &stderr, []string{"-corpus", corpusPath, "-out", invalidOut})
	if err == nil {
		t.Fatal("expected run() to return non-zero error when report write fails, got nil")
	}

	// Must not report false success
	if strings.Contains(stdout.String(), "wrote ") {
		t.Fatalf("stdout reported false success on failed write:\n%s", stdout.String())
	}

	// Must print error to stderr
	if !strings.Contains(stderr.String(), "write report:") {
		t.Fatalf("stderr does not contain write error:\n%s", stderr.String())
	}
}

func TestRunOutputSuccess(t *testing.T) {
	dir := t.TempDir()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeTestCorpus(t, corpusPath)

	outPath := filepath.Join(dir, "report.json")

	var stdout, stderr bytes.Buffer
	err := run(&stdout, &stderr, []string{"-corpus", corpusPath, "-out", outPath})
	if err != nil {
		t.Fatalf("expected run() to succeed, got: %v (stderr: %s)", err, stderr.String())
	}

	if !strings.Contains(stdout.String(), "wrote "+outPath) {
		t.Fatalf("stdout missing success message:\n%s", stdout.String())
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("expected report file to exist: %v", err)
	}

	var rep map[string]any
	if err := json.Unmarshal(data, &rep); err != nil {
		t.Fatalf("expected valid JSON report, got: %v", err)
	}
}
