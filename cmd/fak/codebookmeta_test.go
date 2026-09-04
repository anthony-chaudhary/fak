package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestCodebookmetaCLI_AcceptFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "codebookmeta", "testdata", "integer-grid.json")
	code := runCodebookmeta(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runCodebookmeta returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "supported"`)) {
		t.Fatalf("expected supported outcome, got: %s", stdout.String())
	}
}

func TestCodebookmetaCLI_TextOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "codebookmeta", "testdata", "nf4.json")
	code := runCodebookmeta(&stdout, &stderr, []string{"--input", fixture})
	if code != 0 {
		t.Fatalf("runCodebookmeta returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("CODEBOOK METADATA ADJUDICATION")) {
		t.Fatalf("expected banner, got: %s", stdout.String())
	}
}

func TestCodebookmetaCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCodebookmeta(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}
