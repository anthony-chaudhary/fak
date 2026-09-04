package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestFp4metaCLI_AcceptFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "fp4meta", "testdata", "nvfp4.json")
	code := runFp4meta(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runFp4meta returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "accept"`)) {
		t.Fatalf("expected accept outcome, got: %s", stdout.String())
	}
}

func TestFp4metaCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFp4meta(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}
