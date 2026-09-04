package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestBitnetmetaCLI_AcceptFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "bitnetmeta", "testdata", "native-ternary.json")
	code := runBitnetmeta(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runBitnetmeta returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "accept"`)) {
		t.Fatalf("expected accept outcome, got: %s", stdout.String())
	}
}

func TestBitnetmetaCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBitnetmeta(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}
