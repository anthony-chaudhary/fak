package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestCubicQuantEvalCLI_Supported(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "cubicquanteval", "testdata", "evaluation-v1.json")
	code := runCubicQuantEval(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runCubicQuantEval returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "supported"`)) {
		t.Fatalf("expected supported outcome, got: %s", stdout.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"reason": "CUBICQUANT_EVALUATED"`)) {
		t.Fatalf("expected CUBICQUANT_EVALUATED reason, got: %s", stdout.String())
	}
}

func TestCubicQuantEvalCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runCubicQuantEval(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}

func TestCubicQuantEvalCLI_Delegate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "cubicquanteval", "testdata", "evaluation-v1.json")
	code := runCubicQuantEval(&stdout, &stderr, []string{"--input", fixture, "--scope", "model-quality", "--json"})
	if code != 4 {
		t.Fatalf("expected exit code 4 for delegate, got: %d", code)
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "delegate"`)) {
		t.Fatalf("expected delegate outcome, got: %s", stdout.String())
	}
}
