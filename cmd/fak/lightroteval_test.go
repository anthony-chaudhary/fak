package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestLightRotEvalCLI_Supported(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "lightroteval", "testdata", "case-1.json")
	code := runLightRotEval(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runLightRotEval returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "supported"`)) {
		t.Fatalf("expected supported outcome, got: %s", stdout.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"reason": "LIGHTROT_EVALUATED_MODELED"`)) {
		t.Fatalf("expected LIGHTROT_EVALUATED_MODELED reason, got: %s", stdout.String())
	}
}

func TestLightRotEvalCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLightRotEval(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}

func TestLightRotEvalCLI_Unsupported(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runLightRotEval(&stdout, &stderr, []string{"--input", filepath.Join(repoRoot(), "cmd", "fak", "main.go"), "--json"})
	if code != 3 && code != 1 {
		t.Fatalf("expected exit code 3 or 1 for invalid input, got: %d", code)
	}
}
