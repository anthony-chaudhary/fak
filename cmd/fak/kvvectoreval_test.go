package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvvectoreval"
)

func TestKVVectorEvalInspect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runKVVectorEval(&stdout, &stderr, []string{"inspect"})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, kvvectoreval.ContractID) {
		t.Errorf("missing contract ID: %s", out)
	}

	stdout.Reset()
	stderr.Reset()
	code = runKVVectorEval(&stdout, &stderr, []string{"inspect", "--json"})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	var data map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &data); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if data["contract"] != kvvectoreval.ContractID {
		t.Errorf("contract mismatch in json: %v", data["contract"])
	}
}

func TestKVVectorEvalEvaluate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// Default eval should delegate since --runtime-available is false
	code := runKVVectorEval(&stdout, &stderr, []string{"eval"})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "delegate") {
		t.Errorf("expected delegate, got: %s", stdout.String())
	}

	// Supported eval
	stdout.Reset()
	code = runKVVectorEval(&stdout, &stderr, []string{"eval", "--runtime-available"})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "supported") {
		t.Errorf("expected supported, got: %s", stdout.String())
	}

	// Mismatched eval should return 1
	stdout.Reset()
	stderr.Reset()
	code = runKVVectorEval(&stdout, &stderr, []string{"eval", "--contract-id", "fak.unknown/v1"})
	if code != 1 {
		t.Fatalf("expected 1, got %d", code)
	}
}

func TestKVVectorEvalVerifyArtifact(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "artifact.bin")
	if err := os.WriteFile(path, []byte("test-content"), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	// Unknown artifact or mismatched digest returns 1
	code := runKVVectorEval(&stdout, &stderr, []string{"verify-artifact", "--id", "unknown", "--file", path})
	if code != 1 {
		t.Fatalf("expected 1, got %d", code)
	}
}

func TestKVVectorEvalUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runKVVectorEval(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("expected 0 for help, got %d", code)
	}
	code = runKVVectorEval(&stdout, &stderr, nil)
	if code != 2 {
		t.Fatalf("expected 2 for empty argv, got %d", code)
	}
}
