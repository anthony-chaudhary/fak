package main

import (
	"bytes"
	"path/filepath"
	"testing"
)

func TestKVQuantQualityCLI_Supported(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "kvquantquality", "testdata", "kv-q8-4096.json")
	code := runKVQuantQuality(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 0 {
		t.Fatalf("runKVQuantQuality returned %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "supported"`)) {
		t.Fatalf("expected supported outcome, got: %s", stdout.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"reason": "within_quality_budget"`)) {
		t.Fatalf("expected within_quality_budget reason, got: %s", stdout.String())
	}
}

func TestKVQuantQualityCLI_Refused(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join(repoRoot(), "internal", "kvquantquality", "testdata", "kv-q4-16384.json")
	code := runKVQuantQuality(&stdout, &stderr, []string{"--input", fixture, "--json"})
	if code != 3 {
		t.Fatalf("expected exit code 3 for refused, got: %d; stderr: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "unsupported"`)) {
		t.Fatalf("expected unsupported outcome, got: %s", stdout.String())
	}
}

func TestKVQuantQualityCLI_MissingInput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runKVQuantQuality(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("expected exit code 2, got: %d", code)
	}
}
