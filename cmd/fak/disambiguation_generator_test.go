package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDisambiguationGenerateWritesAndChecksStableArtifact(t *testing.T) {
	output := filepath.Join(t.TempDir(), "index.json")
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"generate", "--output", output, "--json"}); code != 0 {
		t.Fatalf("generate code=%d stderr=%s", code, stderr.String())
	}
	first, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	firstDigest := sha256.Sum256(first)

	stdout.Reset()
	stderr.Reset()
	if code := runDisambiguation(&stdout, &stderr, []string{"generate", "--output", output, "--json"}); code != 0 {
		t.Fatalf("second generate code=%d stderr=%s", code, stderr.String())
	}
	second, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("regeneration changed bytes: %x != %x", firstDigest, sha256.Sum256(second))
	}
	var report disambiguationGenerateReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Changed {
		t.Fatalf("second generation reported changed: %+v", report)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runDisambiguation(&stdout, &stderr, []string{"generate", "--output", output, "--check", "--json"}); code != 0 {
		t.Fatalf("clean check code=%d stderr=%s", code, stderr.String())
	}
}

func TestDisambiguationGenerateCheckDetectsDrift(t *testing.T) {
	output := filepath.Join(t.TempDir(), "index.json")
	if err := os.WriteFile(output, []byte("stale\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"generate", "--output", output, "--check"}); code != 1 {
		t.Fatalf("check code=%d want 1 stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	got, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "stale\n" {
		t.Fatalf("check mutated artifact: %q", got)
	}
}
