package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/localappcert"
)

func TestRunPreservesMatrixValidationModes(t *testing.T) {
	matrixPath := writeValidMatrix(t)
	t.Run("text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := run([]string{"--matrix", matrixPath, "ignored-positional-argument"}, &stdout, &stderr); code != 0 {
			t.Fatalf("run() code = %d, stderr = %q", code, stderr.String())
		}
		if stdout.String() != "localappcert: PASS\n" || stderr.Len() != 0 {
			t.Fatalf("stdout/stderr = %q/%q", stdout.String(), stderr.String())
		}
	})
	t.Run("json", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		if code := run([]string{"--matrix", matrixPath, "--json"}, &stdout, &stderr); code != 0 {
			t.Fatalf("run() code = %d, stderr = %q", code, stderr.String())
		}
		var verdict struct {
			Schema string `json:"schema"`
			OK     bool   `json:"ok"`
			Matrix string `json:"matrix"`
		}
		if err := json.Unmarshal(stdout.Bytes(), &verdict); err != nil {
			t.Fatal(err)
		}
		if verdict.Schema != "fak.local-app-certification-verdict/1" || !verdict.OK || verdict.Matrix != matrixPath || stderr.Len() != 0 {
			t.Fatalf("verdict/stderr = %#v/%q", verdict, stderr.String())
		}
	})
}

func TestRunHelpExitsSuccessfully(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"--help"}, &stdout, &stderr); code != 0 {
		t.Fatalf("run() code = %d, want 0; stderr = %q", code, stderr.String())
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), "Usage of localappcert:") || !strings.Contains(stderr.String(), "-capture") || !strings.Contains(stderr.String(), "-matrix") {
		t.Fatalf("stdout/help = %q/%q", stdout.String(), stderr.String())
	}
}

func TestRunCaptureRefusesIncompleteSpecBeforeCreatingEvidence(t *testing.T) {
	spec := map[string]any{
		"schema":           localappcert.CaptureSchema,
		"runtime_revision": "runtime-r1",
		"artifact":         "artifact@sha256:abc",
		"scenarios": map[string]any{
			localappcert.RequiredScenarios[0]: map[string]any{"command": []string{"must-not-run"}, "timeout": "1s"},
		},
	}
	b, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	specPath := filepath.Join(t.TempDir(), "capture.json")
	if err := os.WriteFile(specPath, b, 0o644); err != nil {
		t.Fatal(err)
	}
	evidenceDir := filepath.Join(t.TempDir(), "must-not-exist")
	var stdout, stderr bytes.Buffer
	if code := run([]string{"--capture", specPath, "--evidence-dir", evidenceDir}, &stdout, &stderr); code != 1 {
		t.Fatalf("run() code = %d, want 1", code)
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), "missing capture scenarios") {
		t.Fatalf("stdout/stderr = %q/%q", stdout.String(), stderr.String())
	}
	if _, err := os.Stat(evidenceDir); !os.IsNotExist(err) {
		t.Fatalf("evidence directory was created: %v", err)
	}
}

func writeValidMatrix(t *testing.T) string {
	t.Helper()
	envelope := localappcert.Envelope{
		ID: "m3", Chip: "Apple M3 Pro", MemoryBytes: 36 << 30, MacOS: "26.6.2", Power: "AC", Thermal: "nominal",
		PackRevision: "pack@sha256:x", RuntimeRevision: "runtime-r1", Supported: true,
	}
	for _, name := range localappcert.RequiredScenarios {
		envelope.Scenarios = append(envelope.Scenarios, localappcert.Scenario{
			Name: name, Status: localappcert.StatusPass, Evidence: "evidence/" + name + ".log",
			Receipt: &localappcert.Receipt{Engine: "fak-native", Fallback: "none", Artifact: "artifact@sha256:abc", RuntimeRevision: "runtime-r1"},
		})
	}
	matrix := localappcert.Matrix{Schema: localappcert.Schema, GeneratedAt: "2026-08-27T00:00:00Z", Envelopes: []localappcert.Envelope{envelope}}
	b, err := json.Marshal(matrix)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "matrix.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
