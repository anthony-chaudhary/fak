package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunRoofline(t *testing.T) {
	t.Run("json schema", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runRoofline(&stdout, &stderr, []string{"--workspace", repoRoot(), "--json"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), `"schema": "fak.roofline-dashboard/1"`) {
			t.Errorf("expected roofline schema in json output, got: %s", stdout.String())
		}
	})

	t.Run("markdown", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runRoofline(&stdout, &stderr, []string{"--workspace", repoRoot(), "--markdown"})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(strings.ToLower(stdout.String()), "roofline dashboard") {
			t.Errorf("expected markdown output to contain 'roofline dashboard', got: %s", stdout.String())
		}
	})

	t.Run("default text", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runRoofline(&stdout, &stderr, []string{"--workspace", repoRoot()})
		if rc != 0 {
			t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
		}
		if !strings.Contains(stdout.String(), "Roofline Dashboard") {
			t.Errorf("expected text output to contain 'Roofline Dashboard', got: %s", stdout.String())
		}
	})

	t.Run("invalid flag", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		rc := runRoofline(&stdout, &stderr, []string{"--invalid-flag-xyz"})
		if rc != 2 {
			t.Fatalf("expected rc 2 on invalid flag, got %d", rc)
		}
	})
}
