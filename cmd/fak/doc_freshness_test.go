package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/docfreshrsi"
	"github.com/anthony-chaudhary/fak/internal/milestonedoc"
	"github.com/anthony-chaudhary/fak/internal/supportmaturityscore"
)

func TestDocFreshnessCheckCleanTempWorkspace(t *testing.T) {
	tmp := t.TempDir()

	// Create fresh milestone status doc
	statusDir := filepath.Join(tmp, "docs", "milestones")
	if err := os.MkdirAll(statusDir, 0755); err != nil {
		t.Fatal(err)
	}
	statusContent := "# Milestone status\n\n" + milestonedoc.Block() + "\n"
	if err := os.WriteFile(filepath.Join(statusDir, "STATUS.md"), []byte(statusContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Create fresh hardware matrix doc
	matrixContent := "# Hardware Matrix\n\n" + supportmaturityscore.MatrixBlock() + "\n"
	if err := os.WriteFile(filepath.Join(tmp, "docs", "HARDWARE-MATRIX.md"), []byte(matrixContent), 0644); err != nil {
		t.Fatal(err)
	}

	// Create VERSION file
	if err := os.WriteFile(filepath.Join(tmp, "VERSION"), []byte("0.54.0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Create a clean guide doc
	guideContent := `# Clean Guide

> Orientation: Orientation guide.

A fresh guide with resolved links.

## Read next

- [STATUS.md](milestones/STATUS.md)
`
	if err := os.WriteFile(filepath.Join(tmp, "docs", "guide.md"), []byte(guideContent), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runDocFreshness(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--generated",
		"--check",
	})
	if code != 0 {
		t.Fatalf("expected exit 0, got %d; stderr=%s, stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "PASS") {
		t.Errorf("expected PASS in stdout, got:\n%s", stdout.String())
	}
}

func TestDocFreshnessCheckStaleAndRefresh(t *testing.T) {
	tmp := t.TempDir()

	statusDir := filepath.Join(tmp, "docs", "milestones")
	if err := os.MkdirAll(statusDir, 0755); err != nil {
		t.Fatal(err)
	}
	staleStatus := "# Milestone status\n\n" +
		milestonedoc.Begin + "\n\nSTALE BLOCK\n\n" +
		milestonedoc.End + "\n"
	statusFile := filepath.Join(statusDir, "STATUS.md")
	if err := os.WriteFile(statusFile, []byte(staleStatus), 0644); err != nil {
		t.Fatal(err)
	}

	guideFile := filepath.Join(tmp, "docs", "guide.md")
	staleGuide := `# Stale Guide

> Orientation: Guide orientation.

Release pinned to v0.10.0.

## Read next

- [STATUS.md](milestones/STATUS.md)
`
	if err := os.WriteFile(guideFile, []byte(staleGuide), 0644); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(tmp, "VERSION"), []byte("0.54.0\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Check should fail with exit code 1
	var stdout, stderr bytes.Buffer
	code := runDocFreshness(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--check",
	})
	if code != 1 {
		t.Fatalf("expected check exit 1 on stale workspace, got %d; stdout=%s", code, stdout.String())
	}
	if !strings.Contains(stdout.String(), "STALE") {
		t.Errorf("expected STALE verdict in check output:\n%s", stdout.String())
	}

	// 2. Refresh should succeed with exit code 0
	stdout.Reset()
	stderr.Reset()
	code = runDocFreshness(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--refresh",
	})
	if code != 0 {
		t.Fatalf("expected refresh exit 0, got %d; stderr=%s, stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "VERDICT: REFRESHED") && !strings.Contains(stdout.String(), "PASS") {
		t.Errorf("expected REFRESHED/PASS in output:\n%s", stdout.String())
	}
	t.Logf("stdout from refresh:\n%s", stdout.String())

	// 3. Verify files were refreshed on disk
	sb, _ := os.ReadFile(statusFile)
	if !strings.Contains(string(sb), milestonedoc.Block()) {
		t.Errorf("STATUS.md was not refreshed to contain milestonedoc.Block()")
	}

	gb, _ := os.ReadFile(guideFile)
	if !strings.Contains(string(gb), "v0.54.0") {
		t.Errorf("guide.md was not refreshed to contain target version v0.54.0:\n%s", string(gb))
	}

	// 4. JSON output verification
	stdout.Reset()
	stderr.Reset()
	code = runDocFreshness(&stdout, &stderr, []string{
		"--workspace", tmp,
		"--generated",
		"--json",
		"--check",
	})
	if code != 0 {
		t.Fatalf("expected json check exit 0, got %d; stdout=%s", code, stdout.String())
	}
	var payload docFreshnessPayload
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("invalid json payload: %v\n%s", err, stdout.String())
	}
	if payload.Schema != "fak.doc-freshness.v1" {
		t.Errorf("schema = %q, want fak.doc-freshness.v1", payload.Schema)
	}
	if !payload.Fresh {
		t.Errorf("expected Fresh=true in json payload")
	}
}

func TestDocFreshnessDefaultTargetVersion(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "VERSION"), []byte("1.2.3\n"), 0644); err != nil {
		t.Fatal(err)
	}
	ver := docfreshrsi.DefaultTargetVersion(tmp)
	if ver != "v1.2.3" {
		t.Errorf("got %q, want v1.2.3", ver)
	}
}
