package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/managedocs"
)

func TestManageDocs(t *testing.T) {
	temp := t.TempDir()
	docDir := filepath.Join(temp, "docs")
	if err := os.MkdirAll(docDir, 0755); err != nil {
		t.Fatal(err)
	}
	shortDoc := filepath.Join(docDir, "short.md")
	if err := os.WriteFile(shortDoc, []byte("# Short Document\nBounded content\n"), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	rc := runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs", "--document-sets"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "document sets audit in") {
		t.Errorf("expected audit message, got: %s", stdout.String())
	}

	// Verify oversized document set rejection
	longDoc := filepath.Join(docDir, "long.md")
	if err := os.WriteFile(longDoc, []byte(strings.Repeat("line\n", managedocs.PageLines+1)), 0644); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs", "--document-sets"})
	if rc != 1 {
		t.Fatalf("expected rc 1 for oversized doc, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "document sets audit failed") {
		t.Errorf("expected audit failed in stderr, got: %s", stderr.String())
	}

	// Verify marked index accepts long document
	markedBody := managedocs.DocumentSetMarker + "\n# Index\n" + strings.Repeat("- [page](pages/page.md)\n", managedocs.PageLines)
	if err := os.WriteFile(longDoc, []byte(markedBody), 0644); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs", "--document-sets"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for marked doc, got %d, stderr: %s", rc, stderr.String())
	}

	// Verify --budget alias triggers document sets audit
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs", "--budget"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for --budget alias, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "document sets audit in") {
		t.Errorf("expected audit message for --budget, got: %s", stdout.String())
	}

	// Verify default audit does not enforce document sets budget and runs retained occurrences audit
	cleanWorkspace := createCleanManagedocsWorkspace(t)
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace})
	if rc != 0 {
		t.Fatalf("expected rc 0 for default audit on clean workspace, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected retained occurrences audit to pass, stdout: %s, stderr: %s", stdout.String(), stderr.String())
	}
	if strings.Contains(stdout.String(), "document sets audit") || strings.Contains(stderr.String(), "document sets audit") {
		t.Errorf("default audit should not enforce document sets budget, stdout: %s, stderr: %s", stdout.String(), stderr.String())
	}

	// Verify default audit fails with injected violation
	violatingDoc := filepath.Join(cleanWorkspace, "docs", "violating.md")
	if err := os.WriteFile(violatingDoc, []byte("unclassified fak guard occurrence\n"), 0644); err != nil {
		t.Fatal(err)
	}
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace})
	if rc != 1 {
		t.Fatalf("expected rc 1 for default audit with injected violation, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "fak managedocs retained audit failed") {
		t.Errorf("expected retained audit failed in stderr, got: %s", stderr.String())
	}
	if err := os.Remove(violatingDoc); err != nil {
		t.Fatal(err)
	}

	// Verify --docs-dir without --document-sets/--budget infers document sets audit
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for --docs-dir with inferred --document-sets, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "document sets audit in") {
		t.Errorf("expected document sets audit message for inferred flag, got: %s", stdout.String())
	}

	// Verify inferred document-sets audit enforces budget on oversized document
	unmarkedLongDoc := filepath.Join(docDir, "unmarked.md")
	if err := os.WriteFile(unmarkedLongDoc, []byte(strings.Repeat("line\n", managedocs.PageLines+1)), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(unmarkedLongDoc)
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "docs"})
	if rc != 1 {
		t.Fatalf("expected rc 1 for oversized doc with inferred --document-sets, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "document sets audit failed") {
		t.Errorf("expected audit failed in stderr, got: %s", stderr.String())
	}
}

func TestManageDocs_CheckRetained(t *testing.T) {
	cleanWorkspace := createCleanManagedocsWorkspace(t)
	var stdout, stderr bytes.Buffer
	rc := runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace, "--check-retained"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for clean workspace with --check-retained, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected retained occurrences audit to pass, stdout: %s, stderr: %s", stdout.String(), stderr.String())
	}

	// Verify --check-retained fails with injected violation
	violatingDoc := filepath.Join(cleanWorkspace, "docs", "violating.md")
	if err := os.WriteFile(violatingDoc, []byte("unclassified fak guard occurrence\n"), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(violatingDoc)
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace, "--check-retained"})
	if rc != 1 {
		t.Fatalf("expected rc 1 for --check-retained with injected violation, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "fak managedocs retained audit failed") {
		t.Errorf("expected retained audit failed in stderr, got: %s", stderr.String())
	}
}

func createCleanManagedocsWorkspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "examples"), 0755); err != nil {
		t.Fatal(err)
	}
	byPath := make(map[string][]string)
	for _, occ := range managedocs.RetainedOccurrences {
		count := occ.Count
		if count <= 0 {
			count = 1
		}
		for i := 0; i < count; i++ {
			byPath[occ.Path] = append(byPath[occ.Path], occ.Line)
		}
	}
	if _, ok := byPath["README.md"]; !ok {
		byPath["README.md"] = []string{"# Readme"}
	}
	for relPath, lines := range byPath {
		fullPath := filepath.Join(root, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			t.Fatal(err)
		}
		content := strings.Join(lines, "\n") + "\n"
		if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestManageDocs_ExplicitNonexistentDirFails(t *testing.T) {
	temp := t.TempDir()
	var stdout, stderr bytes.Buffer
	rc := runManageDocs(&stdout, &stderr, []string{"--workspace", temp, "--docs-dir", "nonexistent-docs-dir-xyz-123"})
	if rc == 0 {
		t.Fatalf("expected non-zero rc for nonexistent --docs-dir, got %d, stdout: %s", rc, stdout.String())
	}
	if !strings.Contains(stderr.String(), "nonexistent-docs-dir-xyz-123") {
		t.Errorf("expected stderr to contain nonexistent docs dir name, got: %s", stderr.String())
	}
}

func TestManageDocs_DocsDirInfersDocumentSets(t *testing.T) {
	cleanWorkspace := createCleanManagedocsWorkspace(t)
	// Inject a retained occurrence violation in the workspace.
	// If runManageDocs falls back to retained occurrences audit, it will fail.
	violatingDoc := filepath.Join(cleanWorkspace, "docs", "violating.md")
	if err := os.WriteFile(violatingDoc, []byte("unclassified fak guard occurrence\n"), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(violatingDoc)

	// Create an explicit specs doc directory with valid bounded docs.
	specsRel := filepath.Join("docs", "specs")
	specsDir := filepath.Join(cleanWorkspace, specsRel)
	if err := os.MkdirAll(specsDir, 0755); err != nil {
		t.Fatal(err)
	}
	specFile := filepath.Join(specsDir, "spec.md")
	if err := os.WriteFile(specFile, []byte("# Spec\nBounded doc\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// 1. Providing --docs-dir without --document-sets executes document sets audit on that dir,
	// and does not silently fall back to AuditRetainedOccurrences (which would fail on violating.md).
	var stdout, stderr bytes.Buffer
	rc := runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace, "--docs-dir", specsRel})
	if rc != 0 {
		t.Fatalf("expected rc 0 for --docs-dir with inferred document sets, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "document sets audit in") {
		t.Errorf("expected document sets audit message, got: %s", stdout.String())
	}
	if strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected no retained occurrences audit execution, got: %s", stdout.String())
	}

	// 2. Providing --docs-dir on directory with an oversized doc executes audit on that directory and fails.
	oversizedDoc := filepath.Join(specsDir, "oversized.md")
	if err := os.WriteFile(oversizedDoc, []byte(strings.Repeat("line\n", managedocs.PageLines+1)), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(oversizedDoc)

	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace, "--docs-dir", specsRel})
	if rc != 1 {
		t.Fatalf("expected rc 1 for oversized doc in explicit --docs-dir, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "document sets audit failed") {
		t.Errorf("expected document sets audit failed in stderr, got: %s", stderr.String())
	}
	if err := os.Remove(oversizedDoc); err != nil {
		t.Fatal(err)
	}

	// 3. When --check-retained is explicitly specified alongside --docs-dir, document-sets is not inferred,
	// and retained occurrences audit runs (and fails due to violating.md).
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", cleanWorkspace, "--docs-dir", specsRel, "--check-retained"})
	if rc != 1 {
		t.Fatalf("expected rc 1 when --check-retained is explicit with injected violation, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "fak managedocs retained audit failed") {
		t.Errorf("expected retained audit failed in stderr, got: %s", stderr.String())
	}
	if strings.Contains(stdout.String(), "document sets audit in") {
		t.Errorf("did not expect document sets audit when only --check-retained requested, stdout: %s", stdout.String())
	}
}
