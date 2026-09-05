package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/managedocs"
)

func TestRunManageDocs(t *testing.T) {
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

	// Verify default audit does not enforce 150-line document-set budget on the repo
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", repoRoot()})
	if rc != 0 {
		t.Fatalf("expected rc 0 for repo default audit, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected retained occurrences audit message, got: %s", stdout.String())
	}
	if strings.Contains(stderr.String(), "document sets audit failed") {
		t.Errorf("default audit should not enforce document sets budget, stderr: %s", stderr.String())
	}

	// Verify --docs-dir without --document-sets/--budget does not enforce 150-line budget
	stdout.Reset()
	stderr.Reset()
	rc = runManageDocs(&stdout, &stderr, []string{"--workspace", repoRoot(), "--docs-dir", "docs"})
	if rc != 0 {
		t.Fatalf("expected rc 0 for --docs-dir without --document-sets, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected retained occurrences audit message, got: %s", stdout.String())
	}
}

func TestRunManageDocs_CheckRetained(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runManageDocs(&stdout, &stderr, []string{"--workspace", repoRoot(), "--check-retained"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "retained occurrences audit passed") {
		t.Errorf("expected audit message, got: %s", stdout.String())
	}
}
