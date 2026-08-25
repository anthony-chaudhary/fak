package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyInventoryJSONMapsLocalCheckout(t *testing.T) {
	root := t.TempDir()
	writeStudyInventoryFixture(t, root, "README.md", "# demo\n")
	writeStudyInventoryFixture(t, root, "cmd/app/main.go", "package main\nfunc main() {}\n")

	var stdout, stderr bytes.Buffer
	code := runStudyInventory(&stdout, &stderr, []string{
		"--root", root,
		"--repository", "owner/repo",
		"--revision", "abc123",
		"--observed-at", "2026-08-25T00:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report studymonitor.InventoryMap
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode output: %v\n%s", err, stdout.String())
	}
	if report.Schema != studymonitor.InventoryMapSchema || report.Repository != "owner/repo" || report.IndexedRevision != "abc123" {
		t.Fatalf("report identity = %+v", report)
	}
	if report.Totals.RuntimeFiles != 1 {
		t.Fatalf("totals = %+v, want one runtime file", report.Totals)
	}
	if !strings.Contains(report.CompletenessNote, "still requires non-tree study artifacts") {
		t.Fatalf("completeness note = %q", report.CompletenessNote)
	}
}

func TestStudyInventoryWritesMarkdown(t *testing.T) {
	root := t.TempDir()
	writeStudyInventoryFixture(t, root, "README.md", "# demo\n")
	out := filepath.Join(t.TempDir(), "inventory.md")

	var stdout, stderr bytes.Buffer
	code := runStudyInventory(&stdout, &stderr, []string{
		"--root", root,
		"--repository", "owner/repo",
		"--revision", "abc123",
		"--observed-at", "2026-08-25T00:00:00Z",
		"--out", out,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "wrote study inventory map") {
		t.Fatalf("stdout = %q, want write confirmation", stdout.String())
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "# Study inventory: owner/repo") || !strings.Contains(string(data), "## Completeness Critic") {
		t.Fatalf("markdown output missing expected sections:\n%s", string(data))
	}
}

func writeStudyInventoryFixture(t *testing.T, root, rel, text string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}
