package managedocs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCanonicalManageFrontDoor(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	if err := Audit(root); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentScannerConfigurationAcceptsLongGeneratedLine(t *testing.T) {
	longLine := strings.Repeat("x", 128*1024)
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "examples"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("ok\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "long.txt"), []byte(longLine+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// RetainedOccurrences belongs to the real repository, so reaching stale-entry
	// validation proves the long line was scanned without Scanner's token error.
	err := Audit(root)
	if err == nil || strings.Contains(err.Error(), "token too long") {
		t.Fatalf("Audit() error = %v, want post-scan stale-entry findings", err)
	}
}

func TestAuditRejectsUnclassifiedGuardExample(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "examples"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("run `fak guard claude`\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := Audit(root)
	if err == nil || !strings.Contains(err.Error(), "README.md:1: unclassified") {
		t.Fatalf("Audit() error = %v, want unclassified occurrence", err)
	}
}

func TestCurrentGuardPosturesAreClassified(t *testing.T) {
	want := map[string]string{
		"README.md":             "fak guard -- codex",
		"docs/cli-reference.md": "also append automatically",
		"docs/generated/disambiguation-index.json": "optionally wraps it with fak guard",
		"docs/integrations/session-new.md":         "always starts behind `fak guard`",
		"docs/notes/graft-study-2026-08-18.md":     "`fak guard` registers capability",
		"docs/response-profiles.md":                "external harness through `fak guard`",
	}
	for path, fragment := range want {
		found := false
		for _, retained := range RetainedOccurrences {
			if retained.Path == path && strings.Contains(retained.Line, fragment) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%s guard posture containing %q is not classified", path, fragment)
		}
	}
}
