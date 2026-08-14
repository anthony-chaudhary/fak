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
