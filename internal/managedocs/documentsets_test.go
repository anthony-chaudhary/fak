package managedocs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditDocumentSetsRejectsOversizedMonolith(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "docs", "long.md")
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(strings.Repeat("line\n", PageLines+1)), 0644); err != nil {
		t.Fatal(err)
	}
	err := AuditDocumentSets(root, "docs")
	if err == nil || !strings.Contains(err.Error(), "docs/long.md") {
		t.Fatalf("got %v", err)
	}
}
func TestAuditDocumentSetsAcceptsMarkedIndex(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "docs", "long.md")
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	body := DocumentSetMarker + "\n# Index\n" + strings.Repeat("- [page](pages/page.md)\n", PageLines)
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	if err := AuditDocumentSets(root, "docs"); err != nil {
		t.Fatal(err)
	}
}
