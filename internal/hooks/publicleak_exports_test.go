package hooks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScanRangePublicLeakResolvesSidecarFromSelectedRevision(t *testing.T) {
	root := t.TempDir()
	gitFixture(t, root, "init", "-q", "-b", "main")
	gitFixture(t, root, "config", "user.email", "fixture@example.com")
	gitFixture(t, root, "config", "user.name", "Fixture")

	needle := "10.11.12.13"
	sidecar := filepath.Join(root, filepath.FromSlash(privateNeedlesRel))
	if err := os.MkdirAll(filepath.Dir(sidecar), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sidecar, []byte(`{"export_audit_needles":["`+needle+`"]}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("tools/_registry/\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", ".gitignore")
	gitFixture(t, root, "add", "-f", privateNeedlesRel)
	gitFixture(t, root, "commit", "-qm", "seed")
	base := gitFixture(t, root, "rev-parse", "HEAD")

	if err := os.WriteFile(filepath.Join(root, "node.json"), []byte(`{"ip":"`+needle+`"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, root, "add", "node.json")
	gitFixture(t, root, "commit", "-qm", "leak")

	findings, err := ScanRangePublicLeak(root, base+"..HEAD")
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 || findings[0].File != "node.json" || !strings.Contains(findings[0].Detail, needle) {
		t.Fatalf("range findings = %+v, want selected-revision sidecar needle in node.json", findings)
	}
}
