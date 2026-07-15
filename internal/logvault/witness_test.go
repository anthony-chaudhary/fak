package logvault

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWitnessedFilesRehashesManifestedMirror(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src")
	if err := os.MkdirAll(filepath.Join(src, "guard-audit"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "guard-audit", "one.jsonl"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := &Vault{Dir: filepath.Join(t.TempDir(), "vault"), Sources: []Source{{ID: "dispatch-runs", Root: src}}}
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	files, err := v.WitnessedFiles("dispatch-runs")
	if err != nil {
		t.Fatal(err)
	}
	if files["guard-audit/one.jsonl"] == "" {
		t.Fatalf("witness=%v", files)
	}
	if err := os.WriteFile(v.mirrorPath("dispatch-runs", "guard-audit/one.jsonl"), []byte("tampered\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := v.WitnessedFiles("dispatch-runs"); err == nil {
		t.Fatal("tampered mirror verified")
	}
}
