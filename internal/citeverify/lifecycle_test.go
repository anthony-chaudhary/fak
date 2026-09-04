package citeverify

import (
	"os"
	"path/filepath"
	"testing"
)

// Invariant: Citation verification must reliably corroborate evidence tokens against ground truth files without leakage.
// Guard: Verify rejects paths outside workspace root and uncorroborated symbols.

func TestCiteVerifyLifecycle(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	sourcePath := filepath.Join(root, "pkg", "sample.go")
	if err := os.MkdirAll(filepath.Dir(sourcePath), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "package pkg\n\n// AlphaFunction performs lifecycle operations.\nfunc AlphaFunction() string {\n\treturn \"alpha\"\n}\n"
	if err := os.WriteFile(sourcePath, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	claim := "`AlphaFunction` performs lifecycle operations"
	evidence := []string{"pkg/sample.go:4"}

	status := Verify(claim, evidence, root)
	if status != Supports {
		t.Fatalf("expected Supports status, got %s", status)
	}
}
