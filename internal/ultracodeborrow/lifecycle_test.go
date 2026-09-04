package ultracodeborrow

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Invariant: Ultracode borrow checks must enforce public hygiene rules and validate workflow companion notes.
// Guard: Parse refuses invalid json payloads or private path disclosures.

func TestUltracodeBorrowLifecycle(t *testing.T) {
	t.Parallel()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	raw, err := os.ReadFile(filepath.Join(root, "docs", "notes", "CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json"))
	if err != nil {
		t.Fatalf("failed reading study json: %v", err)
	}

	artifact, err := Parse(raw)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if artifact.Schema == "" {
		t.Fatal("expected non-empty schema")
	}
}
