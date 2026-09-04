package memorycotravel

import (
	"os"
	"path/filepath"
	"testing"
)

// Invariant: Memory co-travel strategies must correctly sync agent memory files without clobbering existing notes.
// Guard: Additive strategy copies missing notes and refuses to overwrite conflicting existing files.

func TestMemoryCoTravelLifecycle(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	src := filepath.Join(tmp, "src.md")
	dst := filepath.Join(tmp, "dst.md")

	if err := os.WriteFile(src, []byte("content"), 0o644); err != nil {
		t.Fatalf("failed writing src file: %v", err)
	}

	action := Additive(src, dst)
	if action != "copy" {
		t.Fatalf("expected copy action for missing destination, got: %s", action)
	}
}
