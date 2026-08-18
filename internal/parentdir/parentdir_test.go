package parentdir

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEnsure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "file")
	if err := Ensure(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(filepath.Dir(path)); err != nil || !info.IsDir() {
		t.Fatalf("parent = %v, %v", info, err)
	}
}
