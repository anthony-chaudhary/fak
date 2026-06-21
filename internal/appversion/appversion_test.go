package appversion

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFromDirWalksUpToVersionMarker(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "fak", "internal", "bench")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("1.2.3\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, ok := FromDir(nested)
	if !ok {
		t.Fatal("FromDir did not find VERSION")
	}
	if got != "1.2.3" {
		t.Fatalf("version=%q, want 1.2.3", got)
	}
}

func TestCurrentPrefersEnvironment(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "9.9.9-test")
	if got := Current(); got != "9.9.9-test" {
		t.Fatalf("Current()=%q, want environment override", got)
	}
}
