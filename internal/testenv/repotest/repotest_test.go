package repotest

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRootHoldsModuleFile(t *testing.T) {
	root := Root(t)
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("Root() = %q has no go.mod: %v", root, err)
	}
	if got, want := PackageDir(t), filepath.Join(root, "internal", "testenv", "repotest"); got != want {
		t.Fatalf("PackageDir() = %q, want %q", got, want)
	}
	if got, want := Path(t, "internal/testenv/repotest/repotest.go"), filepath.Join(root, "internal", "testenv", "repotest", "repotest.go"); got != want {
		t.Fatalf("Path() = %q, want %q", got, want)
	}
}

func TestPackageDirSurvivesChdir(t *testing.T) {
	want := PackageDir(t)
	t.Chdir(t.TempDir())
	if got := PackageDir(t); got != want {
		t.Fatalf("PackageDir() after chdir = %q, want %q", got, want)
	}
}

func TestFindRootWalksUpAndStops(t *testing.T) {
	top := t.TempDir()
	deep := filepath.Join(top, "a", "b")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(top, "go.mod"), []byte("module x\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := FindRoot(deep)
	if err != nil || got != top {
		t.Fatalf("FindRoot(%q) = %q, %v; want %q", deep, got, err, top)
	}
	if _, err := FindRoot(filepath.VolumeName(top) + string(filepath.Separator)); err == nil {
		t.Fatal("FindRoot at filesystem root returned no error")
	}
}
