package workerworktree

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestSafeRemoveAllClearsUnwritableDirectoryTree pins the removal contract the
// topology-candidate sweep relies on: a tree whose directories were left
// non-writable and whose files are read-only must still be removed. On POSIX the
// directory must stay traversable, so safeRemoveAll may add the owner rwx bits but
// must never chmod a directory down to file-style 0o666 (which strips +x and makes
// every child unlink fail with EACCES -- the WSL commit-gate red).
func TestSafeRemoveAllClearsUnwritableDirectoryTree(t *testing.T) {
	root := t.TempDir()
	tree := filepath.Join(root, "cand", "sub")
	if err := os.MkdirAll(tree, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tree, "f"), []byte("x"), 0o444); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		// Strip the directory's write bit so a naive RemoveAll fails, and its
		// execute bit so a wrong-mode chmod cannot recover traversal.
		if err := os.Chmod(tree, 0o500); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(filepath.Join(tree, "f"), 0o400); err != nil {
			t.Fatal(err)
		}
	}
	if err := safeRemoveAll(filepath.Join(root, "cand")); err != nil {
		t.Fatalf("safeRemoveAll should clear an unwritable tree: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "cand")); !os.IsNotExist(err) {
		t.Fatalf("candidate tree survived safeRemoveAll: %v", err)
	}
}
