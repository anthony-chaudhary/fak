package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// tempLeftovers returns the names of any sibling temporary files the writer
// failed to clean up in dir.
func tempLeftovers(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir %s: %v", dir, err)
	}
	var leftovers []string
	for _, e := range entries {
		if strings.Contains(e.Name(), ".tmp-") {
			leftovers = append(leftovers, e.Name())
		}
	}
	return leftovers
}

func TestWriteFileAtomic_CreatesParentDirsAndContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "etc", "systemd", "system", "fak.service")

	if err := WriteFileAtomic(path, []byte("unit body\n"), 0644); err != nil {
		t.Fatalf("WriteFileAtomic failed: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read written file: %v", err)
	}
	if string(data) != "unit body\n" {
		t.Errorf("content = %q, want %q", string(data), "unit body\n")
	}
	if leftovers := tempLeftovers(t, filepath.Dir(path)); len(leftovers) != 0 {
		t.Errorf("expected no temporary files left behind, got %v", leftovers)
	}
}

func TestWriteFileAtomic_HonorsPermission(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX file modes are not enforced on Windows")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "secret.conf")

	if err := WriteFileAtomic(path, []byte("token\n"), 0600); err != nil {
		t.Fatalf("WriteFileAtomic failed: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat written file: %v", err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Errorf("mode = %o, want 600", got)
	}
}

func TestWriteFileAtomic_ReplacesExistingFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "amdgpu.conf")

	if err := os.WriteFile(path, []byte("old contents\n"), 0644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	if err := WriteFileAtomic(path, []byte("new contents\n"), 0644); err != nil {
		t.Fatalf("WriteFileAtomic failed: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read replaced file: %v", err)
	}
	if string(data) != "new contents\n" {
		t.Errorf("content = %q, want %q", string(data), "new contents\n")
	}
	if leftovers := tempLeftovers(t, dir); len(leftovers) != 0 {
		t.Errorf("expected no temporary files left behind, got %v", leftovers)
	}
}

// TestWriteFileAtomic_FailedWriteLeavesDestinationUntouched proves the atomic
// contract: when the final replace cannot happen, the destination keeps its
// prior state and no partial file is exposed. A non-empty directory occupying
// the destination path makes the rename fail deterministically on every
// platform, after the temporary file has already been written and synced.
func TestWriteFileAtomic_FailedWriteLeavesDestinationUntouched(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "occupied")

	if err := os.MkdirAll(filepath.Join(path, "child"), 0755); err != nil {
		t.Fatalf("seed destination directory: %v", err)
	}
	const marker = "prior state\n"
	if err := os.WriteFile(filepath.Join(path, "child", "keep.txt"), []byte(marker), 0644); err != nil {
		t.Fatalf("seed destination contents: %v", err)
	}

	if err := WriteFileAtomic(path, []byte("replacement\n"), 0644); err == nil {
		t.Fatal("expected WriteFileAtomic to fail when the destination cannot be replaced")
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("destination must survive a failed write: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("destination prior state changed: expected directory, got mode %v", info.Mode())
	}
	data, err := os.ReadFile(filepath.Join(path, "child", "keep.txt"))
	if err != nil {
		t.Fatalf("destination contents must survive a failed write: %v", err)
	}
	if string(data) != marker {
		t.Errorf("destination contents = %q, want %q", string(data), marker)
	}
	if leftovers := tempLeftovers(t, dir); len(leftovers) != 0 {
		t.Errorf("failed write must clean up its temporary file, got %v", leftovers)
	}
}

// TestWriteFileAtomicWithRename_InjectedFailureLeavesPriorBytes proves the
// injectable-rename seam: a caller that owns the replacement step can fail it,
// and the prior destination bytes and the temporary-file cleanup both hold.
func TestWriteFileAtomicWithRename_InjectedFailureLeavesPriorBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "contract.json")
	const marker = "prior revision\n"
	if err := os.WriteFile(path, []byte(marker), 0644); err != nil {
		t.Fatalf("seed file: %v", err)
	}

	replaceFailure := errors.New("injected atomic replacement failure")
	err := WriteFileAtomicWithRename(path, []byte("new revision\n"), 0644, func(_, _ string) error {
		return replaceFailure
	})
	if !errors.Is(err, replaceFailure) {
		t.Fatalf("error = %v, want injected sentinel", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("prior file must survive a failed replacement: %v", err)
	}
	if string(data) != marker {
		t.Errorf("prior contents = %q, want %q", string(data), marker)
	}
	if leftovers := tempLeftovers(t, dir); len(leftovers) != 0 {
		t.Errorf("failed write must clean up its temporary file, got %v", leftovers)
	}
}

// TestWriteFileAtomic_UnwritableDirLeavesPriorFileIntact proves the same
// contract at the temporary-file stage: when the destination directory refuses
// creation of the temporary file, the pre-existing destination is untouched.
// Directory permissions are only enforced on POSIX hosts.
func TestWriteFileAtomic_UnwritableDirLeavesPriorFileIntact(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("directory write permissions are not enforced on Windows")
	}
	if os.Geteuid() == 0 {
		t.Skip("root bypasses directory write permissions")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "fak.conf")
	const marker = "prior contents\n"
	if err := os.WriteFile(path, []byte(marker), 0644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	if err := os.Chmod(dir, 0555); err != nil {
		t.Fatalf("chmod dir read-only: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(dir, 0755) })

	if err := WriteFileAtomic(path, []byte("replacement\n"), 0644); err == nil {
		t.Fatal("expected WriteFileAtomic to fail in a read-only directory")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("prior file must survive a failed write: %v", err)
	}
	if string(data) != marker {
		t.Errorf("prior file contents = %q, want %q", string(data), marker)
	}
}
