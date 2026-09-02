package walkfiles

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
)

// collect is a visit func that gathers every visited path, so the tests can
// assert on the visited set.
func collect(seen *[]string) func(string, fs.DirEntry) error {
	return func(p string, _ fs.DirEntry) error {
		*seen = append(*seen, p)
		return nil
	}
}

func TestFilesVisitsRegularFilesOnly(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "empty_dir"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{"a.txt", "sub/b.txt", "sub/deep/c.txt"} {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var seen []string
	if err := Files(root, collect(&seen)); err != nil {
		t.Fatalf("Files: %v", err)
	}
	var got []string
	for _, p := range seen {
		rel, err := filepath.Rel(root, p)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, filepath.ToSlash(rel))
	}
	sort.Strings(got)
	want := []string{"a.txt", "sub/b.txt", "sub/deep/c.txt"}
	if len(got) != len(want) {
		t.Fatalf("visited %v, want exactly %v (directories must never be visited)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("visited %v, want %v", got, want)
		}
	}
}

func TestFilesSwallowsMissingRoot(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "not-there")
	var seen []string
	if err := Files(missing, collect(&seen)); err != nil {
		t.Fatalf("a missing root is a swallowed walk-step error (every cloning site did), got %v", err)
	}
	if len(seen) != 0 {
		t.Fatalf("missing root must visit nothing, got %v", seen)
	}
}

func TestFilesVisitErrorAbortsWalk(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "f.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	sentinel := errors.New("stop walk")
	calls := 0
	err := Files(root, func(string, fs.DirEntry) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("visit error not propagated: got %v", err)
	}
	if calls != 1 {
		t.Fatalf("visit called %d times, want 1 (the walk must abort)", calls)
	}
}

func TestFilesSwallowsWalkStepErrors(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("permission-bit walk-step errors are POSIX-only")
	}
	if os.Geteuid() == 0 {
		t.Skip("root reads through chmod 0o000, so no walk-step error is producible")
	}
	root := t.TempDir()
	sealed := filepath.Join(root, "sealed")
	if err := os.MkdirAll(sealed, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sealed, "hidden.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "visible.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(sealed, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(sealed, 0o755) })

	var seen []string
	if err := Files(root, collect(&seen)); err != nil {
		t.Fatalf("walk-step error must be swallowed, got %v", err)
	}
	if len(seen) != 1 || filepath.Base(seen[0]) != "visible.txt" {
		t.Fatalf("walk did not continue past the unreadable dir: visited %v, want [visible.txt]", seen)
	}
}
