package pathutil

import (
	"os"
	"path/filepath"
	"testing"
)

// TestExpandTilde pins ~ expansion: a leading ~ becomes $HOME, everything else is
// untouched. This is what lets a flag like "--gguf ~/Downloads/model.gguf" find the
// file under PowerShell and other shells that pass ~ through literally.
// TestReadFileOrStdin pins the ordinary-file path of the shared reader: a named
// file is read in full, and a missing file surfaces the os.ReadFile error. The
// "-"-means-stdin branch is exercised by the CLIs that pipe into it; unit-testing
// it would mean swapping the os.Stdin global, which isn't worth the leakage risk.
func TestReadFileOrStdin(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "payload.json")
	want := []byte(`{"ok":true}`)
	if err := os.WriteFile(p, want, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	got, err := ReadFileOrStdin(p)
	if err != nil {
		t.Fatalf("ReadFileOrStdin(%q): %v", p, err)
	}
	if string(got) != string(want) {
		t.Errorf("ReadFileOrStdin(%q) = %q, want %q", p, got, want)
	}
	if _, err := ReadFileOrStdin(filepath.Join(dir, "missing.json")); err == nil {
		t.Error("ReadFileOrStdin on a missing file: want error, got nil")
	}
}

func TestExpandTilde(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home dir: %v", err)
	}
	cases := []struct{ in, want string }{
		{"~/Downloads/model.gguf", filepath.Join(home, "Downloads", "model.gguf")},
		{"~", home},
		{"", ""},
		{"/abs/path/model.gguf", "/abs/path/model.gguf"},
		{"relative/model.gguf", "relative/model.gguf"},
		{"a/~/b", "a/~/b"},           // ~ only expands as a leading segment, never mid-path
		{"~scratch/x", "~scratch/x"}, // a real name starting with ~ is not a home ref
	}
	for _, tc := range cases {
		if got := ExpandTilde(tc.in); got != tc.want {
			t.Errorf("ExpandTilde(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
