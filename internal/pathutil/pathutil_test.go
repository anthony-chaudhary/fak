package pathutil

import (
	"os"
	"path/filepath"
	"testing"
)

// TestExpandTilde pins ~ expansion: a leading ~ becomes $HOME, everything else is
// untouched. This is what lets a flag like "--gguf ~/Downloads/model.gguf" find the
// file under PowerShell and other shells that pass ~ through literally.
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
