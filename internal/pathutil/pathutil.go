// Package pathutil holds small, dependency-free path helpers shared across the fak
// commands — chiefly normalizing user-supplied path flags before they reach the
// filesystem.
package pathutil

import (
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ExpandTilde turns a leading "~" or "~/" (or "~\" on Windows) into the user's home
// directory. The shell normally does this, but PowerShell and most quoting pass "~"
// through literally, and neither Go's flag parsing nor os.Open expands it — so without
// this a flag like `--gguf ~/Downloads/model.gguf` is read as a literal "~" directory
// and fails with "the system cannot find the path specified".
//
// Only a leading "~" segment is expanded; a bare or mid-path "~" (e.g. "a/~/b", or a
// real filename that merely starts with a tilde like "~scratch") is left untouched.
// An empty string returns empty, so it is safe to call unconditionally on optional
// flags.
func ExpandTilde(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") || strings.HasPrefix(p, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}

// EnsureDir creates p and any missing parents with the repo's standard directory mode.
func EnsureDir(p string) error {
	return os.MkdirAll(p, 0o755)
}

// ReadFileOrStdin reads the whole named file, treating the conventional "-" as
// "read everything from stdin" so CLI flags that take a file path also compose
// with pipes.
func ReadFileOrStdin(path string) ([]byte, error) {
	if path == "-" {
		return io.ReadAll(os.Stdin)
	}
	return os.ReadFile(path)
}
