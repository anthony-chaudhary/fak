// Package repotest locates the package directory and repository root for tests
// that read checked-in files.
//
// It never uses runtime.Caller: under -trimpath the recorded source path is
// module-qualified (or "./file_test.go" for a single-file build), not a
// directory on disk. go test always starts the test binary in the package
// directory, so the working directory captured at init (before any test can
// chdir) is the reliable anchor.
package repotest

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var startDir, startErr = os.Getwd()

// PackageDir returns the directory of the package under test.
func PackageDir(tb testing.TB) string {
	tb.Helper()
	if startErr != nil {
		tb.Fatalf("repotest: working directory: %v", startErr)
	}
	return startDir
}

// Root returns the nearest ancestor of the package directory that holds a
// go.mod file.
func Root(tb testing.TB) string {
	tb.Helper()
	root, err := FindRoot(PackageDir(tb))
	if err != nil {
		tb.Fatalf("repotest: %v", err)
	}
	return root
}

// Path joins slash-separated elements onto the repository root.
func Path(tb testing.TB, elem ...string) string {
	tb.Helper()
	parts := make([]string, 0, len(elem)+1)
	parts = append(parts, Root(tb))
	for _, e := range elem {
		parts = append(parts, filepath.FromSlash(e))
	}
	return filepath.Join(parts...)
}

// FindRoot walks up from dir to the first directory holding a go.mod file.
func FindRoot(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for d := abs; ; {
		if fi, err := os.Stat(filepath.Join(d, "go.mod")); err == nil && !fi.IsDir() {
			return d, nil
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", fmt.Errorf("no go.mod above %s", abs)
		}
		d = parent
	}
}
