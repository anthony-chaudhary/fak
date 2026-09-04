package dependencyquarantine

import (
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	return findRepoRoot(t)
}

func repoRootBenchmark(b *testing.B) string {
	b.Helper()
	return findRepoRoot(b)
}

type failer interface {
	Helper()
	Fatal(args ...any)
}

func findRepoRoot(f failer) string {
	f.Helper()
	dir, err := os.Getwd()
	if err != nil {
		f.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			f.Fatal("repository go.mod not found")
		}
		dir = parent
	}
}

func fixtureBenchmark(b *testing.B, mod, sum string) string {
	b.Helper()
	root := b.TempDir()
	writeHelper(b, filepath.Join(root, "go.mod"), mod)
	writeHelper(b, filepath.Join(root, "go.sum"), sum)
	return root
}

func fixture(t *testing.T, mod, sum string) string {
	t.Helper()
	root := t.TempDir()
	writeHelper(t, filepath.Join(root, "go.mod"), mod)
	writeHelper(t, filepath.Join(root, "go.sum"), sum)
	return root
}

func write(t *testing.T, path, body string) {
	t.Helper()
	writeHelper(t, path, body)
}

func writeHelper(f failer, path, body string) {
	f.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		f.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		f.Fatal(err)
	}
}
func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
func contains(vs []Violation, s string) bool {
	for _, v := range vs {
		if strings.Contains(v.Error(), s) {
			return true
		}
	}
	return false
}
func testCommand(t *testing.T, dir string) *exec.Cmd {
	t.Helper()
	c := exec.Command("go", "test", "./...")
	c.Dir = dir
	c.Env = append(os.Environ(), "GOWORK=off")
	return c
}
