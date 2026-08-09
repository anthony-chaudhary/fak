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
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repository go.mod not found")
		}
		dir = parent
	}
}
func fixture(t *testing.T, mod, sum string) string {
	t.Helper()
	root := t.TempDir()
	write(t, filepath.Join(root, "go.mod"), mod)
	write(t, filepath.Join(root, "go.sum"), sum)
	return root
}
func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
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
