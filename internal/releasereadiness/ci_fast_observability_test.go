package releasereadiness

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFastReleaseGateKeepsLongGoTestObservable(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	data, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci-fast.yml"))
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(data)
	for _, required := range []string{
		"- name: go test ./... (no -race)",
		"timeout-minutes: 30",
		"go test ./... 2>&1 | tee",
		"${PIPESTATUS[0]}",
		"go test ./... is still running",
	} {
		if !strings.Contains(workflow, required) {
			t.Fatalf("ci-fast correctness gate lacks %q", required)
		}
	}
}
