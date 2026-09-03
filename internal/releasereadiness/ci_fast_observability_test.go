package releasereadiness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFastReleaseGateConvergesAndKeepsFullTestObservable(t *testing.T) {
	root := findRepoRoot(t)
	data, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci-fast.yml"))
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(data)
	for _, required := range []string{
		"name: ci-fast",
		"build-vet-test-fast:",
		"name: build · vet · test (no -race, fast release gate)",
		"workflow_dispatch:",
		"concurrency:",
		`GOMAXPROCS: "2"`,
		`GOFLAGS: "-p=2"`,
		"run: go build ./...",
		"run: go vet ./...",
		"- name: go test ./... (no -race)",
		"timeout-minutes: 30",
		"go test ./... 2>&1 | tee",
		`status_file="$RUNNER_TEMP/ci-fast-go-test.status"`,
		`"$RUNNER_TEMP/ci-fast-go-test.log"`,
		"${PIPESTATUS[0]}",
		"go test ./... is still running",
	} {
		if !strings.Contains(workflow, required) {
			t.Fatalf("ci-fast correctness gate lacks %q", required)
		}
	}
	for _, forbidden := range []string{
		"github.head_ref || github.sha",
		`GOFLAGS: "-p=1"`,
	} {
		if strings.Contains(workflow, forbidden) {
			t.Fatalf("ci-fast correctness gate retains obsolete contract %q", forbidden)
		}
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
