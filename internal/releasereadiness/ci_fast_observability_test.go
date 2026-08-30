package releasereadiness

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFastReleaseGateConvergesAndKeepsFullTestObservable(t *testing.T) {
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
		"name: ci-fast",
		"build-vet-test-fast:",
		"name: build · vet · test (no -race, fast release gate)",
		"group: ${{ github.workflow }}-${{ github.ref }}",
		"cancel-in-progress: true",
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
