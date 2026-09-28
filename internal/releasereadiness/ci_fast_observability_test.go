package releasereadiness

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestFastReleaseGateConvergesAndKeepsFullTestObservable(t *testing.T) {
	workflow := readFastReleaseWorkflow(t)
	for _, required := range []string{
		"name: ci-fast",
		"workflow_dispatch:",
		"concurrency:",
		`GOMAXPROCS: "2"`,
		`GOFLAGS: "-p=2"`,
		"run: go build ./...",
		"run: go vet ./...",
		// The go test step stays observable (#6982): tee'd log, the real exit
		// status, and a heartbeat while it runs.
		"- name: go test ./... (no -race)",
		`go test -timeout=20m "${shard_pkgs[@]}" 2>&1 | tee`,
		`status_file="$RUNNER_TEMP/ci-fast-go-test.status"`,
		`"$RUNNER_TEMP/ci-fast-go-test.log"`,
		"${PIPESTATUS[0]}",
		"is still running",
		`FAK_REQUIRE_OPENAPI_VALIDATOR: "1"`,
		// The shards partition ONE package list: every job checks out the event
		// commit (or the pinned dispatch sha), and the partition keys on the
		// runner-provided matrix position so its modulus cannot drift from the
		// matrix size.
		"ref: ${{ github.event_name == 'workflow_dispatch' && inputs.sha || '' }}",
		"go list ./...",
		"SHARD_INDEX: ${{ strategy.job-index }}",
		"SHARD_TOTAL: ${{ strategy.job-total }}",
		"'(NR - 1) % n == i'",
		"fail-fast: false",
		// The verdict job keeps the pre-shard job id and check name and is red
		// unless build/vet and every shard concluded success.
		"build-vet-test-fast:",
		"name: build · vet · test (no -race, fast release gate)",
		"needs: [build-vet-fast, go-test-shard]",
		"if: ${{ !cancelled() }}",
		"BUILD_VET_RESULT: ${{ needs.build-vet-fast.result }}",
		"GO_TEST_SHARDS_RESULT: ${{ needs.go-test-shard.result }}",
		`[ "$BUILD_VET_RESULT" != "success" ] || [ "$GO_TEST_SHARDS_RESULT" != "success" ]`,
	} {
		if !strings.Contains(workflow, required) {
			t.Fatalf("ci-fast correctness gate lacks %q", required)
		}
	}
	for _, forbidden := range []string{
		"github.head_ref || github.sha",
		`GOFLAGS: "-p=1"`,
		// An explicit branch ref checks out the branch tip at fetch time, so
		// parallel shards on a hot trunk could test different trees.
		"inputs.sha || github.ref",
		// The whole suite in one serial step is what never concluded in 30m.
		"go test ./... 2>&1 | tee",
	} {
		if strings.Contains(workflow, forbidden) {
			t.Fatalf("ci-fast correctness gate retains obsolete contract %q", forbidden)
		}
	}
}

// TestFastReleaseGateShardOutlastsPerBinaryTimeout pins the deadline ordering
// that keeps a stuck shard decisive and diagnosable: go test's own per-binary
// -timeout fires (with a goroutine dump) inside the step bound, which leaves
// room for compile/link, and the job bound outlasts the step.
func TestFastReleaseGateShardOutlastsPerBinaryTimeout(t *testing.T) {
	workflow := readFastReleaseWorkflow(t)
	job := workflowJobBlock(t, workflow, "go-test-shard")
	jobTimeout := timeoutMinutesAtIndent(t, job, 4)
	stepAt := strings.Index(job, "      - name: go test ./... (no -race)\n")
	if stepAt < 0 {
		t.Fatal("go-test-shard job has no `go test ./... (no -race)` step")
	}
	step := job[stepAt:]
	stepTimeout := timeoutMinutesAtIndent(t, step, 8)
	m := regexp.MustCompile(`go test -timeout=(\d+)m `).FindStringSubmatch(step)
	if m == nil {
		t.Fatal("go-test-shard step must run go test with an explicit -timeout=<N>m")
	}
	testTimeout, _ := strconv.Atoi(m[1])
	if testTimeout <= 10 {
		t.Fatalf("go test -timeout=%dm does not exceed the 10m default that clipped cmd/fak", testTimeout)
	}
	if stepTimeout < testTimeout+10 {
		t.Fatalf("step timeout-minutes %d must outlast go test -timeout=%dm plus compile/link grace (>= %d)", stepTimeout, testTimeout, testTimeout+10)
	}
	if jobTimeout <= stepTimeout {
		t.Fatalf("job timeout-minutes %d must outlast the go test step bound %d", jobTimeout, stepTimeout)
	}
}

func readFastReleaseWorkflow(t *testing.T) string {
	t.Helper()
	root := findRepoRoot(t)
	data, err := os.ReadFile(filepath.Join(root, ".github", "workflows", "ci-fast.yml"))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(data), "\r\n", "\n")
}

// workflowJobBlock returns the text of one top-level job, from its `  <id>:`
// line up to the next job key at the same indent.
func workflowJobBlock(t *testing.T, workflow, id string) string {
	t.Helper()
	start := strings.Index(workflow, "\n  "+id+":\n")
	if start < 0 {
		t.Fatalf("ci-fast has no %q job", id)
	}
	block := workflow[start+1:]
	lines := strings.SplitAfter(block, "\n")
	var b strings.Builder
	b.WriteString(lines[0])
	for _, line := range lines[1:] {
		if strings.HasPrefix(line, "  ") && !strings.HasPrefix(line, "   ") && !strings.HasPrefix(line, "  #") {
			break
		}
		b.WriteString(line)
	}
	return b.String()
}

// timeoutMinutesAtIndent reads the first `timeout-minutes:` key at exactly the
// given indent (job keys sit at 4 spaces, step keys at 8).
func timeoutMinutesAtIndent(t *testing.T, block string, indent int) int {
	t.Helper()
	prefix := strings.Repeat(" ", indent) + "timeout-minutes:"
	for _, line := range strings.Split(block, "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, prefix)))
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		return v
	}
	t.Fatalf("no %q line in block", strings.TrimSpace(prefix))
	return 0
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
