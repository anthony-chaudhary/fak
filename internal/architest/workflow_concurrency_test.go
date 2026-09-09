package architest

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const workflowConcurrencyContract = `concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'pull_request' && format('pr-{0}', github.event.pull_request.number) || github.event_name == 'push' && format('ref-{0}', github.ref) || format('run-{0}', github.run_id) }}
  cancel-in-progress: ${{ github.event_name == 'push' || github.event_name == 'pull_request' }}`

func TestExpensiveWorkflowsCancelOnlySupersededPushAndPullRequestRuns(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	workflows := []string{
		"bench.yml",
		"dogfood.yml",
		"dogfood-coverage.yml",
		"garden.yml",
		"security-audit.yml",
	}

	if len(workflows) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the test explicitly audits exactly five declared workflows
		t.Fatalf("workflow concurrency gate must cover exactly five workflows; got %d", len(workflows))
	}

	for _, name := range workflows {
		name := name
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(root, ".github", "workflows", name)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			text := strings.ReplaceAll(string(data), "\r\n", "\n")

			if count := strings.Count(text, workflowConcurrencyContract); count != 1 {
				t.Fatalf("%s must contain the exact event-sensitive concurrency contract once; got %d matches", name, count)
			}
			contractAt := strings.Index(text, workflowConcurrencyContract)
			jobsAt := strings.Index(text, "\njobs:\n")
			if jobsAt < 0 {
				t.Fatalf("%s has no top-level jobs block", name)
			}
			if contractAt > jobsAt {
				t.Fatalf("%s concurrency contract is not top-level before jobs", name)
			}
		})
	}
}

func TestRaceJobHasExplicitWallClockDeadline(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	path := filepath.Join(root, ".github", "workflows", "ci.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")

	// Find the race-detector job block in ci.yml
	lines := strings.Split(text, "\n")
	var inRaceDetector bool
	var raceLines []string
	for _, line := range lines {
		if strings.HasPrefix(line, "  race-detector:") {
			inRaceDetector = true
			raceLines = append(raceLines, line)
			continue
		}
		if inRaceDetector {
			// A line starting with exactly 2 spaces followed by a non-space non-comment starts the next job
			if strings.HasPrefix(line, "  ") && !strings.HasPrefix(line, "   ") && !strings.HasPrefix(line, "  #") {
				break
			}
			raceLines = append(raceLines, line)
		}
	}

	if len(raceLines) == 0 {
		t.Fatalf("ci.yml has no race-detector job block")
	}
	raceBlock := strings.Join(raceLines, "\n")

	// 1. Must define an explicit job-level timeout-minutes
	const timeoutKey = "timeout-minutes:"
	var foundTimeout bool
	var jobTimeout int
	for _, line := range raceLines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, timeoutKey) {
			foundTimeout = true
			fields := strings.Fields(trimmed)
			if len(fields) < 2 {
				t.Fatalf("expected key-value for %s, got %q", timeoutKey, trimmed)
			}
			val, err := strconv.Atoi(fields[1])
			if err != nil {
				t.Fatalf("parse timeout-minutes %q: %v", fields[1], err)
			}
			jobTimeout = val
			break
		}
	}

	if !foundTimeout {
		t.Fatalf("race-detector job in ci.yml must define an explicit job-level %s deadline", timeoutKey)
	}

	// 2. The deadline exceeds the observed envelope (test -timeout=25m) plus bounded grace.
	// Contract coverage distinguishes job-level timeout from go test -timeout.
	const testStepTimeout = "-timeout=25m"
	if !strings.Contains(raceBlock, testStepTimeout) {
		t.Fatalf("race-detector job must run go test with %s", testStepTimeout)
	}

	if jobTimeout < 30 {
		t.Fatalf("job timeout-minutes (%d) must exceed go test timeout of 25m with grace (expected >= 30)", jobTimeout)
	}
	if jobTimeout > 45 {
		t.Fatalf("job timeout-minutes (%d) exceeds bounded upper grace limit (expected <= 45)", jobTimeout)
	}
}

