package architest

import (
	"os"
	"path/filepath"
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
