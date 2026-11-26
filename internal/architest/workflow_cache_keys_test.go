package architest

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Cache keys must not embed a per-commit value: each commit would otherwise
// create a fresh multi-GiB entry and evict the warm one.
func TestWorkflowCacheKeysAreNotPerCommit(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	files, err := filepath.Glob(filepath.Join(root, ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no workflow files found")
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(raw), "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "key:") && strings.Contains(trimmed, "github.sha") {
				t.Errorf("%s:%d cache key embeds github.sha: %s", filepath.Base(f), i+1, trimmed)
			}
		}
	}
}

// The gobuild cache is saved only from trunk pushes; other runs are restore-only.
func TestGobuildCacheSaveIsTrunkOnly(t *testing.T) {
	src := readCIWorkflow(t)
	idx := strings.Index(src, "save Go build/test cache (trunk push only)")
	if idx < 0 {
		t.Fatal("ci.yml has no trunk-only gobuild save step")
	}
	block := src[idx:]
	if end := strings.Index(block, "\n      - name:"); end > 0 {
		block = block[:end]
	}
	if !strings.Contains(block, "github.ref == 'refs/heads/main'") || !strings.Contains(block, "github.event_name == 'push'") {
		t.Errorf("gobuild save step is not gated to trunk push:\n%s", block)
	}
}

// A gobuild save persists the cache as the job leaves it, so it must follow every
// build, test and vet command in its own job. Save steps are discovered across all
// jobs rather than pinned to one job's layout, so moving the test step between jobs
// does not break the guard.
func TestGobuildCacheSaveFollowsEveryBuildAndTestCommand(t *testing.T) {
	jobs := ciJobs(t, readCIWorkflow(t))
	trunkSaves := 0
	for _, job := range jobs {
		saves := gobuildSaveSteps(job.body)
		if len(saves) == 0 {
			continue
		}
		trunkSaves++
		for _, save := range saves {
			block := job.body[save.start:save.end]
			if !strings.Contains(block, "github.ref == 'refs/heads/main'") || !strings.Contains(block, "github.event_name == 'push'") {
				t.Errorf("%s: gobuild save step is not gated to trunk push:\n%s", job.name, block)
			}
			if !containsAny(job.body[:save.start], gobuildCommands) {
				t.Errorf("%s saves the gobuild cache with no go build, test or vet command before the save", job.name)
			}
			for _, cmd := range gobuildCommands {
				if last := strings.LastIndex(job.body, cmd); last > save.start {
					t.Errorf("%s runs %q after the gobuild save step; move the save to the end of the job", job.name, cmd)
				}
			}
		}
	}
	if trunkSaves == 0 {
		t.Fatal("ci.yml has no job that saves the gobuild cache")
	}
}

var gobuildCommands = []string{"go test ", "go build ", "go run ", "go vet ", "go install "}

var ciJobHeader = regexp.MustCompile(`(?m)^  [A-Za-z0-9_-]+:$`)

var ciStepStart = regexp.MustCompile(`(?m)^      - `)

type ciJob struct {
	name string
	body string
}

type ciStep struct {
	start, end int
}

func readCIWorkflow(t *testing.T) string {
	t.Helper()
	path := filepath.Join(filepath.Dir(internalDir(t)), ".github", "workflows", "ci.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.ReplaceAll(string(raw), "\r\n", "\n")
}

// ciJobs returns every top-level job in ci.yml in file order, each body with
// comment-only lines removed, so commands quoted in comments cannot satisfy or
// break the ordering check.
func ciJobs(t *testing.T, src string) []ciJob {
	t.Helper()
	at := strings.Index(src, "\njobs:\n")
	if at < 0 {
		t.Fatal("ci.yml has no jobs section")
	}
	body := src[at:]
	headers := ciJobHeader.FindAllStringIndex(body, -1)
	if len(headers) == 0 {
		t.Fatal("ci.yml jobs section has no jobs")
	}
	jobs := make([]ciJob, 0, len(headers))
	for i, h := range headers {
		end := len(body)
		if i+1 < len(headers) {
			end = headers[i+1][0]
		}
		name := strings.TrimSuffix(strings.TrimSpace(body[h[0]:h[1]]), ":")
		jobs = append(jobs, ciJob{name: name, body: stripCommentLines(body[h[1]:end])})
	}
	return jobs
}

// gobuildSaveSteps returns the steps of a job body that save the gobuild cache.
func gobuildSaveSteps(job string) []ciStep {
	starts := ciStepStart.FindAllStringIndex(job, -1)
	var saves []ciStep
	for i, loc := range starts {
		end := len(job)
		if i+1 < len(starts) {
			end = starts[i+1][0]
		}
		block := job[loc[0]:end]
		if strings.Contains(block, "uses: actions/cache/save@") && strings.Contains(block, "-gobuild-") {
			saves = append(saves, ciStep{start: loc[0], end: end})
		}
	}
	return saves
}

func containsAny(s string, needles []string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}

func stripCommentLines(s string) string {
	var kept []string
	for _, line := range strings.Split(s, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "#") {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}
