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

// A save that runs before the job's last build or test command persists a cache
// that does not contain what the job compiled, so the save must come last.
func TestGobuildCacheSaveFollowsEveryBuildAndTestCommand(t *testing.T) {
	job := ciJobBlock(t, readCIWorkflow(t), "build-vet-test")
	saveAt := strings.Index(job, "save Go build/test cache (trunk push only)")
	if saveAt < 0 {
		t.Fatal("build-vet-test has no gobuild save step")
	}
	if testAt := strings.Index(job, "name: root module zip budget (uncached)"); testAt < 0 || testAt > saveAt {
		t.Fatalf("gobuild save step must come after the test step (test at %d, save at %d)", testAt, saveAt)
	}
	for _, cmd := range []string{"go test ", "go build ", "go run ", "go vet ", "go install "} {
		if last := strings.LastIndex(job, cmd); last > saveAt {
			t.Errorf("build-vet-test runs %q after the gobuild save step; move the save to the end of the job", cmd)
		}
	}
}

var ciJobHeader = regexp.MustCompile(`(?m)^  [A-Za-z0-9_-]+:$`)

func readCIWorkflow(t *testing.T) string {
	t.Helper()
	path := filepath.Join(filepath.Dir(internalDir(t)), ".github", "workflows", "ci.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.ReplaceAll(string(raw), "\r\n", "\n")
}

// ciJobBlock returns the named top-level job with comment-only lines removed, so
// commands quoted in comments cannot satisfy or break the ordering check.
func ciJobBlock(t *testing.T, src, name string) string {
	t.Helper()
	header := "\n  " + name + ":\n"
	start := strings.Index(src, header)
	if start < 0 {
		t.Fatalf("ci.yml has no %s job", name)
	}
	rest := src[start+len(header):]
	if loc := ciJobHeader.FindStringIndex(rest); loc != nil {
		rest = rest[:loc[0]]
	}
	var kept []string
	for _, line := range strings.Split(rest, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "#") {
			kept = append(kept, line)
		}
	}
	return strings.Join(kept, "\n")
}
