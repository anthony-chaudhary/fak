package architest

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestWorkflowCommandsResolve is the CI-hygiene gate that would have caught the garden.yml
// rot. When tools/garden_bundle.py was ported to Go (`fak garden`) and the .py deleted, the
// garden workflow kept running `python tools/garden_bundle.py` and failed on EVERY push for
// days with "No such file or directory" — because nothing checked that a workflow's commands
// still resolve to files that exist. A red workflow that nobody is forced to look at is the
// "guessing" failure mode: the breakage is real but invisible at commit time.
//
// This walks .github/workflows/*.yml and asserts every referenced in-repo script
// (`python tools/X.py`, `bash tools/Y.sh`) and `go run ./cmd/Z` target exists in the tree.
// A port-then-delete or a rename now fails HERE — in the always-on `go test ./...` gate,
// fast and legibly, naming the workflow and the dangling path — instead of rotting a
// workflow red until someone happens to read the Actions tab. Pure-stdlib, hermetic
// (reads tracked files, no network, no subprocess), so it costs nothing to keep on.
func TestWorkflowCommandsResolve(t *testing.T) {
	root := filepath.Dir(internalDir(t)) // repo root = parent of internal/
	wfDir := filepath.Join(root, ".github", "workflows")
	entries, err := os.ReadDir(wfDir)
	if err != nil {
		t.Fatalf("read workflows dir %s: %v", wfDir, err)
	}

	// `python tools/x.py`, `python3 tools/x.py`, `bash tools/x.sh`, and `go run ./cmd/x`.
	reScript := regexp.MustCompile(`(?:python3?|bash)\s+(tools/[\w./-]+\.(?:py|sh))`)
	reGoRun := regexp.MustCompile(`go run\s+(\./cmd/[\w/-]+)`)

	checked := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".yml") && !strings.HasSuffix(name, ".yaml") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(wfDir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		text := string(data)

		for _, m := range reScript.FindAllStringSubmatch(text, -1) {
			rel := m[1]
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err != nil {
				t.Errorf("%s references %q, which does not exist in the tree — a workflow command "+
					"points at a removed or renamed script. Update the workflow (e.g. to the Go verb "+
					"the tool was ported to) or restore the file. This is the garden.yml rot: a tool "+
					"deleted under a workflow that kept calling it.", name, rel)
			}
			checked++
		}
		for _, m := range reGoRun.FindAllStringSubmatch(text, -1) {
			rel := strings.TrimPrefix(m[1], "./")
			fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
			if err != nil || !fi.IsDir() {
				t.Errorf("%s runs %q, but that command directory does not exist — a `go run ./cmd/...` "+
					"target was removed or renamed. Update the workflow or restore cmd/%s.",
					name, m[1], strings.TrimPrefix(rel, "cmd/"))
			}
			checked++
		}
	}

	// Fail closed: if the extractor matched nothing the gate is silently inert (the regex
	// drifted or the workflows moved), which is exactly how a checker rots into a no-op.
	if checked == 0 {
		t.Fatal("no workflow commands were checked — the extractor matched nothing, so this gate " +
			"would be silently inert; the regex or the .github/workflows layout changed")
	}
	t.Logf("workflow command references checked across .github/workflows: %d", checked)
}
