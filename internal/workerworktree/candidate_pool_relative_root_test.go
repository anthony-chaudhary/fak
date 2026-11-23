package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// A land invoked with --root . from the main checkout gets a relative git
// common dir back from git. The pool must still recognize its own checkouts,
// or the cap counts zero and every land parks a fresh one.
// fak-test:justify why=regression when=changed:internal/workerworktree/land_topology.go
// fak-test:runtime medium est=60s lane=default
func TestCandidatePoolRecognizesItsCheckoutsFromARelativeRoot(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root, parent := newSiblingTopologyRepo(t)
	t.Chdir(root)
	const rel = "."

	if got := rootCommonGitDir(rel, defaultGit); !filepath.IsAbs(got) {
		t.Fatalf("rootCommonGitDir(%q) = %q; want an absolute path", rel, got)
	}

	hook := func(string) (bool, string) { return true, "" }
	first := newTopologyCandidateSession(rel, defaultGit)
	defer first.close()
	if ok, detail := first.verify("HEAD", "", hook); !ok {
		t.Fatalf("first verify: %s", detail)
	}
	kept := first.dir
	first.close()
	if _, err := os.Stat(kept); err != nil {
		t.Fatalf("pooled checkout %q should be kept after close: %v", kept, err)
	}
	if got := livePoolCount(rel, parent, defaultGit); got != 1 {
		t.Fatalf("live pool count from a relative root = %d, want 1", got)
	}

	second := newTopologyCandidateSession(rel, defaultGit)
	defer second.close()
	if ok, detail := second.verify("HEAD", "", hook); !ok {
		t.Fatalf("second verify: %s", detail)
	}
	if !samePath(second.dir, kept) {
		t.Fatalf("second session materialized %q, want adoption of %q", second.dir, kept)
	}
	second.close()
	if got := len(registeredPoolCandidates(t, root)); got != 1 {
		t.Fatalf("registered pool checkouts = %d, want 1", got)
	}
}
