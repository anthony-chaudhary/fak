package workerworktree

import (
	"os"
	"path/filepath"
	"strings"
)

// verifyTopologyCandidate materializes the exact tree being verified beside the
// selected repository root. Checked-in workspaces may refer to sibling modules
// via paths such as ../module; fleet worktrees and system temp directories do
// not preserve that relationship. prospectiveDiff is applied only for the
// pre-land worker candidate; post-merge commits already contain the full tree.
func verifyTopologyCandidate(root, ref, prospectiveDiff string, verify VerifyHook, git GitRunner) (bool, string) {
	cand := newTopologyCandidateSession(root, git)
	defer cand.close()
	return cand.verify(ref, prospectiveDiff, verify)
}

// topologyCandidateSession is one land's verify-only checkout, reused across its
// verifications. A lost CAS re-verifies a candidate that differs from the last
// one by the peer commits it lost to, so moving the existing checkout to the new
// commit rewrites only those files; a fresh `git worktree add` rewrote the whole
// repository on every attempt. The verified tree is the same either way: the
// forced detach checkout resets every tracked file to the commit, and the clean
// removes everything untracked or ignored except the worktree-local Go cache,
// which is content-addressed and cannot change a build's result.
type topologyCandidateSession struct {
	root string
	git  GitRunner
	dir  string
}

func newTopologyCandidateSession(root string, git GitRunner) *topologyCandidateSession {
	return &topologyCandidateSession{root: root, git: git}
}

// verify materializes ref (plus prospectiveDiff, when non-empty) in the
// session's checkout and runs the hook there. Reuse that fails falls back to a
// fresh checkout rather than verifying a tree of unknown content.
func (s *topologyCandidateSession) verify(ref, prospectiveDiff string, verify VerifyHook) (bool, string) {
	if s.dir != "" && !s.moveTo(ref) {
		s.close()
	}
	if s.dir == "" {
		if ok, detail := s.create(ref); !ok {
			return false, detail
		}
	}
	if prospectiveDiff != "" {
		patch, cleanupPatch, err := writePatch(prospectiveDiff)
		if err != nil {
			return false, "failed to materialize prospective patch: " + err.Error()
		}
		defer cleanupPatch()
		if rc, out := run(s.git, s.dir, []string{"apply", "--whitespace=nowarn", patch}); rc != 0 {
			return false, "failed to apply prospective patch: " + tail(out, 200)
		}
	}
	return verify(s.dir)
}

func (s *topologyCandidateSession) create(ref string) (bool, string) {
	parent := ""
	if workspaceNeedsSiblingTopology(s.root) {
		rootAbs, err := filepath.Abs(s.root)
		if err != nil {
			return false, "failed to resolve selected root: " + err.Error()
		}
		parent = filepath.Dir(rootAbs)
	}
	// A killed land never runs close; its owner-named candidate is collected by
	// a later land's sweep (candidate.go).
	sweepTopologyCandidatesBeforeCreate(s.root, parent, s.git)
	candDir, err := os.MkdirTemp(parent, topologyCandidatePattern())
	if err != nil {
		return false, "failed to create topology-preserving candidate temp dir: " + err.Error()
	}
	_ = os.Remove(candDir)
	s.dir = candDir
	if rc, out := run(s.git, s.root, []string{"-c", "core.longpaths=true", "worktree", "add", "--detach", candDir, ref}); rc != 0 {
		s.close()
		return false, "git candidate checkout failed: " + tail(out, 200)
	}
	return true, ""
}

func (s *topologyCandidateSession) moveTo(ref string) bool {
	if rc, _ := run(s.git, s.dir, []string{"-c", "core.longpaths=true", "checkout", "--detach", "--force", ref}); rc != 0 {
		return false
	}
	rc, _ := run(s.git, s.dir, []string{"clean", "-ffdx", "-e", "/.gocache"})
	return rc == 0
}

// close removes the checkout; safe to call more than once.
func (s *topologyCandidateSession) close() {
	if s == nil || s.dir == "" {
		return
	}
	cleanupTopologyCandidate(s.root, s.dir, s.git)
	s.dir = ""
}

// workspaceNeedsSiblingTopology identifies checked-in Go workspaces whose module
// graph escapes the selected repository root. Those candidates must be verified
// beside root; ordinary repositories retain direct worker verification and the
// system-temp post-merge checkout used before #12447.
func workspaceNeedsSiblingTopology(root string) bool {
	b, err := os.ReadFile(filepath.Join(root, "go.work"))
	if err != nil {
		return false
	}
	for _, field := range strings.Fields(string(b)) {
		field = strings.Trim(field, "()\"")
		if field == ".." || strings.HasPrefix(field, "../") || strings.HasPrefix(field, `..\`) {
			return true
		}
	}
	return false
}
