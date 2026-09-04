package workerworktree

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// Concurrent-land race soak for #3619 (child of epic #3165). #3547 built two
// layers against the shared-index land race but shipped them default-OFF; this
// soak is the evidence that lets the defaults flip on.
//
// THE RACE (#3547, witnessed): Land's baseline path `git apply`s the worker diff
// into the SHARED trunk working tree, then `git commit -- <paths>`. In the window
// between those two steps a PEER's non-path-scoped commit (`git add -A && commit`
// from another session) can sweep the worker's apply-staged file into its own
// unrelated commit — so Land false-succeeds while the worker's change lands
// nowhere, or lands mis-attributed (observed swallowing #3153's dosobserve.go
// into a docs commit). The isolated-index path (FAK_LAND_ISOLATED_INDEX) stages
// into a THROWAWAY GIT_INDEX_FILE and moves the branch by compare-and-swap, so the
// worker's change is NEVER in the shared working tree for a peer to sweep.
//
// HOW THE SOAK IS FAITHFUL AND DETERMINISTIC:
//   - Real throwaway git repo, real worktrees, real `git` — no reimplementation of
//     Land; the production Land()/landIsolated() run verbatim.
//   - N worker goroutines each Land a DISJOINT path. Production always calls Land
//     under the dispatcher's lane lease (see Land's doc comment: "The caller holds
//     the lane lease, which serializes this"), so the soak holds a lease mutex
//     around each Land — the concurrency under test is land-vs-peer-sweep, exactly
//     the #3547 hazard, not the lease-forbidden land-vs-land.
//   - The peer sweep is injected DETERMINISTICALLY at the one vulnerable instant:
//     the `git` runner handed to Land fires one non-path-scoped `git add -A &&
//     commit` immediately after any BASELINE working-tree `apply` succeeds. That is
//     the precise shared-index window the race lives in. The isolated path never
//     calls that baseline apply (it stages through GIT_INDEX_FILE via a separate
//     env-runner), so with isolation ON the sweep never fires and every land is
//     atomic; with isolation OFF every land is swept. A forced interleaving that
//     makes a real, timing-dependent race reliably observable is the standard shape
//     of a concurrency regression test.
//
// Run under `-race` (needs cgo; on this repo's native-Windows host that means WSL:
// `./test.ps1 -race ./internal/workerworktree/`).

const soakWorkers = 8

// soakRepo is a real throwaway trunk repo plus N detached worker worktrees, each
// having modified its own disjoint tracked file. It drives the production Land().
type soakRepo struct {
	t        *testing.T
	root     string   // trunk repo path
	base     string   // base commit sha every worktree is pinned at
	files    []string // files[i] is worker i's disjoint tracked path
	wts      []string // wts[i] is worker i's detached worktree path
	sweeps   atomic.Int32
	lease    sync.Mutex // models the dispatcher's lane lease (serializes lands)
	sweepLog []string
	sweepMu  sync.Mutex
}

func rawGit(t *testing.T, dir string, args ...string) (int, string) {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	out, err := c.CombinedOutput()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		return 127, string(out)
	}
	return 0, string(out)
}

func mustGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	rc, out := rawGit(t, dir, args...)
	if rc != 0 {
		t.Fatalf("git %s (in %s): rc=%d\n%s", strings.Join(args, " "), dir, rc, out)
	}
	return out
}

func newSoakRepo(t *testing.T, n int) *soakRepo {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "trunk@test")
	mustGit(t, root, "config", "user.name", "trunk")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	// Every worker's path is a pre-existing TRACKED file: the baseline land path
	// commits a MODIFICATION by pathspec (a brand-new untracked file cannot be
	// staged by `git commit -- newfile`, which is a fixture artifact, not the race).
	files := make([]string, n)
	for i := 0; i < n; i++ {
		files[i] = fmt.Sprintf("w%02d.txt", i)
		if err := os.WriteFile(filepath.Join(root, files[i]), []byte(fmt.Sprintf("base %d\n", i)), 0o644); err != nil {
			t.Fatalf("seed %s: %v", files[i], err)
		}
		mustGit(t, root, "add", files[i])
	}
	mustGit(t, root, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	s := &soakRepo{t: t, root: root, base: base, files: files, wts: make([]string, n)}
	wtRoot := t.TempDir()
	for i := 0; i < n; i++ {
		res := Prepare(root, "soak", fmt.Sprintf("%d", i), base, wtRoot, nil)
		if !res.OK {
			t.Fatalf("prepare worker %d: %+v", i, res)
		}
		s.wts[i] = res.Path
		if err := os.WriteFile(filepath.Join(res.Path, files[i]), []byte(fmt.Sprintf("worker %d\n", i)), 0o644); err != nil {
			t.Fatalf("edit worker %d: %v", i, err)
		}
		mustGit(t, res.Path, "config", "user.email", "w@test")
		mustGit(t, res.Path, "config", "user.name", "w")
		mustGit(t, res.Path, "config", "commit.gpgsign", "false")
		mustGit(t, res.Path, "add", files[i])
		mustGit(t, res.Path, "commit", "-q", "-m", fmt.Sprintf("fix(w): modify %s (#3619) (fak workerworktree)", files[i]))
	}
	return s
}

// peerGit is the git runner handed to Land. It runs real git AND, right after a
// successful BASELINE working-tree `apply`, fires one non-path-scoped peer sweep —
// the deterministic realization of a peer session committing during the land's
// shared-index window. landIsolated stages via GIT_INDEX_FILE through a DIFFERENT
// env-runner, so its `apply --cached` never routes here — the sweep only ever fires
// on the baseline (isolation-OFF or fallback) path.
func (s *soakRepo) peerGit(root string, args []string) (int, string) {
	rc, out := rawGit(s.t, root, args...)
	if rc == 0 && len(args) > 0 && args[0] == "apply" {
		// The worker's change now sits unstaged in the SHARED working tree. A peer's
		// non-path-scoped commit sweeps it into an unrelated commit (#3547).
		_, _ = rawGit(s.t, s.root, "add", "-A")
		src, cout := rawGit(s.t, s.root, "commit", "-q", "-m", "docs(peer): unrelated non-path-scoped sweep")
		if src == 0 {
			s.sweeps.Add(1)
			s.sweepMu.Lock()
			s.sweepLog = append(s.sweepLog, strings.TrimSpace(cout))
			s.sweepMu.Unlock()
		}
	}
	return rc, out
}

// land drives one worker's Land under the lane lease (as production does).
func (s *soakRepo) land(i int) Result {
	s.lease.Lock()
	defer s.lease.Unlock()
	return Land(s.root, s.wts[i], s.base, "", []string{s.files[i]}, nil, s.peerGit)
}

// landParallel lands without holding the caller lease mutex, driving production
// LandingQueue and CAS retry under high concurrency (#11235).
func (s *soakRepo) landParallel(i int) Result {
	return Land(s.root, s.wts[i], s.base, "", []string{s.files[i]}, nil, s.peerGit)
}

// runConcurrent spins N goroutines through land() and collects the per-worker
// Results. The lease serializes the lands; the goroutines + `-race` prove the
// land path carries no Go data race.
func (s *soakRepo) runConcurrent() []Result {
	res := make([]Result, len(s.files))
	var wg sync.WaitGroup
	for i := range s.files {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res[i] = s.land(i)
		}(i)
	}
	wg.Wait()
	return res
}

// runHighVelocity runs lands concurrently without caller-side serialization,
// verifying that LandingQueue and in-memory 3-way merge resolution complete
// isolated lands without falling back to the racy shared index (#11235).
func (s *soakRepo) runHighVelocity() []Result {
	res := make([]Result, len(s.files))
	var wg sync.WaitGroup
	for i := range s.files {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			res[i] = s.landParallel(i)
		}(i)
	}
	wg.Wait()
	return res
}

// commitInfo is one commit on HEAD's lineage above base: its subject and the
// exact set of paths it changed.
type commitInfo struct {
	sha     string
	subject string
	files   []string
}

// lineageAboveBase returns the commits from base..HEAD (child-most first) with
// their subject and changed-file set, read straight from git.
func (s *soakRepo) lineageAboveBase() []commitInfo {
	out := mustGit(s.t, s.root, "rev-list", s.base+"..HEAD")
	var infos []commitInfo
	for _, sha := range strings.Fields(out) {
		subj := strings.TrimSpace(mustGit(s.t, s.root, "show", "-s", "--format=%s", sha))
		names := mustGit(s.t, s.root, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
		var fs []string
		for _, ln := range strings.Split(names, "\n") {
			if p := normalizeSlash(strings.TrimSpace(ln)); p != "" {
				fs = append(fs, p)
			}
		}
		infos = append(infos, commitInfo{sha: sha, subject: subj, files: fs})
	}
	return infos
}

// headContent returns the HEAD-tree content of one path (or "" / not-ok).
func (s *soakRepo) headContent(path string) (string, bool) {
	rc, out := rawGit(s.t, s.root, "show", "HEAD:"+path)
	return out, rc == 0
}

// TestConcurrentLandSoakIsolatedIsRaceFree is the ON half of the #3619 A/B: with
// the isolated-index default in force, N concurrent lease-held lands of disjoint
// paths, each racing a peer non-path-scoped sweep, all land atomically — every
// land's HEAD commit carries EXACTLY its own path, no land false-succeeds, no
// cross-contamination, and the sweep never fires (the isolated path keeps every
// worker's change out of the shared working tree). Run under `-race`.
func TestConcurrentLandSoakIsolatedIsRaceFree(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1") // pin ON regardless of ambient default
	t.Setenv(LandReadbackEnv, "1") // layer-1 honest-refusal also on
	s := newSoakRepo(t, soakWorkers)
	res := s.runConcurrent()

	// (a) every land cleanly committed — no lost land, no false refusal.
	for i, r := range res {
		if !r.OK || !r.Committed {
			t.Fatalf("worker %d land not clean: OK=%v Committed=%v reason=%q detail=%q",
				i, r.OK, r.Committed, r.Reason, r.Detail)
		}
		if !strings.Contains(r.Reason, "isolated-index") {
			t.Fatalf("worker %d did not take the isolated path: reason=%q", i, r.Reason)
		}
	}
	// (b) the shared-index window never opened: the isolated path staged through a
	// throwaway index, so the injected peer sweep had nothing to sweep and never
	// moved HEAD.
	if got := s.sweeps.Load(); got != 0 {
		t.Fatalf("isolated lands must never expose the shared index; peer sweep fired %d time(s)", got)
	}
	// (c) per-land HEAD-carries-exactly-my-paths + no cross-contamination: every
	// commit above base changes exactly one worker file, each worker file is
	// carried by exactly one commit, and all N are present.
	lineage := s.lineageAboveBase()
	carriedBy := map[string]int{}
	for _, ci := range lineage {
		if len(ci.files) != 1 {
			t.Fatalf("commit %s carries %d files %v — a land must carry exactly its own path",
				ci.sha[:8], len(ci.files), ci.files)
		}
		carriedBy[ci.files[0]]++
	}
	for i, f := range s.files {
		if carriedBy[f] != 1 {
			t.Fatalf("worker %d file %s carried by %d commits, want exactly 1", i, f, carriedBy[f])
		}
		// (d) zero false-success: the content on HEAD is actually the worker's.
		content, ok := s.headContent(f)
		if !ok || content != fmt.Sprintf("worker %d\n", i) {
			t.Fatalf("worker %d: HEAD %s = %q (ok=%v), want %q", i, f, content, ok, fmt.Sprintf("worker %d\n", i))
		}
	}
}

// TestConcurrentWorkerLandSoakUnderHighVelocity verifies that parallel workers landing
// simultaneously complete their lands via landIsolated using LandingQueue and in-memory
// 3-way merge tree resolution without falling back to racy shared index commits (#11235).
func TestConcurrentWorkerLandSoakUnderHighVelocity(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	t.Setenv(LandReadbackEnv, "1")
	t.Setenv(IsolatedLandRetryEnv, "10")
	stubCASSleep(t)
	s := newSoakRepo(t, soakWorkers)
	res := s.runHighVelocity()

	// (a) every land cleanly committed via the isolated path — no lost land, no fallback.
	for i, r := range res {
		if !r.OK || !r.Committed {
			t.Fatalf("worker %d land not clean: OK=%v Committed=%v reason=%q detail=%q",
				i, r.OK, r.Committed, r.Reason, r.Detail)
		}
		if !strings.Contains(r.Reason, "isolated-index") {
			t.Fatalf("worker %d did not take the isolated path: reason=%q", i, r.Reason)
		}
	}
	// (b) the shared-index window never opened: peer sweeps never fired.
	if got := s.sweeps.Load(); got != 0 {
		t.Fatalf("isolated lands must never expose the shared index; peer sweep fired %d time(s)", got)
	}
	// (c) every worker's file was committed and is intact on HEAD.
	lineage := s.lineageAboveBase()
	carriedBy := map[string]int{}
	for _, ci := range lineage {
		for _, f := range ci.files {
			carriedBy[f]++
		}
	}
	for i, f := range s.files {
		if carriedBy[f] != 1 {
			t.Fatalf("worker %d file %s carried by %d commits, want exactly 1", i, f, carriedBy[f])
		}
		content, ok := s.headContent(f)
		if !ok || content != fmt.Sprintf("worker %d\n", i) {
			t.Fatalf("worker %d: HEAD %s = %q (ok=%v), want %q", i, f, content, ok, fmt.Sprintf("worker %d\n", i))
		}
	}
}

// TestConcurrentLandSoakBaselineRaceReproduced is the OFF half of the A/B: the
// same soak with the isolated path forced off reproduces the #3547 sweep — the
// peer's non-path-scoped commit swallows worker changes out of the shared working
// tree, so lands false-succeed / mis-attribute. This is the baseline that proves
// the isolated default is load-bearing, not decorative. It PASSES by witnessing
// the baseline's brokenness (so the committed suite stays green while still
// encoding the A/B).
func TestConcurrentLandSoakBaselineRaceReproduced(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0") // force the shared-index baseline
	t.Setenv(LandReadbackEnv, "0") // and no honest-refusal, as pre-#3619 prod
	s := newSoakRepo(t, soakWorkers)
	res := s.runConcurrent()

	// The shared-index window opened for real: the peer sweep fired.
	if s.sweeps.Load() == 0 {
		t.Fatalf("baseline soak did not exercise the shared-index window (no peer sweep fired)")
	}
	// The race manifests two ways; require at least one worker to exhibit it, so the
	// witness is robust to interleave-count without being flaky.
	lineage := s.lineageAboveBase()
	fileToSubject := map[string]string{}
	for _, ci := range lineage {
		for _, f := range ci.files {
			fileToSubject[f] = ci.subject
		}
	}
	swept, falseSucceeded := 0, 0
	for i, r := range res {
		f := normalizeSlash(s.files[i])
		subj, onHead := fileToSubject[f]
		// A worker change carried by a PEER commit (not the worker's own subject)
		// is the #3547 sweep — the change landed mis-attributed.
		if onHead && strings.HasPrefix(subj, "docs(peer)") {
			swept++
		}
		// A land that reported OK+Committed while its path is missing/mis-attributed
		// on HEAD is a false-success (the failure #3547's layer-1 readback catches).
		if r.OK && r.Committed && (!onHead || strings.HasPrefix(subj, "docs(peer)")) {
			falseSucceeded++
		}
	}
	if swept == 0 && falseSucceeded == 0 {
		t.Fatalf("baseline soak did not reproduce the #3547 shared-index race "+
			"(swept=%d falseSucceeded=%d sweeps=%d) — the A/B is not witnessed",
			swept, falseSucceeded, s.sweeps.Load())
	}
	t.Logf("#3547 baseline race reproduced: swept=%d falseSucceeded=%d peerSweeps=%d",
		swept, falseSucceeded, s.sweeps.Load())
}

// TestLandDefaultsAreOnAbsentEnv pins the #3619 default flip: with no env override,
// BOTH land-safety gates are ON, and the explicit 0/false/off escape still forces
// them off (an operator can always fall back to the shared-index baseline).
func TestLandDefaultsAreOnAbsentEnv(t *testing.T) {
	// t.Setenv registers restoration of any ambient value; then truly unset so the
	// pin checks the genuine absent-env default, not an inherited override.
	t.Setenv(IsolatedLandEnv, "")
	t.Setenv(LandReadbackEnv, "")
	os.Unsetenv(IsolatedLandEnv)
	os.Unsetenv(LandReadbackEnv)
	if !isolatedLandEnabled() {
		t.Fatal("isolatedLandEnabled() must default ON absent the env override (#3619)")
	}
	if !landReadbackEnabled() {
		t.Fatal("landReadbackEnabled() must default ON absent the env override (#3619)")
	}
	for _, off := range []string{"0", "false", "off", "OFF", "False"} {
		t.Setenv(IsolatedLandEnv, off)
		t.Setenv(LandReadbackEnv, off)
		if isolatedLandEnabled() {
			t.Fatalf("IsolatedLandEnv=%q must force the isolated path off (escape hatch)", off)
		}
		if landReadbackEnabled() {
			t.Fatalf("LandReadbackEnv=%q must force the readback off (escape hatch)", off)
		}
	}
	// A truthy explicit value keeps it on.
	t.Setenv(IsolatedLandEnv, "1")
	t.Setenv(LandReadbackEnv, "1")
	if !isolatedLandEnabled() || !landReadbackEnabled() {
		t.Fatal("explicit truthy env must keep the gates on")
	}
}
