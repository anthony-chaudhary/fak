package workerworktree

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const witnessSnapshotPoolPrefix = ".fak-witness-pool-"

// WitnessSnapshotEvent records source preparation separately from test outcomes.
type WitnessSnapshotEvent struct {
	Phase    string        `json:"phase"`
	Ref      string        `json:"ref"`
	Dir      string        `json:"dir"`
	Reused   bool          `json:"reused"`
	Duration time.Duration `json:"duration_ns"`
	Error    string        `json:"error,omitempty"`
}
type witnessSnapshotObserverKey struct{}

// WithWitnessSnapshotObserver receives acquisition and release evidence.
func WithWitnessSnapshotObserver(ctx context.Context, observe func(WitnessSnapshotEvent)) context.Context {
	return context.WithValue(ctx, witnessSnapshotObserverKey{}, observe)
}

// AcquireWitnessSnapshot leases a full clean detached snapshot. Active leases
// remain disjoint from each other and from the land candidate namespace.
func AcquireWitnessSnapshot(ctx context.Context, root, exactSHA string) (string, func(), error) {
	started := time.Now()
	emit := func(phase, dir string, reused bool, start time.Time, err error) {
		if observer, ok := ctx.Value(witnessSnapshotObserverKey{}).(func(WitnessSnapshotEvent)); ok && observer != nil {
			event := WitnessSnapshotEvent{Phase: phase, Ref: exactSHA, Dir: dir, Reused: reused, Duration: time.Since(start)}
			if err != nil {
				event.Error = err.Error()
			}
			observer(event)
		}
	}
	if err := ctx.Err(); err != nil {
		emit("prepare", "", false, started, err)
		return "", nil, err
	}
	if len(exactSHA) != 40 || strings.IndexFunc(exactSHA, func(r rune) bool { return !(r >= '0' && r <= '9' || r >= 'a' && r <= 'f') }) >= 0 {
		err := fmt.Errorf("witness snapshot requires exact commit SHA")
		emit("prepare", "", false, started, err)
		return "", nil, err
	}
	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		emit("prepare", "", false, started, err)
		return "", nil, err
	}
	root = absoluteRoot
	session := newTopologyCandidateSession(root, witnessSnapshotGitRunner(ctx))
	session.poolPrefix = witnessSnapshotPoolPrefix
	if ok, detail := session.create(exactSHA); !ok {
		err := ctx.Err()
		if err == nil {
			err = fmt.Errorf("witness snapshot preparation: %s", detail)
		}
		dir := session.dir
		err = errors.Join(err, session.discardWitnessSnapshot(), session.cleanupErr)
		emit("prepare", dir, session.reused, started, err)
		return "", nil, err
	}
	dir := session.dir
	if rc, _ := run(session.git, dir, []string{"sparse-checkout", "disable"}); rc != 0 {
		err := errors.Join(fmt.Errorf("witness snapshot full checkout failed"), ctx.Err(), session.discardWitnessSnapshot())
		emit("prepare", dir, session.reused, started, err)
		return "", nil, err
	}
	if err := session.validateWitnessSnapshot(exactSHA); err != nil {
		err = errors.Join(err, ctx.Err(), session.discardWitnessSnapshot())
		emit("prepare", dir, session.reused, started, err)
		return "", nil, err
	}
	if err := ctx.Err(); err != nil {
		err = errors.Join(err, session.discardWitnessSnapshot())
		emit("prepare", dir, session.reused, started, err)
		return "", nil, err
	}
	if err := os.Remove(witnessSnapshotUnavailablePath(dir)); err != nil {
		err = errors.Join(err, session.discardWitnessSnapshot())
		emit("prepare", dir, session.reused, started, err)
		return "", nil, err
	}
	emit("prepare", dir, session.reused, started, nil)
	var once sync.Once
	return dir, func() {
		once.Do(func() {
			start := time.Now()
			cleanupCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			session.git = witnessSnapshotGitRunner(cleanupCtx)
			var cleanupErr error
			if session.pooled && livePoolCountPrefix(root, filepath.Dir(dir), session.git, session.poolName()) <= TopologyCandidatePoolCap {
				session.close()
			} else {
				cleanupErr = session.discardWitnessSnapshot()
			}
			emit("release", dir, session.reused, start, cleanupErr)
		})
	}, nil
}

func (s *topologyCandidateSession) validateWitnessSnapshot(ref string) error {
	for _, probe := range []struct {
		args []string
		want string
	}{
		{[]string{"rev-parse", "HEAD"}, ref},
		{[]string{"status", "--porcelain", "--untracked-files=all", "--ignored"}, ""},
	} {
		rc, out := run(s.git, s.dir, probe.args)
		if rc != 0 || strings.TrimSpace(out) != probe.want {
			return fmt.Errorf("witness snapshot source or cleanliness probe failed")
		}
	}
	if rc, _ := run(s.git, s.dir, []string{"symbolic-ref", "-q", "HEAD"}); rc != 1 {
		return fmt.Errorf("witness snapshot is not detached")
	}
	return nil
}

func witnessSnapshotUnavailablePath(dir string) string {
	return candidatePoolSidecarPath(dir) + ".witness-unavailable"
}

// Persist exclusion outside the checkout before any source mutation. Failure
// paths retain this marker even if removing or renaming the checkout fails.
func beginWitnessSnapshotMutation(dir string) error {
	path := witnessSnapshotUnavailablePath(dir)
	if err := os.MkdirAll(filepath.Dir(path), poolStateDirPerm); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, poolLockFilePerm)
	if err != nil {
		return err
	}
	_, writeErr := f.WriteString("witness snapshot preparation incomplete\n")
	syncErr := f.Sync()
	return errors.Join(writeErr, syncErr, f.Close())
}

func (s *topologyCandidateSession) discardWitnessSnapshot() error {
	if s.dir == "" {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := os.Stat(witnessSnapshotUnavailablePath(s.dir)); os.IsNotExist(err) {
		if err := beginWitnessSnapshotMutation(s.dir); err != nil {
			// This checkout was validated before release. Do not mutate it when
			// exclusion cannot be persisted; the next acquisition must reset it.
			releaseCandidatePool(s.poolLock)
			s.poolLock, s.dir = nil, ""
			return err
		}
	}
	var cleanupErr error
	if rc, out := run(witnessSnapshotGitRunner(ctx), s.root, []string{"worktree", "remove", "--force", s.dir}); rc != 0 {
		cleanupErr = fmt.Errorf("witness snapshot removal exited %d: %s", rc, tail(out, 200))
		// Failed preparation must leave the reusable namespace before unlocking.
		if err := os.Rename(s.dir, filepath.Join(filepath.Dir(s.dir), ".fak-witness-quarantine-"+filepath.Base(s.dir))); err != nil && !os.IsNotExist(err) {
			cleanupErr = fmt.Errorf("%w; quarantine rename: %v", cleanupErr, err)
		}
	}
	releaseCandidatePool(s.poolLock)
	s.poolLock = nil
	s.dir = ""
	s.cleanupErr = cleanupErr
	return cleanupErr
}

func witnessSnapshotGitRunner(ctx context.Context) GitRunner {
	return func(root string, args []string) (int, string) {
		if err := ctx.Err(); err != nil {
			return ReapTimeoutExitCode, err.Error()
		}
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = root
		cmd.WaitDelay = 100 * time.Millisecond
		windowgate.ConfigureBackgroundCommand(cmd)
		procguard.ConfigureProcessTreeCancel(cmd)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		job, err := windowgate.StartInNewJob(cmd)
		if err != nil {
			return 127, err.Error()
		}
		err = cmd.Wait()
		_ = job.Close()
		if ctx.Err() != nil {
			return ReapTimeoutExitCode, ctx.Err().Error()
		}
		if err != nil {
			if exit, ok := err.(*exec.ExitError); ok {
				return exit.ExitCode(), stdout.String() + stderr.String()
			}
			return 127, err.Error()
		}
		return 0, stdout.String()
	}
}

// Pool constants. topologyCandidatePoolPrefix names the kept-checkout family;
// it is deliberately OUTSIDE topologyCandidatePrefixes so the owner-aware sweep
// never collects a pooled checkout whose creator has merely moved on — the pool
// cap, not the sweep, bounds this family.
const (
	topologyCandidatePoolPrefix = ".fak-cand-pool-"

	// TopologyCandidatePoolCap bounds the pool: at most this many kept
	// candidate checkouts per parent directory per repository. Each is a full
	// working-tree checkout, so the cap bounds the disk the pool can hold.
	// The bound is self-healed by trimPoolExcess rather than enforced by a
	// creation lock, so concurrent cold lands never queue on each other.
	TopologyCandidatePoolCap = 2

	// CandidatePoolEnv disables the pool: values 0, false, off (case-insensitive)
	// make every candidate transient again (created fresh, removed on close,
	// collected by the owner-aware sweep). Anything else, including unset,
	// enables the pool.
	CandidatePoolEnv = "FAK_CANDIDATE_POOL"

	// candidatePoolLockName is the advisory flock file inside a pooled
	// checkout's git admin dir. It lives there — not in the working tree — so
	// `clean -ffdx` can never unlink a lock another process holds (the
	// prepare-lock hazard), and it dies with the registration.
	candidatePoolLockName = "fak-pool.lock"

	// candidatePoolSidecarDir is the per-parent directory of creation locks.
	// A creator holds its checkout's sidecar lock from MkdirTemp until the
	// admin-dir flock above is taken, because the admin dir — and with it the
	// only in-tree exclusion — does not exist until `worktree add` creates it:
	// without the sidecar, an adopter could watch a half-materialized checkout
	// pass every registration gate and reset the tree under the running add.
	candidatePoolSidecarDir = ".fak-cand-pool-locks"
)

// candidatePoolEnabled reports whether kept-checkout pooling is on.
func candidatePoolEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(CandidatePoolEnv))) {
	case "0", "false", "off":
		return false
	}
	return true
}

// topologyCandidatePoolPattern is the os.MkdirTemp pattern for a pooled
// candidate owned by this process: owner-named like the transient family so a
// directory's creator stays diagnosable, and so a git `locked` marker left by
// a killed creator can be judged stale by owner liveness.
func topologyCandidatePoolPattern() string {
	return OwnerNamedScratchPattern(topologyCandidatePoolPrefix)
}

func isTopologyCandidatePoolName(name string) bool {
	return strings.HasPrefix(name, topologyCandidatePoolPrefix)
}

// parsePoolCandidateOwner mirrors parseTopologyCandidateOwner for the pool
// prefix, whose names carry the same owner identity shape.
func parsePoolCandidateOwner(name string) (candidateOwner, bool) {
	rest, ok := strings.CutPrefix(name, topologyCandidatePoolPrefix)
	if !ok {
		return candidateOwner{}, false
	}
	parts := strings.Split(rest, "-")
	if len(parts) != 3 || parts[2] == "" {
		return candidateOwner{}, false
	}
	pid, err := strconv.Atoi(parts[0])
	if err != nil || pid <= 0 {
		return candidateOwner{}, false
	}
	start, err := strconv.ParseInt(parts[1], 36, 64)
	if err != nil || start < 0 {
		return candidateOwner{}, false
	}
	return candidateOwner{PID: pid, Start: start}, true
}

// poolLockMarkerStale reports whether a pooled checkout's git `locked` marker
// may be cleared: only when its named creator is provably gone. The marker on
// a pool dir is then either git's mid-add initializing lock left by a killed
// land, or a leftover on an abandoned checkout. A live creator's dir is
// excluded earlier by the sidecar lock; a same-named live creator's explicit
// marker is respected.
func poolLockMarkerStale(name string) bool {
	owner, ok := parsePoolCandidateOwner(name)
	if !ok {
		return false
	}
	return candidateOwnerState(owner, CandidateSweepOptions{}) != "owner_process_live"
}

// THE CANDIDATE CHECKOUT POOL
//
// A fresh verify checkout is a full `git worktree add` of the whole repository.
// Measured on the Windows fleet host (2026-11-10): 489.5s for 33,098 files,
// because every file creation on that host costs 15-30ms (per-file write tax;
// the go-build cache path, Defender-excluded and quiet, writes at 6.2ms/file
// while the workspace parent writes at 29.6ms/file). That checkout is the
// dominant cost of a prepared land: the invocation ledger's prepare phase is
// p50 787.6s across 255 lands while the build it exists to run takes 6.5s warm.
//
// A candidate is verification scratch by construction: its content is a pure
// function of a commit that already exists and a diff the worker worktree
// still holds. So a checkout left behind by one land is exactly as good as a
// fresh one for the next land, provided the next user resets it to its ref
// first — which moveTo already does (checkout --force + clean -ffdx), the same
// reset a lost-CAS retry inside one session already trusted.
//
// The pool makes that reuse durable. A pooled checkout is created under
// topologyCandidatePoolPrefix and KEPT on close; a later session adopts it.
// Adoption is layered so no window is unguarded:
//
//  1. the creation sidecar lock (candidatePoolSidecarDir beside the checkout)
//     covers MkdirTemp through the admin-flock handoff — the window in which
//     the checkout is registered-but-mid-add and would otherwise pass every
//     registration gate while `worktree add` is still writing it;
//  2. the admin-dir flock (candidatePoolLockName, inside the worktree's git
//     admin dir, outside clean's reach) covers active use; it cannot be
//     unlinked out from under its holder;
//  3. git's `locked` marker — a killed creator's mid-add leftover — is
//     cleared only when the checkout's named creator is provably gone, so a
//     live creator's (or operator's) marker is respected.
//
// After locking, moveToFrom(ref) resets the tree (checkout --force + clean
// -ffdx), the same reset a lost-CAS retry inside one session already trusted.
// A creator that loses its fresh checkout's admin lock abandons the dir to
// that lock holder and materializes its own transient checkout — it never
// uses a tree it does not hold. The cap is self-healed by trimPoolExcess
// (concurrent cold creators may briefly overshoot; the next creator trims the
// oldest extras it can fully exclude), so cold creations never queue on each
// other. Excess-cap creations fall back to the transient behavior, and the
// owner-aware sweep still collects transient candidates.
// FAK_CANDIDATE_POOL=0/off/false restores the all-transient behavior.

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
//
// When pooling is on, the session may also adopt a checkout a previous land
// kept (pooled) instead of materializing a fresh one, and close() keeps a
// pooled checkout for the next land. poolLock is the advisory exclusion held
// for as long as this session may mutate the checkout; a session that could
// not take it never marks the dir pooled-holding (it either skips the dir or
// leaves it unlocked-kept for its holder to release).
type topologyCandidateSession struct {
	poolPrefix string
	reused     bool
	failed     string
	cleanupErr error
	root       string
	git        GitRunner
	dir        string
	// pooled reports that dir is a kept pool checkout: close releases the
	// advisory lock (when held) and leaves the checkout in place.
	pooled bool
	// poolLock is the held flock file for a pooled dir; nil when the dir is
	// transient or was kept unlocked.
	poolLock *os.File
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

func (s *topologyCandidateSession) poolName() string {
	if s.poolPrefix != "" {
		return s.poolPrefix
	}
	return topologyCandidatePoolPrefix
}
func (s *topologyCandidateSession) poolMarkerStale(name string) bool {
	return poolLockMarkerStale(strings.Replace(name, s.poolName(), topologyCandidatePoolPrefix, 1))
}

// A managed caller's generated workspace describes plumbing, not the pinned tree.
func (s *topologyCandidateSession) witnessSnapshotParent(ref string) (string, error) {
	rc, files := run(s.git, s.root, []string{"ls-tree", "--full-tree", "--name-only", ref, "--", "go.work"})
	if rc != 0 {
		return "", fmt.Errorf("inspect committed witness workspace")
	}
	if strings.TrimSpace(files) == "" {
		return "", nil
	}
	rc, workspace := run(s.git, s.root, []string{"show", ref + ":go.work"})
	if rc != 0 {
		return "", fmt.Errorf("read committed witness workspace")
	}
	sibling := false
	for _, field := range strings.Fields(workspace) {
		field = strings.Trim(field, "()\"")
		if field == ".." || strings.HasPrefix(field, "../") || strings.HasPrefix(field, "..\\") {
			sibling = true
			break
		}
	}
	if !sibling {
		return "", nil
	}
	common := rootCommonGitDir(s.root, s.git)
	info, err := os.Stat(common)
	if common == "" || filepath.Base(common) != ".git" || err != nil || !info.IsDir() {
		return "", fmt.Errorf("resolve canonical witness repository root")
	}
	canonical := filepath.Dir(common)
	parent := filepath.Dir(canonical)
	if parent == canonical {
		return "", fmt.Errorf("canonical witness repository has no parent")
	}
	return parent, nil
}

func (s *topologyCandidateSession) create(ref string) (bool, string) {
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		if rc, out := run(s.git, s.root, []string{"rev-parse", ref + "^{commit}"}); rc != 0 || strings.TrimSpace(out) != ref {
			return false, "witness source commit unavailable"
		}
	}
	parent := ""
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		var err error
		parent, err = s.witnessSnapshotParent(ref)
		if err != nil {
			return false, err.Error()
		}
	} else if workspaceNeedsSiblingTopology(s.root) {
		rootAbs, err := filepath.Abs(s.root)
		if err != nil {
			return false, "failed to resolve selected root: " + err.Error()
		}
		parent = filepath.Dir(rootAbs)
	}
	// A killed land never runs close; its owner-named candidate is collected by
	// a later land's sweep (candidate.go).
	if s.poolPrefix != witnessSnapshotPoolPrefix {
		sweepTopologyCandidatesBeforeCreate(s.root, parent, s.git)
	}
	if candidatePoolEnabled() || s.poolPrefix == witnessSnapshotPoolPrefix {
		if s.adoptPooled(parent, ref) {
			return true, ""
		}
		if s.failed != "" {
			return false, s.failed
		}

		if livePoolCountPrefix(s.root, parent, s.git, s.poolName()) < TopologyCandidatePoolCap {
			ok, hard := s.createPooled(parent, ref)
			if ok {
				return true, ""
			}
			if hard != "" {
				return false, hard
			}
			// The fresh checkout was adopted mid-creation by whoever won its
			// admin lock; fall through and materialize this session's own
			// transient checkout rather than use a tree it does not hold.
		}
	}
	return s.createTransient(parent, ref)
}

// createPooled materializes a new pooled checkout. It reports (true, "") when
// the session owns the new checkout (admin lock held), (false, "") when the
// dir must be left to whoever won its lock or pooling could not be
// established (the caller falls back to a transient checkout), and
// (false, hard) when the materialization itself failed.
func (s *topologyCandidateSession) createPooled(parent, ref string) (bool, string) {
	candDir, err := os.MkdirTemp(parent, OwnerNamedScratchPattern(s.poolName()))
	if err != nil {
		return false, ""
	}
	_ = os.Remove(candDir)
	// The sidecar covers the window the admin dir cannot: from MkdirTemp until
	// the admin-dir flock is taken, the checkout is registered-but-mid-add and
	// would otherwise pass every adoption gate.
	sidecar, ok := lockCandidateSidecar(candDir)
	if !ok {
		return false, ""
	}
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		if err := beginWitnessSnapshotMutation(candDir); err != nil {
			releaseCandidateSidecar(sidecar)
			return false, "witness exclusion marker failed: " + err.Error()
		}
	}
	rc, out := run(s.git, s.root, []string{"-c", "core.longpaths=true", "worktree", "add", "--detach", candDir, ref})
	if rc != 0 {
		if s.poolPrefix == witnessSnapshotPoolPrefix {
			s.dir = candDir
			s.discardWitnessSnapshot()
			releaseCandidateSidecar(sidecar)
			return false, "git witness checkout failed: " + tail(out, 200)
		}
		releaseCandidateSidecar(sidecar)
		cleanupTopologyCandidate(s.root, candDir, s.git)
		return false, "git candidate checkout failed: " + tail(out, 200)
	}
	lock, lockOK := lockCandidatePoolDir(candDir)
	releaseCandidateSidecar(sidecar)
	if !lockOK {
		return false, ""
	}
	s.dir = candDir
	s.pooled = true
	s.poolLock = lock
	if s.poolPrefix != witnessSnapshotPoolPrefix {
		trimPoolExcessPrefix(s.root, parent, s.git, s.poolName())
	}
	return true, ""
}

// createTransient is the pre-pool behavior: a fresh owner-named checkout that
// close removes and the owner-aware sweep collects if the land is killed.
func (s *topologyCandidateSession) createTransient(parent, ref string) (bool, string) {
	pattern := topologyCandidatePattern()
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		pattern = OwnerNamedScratchPattern(".fak-witness-once-")
	}
	candDir, err := os.MkdirTemp(parent, pattern)
	if err != nil {
		return false, "failed to create topology-preserving candidate temp dir: " + err.Error()
	}
	_ = os.Remove(candDir)
	s.dir = candDir
	s.pooled = false
	s.poolLock = nil
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		if err := beginWitnessSnapshotMutation(candDir); err != nil {
			s.dir = ""
			return false, "witness exclusion marker failed: " + err.Error()
		}
	}
	if rc, out := run(s.git, s.root, []string{"-c", "core.longpaths=true", "worktree", "add", "--detach", candDir, ref}); rc != 0 {
		if s.poolPrefix == witnessSnapshotPoolPrefix {
			s.discardWitnessSnapshot()
		} else {
			s.close()
		}
		return false, "git candidate checkout failed: " + tail(out, 200)
	}
	return true, ""
}

// adoptPooled looks for a kept pool checkout of this repository in parent that
// no session holds, resets it to ref, and adopts it. It reports whether one was
// adopted; every failure falls through to a fresh materialization, because a
// land must never fail or slow down because reuse was unavailable.
//
// Per-candidate exclusion is layered, each layer covering a window the next
// cannot see: the sidecar lock (creation, from MkdirTemp to the admin-flock
// handoff), the admin-dir flock (active use), and git's `locked` marker (a
// killed creator's mid-add leftover, cleared only when the named creator is
// provably gone).
func (s *topologyCandidateSession) adoptPooled(parent, ref string) bool {
	if parent == "" {
		parent = os.TempDir()
	}
	common := rootCommonGitDir(s.root, s.git)
	if common == "" {
		return false
	}
	registered, prunable := registeredWorktrees(s.root, s.git)
	staleRegistration := false
	entries, err := os.ReadDir(parent)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), s.poolName()) {
			continue
		}
		cand := filepath.Join(parent, entry.Name())
		sidecar, ok := lockCandidateSidecar(cand)
		if !ok {
			// Mid-creation (or its creator sits between MkdirTemp and the
			// handoff): never touch it.
			continue
		}
		if s.poolPrefix == witnessSnapshotPoolPrefix {
			if _, err := os.Stat(witnessSnapshotUnavailablePath(cand)); !os.IsNotExist(err) {
				releaseCandidateSidecar(sidecar)
				continue
			}
		}
		candCommon := candidateCommonGitDir(cand)
		if candCommon == "" || !samePath(candCommon, common) {
			releaseCandidateSidecar(sidecar)
			// No .git link: either an orphaned pre-add empty dir, or another
			// repository's candidate. Remove only the provably inert empty
			// case; never touch a dir whose repository cannot be named.
			if candCommon == "" && emptyDir(cand) {
				_ = os.Remove(cand)
			}
			continue
		}
		if !containsPath(registered, cand) || containsPath(prunable, cand) {
			if s.poolPrefix == witnessSnapshotPoolPrefix {
				releaseCandidateSidecar(sidecar)
				continue
			}
			// Registered-but-gutted (or unregistered) scratch of this repo:
			// remove the dir and let the prune collect the registration, the
			// same verdict the owner-aware sweep reaches for the transient
			// families. Scratch by construction: its content is a pure
			// function of a commit that still exists.
			removeCandidateWithRegistration(s.root, cand, s.git)
			releaseCandidateSidecar(sidecar)
			staleRegistration = true
			continue
		}
		lock, lockOK := lockCandidatePoolDir(cand)
		releaseCandidateSidecar(sidecar)
		if !lockOK {
			continue
		}
		if candidateWorktreeLockPresent(cand) && !s.poolMarkerStale(entry.Name()) {
			// A live creator's (or operator's) explicit marker: respect it.
			releaseCandidatePool(lock)
			continue
		}
		if s.poolPrefix == witnessSnapshotPoolPrefix {
			if err := beginWitnessSnapshotMutation(cand); err != nil {
				releaseCandidatePool(lock)
				s.failed = "witness exclusion marker failed: " + err.Error()
				return false
			}
		}
		clearCandidateWorktreeLock(cand)
		if !s.moveToFrom(cand, ref) {
			if s.poolPrefix == witnessSnapshotPoolPrefix {
				s.dir, s.poolLock, s.pooled = cand, lock, true
				s.failed = "git witness reset failed"
				return false
			}
			// The flock proves no pool-aware session is using the checkout, so
			// removing it is safe; a git "initializing" marker left by a dead
			// creator is cleared by removeCandidateWithRegistration.
			releaseCandidatePool(lock)
			removeCandidateWithRegistration(s.root, cand, s.git)
			staleRegistration = true
			continue
		}
		s.reused = true
		s.dir = cand
		s.pooled = true
		s.poolLock = lock
		if staleRegistration {
			run(s.git, s.root, []string{"worktree", "prune"})
		}
		if s.poolPrefix != witnessSnapshotPoolPrefix {
			s.pruneGoneRegistrations()
		}
		return true
	}
	if staleRegistration {
		run(s.git, s.root, []string{"worktree", "prune"})
	}
	if s.poolPrefix != witnessSnapshotPoolPrefix {
		s.pruneGoneRegistrations()
	}
	return false
}

// trimPoolExcess self-heals the cap: concurrent cold creators can each observe
// room and create, parking more kept checkouts than TopologyCandidatePoolCap.
// The creator that just pushed the count past the cap removes the oldest
// extras it can fully exclude (sidecar plus admin flock), so the pool returns
// to the cap without ever serializing cold creations behind each other.
func trimPoolExcess(root, parent string, git GitRunner) {
	trimPoolExcessPrefix(root, parent, git, topologyCandidatePoolPrefix)
}
func trimPoolExcessPrefix(root, parent string, git GitRunner, prefix string) {
	if parent == "" {
		parent = os.TempDir()
	}
	common := rootCommonGitDir(root, git)
	if common == "" {
		return
	}
	registered, prunable := registeredWorktrees(root, git)
	type poolDir struct {
		path string
		mod  time.Time
	}
	var dirs []poolDir
	entries, err := os.ReadDir(parent)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}
		cand := filepath.Join(parent, entry.Name())
		if candCommon := candidateCommonGitDir(cand); candCommon == "" || !samePath(candCommon, common) {
			continue
		}
		if !containsPath(registered, cand) || containsPath(prunable, cand) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		dirs = append(dirs, poolDir{path: cand, mod: info.ModTime()})
	}
	if len(dirs) <= TopologyCandidatePoolCap {
		return
	}
	sort.Slice(dirs, func(i, j int) bool { return dirs[i].mod.Before(dirs[j].mod) })
	for _, d := range dirs[:len(dirs)-TopologyCandidatePoolCap] {
		sidecar, ok := lockCandidateSidecar(d.path)
		if !ok {
			continue
		}
		lock, lockOK := lockCandidatePoolDir(d.path)
		releaseCandidateSidecar(sidecar)
		if !lockOK {
			continue
		}
		releaseCandidatePool(lock)
		removeCandidateWithRegistration(root, d.path, git)
	}
}

// pruneGoneRegistrations collects pool registrations whose checkout directory
// is gone — invisible to the parent-directory scan above (a ReadDir cannot
// list a removed dir), and never pruned for the transient families' reasons:
// git skips a registration carrying its "initializing" lock, so the sweep
// removes that lock only after proving the creator gone. The pool applies the
// same proof twice over: the named creator must be gone AND the checkout's own
// flock must be free (a held one is left for its holder).
func (s *topologyCandidateSession) pruneGoneRegistrations() {
	common := rootCommonGitDir(s.root, s.git)
	if common == "" {
		return
	}
	admins, err := os.ReadDir(filepath.Join(common, "worktrees"))
	if err != nil {
		return
	}
	collected := false
	for _, admin := range admins {
		adminDir := filepath.Join(common, "worktrees", admin.Name())
		gitdir, err := os.ReadFile(filepath.Join(adminDir, "gitdir"))
		if err != nil {
			continue
		}
		wt := filepath.Dir(filepath.Clean(strings.TrimSpace(string(gitdir))))
		if !strings.HasPrefix(filepath.Base(wt), s.poolName()) {
			continue
		}
		if _, err := os.Stat(wt); !os.IsNotExist(err) {
			continue
		}
		if !s.poolMarkerStale(filepath.Base(wt)) {
			continue
		}
		lockPath := filepath.Join(adminDir, candidatePoolLockName)
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, poolLockFilePerm)
		if err != nil {
			continue
		}
		if err := flock.TryLock(f); err != nil {
			_ = f.Close()
			continue
		}
		_ = os.Remove(filepath.Join(adminDir, "locked"))
		_ = flock.Unlock(f)
		_ = f.Close()
		collected = true
	}
	if collected {
		run(s.git, s.root, []string{"worktree", "prune"})
	}
}

// moveToFrom is moveTo against an explicit directory (the adopt path moves the
// adopted checkout before the session owns s.dir).
func (s *topologyCandidateSession) moveToFrom(dir, ref string) bool {
	if rc, _ := run(s.git, dir, []string{"-c", "core.longpaths=true", "checkout", "--detach", "--force", ref}); rc != 0 {
		return false
	}
	args := []string{"clean", "-ffdx", "-e", "/.gocache"}
	if s.poolPrefix == witnessSnapshotPoolPrefix {
		if rc, _ := run(s.git, dir, []string{"reset", "--hard", ref}); rc != 0 {
			return false
		}
		args = []string{"clean", "-ffdx"}
	}
	rc, _ := run(s.git, dir, args)
	return rc == 0
}

func (s *topologyCandidateSession) moveTo(ref string) bool {
	return s.moveToFrom(s.dir, ref)
}

// close releases the session's checkout: a pooled one is kept (lock released)
// for a later land to adopt, a transient one is removed. Safe to call more
// than once.
func (s *topologyCandidateSession) close() {
	if s == nil || s.dir == "" {
		return
	}
	if s.pooled {
		releaseCandidatePool(s.poolLock)
		s.poolLock = nil
		s.dir = ""
		return
	}
	cleanupTopologyCandidate(s.root, s.dir, s.git)
	s.dir = ""
}

// rootCommonGitDir resolves root's git common directory, absolutized.
func rootCommonGitDir(root string, git GitRunner) string {
	rc, out := run(git, root, []string{"rev-parse", "--git-common-dir"})
	if rc != 0 {
		return ""
	}
	dir := strings.TrimSpace(out)
	if dir == "" {
		return ""
	}
	if !filepath.IsAbs(dir) {
		dir = filepath.Join(root, dir)
	}
	return filepath.Clean(dir)
}

// registeredWorktrees parses one `git worktree list --porcelain` into
// path -> prunable maps. listOK is false when git could not produce a listing.
func registeredWorktrees(root string, git GitRunner) (registered map[string]bool, prunable map[string]bool) {
	registered, prunable = map[string]bool{}, map[string]bool{}
	rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
	if rc != 0 {
		return registered, prunable
	}
	for _, listing := range parseWorktreeListing(out) {
		registered[filepath.Clean(listing.Path)] = true
		if listing.Prunable {
			prunable[filepath.Clean(listing.Path)] = true
		}
	}
	return registered, prunable
}

// livePoolCount counts this repository's live registered pool checkouts in
// parent — the pool's disk bound. Unregistered or other-repository dirs do not
// consume the cap.
func livePoolCount(root, parent string, git GitRunner) int {
	return livePoolCountPrefix(root, parent, git, topologyCandidatePoolPrefix)
}
func livePoolCountPrefix(root, parent string, git GitRunner, prefix string) int {
	if parent == "" {
		parent = os.TempDir()
	}
	common := rootCommonGitDir(root, git)
	if common == "" {
		return 0
	}
	registered, prunable := registeredWorktrees(root, git)
	n := 0
	entries, err := os.ReadDir(parent)
	if err != nil {
		return 0
	}
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), prefix) {
			continue
		}
		cand := filepath.Clean(filepath.Join(parent, entry.Name()))
		if candCommon := candidateCommonGitDir(cand); candCommon == "" || !samePath(candCommon, common) {
			continue
		}
		if containsPath(registered, cand) && !containsPath(prunable, cand) {
			n++
		}
	}
	return n
}

// candidateAdminDir returns the git admin dir a candidate worktree is
// registered under, read from its .git file (gitdir: <common>/worktrees/<id>),
// or "" when the file is absent or does not have that shape.
func candidateAdminDir(dir string) string {
	common := candidateCommonGitDir(dir)
	if common == "" {
		return ""
	}
	b, err := os.ReadFile(filepath.Join(dir, ".git"))
	if err != nil {
		return ""
	}
	admin, found := strings.CutPrefix(strings.TrimSpace(string(b)), "gitdir:")
	if !found {
		return ""
	}
	admin = filepath.Clean(strings.TrimSpace(admin))
	if info, err := os.Stat(admin); err != nil || !info.IsDir() {
		return ""
	}
	return admin
}

// lockCandidatePoolDir takes the pooled checkout's advisory flock. ok is false
// when another session holds it or the lock file cannot be established; the
// caller must then treat the checkout as unavailable (adopt) or abandon it to
// its holder (create). It is a package variable so tests can inject failures.
var lockCandidatePoolDir = lockCandidatePoolDirAt

func lockCandidatePoolDirAt(dir string) (lock *os.File, ok bool) {
	admin := candidateAdminDir(dir)
	if admin == "" {
		return nil, false
	}
	f, err := os.OpenFile(filepath.Join(admin, candidatePoolLockName), os.O_CREATE|os.O_RDWR, poolLockFilePerm)
	if err != nil {
		return nil, false
	}
	if err := flock.TryLock(f); err != nil {
		_ = f.Close()
		return nil, false
	}
	return f, true
}

// releaseCandidatePool releases and closes a held pool lock; nil-safe.
func releaseCandidatePool(lock *os.File) {
	if lock == nil {
		return
	}
	_ = flock.Unlock(lock)
	_ = lock.Close()
}

// lockCandidateSidecar takes a pool checkout's creation lock, held from
// MkdirTemp until the admin-dir flock handoff. The lock file lives in a
// sidecar directory beside the checkout — the admin dir it would otherwise
// live in does not exist until `worktree add` creates it — and is never
// unlinked, per the prepare-lock rule (unlinking a held lock file would let a
// third process lock a fresh file and break the exclusion).
func lockCandidateSidecar(dir string) (*os.File, bool) {
	lockPath := candidatePoolSidecarPath(dir)
	if err := os.MkdirAll(filepath.Dir(lockPath), poolStateDirPerm); err != nil {
		return nil, false
	}
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, poolLockFilePerm)
	if err != nil {
		return nil, false
	}
	if err := flock.TryLock(f); err != nil {
		_ = f.Close()
		return nil, false
	}
	return f, true
}

// releaseCandidateSidecar releases and closes a held creation lock; nil-safe.
func releaseCandidateSidecar(sidecar *os.File) {
	if sidecar == nil {
		return
	}
	_ = flock.Unlock(sidecar)
	_ = sidecar.Close()
}

// candidatePoolSidecarPath is the creation-lock file for a pool checkout dir.
func candidatePoolSidecarPath(dir string) string {
	return filepath.Join(filepath.Dir(dir), candidatePoolSidecarDir, filepath.Base(dir)+prepareLockExt)
}

// candidateWorktreeLockPresent reports whether the checkout's registration
// carries git's `locked` marker (git's own mid-add initializing lock, or an
// explicit `git worktree lock`).
func candidateWorktreeLockPresent(dir string) bool {
	admin := candidateAdminDir(dir)
	if admin == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(admin, "locked"))
	return err == nil
}

// clearCandidateWorktreeLock removes the checkout's git `locked` marker (all
// `git worktree unlock` does); the caller must have proven no live user.
func clearCandidateWorktreeLock(dir string) {
	if admin := candidateAdminDir(dir); admin != "" {
		_ = os.Remove(filepath.Join(admin, "locked"))
	}
}

// removeCandidateWithRegistration removes a candidate directory whose
// registration may carry git's "initializing" lock (a land killed mid-add):
// the admin dir is read BEFORE the removal (the .git link dies with the
// directory), then the directory, then the lock file (all `git worktree
// unlock` does), so the caller's prune can collect the entry — the sweep's
// recipe, applied inline by the adopt path.
func removeCandidateWithRegistration(root, cand string, git GitRunner) {
	admin := candidateAdminDir(cand)
	if removeTopologyCandidateDir(cand) != "" {
		return
	}
	if admin != "" {
		_ = os.Remove(filepath.Join(admin, "locked"))
	}
	run(git, root, []string{"worktree", "prune"})
}

// containsPath reports whether the key set contains path, compared with the
// codebase's case-insensitive, symlink-resolving path equality.
func containsPath(set map[string]bool, path string) bool {
	for key := range set {
		if samePath(key, path) {
			return true
		}
	}
	return false
}

// emptyDir reports whether dir exists and holds no entries.
func emptyDir(dir string) bool {
	entries, err := os.ReadDir(dir)
	return err == nil && len(entries) == 0
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
