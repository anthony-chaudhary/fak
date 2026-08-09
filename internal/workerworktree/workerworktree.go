// Package workerworktree is the native Go port of tools/worker_worktree.py: the
// per-worker git worktree isolation primitive that the live dispatch spawn wires
// in for #3168.
//
// THE PROBLEM (#1334 / #1333)
// Every dispatch worker launches with cwd = repo root, so N concurrent workers
// share ONE working tree, ONE index, and ONE Go build cache on the trunk: a
// worker mid-edit leaves a half-built package that reds another's `go build`, two
// `git commit -- <paths>` race on the shared index, and a stalled worker's WIP
// entangles the next worker's diff. That is the dominant throughput killer past
// ~4 concurrent workers.
//
// THE RECONCILIATION with the trunk-only commit rule (OFF_TRUNK):
//  1. Each worker EDITS in its own throwaway worktree at a DETACHED HEAD pinned to
//     the current trunk SHA (`git worktree add --detach <dir> <sha>`). A detached
//     worktree is not on `main` (git does not refuse it) and not on a feature
//     branch (it can never trip OFF_TRUNK); GOCACHE/GOTMPDIR point inside it so a
//     broken build in one worktree cannot red another's.
//  2. The change LANDS on the trunk through a serialized commit-to-trunk step
//     (Land): the worktree's diff-since-base is applied to the trunk worktree and
//     committed there as a normal stamped, signed-off commit ON `main`. Nothing
//     commits off-trunk; the lane lease the dispatcher already holds serializes
//     the apply so two workers never touch the same leaf tree at once.
//
// SAFETY STANCE: everything here is FAIL-OPEN and idempotent — a git error is
// reported in the returned Result, never surfaced as a wedge, so wiring the
// isolation in can only ADD isolation, never break a spawn or a sweep. The pure
// planners (DirName/Path/WorktreeEnv/IsWorkerWorktree) are unit-tested without
// git; the git-touching Prepare/Land/Reap take an injectable GitRunner so the
// whole acquire→edit→land→reap path is exercised against a fake. See
// tools/worker_worktree.py for the reference implementation this mirrors 1:1.
package workerworktree

import (
	"crypto/sha1"
	"encoding/hex"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	// WorktreeRootEnv overrides where per-worker worktrees are created.
	WorktreeRootEnv = "FLEET_WORKER_WORKTREE_ROOT"
	// WorktreeMarker is the dir-name segment that identifies OUR worktrees, so an
	// auditor (and Reap's guardrail) recognizes them without any worker's
	// self-report. Matches tools/worker_worktree.py and worktree_doctor.py.
	WorktreeMarker = "fak-worker-wt"
	// WorktreeDirEnv is the child-side env naming the worktree the worker runs in
	// (distinct from the FLEET_WORKER_WORKTREE enable switch the cmd/fak gate reads,
	// so the switch is never overloaded as a path). The Python module reuses
	// FLEET_WORKER_WORKTREE as the path marker; the Go wiring keeps them separate.
	WorktreeDirEnv = "FLEET_WORKER_WORKTREE_DIR"
	// LandReadbackEnv makes Land re-read trunk HEAD after a path-scoped commit and
	// confirm it actually carries the worker's intended paths — turning the silent
	// shared-index-race false-success (#3547) into an honest LAND_READBACK_MISMATCH
	// refusal. DEFAULT ON since #3619 (the concurrent-land soak proved the layer-2
	// isolated path race-free; this layer-1 honest refusal rides on by default too);
	// set it to 0/false/off to force the pre-#3619 byte-for-byte baseline as an escape.
	LandReadbackEnv = "FAK_LAND_READBACK_VERIFY"
	// IsolatedLandEnv makes Land stage+commit through a THROWAWAY index (GIT_INDEX_FILE)
	// and move the branch with a compare-and-swap ref update, so a worker's change is
	// never in the SHARED index for a concurrent commit to sweep — the race-free layer-2
	// fix for #3547 (layer 1 is LandReadbackEnv's honest refusal). DEFAULT ON since #3619
	// (TestConcurrentLandSoakIsolatedIsRaceFree witnesses it clean under -race while the
	// baseline soak reproduces the sweep); on ANY hiccup (detached HEAD, apply conflict,
	// lost CAS, unresolved identity, git error) it FALLS BACK to the baseline shared path,
	// so it can only ever REDUCE race exposure, never do worse. Set 0/false/off to force
	// the shared-index baseline as an escape.
	IsolatedLandEnv = "FAK_LAND_ISOLATED_INDEX"
	// IsolatedLandRetryEnv bounds how many compare-and-swap attempts landIsolated
	// makes before giving up and falling back to the baseline shared path (#3570).
	// A lost CAS (a peer landed in the gap) re-resolves HEAD, re-seeds the throwaway
	// index from the new base, re-applies the captured diff and re-attempts the CAS
	// instead of dropping straight into the racy shared-index fallback — the fallback
	// #3547 exists to avoid, which under contention would fire on nearly every land.
	// Unset/invalid/<1 → defaultIsolatedLandRetry attempts.
	IsolatedLandRetryEnv = "FAK_LAND_ISOLATED_RETRY"
	// defaultIsolatedLandRetry is the total CAS attempts (first try + retries) when
	// IsolatedLandRetryEnv is unset. Small: each retry means a peer is actively
	// landing, and the baseline fallback below still exists as the final resort.
	defaultIsolatedLandRetry = 5
	// LandReadbackMismatchToken is the structured refusal token landReadbackVerify
	// stamps on Result.Reason when a path-scoped commit does not carry the worker's
	// intended paths (the #3547 shared-index sweep). Exported so the dispatch land
	// seam routes exactly this transient race class into a bounded re-land (#3613)
	// instead of matching free text.
	LandReadbackMismatchToken = "LAND_READBACK_MISMATCH"
	keyHashLen                = 12
)

// LandRefusalRetryable reports whether a refused Land is worth re-attempting on
// the same worktree (#3613). True only for the readback-mismatch race class: a
// concurrent commit on the shared index swept the worker's paths, so the refusal
// is TRANSIENT — the worktree still holds the only copy of the diff, and a
// re-land on the moved HEAD re-applies it cleanly. Every other refusal (red
// verify, apply conflict, disambiguation invariant, git fault) is deterministic
// for the same inputs: replaying it cannot change the verdict.
func LandRefusalRetryable(reason string) bool {
	return strings.Contains(reason, LandReadbackMismatchToken)
}

// casRetrySleep sleeps a short jittered backoff before a lost-CAS retry so N
// colliding landers spread out instead of re-contending the ref in lockstep.
// Package var so unit tests stub it to a no-op (#3570).
var casRetrySleep = func(attempt int) {
	time.Sleep(time.Duration(rand.Intn(15*attempt)+1) * time.Millisecond)
}

// isolatedLandRetryCap reads IsolatedLandRetryEnv: the total bounded CAS attempts
// for one landIsolated call. Unset, unparsable, or <1 falls back to the default —
// fail-open to a sane bound, never to an unbounded loop.
func isolatedLandRetryCap() int {
	v := strings.TrimSpace(os.Getenv(IsolatedLandRetryEnv))
	if v == "" {
		return defaultIsolatedLandRetry
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return defaultIsolatedLandRetry
	}
	return n
}

// GitRunner runs one `git` subcommand under root and returns (rc, stdout). It
// never raises: an exec failure is reported as a non-zero rc so every caller
// fails open. Injectable so the whole path is testable against a fake.
type GitRunner func(root string, args []string) (int, string)

// IsolationBackend materializes and releases a worker's private writable tree.
// Materialize must preserve git linkage so `git diff <baseSHA>` run in the
// returned path observes the worker's complete patch; Release must remove only
// that materialization. The seam exists so follow-on backends can attack the
// roughly 450 MB per-worker cost of detached git worktrees tracked by #3165.
type IsolationBackend interface {
	Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result
	Release(root, wtPath string, git GitRunner) Result
}

type gitWorktree struct{}

var defaultIsolationBackend IsolationBackend = gitWorktree{}

// IsolationBackends returns all registered implementations for conformance tests.
func IsolationBackends() []IsolationBackend { return []IsolationBackend{gitWorktree{}} }

// Result is the fail-open outcome of a git-touching op. OK is the one bit callers
// branch on; the rest carries evidence for the record/log.
type Result struct {
	OK        bool   `json:"ok"`
	Path      string `json:"path,omitempty"`
	BaseSHA   string `json:"base_sha,omitempty"`
	Reused    bool   `json:"reused,omitempty"`
	Applied   bool   `json:"applied,omitempty"`
	Committed bool   `json:"committed,omitempty"`
	Removed   bool   `json:"removed,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
	// DroppedOutOfLane is the number of changed worktree paths outside the
	// caller-declared lease tree. Path-scoped land commits omit those paths;
	// surfacing the count makes that otherwise-silent loss observable (#4599).
	DroppedOutOfLane int                      `json:"dropped_out_of_lane,omitempty"`
	Disambiguation   *DisambiguationWitnesses `json:"disambiguation,omitempty"`
}

// VerifyHook is a build/adjudication witness run IN the isolated worktree before
// anything is applied to the trunk. It receives the worktree path and returns
// (ok, detail); a non-ok result refuses the land. Nil skips the witness (the
// caller's downstream `dos commit-audit` is then the only arm) — the dispatcher
// passes nil initially, exactly as the Python CLI's default `--verify off`.
type VerifyHook func(wtPath string) (ok bool, detail string)

// defaultGit shells `git` under root with the no-window flag on Windows, mirroring
// tools/worker_worktree.py._git. Never raises: an exec failure is rc 127.
func defaultGit(root string, args []string) (int, string) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		return 127, string(out)
	}
	return 0, string(out)
}

func run(git GitRunner, root string, args []string) (int, string) {
	if git == nil {
		git = defaultGit
	}
	return git(root, args)
}

// GitEnvRunner is a GitRunner that also overlays environment variables. Only the
// opt-in isolated-index land path needs it — to set GIT_INDEX_FILE so staging lands
// in a throwaway index instead of the shared one. Kept as a separate type so every
// existing caller and the entire default land path stay byte-for-byte untouched.
type GitEnvRunner func(root string, env map[string]string, args []string) (int, string)

// defaultGitEnv is defaultGit plus KEY=VALUE overlays on the inherited environment.
func defaultGitEnv(root string, env map[string]string, args []string) (int, string) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	base := os.Environ()
	for k, v := range env {
		base = append(base, k+"="+v)
	}
	cmd.Env = base
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		return 127, string(out)
	}
	return 0, string(out)
}

func runEnv(genv GitEnvRunner, root string, env map[string]string, args []string) (int, string) {
	if genv == nil {
		genv = defaultGitEnv
	}
	return genv(root, env, args)
}

// isolatedGitEnv is the env-aware runner Land's opt-in layer-2 path uses. A package
// seam (not a Land parameter) so the exported Land signature and all its dispatch
// callers stay unchanged; tests swap it to drive the isolated path against a fake.
var isolatedGitEnv GitEnvRunner = defaultGitEnv

// --------------------------------------------------------------------------- //
// PURE planners — path / name / env composition. Unit-tested without git.
// --------------------------------------------------------------------------- //

// safeKey hashes an arbitrary worker key (issue number, wave id, pid) to a flat
// path-safe token, so a hostile/odd key ("/", "\\", "..") can never escape the
// worktree root or collide a sibling.
func safeKey(key string) string {
	if key == "" {
		key = "worker"
	}
	sum := sha1.Sum([]byte(key))
	return hex.EncodeToString(sum[:])[:keyHashLen]
}

// DirName is the flat directory name for one worker's worktree:
// <marker>-<lane>-<hashed-key>. lane is sanitized to [A-Za-z0-9_.-]; the key is
// hashed, so the result is always one safe path segment.
func DirName(lane, key string) string {
	if lane == "" {
		lane = "lane"
	}
	var b strings.Builder
	for _, c := range lane {
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-' {
			b.WriteRune(c)
		} else {
			b.WriteRune('-')
		}
	}
	safeLane := b.String()
	if safeLane == "" {
		safeLane = "lane"
	}
	return WorktreeMarker + "-" + safeLane + "-" + safeKey(key)
}

// DefaultRoot is the parent dir under which per-worker worktrees are created:
// FLEET_WORKER_WORKTREE_ROOT if set, else a per-OS scratch location OUTSIDE the
// repo (LOCALAPPDATA/Fleet/worker-worktrees on Windows, TMP/Fleet/... elsewhere),
// so a worktree never shows up in the trunk's own `git status`.
func DefaultRoot() string {
	if override := strings.TrimSpace(os.Getenv(WorktreeRootEnv)); override != "" {
		return override
	}
	base := ""
	if v := os.Getenv("LOCALAPPDATA"); v != "" {
		base = v
	}
	if base == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, "Fleet", "worker-worktrees")
}

// Path is the absolute path one worker's isolated worktree will live at. An empty
// root uses DefaultRoot().
func Path(lane, key, root string) string {
	if root == "" {
		root = DefaultRoot()
	}
	return filepath.Join(root, DirName(lane, key))
}

// WorktreeEnv returns the child env that isolates a worker's BUILD to its own
// worktree, layered on top of whatever base the dispatcher already composed.
// Pointing GOCACHE/GOTMPDIR INSIDE the worktree is what makes "a broken build in
// one worker's worktree does not red another's" true. DISPATCH_WORKSPACE is
// repointed at the worktree so a worker that reads it operates on its isolated
// tree. Does not mutate base.
func WorktreeEnv(base map[string]string, wtDir string) map[string]string {
	env := make(map[string]string, len(base)+4)
	for k, v := range base {
		env[k] = v
	}
	env["DISPATCH_WORKSPACE"] = wtDir
	env[WorktreeDirEnv] = wtDir
	env["GOCACHE"] = filepath.Join(wtDir, ".gocache")
	env["GOTMPDIR"] = filepath.Join(wtDir, ".gotmp")
	return env
}

// parseWorktreePaths extracts the worktree paths from `git worktree list
// --porcelain` output — the pure half of the tracked-check and enumeration.
func parseWorktreePaths(porcelain string) []string {
	var out []string
	for _, line := range strings.Split(porcelain, "\n") {
		if strings.HasPrefix(line, "worktree ") {
			out = append(out, strings.TrimSpace(line[len("worktree "):]))
		}
	}
	return out
}

// IsWorkerWorktree is true when path is one of OUR per-worker worktrees (its
// basename carries the marker), so Reap only ever removes our worktrees and an
// auditor can enumerate the live wave without trusting a self-report.
func IsWorkerWorktree(path string) bool {
	name := filepath.Base(filepath.Clean(path))
	return name == WorktreeMarker || strings.HasPrefix(name, WorktreeMarker+"-")
}

func samePath(a, b string) bool {
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}

// --------------------------------------------------------------------------- //
// Git-touching create / reap / land — fail-open, injectable git runner.
// --------------------------------------------------------------------------- //

// TrunkHeadSHA is the current trunk HEAD sha to pin a detached worktree to, or ""
// on any git error (caller fails open: no worktree, worker runs in the shared
// trunk exactly as before).
func TrunkHeadSHA(root string, git GitRunner) string {
	rc, out := run(git, root, []string{"rev-parse", "HEAD"})
	sha := strings.TrimSpace(out)
	if rc == 0 && sha != "" {
		return sha
	}
	return ""
}

// Prepare creates ONE worker's isolated, DETACHED worktree at baseSHA (or trunk
// HEAD when baseSHA is empty). On OK the worker should run with cwd = Result.Path
// and the env from WorktreeEnv; on failure the dispatcher FAILS OPEN — runs the
// worker in the shared trunk exactly as before, so a worktree-layer fault never
// wedges a spawn. Detached on purpose: pinned to a SHA, never a branch, so git
// does not refuse it and it can never trip OFF_TRUNK. Idempotent: a target path
// already tracked as this worktree is reported Reused rather than re-added.
//
// With the warm pool on (PoolCapEnv, off by default) a miss on that same-key reuse
// falls to a POOL lease first: an idle worktree of this lane, re-pointed at base, also
// reported Reused — so a NEW worker can inherit a materialized tree, which the same-key
// check alone never allowed (#3572). Result.Path is authoritative; a leased member does
// NOT sit at Path(lane, key, wtRoot).
func Prepare(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	return PrepareWithBackend(root, lane, key, baseSHA, wtRoot, git, defaultIsolationBackend)
}

// PrepareWithBackend is the injectable form of Prepare.
func PrepareWithBackend(root, lane, key, baseSHA, wtRoot string, git GitRunner, backend IsolationBackend) Result {
	if backend == nil {
		backend = defaultIsolationBackend
	}
	return backend.Materialize(root, lane, key, baseSHA, wtRoot, git)
}

func (gitWorktree) Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	base := baseSHA
	if base == "" {
		base = TrunkHeadSHA(root, git)
	}
	if base == "" {
		return Result{OK: false, Reason: "could not resolve trunk HEAD (git error) — fail open"}
	}
	wt := Path(lane, key, wtRoot)
	if _, err := os.Stat(wt); err == nil {
		// Already prepared (a retry / re-dispatch). Reuse only if git still tracks
		// it, rather than erroring on `worktree add` over an existing dir.
		rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
		if rc == 0 {
			for _, p := range parseWorktreePaths(out) {
				if samePath(p, wt) {
					return Result{OK: true, Path: wt, BaseSHA: base, Reused: true}
				}
			}
		}
	}
	// #3572 warm pool: hand this worker an ALREADY-materialized idle worktree of its
	// lane (a fast reset to the new base) instead of paying `worktree add`'s full
	// checkout. Runs AFTER the same-lane+key reuse check above, which is the cheaper
	// hit, and falls straight through to the create path below on any miss — so with
	// the pool off (the default) this is one env read and nothing else changes.
	if k := PoolCap(); k > 0 {
		if res, ok := leasePooled(root, lane, base, wtRoot, git); ok {
			return res
		}
	}
	if err := os.MkdirAll(filepath.Dir(wt), 0o755); err != nil {
		return Result{OK: false, Path: wt, BaseSHA: base,
			Reason: "could not create worktree root: " + err.Error() + " — fail open"}
	}
	rc, out := run(git, root, []string{"worktree", "add", "--detach", wt, base})
	if rc != 0 {
		return Result{OK: false, Path: wt, BaseSHA: base,
			Reason: "git worktree add failed — fail open", Detail: tail(out, 500)}
	}
	return Result{OK: true, Path: wt, BaseSHA: base, Reused: false}
}

// Reap force-removes ONE worker's worktree after its change has LANDED (or it
// crashed). --force is honest: the worktree is throwaway editing space and its
// only durable output is the commit Land already placed on the trunk. Best-effort:
// a removal failure is reported, never raised, and a trailing `worktree prune`
// clears the admin record. Refuses any non-marker path as a guardrail.
//
// With the warm pool on (PoolCapEnv, off by default) the worktree is instead RETURNED
// to its lane's idle set while that lane is under the cap — reset clean and marked
// idle, so the next Prepare leases it rather than re-adding one (#3572). A return
// reports OK with Removed=false; overflow and every failure path force-remove as above.
func Reap(root, wtPath string, git GitRunner) Result {
	return ReapWithBackend(root, wtPath, git, defaultIsolationBackend)
}

// ReapWithBackend is the injectable form of Reap.
func ReapWithBackend(root, wtPath string, git GitRunner, backend IsolationBackend) Result {
	if backend == nil {
		backend = defaultIsolationBackend
	}
	return backend.Release(root, wtPath, git)
}

func (gitWorktree) Release(root, wtPath string, git GitRunner) Result {
	if !IsWorkerWorktree(wtPath) {
		return Result{OK: false, Path: wtPath, Removed: false,
			Reason: "refusing to reap a non-worker worktree"}
	}
	// #3572 warm pool: RETURN the worktree to its lane's idle set instead of destroying
	// it, up to the cap; past the cap (or on any hiccup) it force-removes exactly as
	// below. Removed stays false for a return — the worktree really is still there, and
	// a caller counting reclaimed dirs must not be told otherwise.
	if k := PoolCap(); k > 0 {
		if res, ok := returnPooled(root, wtPath, k, git); ok {
			return res
		}
	}
	rc, out := run(git, root, []string{"worktree", "remove", "--force", wtPath})
	removed := rc == 0
	run(git, root, []string{"worktree", "prune"})
	res := Result{OK: removed, Path: wtPath, Removed: removed}
	if !removed {
		res.Detail = tail(out, 300)
	}
	return res
}

// Land applies a worker's edits from its isolated worktree onto the TRUNK as one
// stamped, signed-off commit ON main — the serialized commit-to-trunk step that
// keeps OFF_TRUNK from ever tripping (#1334).
//
// It captures the worktree's FULL delta since baseSHA (not HEAD — a guarded
// worker commits IN its detached worktree, so `git diff HEAD` would be empty and
// the change would silently evaporate; diffing against the pinned base captures it
// whether committed, staged, or unstaged) and applies it to the trunk worktree,
// then commits -s by explicit path. The caller holds the lane lease, which
// serializes this so two workers never apply to the same leaf tree at once.
//
// commitMsgFile is the message for `git commit -s -F`; when empty, Land derives
// the message from the worktree tip (`git log -1 --format=%B`) so the landed trunk
// commit keeps the worker's own authored, stampable subject (which cites #N and
// carries the (fak <leaf>) trailer — the worker's acceptance contract). paths,
// when non-empty, scopes the commit to the worker's declared region — never an
// add -A. verify (when non-nil) is a witness run in the worktree before anything
// touches the trunk; a failed witness refuses the land. FAIL-OPEN on git errors.
//
// ONE exception to fail-open: the hard-self core lock (#5392, corelock.go). A diff
// touching a core-locked kernel path is REFUSED unless a maintenance witness
// resolves CONFIRMED — supplied by the WithCoreLockWitness option or by the
// CoreLockWitnessTrailer in the commit message. That gate runs before any apply, so
// a refused land leaves the trunk exactly as it found it.
// CountPathsOutsideTrees counts how many of `changed` fall outside every declared tree in
// `trees`. Both sides are normalised to forward slashes and stripped of leading/trailing
// separators first, and a tree matches a path exactly or as a "<tree>/" prefix; an empty
// tree entry matches nothing (so a stray "" can never swallow the whole diff).
//
// Exported because the out-of-lane count is a LEASE-SCOPE verdict, and `fak dispatch tick`'s
// witness sweep grades the same worker diffs with it. The two used to carry byte-identical
// copies, which is how a land-time refusal and its post-hoc witness silently disagree.
func CountPathsOutsideTrees(changed, trees []string) int {
	outside := 0
	for _, path := range changed {
		path = strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/")
		inside := false
		for _, tree := range trees {
			tree = strings.Trim(strings.ReplaceAll(tree, "\\", "/"), "/")
			if tree != "" && (path == tree || strings.HasPrefix(path, tree+"/")) {
				inside = true
				break
			}
		}
		if !inside {
			outside++
		}
	}
	return outside
}

func Land(root, wtPath, baseSHA, commitMsgFile string, paths []string, verify VerifyHook, git GitRunner, opts ...LandOption) Result {
	cfg := newLandConfig(opts)
	diffRef := baseSHA
	if diffRef == "" {
		diffRef = "HEAD"
	}
	rc, diff := run(git, wtPath, []string{"diff", diffRef})
	if rc != 0 {
		return Result{OK: false, Reason: "could not read worktree diff vs " + diffRef + " (git error) — fail open"}
	}
	if strings.TrimSpace(diff) == "" {
		// No net change since the base: the worker landed nothing. The caller's
		// commit-witness (dos commit-audit) decides whether the slot was productive.
		return Result{OK: true, Applied: false, Committed: false,
			Reason: "no net diff in worktree vs " + diffRef + " to land"}
	}
	// The landing pathset, read ONCE: it feeds both the out-of-lane ledger and the
	// hard-self core-lock gate below, so the two can never disagree about what this
	// land actually carries. An unreadable name list leaves it empty — the ledger
	// then counts 0 exactly as before, and the gate falls back to the captured
	// diff's own headers rather than treating "unreadable" as "nothing locked".
	namesRC, names := run(git, wtPath, []string{"diff", "--name-only", diffRef})
	if namesRC != 0 {
		names = ""
	}
	droppedOutOfLane := 0
	if len(paths) > 0 && names != "" {
		droppedOutOfLane = CountPathsOutsideTrees(strings.Fields(names), paths)
	}
	if verify != nil {
		if ok, detail := verify(wtPath); !ok {
			return Result{OK: false, Applied: false, Committed: false,
				Reason: "worktree verify failed, refusing to land: " + detail}
		}
	}
	// Resolve a message file: use the caller's, else materialize the worktree tip's
	// message to a temp file so the landed commit keeps the worker's own subject.
	msgFile, cleanup, err := resolveMsgFile(wtPath, commitMsgFile, git)
	if err != nil {
		return Result{OK: false, Applied: false, Committed: false,
			Reason: "could not resolve commit message: " + err.Error()}
	}
	defer cleanup()

	// HARD-SELF CORE LOCK (#5392). The last gate before anything touches the trunk:
	// a diff carrying a core-locked kernel path (internal/adjudicator/**,
	// internal/abi/**, internal/corelocks/**) lands only with a maintenance witness
	// that resolves CONFIRMED — supplied by WithCoreLockWitness (the CLI flag) or by
	// the CoreLockWitnessTrailer in the commit message. This runs HERE, in the
	// lander, because the sanctioned worktree cannot use `fak commit` (OFF_TRUNK on a
	// detached HEAD) and the default isolated path commits through commit-tree, which
	// runs no git hook. Refusing here leaves the trunk index, worktree and HEAD
	// untouched; the worker's diff stays in its worktree.
	if refusal, fired := coreLockLandGate(root, names, diff, msgFile, cfg, git); fired {
		refusal.DroppedOutOfLane = droppedOutOfLane
		return refusal
	}

	// Opt-in race-free layer-2 land (default OFF): stage+commit through a THROWAWAY
	// index so the shared index is never a sweep target. handled=false means it could
	// not isolate safely (detached HEAD, apply conflict, lost CAS, …) and falls through
	// to the baseline shared path below — so enabling it only ever reduces the #3547
	// race window, never regresses it. Path-scoped lands only (a whole-tree land has no
	// safe isolated form here).
	if isolatedLandEnabled() && len(paths) > 0 {
		if res, handled := landIsolated(root, wtPath, diff, msgFile, paths, git, isolatedGitEnv); handled {
			res.DroppedOutOfLane = droppedOutOfLane
			return res
		}
	}

	applied := gitApply(root, diff, git)
	if !applied.OK {
		return Result{OK: false, Applied: false, Committed: false,
			Reason: "git apply to trunk failed", Detail: applied.Detail}
	}
	commitArgs := []string{"commit", "-s", "-F", msgFile}
	if len(paths) > 0 {
		commitArgs = append(commitArgs, "--")
		commitArgs = append(commitArgs, paths...)
	}
	rc, out := run(git, root, commitArgs)
	res := Result{OK: rc == 0, Applied: true, Committed: rc == 0, Detail: tail(out, 300), DroppedOutOfLane: droppedOutOfLane}
	// Opt-in honest-refusal readback (default OFF): confirm the commit we just made
	// actually carries our intended paths. A missing path means our staged change
	// was swept into a concurrent commit on the shared index (#3547); refuse rather
	// than return a false success. FAIL-OPEN — only a positive mismatch flips OK.
	if rc == 0 && len(paths) > 0 && landReadbackEnabled() {
		if ok, reason := landReadbackVerify(root, paths, git); !ok {
			res.OK = false
			res.Reason = reason
		}
	}
	return res
}

// resolveMsgFile returns a commit-message file path and a cleanup func. A non-empty
// commitMsgFile is used verbatim (no cleanup). Otherwise the worktree tip message
// (git log -1 --format=%B) is written to a temp file so the trunk commit preserves
// the worker's own stamped subject; the temp file is removed by cleanup.
func resolveMsgFile(wtPath, commitMsgFile string, git GitRunner) (string, func(), error) {
	if strings.TrimSpace(commitMsgFile) != "" {
		return commitMsgFile, func() {}, nil
	}
	rc, out := run(git, wtPath, []string{"log", "-1", "--format=%B"})
	msg := strings.TrimRight(out, "\n")
	if rc != 0 || strings.TrimSpace(msg) == "" {
		// No worktree commit to borrow a subject from (worker left the change
		// staged/unstaged). A conservative fallback keeps the land honest: it is a
		// plain, un-stamped subject that dos commit-audit will grade as ABSTAIN
		// rather than a forged claim.
		msg = "chore(dispatch): land worker worktree diff"
	}
	f, err := os.CreateTemp("", "fak-wt-msg-*.txt")
	if err != nil {
		return "", func() {}, err
	}
	if _, err := f.WriteString(msg + "\n"); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// gitApply applies a captured diff to root's working tree via `git apply`, reading
// the patch from a temp file (a long diff exceeds argv limits) and removing it
// after. Kept separate so the apply step is isolated and testable.
func gitApply(root, diff string, git GitRunner) Result {
	patch, cleanup, err := writePatch(diff)
	if err != nil {
		return Result{OK: false, Detail: "could not stage patch temp: " + err.Error()}
	}
	defer cleanup()
	rc, out := run(git, root, []string{"apply", "--whitespace=nowarn", patch})
	return Result{OK: rc == 0, Detail: tail(out, 300)}
}

// writePatch writes a captured diff (newline-terminated) to a temp file and returns
// its path plus a cleanup func. Shared by the baseline working-tree apply and the
// isolated-index `apply --cached`, so both stage byte-identical patch content.
func writePatch(diff string) (string, func(), error) {
	f, err := os.CreateTemp("", "fak-wt-land-*.patch")
	if err != nil {
		return "", func() {}, err
	}
	body := diff
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// landIsolated is the race-free layer-2 land (#3547). It stages the worker diff into a
// THROWAWAY index seeded from the current trunk HEAD, commits it as a child of that
// exact HEAD via commit-tree, then advances the branch with a compare-and-swap
// update-ref. Two properties fall out:
//   - Staging never touches the SHARED index, so no concurrent `commit -- paths` can
//     sweep our change into its own commit (the false-success layer 1 only detects).
//   - The ref moves ONLY if HEAD has not advanced since we seeded, so a commit built on
//     a stale base can never silently revert a peer's concurrent work — a lost CAS
//     falls back instead.
//
// A LOST CAS is not a fallback trigger anymore (#3570): a peer landing in the gap
// just moved the base, so the loop below re-resolves HEAD, re-seeds the throwaway
// index from the new base, re-applies the same captured diff and re-attempts the
// CAS — optimistic concurrency, bounded by IsolatedLandRetryEnv with a short
// jittered backoff. Under contention the old behavior collapsed nearly every land
// into the racy shared-index fallback precisely when isolation mattered most.
//
// Returns (result, handled). handled=false means "could not isolate safely — use the
// baseline shared path": detached HEAD, unresolved identity, apply conflict (a
// GENUINE same-path overlap, including a conflicting re-apply after a lost CAS),
// exhausted CAS attempts, or any git error. Thus enabling this can only REDUCE the
// race window on the happy path, never regress the baseline. On success the shared
// working tree is synced for `paths` (git checkout <new> -- paths) so trunk builders
// see the landed change, matching the baseline post-state; a sync hiccup is reported
// but does NOT unland.
func landIsolated(root, wtPath, diff, msgFile string, paths []string, git GitRunner, genv GitEnvRunner) (Result, bool) {
	// The branch to move. Detached HEAD → no branch ref to CAS safely; fall back.
	rc, ref := run(git, root, []string{"symbolic-ref", "--quiet", "HEAD"})
	branch := strings.TrimSpace(ref)
	if rc != 0 || branch == "" {
		return Result{}, false
	}
	// The exact base our commit parents AND the compare-and-swap old-value.
	rc, head := run(git, root, []string{"rev-parse", "HEAD"})
	oldHEAD := strings.TrimSpace(head)
	if rc != 0 || oldHEAD == "" {
		return Result{}, false
	}
	// commit-tree runs no hook and adds no signoff; compose Signed-off-by ourselves to
	// preserve the baseline `commit -s`. Unresolved identity → fall back (can't honor -s).
	_, name := run(git, root, []string{"config", "user.name"})
	_, email := run(git, root, []string{"config", "user.email"})
	nm, em := strings.TrimSpace(name), strings.TrimSpace(email)
	if nm == "" || em == "" {
		return Result{}, false
	}

	// Throwaway index at a fresh path git creates via read-tree; removed on return.
	idxF, err := os.CreateTemp("", "fak-land-*.index")
	if err != nil {
		return Result{}, false
	}
	idx := idxF.Name()
	idxF.Close()
	os.Remove(idx) // read-tree writes it fresh; a pre-existing empty file would also do
	defer os.Remove(idx)
	env := map[string]string{"GIT_INDEX_FILE": idx}

	// The captured diff and the signed message are attempt-invariant: write them once
	// so every CAS attempt stages byte-identical content under the same subject.
	patch, cleanupPatch, err := writePatch(diff)
	if err != nil {
		return Result{}, false
	}
	defer cleanupPatch()
	ctMsg, cleanupMsg, err := composeSignedMsg(msgFile, nm, em)
	if err != nil {
		return Result{}, false
	}
	defer cleanupMsg()

	// Bounded optimistic-concurrency loop (#3570): each attempt seeds the throwaway
	// index from the CURRENT base, builds the commit as a child of that exact base,
	// and CASes the branch forward. Only a lost CAS loops; every other hiccup still
	// falls back immediately, exactly as before.
	attempts := isolatedLandRetryCap()
	var disambiguation *DisambiguationWitnesses
	for attempt := 1; attempt <= attempts; attempt++ {
		if attempt > 1 {
			// A peer is actively landing: back off briefly, then re-resolve the base
			// the peer just moved so this attempt re-builds on the NEW HEAD.
			casRetrySleep(attempt)
			rc, head := run(git, root, []string{"rev-parse", "HEAD"})
			oldHEAD = strings.TrimSpace(head)
			if rc != 0 || oldHEAD == "" {
				return Result{}, false
			}
		}
		// Seed the throwaway index with the current trunk HEAD's tree.
		if rc, _ := runEnv(genv, root, env, []string{"read-tree", oldHEAD}); rc != 0 {
			return Result{}, false
		}
		// Stage the worker diff into the throwaway index ONLY (--cached never touches
		// the working tree). A conflict here — first try or re-apply after a lost CAS —
		// means a concurrent change to the SAME paths; let the baseline path adjudicate
		// it exactly as today rather than force it.
		if rc, _ := runEnv(genv, root, env, []string{"apply", "--cached", "--whitespace=nowarn", patch}); rc != 0 {
			return Result{}, false
		}
		rc, tree := runEnv(genv, root, env, []string{"write-tree"})
		treeSHA := strings.TrimSpace(tree)
		if rc != 0 || treeSHA == "" {
			return Result{}, false
		}
		disambiguation = nil
		if disambiguationRelevant(paths) {
			var valid bool
			disambiguation, valid = verifyAppliedDisambiguation(root, wtPath, treeSHA)
			if !valid {
				return Result{OK: false, Path: root, Reason: "post-apply disambiguation invariant failed", Detail: disambiguation.compactDetail(), Disambiguation: disambiguation}, true
			}
		}
		rc, commit := runEnv(genv, root, env, []string{"commit-tree", treeSHA, "-p", oldHEAD, "-F", ctMsg})
		newCommit := strings.TrimSpace(commit)
		if rc != 0 || newCommit == "" {
			return Result{}, false
		}
		// Name the off-branch commit before trunk CAS. A process crash from here on
		// leaves an observable, GC-safe recovery candidate instead of a dangling SHA.
		recoveryRef, anchorErr := AnchorRecoveryEntry(root, wtPath, newCommit, func(r string, a []string) (int, string) { return runEnv(genv, r, env, a) })
		if anchorErr != nil {
			return Result{OK: false, Reason: "isolated land recovery anchor failed — trunk unchanged", Detail: anchorErr.Error()}, true
		}
		// Compare-and-swap: move the branch ONLY if HEAD is still oldHEAD. A peer commit
		// in the gap fails this → retry on the peer's new HEAD (#3570); the throwaway
		// commit built on the stale base is simply abandoned, unreferenced.
		if rc, _ := run(git, root, []string{"update-ref", branch, newCommit, oldHEAD}); rc != 0 {
			continue
		}
		// The ref moved but the shared working tree still holds OLD content for `paths`
		// (we never touched it). Sync just those paths so trunk builders see the landed
		// change, matching the baseline post-state. A sync failure does NOT unland.
		detail := "cas-attempts=" + strconv.Itoa(attempt) + "/" + strconv.Itoa(attempts) + "; recovery-ref=" + recoveryRef
		coArgs := append([]string{"checkout", newCommit, "--"}, paths...)
		if rc, out := run(git, root, coArgs); rc != 0 {
			detail += "; landed " + shortSHA(newCommit) + " but working-tree sync failed: " + tail(out, 200)
		}
		return Result{OK: true, Applied: true, Committed: true,
			Reason: "isolated-index land " + shortSHA(newCommit) + " (race-free, #3547)",
			Detail: detail, Disambiguation: disambiguation}, true
	}
	// Every bounded attempt lost its CAS — genuine sustained contention. Fall back to
	// the baseline shared path as the final resort rather than loop unbounded.
	return Result{}, false
}

// composeSignedMsg writes msgFile's content to a new temp file with a Signed-off-by
// trailer appended when absent, so a commit-tree land keeps the baseline `-s` signoff.
// Returns the temp path and a cleanup func.
func composeSignedMsg(msgFile, name, email string) (string, func(), error) {
	raw, err := os.ReadFile(msgFile)
	if err != nil {
		return "", func() {}, err
	}
	body := string(raw)
	signoff := "Signed-off-by: " + name + " <" + email + ">"
	if !strings.Contains(body, signoff) {
		if !strings.HasSuffix(body, "\n") {
			body += "\n"
		}
		body += "\n" + signoff + "\n"
	}
	f, err := os.CreateTemp("", "fak-land-msg-*.txt")
	if err != nil {
		return "", func() {}, err
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", func() {}, err
	}
	f.Close()
	return f.Name(), func() { os.Remove(f.Name()) }, nil
}

// landReadbackEnabled reports whether the post-commit readback (LandReadbackEnv)
// is on. DEFAULT ON since #3619; only an explicit 0/false/off forces it off.
func landReadbackEnabled() bool { return envDefaultOn(LandReadbackEnv) }

// isolatedLandEnabled reports whether the race-free isolated-index land
// (IsolatedLandEnv) is on. DEFAULT ON since #3619; only an explicit 0/false/off
// forces the shared-index baseline.
func isolatedLandEnabled() bool { return envDefaultOn(IsolatedLandEnv) }

// envDefaultOn reads a boolean gate that DEFAULTS ON: unset/empty counts as on;
// only an explicit 0/false/off (any case) forces it off. The shape a gate takes
// once its soak has proven it safe to default on (#3619), keeping the env as an
// operator escape hatch back to the old behavior.
func envDefaultOn(name string) bool {
	v := strings.TrimSpace(os.Getenv(name))
	return v == "" || (v != "0" && !strings.EqualFold(v, "false") && !strings.EqualFold(v, "off"))
}

// landReadbackVerify confirms trunk HEAD, right after a path-scoped commit, carries
// EVERY intended path. A missing path means the worker's staged change was swept
// into a concurrent commit on the shared index (the #3547 race) — reported as a
// LAND_READBACK_MISMATCH refusal. FAIL-OPEN: any git error yields ok=true, so a
// readback that cannot run never manufactures a refusal. A directory pathspec is
// satisfied by any committed file beneath it.
func landReadbackVerify(root string, paths []string, git GitRunner) (bool, string) {
	rc, head := run(git, root, []string{"rev-parse", "HEAD"})
	if rc != 0 {
		return true, ""
	}
	sha := strings.TrimSpace(head)
	rc, out := run(git, root, []string{"diff-tree", "--no-commit-id", "--name-only", "-r", sha})
	if rc != 0 {
		return true, ""
	}
	committed := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		if p := normalizeSlash(strings.TrimSpace(line)); p != "" {
			committed[p] = true
		}
	}
	var missing []string
	for _, p := range paths {
		want := normalizeSlash(strings.TrimSpace(p))
		if want == "" {
			continue
		}
		found := committed[want]
		if !found {
			for c := range committed {
				if strings.HasPrefix(c, want+"/") {
					found = true
					break
				}
			}
		}
		if !found {
			missing = append(missing, p)
		}
	}
	if len(missing) > 0 {
		return false, LandReadbackMismatchToken + ": trunk HEAD " + shortSHA(sha) +
			" does not carry intended path(s) " + strings.Join(missing, ", ") +
			" after commit — shared-index race, land not trusted (#3547)"
	}
	return true, ""
}

func normalizeSlash(p string) string { return strings.ReplaceAll(p, "\\", "/") }

func shortSHA(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// Count enumerates the live per-worker worktrees from `git worktree list` — the
// direct check of the done-condition, read from git not a self-report.
func Count(root string, git GitRunner) (int, []string) {
	rc, out := run(git, root, []string{"worktree", "list", "--porcelain"})
	if rc != 0 {
		return 0, nil
	}
	var ours []string
	for _, p := range parseWorktreePaths(out) {
		if IsWorkerWorktree(p) {
			ours = append(ours, p)
		}
	}
	return len(ours), ours
}

// tail clamps a git/gh transcript to its last n bytes for a Result.Detail field —
// the diagnosis is at the END of a failing command's output. strmatch.Tail owns the
// one definition; `fak release status` carried a byte-identical private copy.
func tail(s string, n int) string { return strmatch.Tail(s, n) }
