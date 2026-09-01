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
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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
	// ReapTimeoutExitCode is returned by BoundedGitRunner when the shared reap
	// deadline expires. It is intentionally distinct from ordinary git failures so
	// the CLI can emit a stable refusal instead of free-text diagnosis.
	ReapTimeoutExitCode = 124

	ReapCodeDirtyWorktreeRefused   = "DIRTY_WORKTREE_REFUSED"
	ReapCodeProofRefused           = "REAP_PROOF_REFUSED"
	ReapCodeVerifiedWorktreeReaped = "VERIFIED_WORKTREE_REAPED"
	ReapCodeTimeout                = "REAP_TIMEOUT"
	ReapCodeRemoved                = "WORKTREE_REAPED"
	ReapCodeReleased               = "WORKTREE_RELEASED"
	keyHashLen                     = 12
)

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

// ownedIsolationBackend lets a backend persist the exact owner as part of an
// atomic lease reservation before it touches a pooled path. Backends that do not
// implement it retain the original Materialize contract and are stamped afterward.
type ownedIsolationBackend interface {
	MaterializeOwned(root, lane, key, baseSHA, wtRoot string, git GitRunner, owner OwnerStamp) Result
}

type gitWorktree struct{}

var defaultIsolationBackend IsolationBackend = nativeIsolationBackend()

// IsolationBackends returns all registered implementations for conformance tests.
func IsolationBackends() []IsolationBackend {
	backends := []IsolationBackend{gitWorktree{}}
	if _, ok := nativeIsolationBackend().(blockClone); ok {
		backends = append(backends, newBlockCloneBackend())
	}
	return backends
}

// Result is the fail-open outcome of a git-touching op. OK is the one bit callers
// branch on; the rest carries evidence for the record/log.
type Result struct {
	OK        bool   `json:"ok"`
	Code      string `json:"code,omitempty"`
	Backend   string `json:"backend,omitempty"`
	Path      string `json:"path,omitempty"`
	BaseSHA   string `json:"base_sha,omitempty"`
	Reused    bool   `json:"reused,omitempty"`
	Applied   bool   `json:"applied,omitempty"`
	Committed bool   `json:"committed,omitempty"`
	Removed   bool   `json:"removed,omitempty"`
	Preserved bool   `json:"preserved,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
	// SupersededBy is populated only after the named commit is proven to be on
	// current trunk ancestry and byte-equivalent to the managed worktree.
	SupersededBy string `json:"superseded_by,omitempty"`
	// DroppedOutOfLane is the number of changed worktree paths outside the
	// caller-declared lease tree. Path-scoped land commits omit those paths;
	// surfacing the count makes that otherwise-silent loss observable (#4599).
	DroppedOutOfLane int                      `json:"dropped_out_of_lane,omitempty"`
	Disambiguation   *DisambiguationWitnesses `json:"disambiguation,omitempty"`
	RecoveryRef      string                   `json:"recovery_ref,omitempty"`
	RemoteRecovery   *RemoteReadback          `json:"remote_recovery,omitempty"`
	Cost             *LandCostReceipt         `json:"cost,omitempty"`
	// pooled is internal lifecycle evidence: unlike generic same-key reuse, this
	// Prepare exclusively reserved an idle member and may safely destroy it if the
	// post-materialization owner/state write fails.
	pooled bool
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
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err != nil {
		out := stdout.String() + stderr.String()
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), out
		}
		return 127, out
	}
	// Successful git commands may still emit advisory warnings on stderr (for
	// example core.autocrlf conversion notices). Machine-readable callers consume
	// stdout as names, SHAs, or patches, so mixing those warnings into success data
	// can corrupt a patch and invent paths. Preserve stderr only on failure.
	return 0, stdout.String()
}

// BoundedGitRunner binds every git subprocess to one caller-owned deadline. The
// shared context makes the whole checked reap bounded rather than granting each
// probe a fresh timeout. WaitDelay also closes inherited pipes after cancellation,
// so a stuck helper grandchild cannot keep CombinedOutput waiting indefinitely.
func BoundedGitRunner(ctx context.Context) GitRunner {
	return func(root string, args []string) (int, string) {
		if err := ctx.Err(); err != nil {
			return ReapTimeoutExitCode, err.Error()
		}
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = root
		cmd.WaitDelay = 100 * time.Millisecond
		windowgate.ConfigureBackgroundCommand(cmd)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		err := cmd.Run()
		if ctx.Err() != nil {
			return ReapTimeoutExitCode, ctx.Err().Error()
		}
		if err != nil {
			out := stdout.String() + stderr.String()
			if ee, ok := err.(*exec.ExitError); ok {
				return ee.ExitCode(), out
			}
			return 127, out
		}
		return 0, stdout.String()
	}
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
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err != nil {
		out := stdout.String() + stderr.String()
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), out
		}
		return 127, out
	}
	return 0, stdout.String()
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

// EnsureBuildDirs recreates the disposable Go build directories owned by a
// managed worktree and returns the matching child environment. Reapers and
// manual disk maintenance may remove either directory while the source
// worktree remains valid, so callers must run this immediately before invoking
// Go rather than relying on prepare-time state. The worktree itself must already
// exist: refusing to recreate it keeps a reaped checkout from being resurrected
// as an empty directory.
func EnsureBuildDirs(wtDir string) (map[string]string, error) {
	info, err := os.Stat(wtDir)
	if err != nil {
		return nil, fmt.Errorf("stat managed worktree %q: %w", wtDir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("managed worktree %q is not a directory", wtDir)
	}

	env := WorktreeEnv(nil, wtDir)
	for _, name := range []string{"GOCACHE", "GOTMPDIR"} {
		path := env[name]
		if err := os.MkdirAll(path, 0o755); err != nil {
			return nil, fmt.Errorf("create %s directory %q: %w", name, path, err)
		}
	}
	return env, nil
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
	return strings.EqualFold(canonicalComparisonPath(a), canonicalComparisonPath(b))
}

func canonicalComparisonPath(path string) string {
	clean := filepath.Clean(path)
	if resolved, err := filepath.EvalSymlinks(clean); err == nil {
		return filepath.Clean(resolved)
	}
	return clean
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
// With the warm pool on (PoolCapEnv, a small default; 0 disables) a miss on that same-key reuse
// falls to a POOL lease first: an idle worktree of this lane, re-pointed at base, also
// reported Reused — so a NEW worker can inherit a materialized tree, which the same-key
// check alone never allowed (#3572). Result.Path is authoritative; a leased member does
// NOT sit at Path(lane, key, wtRoot).
func Prepare(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	return PrepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot, git, defaultIsolationBackend, defaultOwnerStamp(lane))
}

// PrepareOwned is Prepare with an explicit owner stamp. Dispatch/CLI seams that know
// the exact lease identity use this form; older callers retain Prepare's conservative
// coarse-lane stamp.
func PrepareOwned(root, lane, key, baseSHA, wtRoot string, git GitRunner, owner OwnerStamp) Result {
	return PrepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot, git, defaultIsolationBackend, owner)
}

// PrepareOwnedBounded is the launch-facing prepare path. One total budget bounds
// materialization and readiness checks, and metadata is withheld until the detached
// checkout is proven complete and clean.
func PrepareOwnedBounded(root, lane, key, baseSHA, wtRoot string, owner OwnerStamp, budget time.Duration) Result {
	if budget <= 0 {
		budget = 2 * time.Minute
	}
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	git := BoundedGitRunner(ctx)
	base := strings.TrimSpace(baseSHA)
	revision := "HEAD"
	if base != "" {
		revision = base + "^{commit}"
	}
	rc, out := run(git, root, []string{"rev-parse", revision})
	if rc == ReapTimeoutExitCode || ctx.Err() != nil {
		return prepareTimeoutResult(Path(lane, key, wtRoot), baseSHA, "resolving base", ctx.Err())
	}
	if rc == 0 {
		base = strings.TrimSpace(out)
	}
	res := prepareOwnedWithBackend(root, lane, key, base, wtRoot, git, defaultIsolationBackend, owner, true)
	if !res.OK && res.Path != "" && (res.Code == "PREPARE_TIMEOUT" || ctx.Err() != nil) {
		cleanupPartialPrepareBounded(root, res.Path)
	}
	if ctx.Err() != nil && !res.OK {
		return prepareTimeoutResult(res.Path, base, "materialization/readiness", ctx.Err())
	}
	return res
}

func prepareTimeoutResult(path, base, phase string, err error) Result {
	detail := "prepare deadline exceeded"
	if err != nil {
		detail = err.Error()
	}
	return Result{OK: false, Code: "PREPARE_TIMEOUT", Path: path, BaseSHA: base,
		Reason: "worker worktree prepare timed out during " + phase + " - no ready receipt emitted",
		Detail: detail}
}

// PrepareWithBackend is the injectable form of Prepare.
func PrepareWithBackend(root, lane, key, baseSHA, wtRoot string, git GitRunner, backend IsolationBackend) Result {
	return PrepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot, git, backend, defaultOwnerStamp(lane))
}

// PrepareOwnedWithBackend combines the backend seam with an explicit owner stamp.
func PrepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot string, git GitRunner, backend IsolationBackend, owner OwnerStamp) Result {
	return prepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot, git, backend, owner, false)
}

func prepareOwnedWithBackend(root, lane, key, baseSHA, wtRoot string, git GitRunner, backend IsolationBackend, owner OwnerStamp, verifyReady bool) Result {
	if backend == nil {
		backend = defaultIsolationBackend
	}
	owner = normalizeOwnerStamp(owner)
	var res Result
	if owned, ok := backend.(ownedIsolationBackend); ok {
		res = owned.MaterializeOwned(root, lane, key, baseSHA, wtRoot, git, owner)
	} else {
		res = backend.Materialize(root, lane, key, baseSHA, wtRoot, git)
	}
	if !res.OK || res.Path == "" {
		if verifyReady && res.Path != "" && res.Code == "PREPARE_TIMEOUT" {
			cleanupPartialPrepare(root, res.Path, git)
			return prepareTimeoutResult(res.Path, res.BaseSHA, "materialization", fmt.Errorf("%s", res.Detail))
		}
		return res
	}
	if verifyReady {
		if ready := verifyPreparedWorktree(res, git); !ready.OK {
			cleanupPartialPrepare(root, res.Path, git)
			return ready
		}
	}
	metadataErr := writeOwnerStamp(res.Path, owner)
	if metadataErr == nil && PoolCap() > 0 && isGitWorktreeBackend(backend) {
		metadataErr = recordPoolLease(res.Path, lane, owner)
	}
	if metadataErr != nil {
		// A newly-materialized worktree without its owner stamp is invisible to the
		// owner-dead GC, and a pooled tree without its leased state can be handed to a
		// second worker. Clean new or exclusively-reserved pool members at the source.
		// A generic same-key reuse may still carry live WIP, so fail toward keeping it.
		cleanupDetail := ""
		if !res.Reused || res.pooled {
			var cleanup Result
			if isGitWorktreeBackend(backend) {
				cleanup = ForceReap(root, res.Path, git)
			} else {
				cleanup = backend.Release(root, res.Path, git)
			}
			if !cleanup.Removed {
				cleanupDetail = "; cleanup failed: " + cleanup.Reason + " " + cleanup.Detail
			}
		}
		res.OK = false
		res.Reason = "could not persist worktree owner/pool metadata — fail open"
		res.Detail = metadataErr.Error() + cleanupDetail
	}
	return res
}

func verifyPreparedWorktree(res Result, git GitRunner) Result {
	fail := func(code, reason, detail string) Result {
		return Result{OK: false, Code: code, Path: res.Path, BaseSHA: res.BaseSHA,
			Reason: reason + " - no ready receipt emitted", Detail: detail}
	}
	rc, out := run(git, res.Path, []string{"rev-parse", "HEAD"})
	if rc == ReapTimeoutExitCode {
		return fail("PREPARE_TIMEOUT", "worker worktree prepare timed out during HEAD verification", tail(out, 500))
	}
	if rc != 0 || strings.TrimSpace(out) != strings.TrimSpace(res.BaseSHA) {
		return fail("PREPARE_NOT_READY", "worker worktree HEAD does not match requested base", tail(out, 500))
	}
	rc, out = run(git, res.Path, []string{"status", "--porcelain"})
	if rc == ReapTimeoutExitCode {
		return fail("PREPARE_TIMEOUT", "worker worktree prepare timed out during clean-index verification", tail(out, 500))
	}
	if rc != 0 || strings.TrimSpace(out) != "" {
		return fail("PREPARE_NOT_READY", "worker worktree index or checkout is not clean", tail(out, 500))
	}
	rc, gitDir := run(git, res.Path, []string{"rev-parse", "--git-dir"})
	if rc != 0 {
		return fail("PREPARE_NOT_READY", "worker worktree git directory could not be resolved", tail(gitDir, 500))
	}
	gitDir = strings.TrimSpace(gitDir)
	if !filepath.IsAbs(gitDir) {
		gitDir = filepath.Join(res.Path, gitDir)
	}
	lock := filepath.Join(gitDir, "index.lock")
	if _, err := os.Stat(lock); err == nil {
		return fail("PREPARE_NOT_READY", "worker worktree index lock is still present", lock)
	} else if !os.IsNotExist(err) {
		return fail("PREPARE_NOT_READY", "worker worktree index lock state is unknown", err.Error())
	}
	res.OK = true
	return res
}

// cleanupPartialPrepare names only the path derived for this prepare attempt.
// It never scans or reaps neighboring worker trees.
func cleanupPartialPrepareBounded(root, path string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cleanupPartialPrepare(root, path, BoundedGitRunner(ctx))
}

func cleanupPartialPrepare(root, path string, git GitRunner) {
	if path == "" || !IsWorkerWorktree(path) {
		return
	}
	_, _ = run(git, root, []string{"worktree", "remove", "--force", path})
	_, _ = run(git, root, []string{"worktree", "prune"})

}

func isGitWorktreeBackend(backend IsolationBackend) bool {
	switch backend.(type) {
	case gitWorktree, *gitWorktree:
		return true
	default:
		return false
	}
}

func (gitWorktree) Materialize(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
	return gitWorktree{}.MaterializeOwned(root, lane, key, baseSHA, wtRoot, git, defaultOwnerStamp(lane))
}

func (gitWorktree) MaterializeOwned(root, lane, key, baseSHA, wtRoot string, git GitRunner, owner OwnerStamp) Result {
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
					if PoolCap() > 0 {
						// An idle member at the same derived path is a NEW lease, not
						// the historical retry case: reserve and sanitize it first.
						if meta, err := readPoolMember(wt); err == nil && meta.State == poolStateIdle {
							if res, ok := leaseSpecificPooled(root, wt, base, git, owner); ok {
								return res
							}
							return Result{OK: false, Path: wt, BaseSHA: base,
								Reason: "same-key idle pool member could not be leased — fail open"}
						}
					}
					return Result{OK: true, Path: wt, BaseSHA: base, Reused: true}
				}
			}
		}
	}
	// #3572 warm pool: hand this worker an ALREADY-materialized idle worktree of its
	// lane (a fast reset to the new base) instead of paying `worktree add`'s full
	// checkout. Runs AFTER the same-lane+key reuse check above, which is the cheaper
	// hit, and falls straight through to the create path below on any miss. Setting
	// PoolCapEnv=0 preserves the old create path without pool state or reset/clean.
	if k := PoolCap(); k > 0 {
		if res, ok := leasePooled(root, lane, base, wtRoot, git, owner); ok {
			return res
		}
	}
	if err := os.MkdirAll(filepath.Dir(wt), 0o755); err != nil {
		return Result{OK: false, Path: wt, BaseSHA: base,
			Reason: "could not create worktree root: " + err.Error() + " — fail open"}
	}
	rc, out := run(git, root, []string{"worktree", "add", "--detach", wt, base})
	if rc != 0 {
		code := ""
		if rc == ReapTimeoutExitCode {
			code = "PREPARE_TIMEOUT"
		}
		return Result{OK: false, Code: code, Path: wt, BaseSHA: base,
			Reason: "git worktree add failed - fail open", Detail: tail(out, 500)}
	}
	return Result{OK: true, Path: wt, BaseSHA: base, Reused: false}
}

// Reap force-removes ONE worker's worktree after its change has LANDED (or it
// crashed). --force is honest: the worktree is throwaway editing space and its
// only durable output is the commit Land already placed on the trunk. Best-effort:
// a removal failure is reported, never raised, and a trailing `worktree prune`
// clears the admin record. Refuses any non-marker path as a guardrail.
//
// With the warm pool on (PoolCapEnv, small by default; 0 disables) the worktree is instead RETURNED
// to its lane's idle set while that lane is under the cap — reset clean and marked
// idle, so the next Prepare leases it rather than re-adding one (#3572). A return
// reports OK with Removed=false; overflow and every failure path force-remove as above.
func Reap(root, wtPath string, git GitRunner) Result {
	return ReapWithBackend(root, wtPath, git, defaultIsolationBackend)
}

// ReapChecked is the path-local safety gate for the single-worktree CLI. A dirty
// worktree is preserved unless the caller names a commit that is independently
// proven to be on current trunk ancestry and to contain the exact checkout state.
// The caller supplies BoundedGitRunner when it needs a wall-clock deadline.
func ReapChecked(root, wtPath, supersededBy string, git GitRunner) Result {
	wtPath = filepath.Clean(strings.TrimSpace(wtPath))
	refuse := func(code, reason, detail string) Result {
		return Result{
			OK: false, Code: code, Path: wtPath, Preserved: true,
			Reason: reason, Detail: strings.TrimSpace(detail),
		}
	}
	gitFailure := func(reason string, rc int, detail string) Result {
		if rc == ReapTimeoutExitCode {
			return refuse(ReapCodeTimeout, "single-worktree reap deadline exceeded", detail)
		}
		return refuse(ReapCodeProofRefused, reason, detail)
	}
	if !IsWorkerWorktree(wtPath) {
		return refuse("REAP_PATH_REFUSED", "refusing to reap a non-worker worktree", "")
	}

	rc, status := run(git, wtPath, []string{"status", "--porcelain=v1", "--untracked-files=all"})
	if rc != 0 {
		return gitFailure("cannot inspect worktree status", rc, status)
	}
	if strings.TrimSpace(status) == "" {
		res := Reap(root, wtPath, git)
		if res.Code == "" {
			switch {
			case res.OK && res.Removed:
				res.Code = ReapCodeRemoved
			case res.OK:
				res.Code = ReapCodeReleased
			default:
				res.Code = "REAP_FAILED"
			}
		}
		if !res.OK && !res.Removed {
			res.Preserved = true
		}
		return res
	}

	supersededBy = strings.TrimSpace(supersededBy)
	if supersededBy == "" {
		return refuse(ReapCodeDirtyWorktreeRefused,
			"dirty worktree preserved; pass --superseded-by only after its effect is independently present on trunk", "")
	}
	rc, resolved := run(git, root, []string{"rev-parse", "--verify", "--end-of-options", supersededBy + "^{commit}"})
	if rc != 0 {
		return gitFailure("supersession commit cannot be resolved", rc, resolved)
	}
	resolved = strings.TrimSpace(resolved)
	rc, detail := run(git, root, []string{"merge-base", "--is-ancestor", resolved, "HEAD"})
	if rc != 0 {
		if rc == ReapTimeoutExitCode {
			return gitFailure("supersession ancestry check timed out", rc, detail)
		}
		return refuse(ReapCodeProofRefused,
			"supersession commit is not on current trunk ancestry", detail)
	}
	rc, untracked := run(git, wtPath, []string{"ls-files", "--others", "--exclude-standard", "-z"})
	if rc != 0 {
		return gitFailure("cannot inspect untracked worktree paths", rc, untracked)
	}
	if len(untracked) != 0 {
		return refuse(ReapCodeProofRefused,
			"untracked work cannot be proven byte-equivalent to a supersession commit", "")
	}
	// Compare only paths changed from this worker's pinned HEAD. The rest of its
	// checkout is intentionally stale and may differ from newer, unrelated trunk
	// commits. Keep rename detection off so a rename contributes both its deletion
	// and addition to the proof instead of hiding one side behind a similarity guess.
	rc, dirtyOutput := run(git, wtPath, []string{
		"diff", "--no-ext-diff", "--no-renames", "--name-only", "-z", "HEAD", "--",
	})
	if rc != 0 {
		return gitFailure("cannot identify tracked dirty worktree paths", rc, dirtyOutput)
	}
	dirtyPaths := splitNULPaths(dirtyOutput)
	if len(dirtyPaths) == 0 {
		return refuse(ReapCodeProofRefused,
			"dirty worktree has no provable tracked path set", "")
	}
	compareArgs := []string{"diff", "--no-ext-diff", "--quiet", resolved, "--"}
	for _, path := range dirtyPaths {
		// Git still interprets pathspec magic after `--`; force every status-derived
		// name literal so a hostile filename cannot exclude another dirty path.
		compareArgs = append(compareArgs, ":(literal)"+path)
	}
	rc, detail = run(git, wtPath, compareArgs)
	if rc != 0 {
		if rc == ReapTimeoutExitCode {
			return gitFailure("supersession content check timed out", rc, detail)
		}
		if rc == 1 {
			return refuse(ReapCodeProofRefused,
				"worktree bytes differ from the supersession commit", "")
		}
		return refuse(ReapCodeProofRefused,
			"cannot compare worktree bytes to the supersession commit", detail)
	}

	res := ForceReap(root, wtPath, git)
	if !res.OK {
		if res.Code == "" {
			res.Code = "REAP_FAILED"
		}
		res.Preserved = !res.Removed
		return res
	}
	res.Code = ReapCodeVerifiedWorktreeReaped
	res.SupersededBy = resolved
	return res
}

func splitNULPaths(output string) []string {
	fields := strings.Split(output, "\x00")
	paths := make([]string, 0, len(fields))
	for _, path := range fields {
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

// ReapWithBackend is the injectable form of Reap.
func ReapWithBackend(root, wtPath string, git GitRunner, backend IsolationBackend) Result {
	if backend == nil {
		backend = defaultIsolationBackend
	}
	res := backend.Release(root, wtPath, git)
	if res.Removed {
		removeOwnerStamp(wtPath)
	}
	return res
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
		if res, ok, fallback := returnPooled(root, wtPath, k, git); ok {
			return res
		} else if fallback != "" {
			res := ForceReap(root, wtPath, git)
			if res.OK {
				res.Reason = "pool return skipped (" + fallback + "); force-removed"
			}
			return res
		}
	}
	return ForceReap(root, wtPath, git)
}

// ForceReap destroys one worker worktree even when the warm pool is enabled. Normal
// Reap preserves the pool's return-on-release behavior; owner-stamped GC uses this
// explicit destructive path because its selected member is old, owner-dead,
// lease-released, and clean. It also clears pool/owner sidecars after a successful
// removal so a collected member cannot remain advertised as reusable.
func ForceReap(root, wtPath string, git GitRunner) Result {
	if !IsWorkerWorktree(wtPath) {
		return Result{OK: false, Path: wtPath, Removed: false,
			Reason: "refusing to reap a non-worker worktree"}
	}
	rc, out := run(git, root, []string{"worktree", "remove", "--force", wtPath})
	removed := rc == 0
	run(git, root, []string{"worktree", "prune"})
	res := Result{OK: removed, Path: wtPath, Removed: removed}
	if !removed {
		_, statErr := os.Stat(wtPath)
		res.Preserved = statErr == nil || !os.IsNotExist(statErr)
		if rc == ReapTimeoutExitCode {
			res.Code = ReapCodeTimeout
			res.Reason = "single-worktree reap deadline exceeded"
		}
		res.Detail = tail(out, 300)
		return res
	}
	removePoolMemberState(wtPath)
	removeOwnerStamp(wtPath)
	return res
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
