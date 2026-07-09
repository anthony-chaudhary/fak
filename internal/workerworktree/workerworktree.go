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
	"os"
	"os/exec"
	"path/filepath"
	"strings"

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
	// LandReadbackEnv, when truthy, makes Land re-read trunk HEAD after a
	// path-scoped commit and confirm it actually carries the worker's intended
	// paths — turning the silent shared-index-race false-success (#3547) into an
	// honest LAND_READBACK_MISMATCH refusal. DEFAULT OFF so the baseline land path
	// is byte-for-byte unchanged (the #3165 opt-in, fail-open, A/B-able doctrine);
	// the epic owner flips it on to measure the false-refuse rate under real load.
	LandReadbackEnv = "FAK_LAND_READBACK_VERIFY"
	keyHashLen      = 12
)

// GitRunner runs one `git` subcommand under root and returns (rc, stdout). It
// never raises: an exec failure is reported as a non-zero rc so every caller
// fails open. Injectable so the whole path is testable against a fake.
type GitRunner func(root string, args []string) (int, string)

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
func Prepare(root, lane, key, baseSHA, wtRoot string, git GitRunner) Result {
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
func Reap(root, wtPath string, git GitRunner) Result {
	if !IsWorkerWorktree(wtPath) {
		return Result{OK: false, Path: wtPath, Removed: false,
			Reason: "refusing to reap a non-worker worktree"}
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
func Land(root, wtPath, baseSHA, commitMsgFile string, paths []string, verify VerifyHook, git GitRunner) Result {
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
	res := Result{OK: rc == 0, Applied: true, Committed: rc == 0, Detail: tail(out, 300)}
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
	f, err := os.CreateTemp("", "fak-wt-land-*.patch")
	if err != nil {
		return Result{OK: false, Detail: "could not create patch temp: " + err.Error()}
	}
	patch := f.Name()
	defer os.Remove(patch)
	body := diff
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		return Result{OK: false, Detail: "could not write patch temp: " + err.Error()}
	}
	f.Close()
	rc, out := run(git, root, []string{"apply", "--whitespace=nowarn", patch})
	return Result{OK: rc == 0, Detail: tail(out, 300)}
}

// landReadbackEnabled reports whether the opt-in post-commit readback (LandReadbackEnv)
// is on. Truthy = any value other than "", "0", "false", "off" (case-insensitive).
func landReadbackEnabled() bool {
	v := strings.TrimSpace(os.Getenv(LandReadbackEnv))
	return v != "" && v != "0" && !strings.EqualFold(v, "false") && !strings.EqualFold(v, "off")
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
		return false, "LAND_READBACK_MISMATCH: trunk HEAD " + shortSHA(sha) +
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

func tail(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
