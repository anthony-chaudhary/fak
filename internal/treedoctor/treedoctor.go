// Package treedoctor diagnoses and (optionally) sweeps a fak working tree that has
// gone un-tidy under a permanently-on agent fleet, where the trunk is never quiescent.
//
// It encodes, as a reusable tool, the manual recovery a 2026-06-28 session did by hand
// after auto-gardening fell behind:
//
//   - A DEAD committer's orphaned .git/fak-commit.lock had wedged the whole commit lane
//     for ~56 minutes (Windows LockFileEx does not promptly release on an abnormal exit).
//     The fix is the safecommit dead-PID reap; treedoctor exposes it as an on-demand check.
//   - 18 stale worktrees from a prior multi-agent run lingered, all already merged into the
//     trunk, their only dirty content disposable scratch or stale-base deletions.
//
// The doctrine treedoctor enforces is the one that session learned the hard way: in an
// always-on tree you must NEVER delete a peer's live work. So treedoctor only ever reclaims
// the things that are provably safe — a lockfile whose holder PID is dead; a renamed-aside
// lock residue file (its name carries something AFTER the `.lock` an active git lock ends
// in, so it never races a live git op) aged past the residue floor; a worktree whose HEAD is an
// ancestor of the trunk (already merged) AND has not been touched recently; and an orphaned
// per-worker isolation worktree (its dir carries WorkerWorktreeMarker, `fak-worker-wt-*`)
// that is not live — disposable editing space whose only durable output already landed on
// the trunk, so a crashed worker's leaked tree is reclaimed even when its in-worktree commit
// left HEAD off the trunk. Everything else is reported, never removed.
//
// Like safecommit, every git effect goes through an injected Runner so the whole decision
// tree is testable with no git and no repo. Diagnose is read-only; Sweep performs the
// reclaim only when apply is true.
package treedoctor

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/selfinstall"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// Runner runs a git command in dir and returns combined stdout/stderr, the exit code, and
// an error only when git could not be executed at all (mirrors safecommit.Runner).
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// Options configures Diagnose / Sweep.
type Options struct {
	// RepoRoot is the main checkout (the worktree whose .git holds fak-commit.lock).
	RepoRoot string
	// Trunk is the ref a worktree must be merged into to be prunable ("" => origin/main).
	Trunk string
	// LiveWindow guards against pruning an ACTIVELY-USED worktree: one whose newest file
	// was modified within this window is treated as live and kept, even if merged. Zero
	// => DefaultLiveWindow.
	LiveWindow time.Duration
	// Now is the reference time for the live-window check (injectable for tests). Zero =>
	// time.Now() at call.
	Now time.Time
	// ResidueMinAge is the floor under which renamed-aside lock residue is never swept,
	// independent of LiveWindow. Zero => DefaultResidueMinAge; negative disables the floor
	// (the test seam) and leaves LiveWindow as the only gate.
	ResidueMinAge time.Duration
	// WIP carries the optional build/owner probes for the untracked-WIP inventory. Its zero
	// value inventories age only (no build probe, owner undiscoverable) — still enough to
	// classify live vs abandoned. Diagnose always runs the inventory; it is read-only.
	WIP WIPOptions
	// RefLock carries the thresholds and seams for the packed-refs.lock / AUTO_MERGE.lock
	// diagnosis and the loose-ref pressure count. Its zero value is the production
	// configuration (see reflocks.go).
	RefLock RefLockOptions
	// ProcessAlive protects owner-stamped worker worktrees across launcher PID handoff.
	ProcessAlive func(int) bool
	// LocksOnly narrows Diagnose/Sweep to the LOCK half: the commit lock, the renamed-aside
	// lock residue, and the ref locks. The worktree classification and the untracked-WIP
	// inventory are skipped entirely — so Sweep's `git worktree remove` loop has nothing to
	// iterate and CANNOT fire, and the two whole-tree walks those diagnoses cost are not paid.
	//
	// It exists for the UNATTENDED caller (`fak git-daily`, internal/gitdaily): every lock
	// reap here is provably safe without a human — a dead holder PID, or a name that is
	// structurally not a lock git can be holding — whereas a worktree prune weighs a
	// merged/live judgement whose false positive destroys a peer's in-flight work. Keeping
	// the unattended tick locks-only makes that blast radius a property of the CALL, not of
	// a reviewer remembering which report fields Sweep acts on. The interactive
	// `fak tree-doctor --apply` leaves it false and keeps the full sweep.
	LocksOnly bool
}

// DefaultLiveWindow: a worktree touched within this long is assumed to belong to a live
// session and is never pruned. Generous on purpose — a false keep costs disk, a false
// prune destroys a peer's in-flight work.
const DefaultLiveWindow = 10 * time.Minute

// DefaultTrunk is the merge target a worktree must be folded into to be prunable.
const DefaultTrunk = "origin/main"

// WorkerWorktreeMarker is the leading path segment fak's per-worker isolation
// stamps onto a throwaway worktree dir (`fak-worker-wt-<lane>-<key>`), mirrored
// from tools/worker_worktree.py's WORKTREE_MARKER. A worktree carrying this marker
// is disposable editing space whose only durable output already LANDED on the trunk
// via land_worktree_diff — so an orphan (worker crashed, or the host died mid-wave)
// is safe to reap regardless of merge status. The marker is the load-bearing
// guardrail: only a marker worktree is ever reaped on this relaxed rule, never the
// primary or a peer's non-marker scratch tree.
const WorkerWorktreeMarker = "fak-worker-wt"

// isWorkerWorktree reports whether path is one of fak's per-worker isolation
// worktrees (its basename carries WorkerWorktreeMarker). Mirrors
// worker_worktree.is_worker_worktree so the sweep reuses the SAME authoritative
// match rather than re-deriving it.
func isWorkerWorktree(path string) bool {
	name := filepath.Base(filepath.Clean(path))
	return name == WorkerWorktreeMarker || strings.HasPrefix(name, WorkerWorktreeMarker+"-")
}

// LockState is the diagnosis of the commit lock.
type LockState struct {
	Path      string `json:"path"`
	Present   bool   `json:"present"`
	HolderPID int    `json:"holder_pid,omitempty"`
	Stale     bool   `json:"stale"` // a dead holder still owns it — wedges the commit lane
}

// LockResidueState is a renamed-aside git lock file a lock-recovery mechanism left in the
// git common dir (e.g. `fak-commit.lock.stale-20260716-044444`, `index.lock.stale-<stamp>`).
// The suffix AFTER `.lock` is the load-bearing safety marker: an ACTIVE git lock ends in
// exactly `.lock` (`HEAD.lock`, `packed-refs.lock`, `index.lock`, …), so a name with
// anything trailing that is provably NOT a lock git is currently holding — removing it can
// never race a live transaction. Left unswept, this residue accumulates in the hot shared
// `.git`. It is swept only once aged past the residue floor (a just-created one might
// belong to an in-flight recovery, so it is reported but kept).
type LockResidueState struct {
	Path       string `json:"path"`
	AgeSeconds int64  `json:"age_seconds"`
	Sweepable  bool   `json:"sweepable"` // aged past the residue floor — safe to remove
}

// WorktreeState classifies one worktree for the prune decision.
type WorktreeState struct {
	Path         string `json:"path"`
	Head         string `json:"head,omitempty"`
	IsMain       bool   `json:"is_main"`                  // the RepoRoot itself — never a prune candidate
	IsWorker     bool   `json:"is_worker,omitempty"`      // carries WorkerWorktreeMarker — disposable, reap when not live
	IsSelfUpdate bool   `json:"is_self_update,omitempty"` // owned and collected by selfinstall's stricter GC
	Merged       bool   `json:"merged"`                   // HEAD is an ancestor of Trunk (commits already on the trunk)
	Live         bool   `json:"live"`                     // touched within LiveWindow — an active session, keep
	DirtyN       int    `json:"dirty_n"`                  // count of uncommitted entries (informational)
	Archive      bool   `json:"archive,omitempty"`        // dirty worker content must be archived before removal
	Prunable     bool   `json:"prunable"`                 // safe to remove: (Merged || IsWorker) && !Live && !IsMain
	Keep         string `json:"keep,omitempty"`
}

// Report is the full read-only diagnosis.
type Report struct {
	Lock        LockState          `json:"lock"`
	LockResidue []LockResidueState `json:"lock_residue,omitempty"`
	// RefLocks is the packed-refs.lock / AUTO_MERGE.lock diagnosis plus the loose-ref
	// pressure that builds while ref packing is blocked (see reflocks.go). Report-only by
	// default; Sweep reaps a stale ref lock only under apply.
	RefLocks  RefLockReport   `json:"ref_locks"`
	Worktrees []WorktreeState `json:"worktrees"`
	// WIP is the untracked durable-artifact inventory: source, .claude control fuel, and
	// testdata fixtures classified with a typed cleanup action. Read-only — Sweep never touches it (a load-bearing
	// unlanded file is byte-indistinguishable from cruft, so acting on it is a human's call).
	WIP            []WIPFile            `json:"wip,omitempty"`
	ScratchHygiene ScratchHygieneReport `json:"scratch_hygiene"`
}

// StaleLockWedged reports whether the commit lock is currently wedged by a dead holder.
func (r Report) StaleLockWedged() bool { return r.Lock.Stale }

// PrunableWorktrees returns the worktrees Sweep would remove.
func (r Report) PrunableWorktrees() []WorktreeState {
	var out []WorktreeState
	for _, w := range r.Worktrees {
		if w.Prunable {
			out = append(out, w)
		}
	}
	return out
}

// SweepableLockResidue returns the orphan lock residue files Sweep would remove.
func (r Report) SweepableLockResidue() []LockResidueState {
	var out []LockResidueState
	for _, f := range r.LockResidue {
		if f.Sweepable {
			out = append(out, f)
		}
	}
	return out
}

// LandOrParkWIP returns the untracked source surfaced for a human to land or park (build
// poison or aged-and-unowned). Sweep never acts on these — the surface is read-only.
func (r Report) LandOrParkWIP() []WIPFile {
	var out []WIPFile
	for _, f := range r.WIP {
		if f.LandOrPark {
			out = append(out, f)
		}
	}
	return out
}

// PoisonWIP returns the untracked source whose package will not compile — the shared-trunk
// build poison that crash-loops peers on rebuild.
func (r Report) PoisonWIP() []WIPFile {
	var out []WIPFile
	for _, f := range r.WIP {
		if f.Poison {
			out = append(out, f)
		}
	}
	return out
}

// Diagnose inspects the tree read-only: the commit lock's staleness and every worktree's
// prune classification. It never mutates anything.
func Diagnose(ctx context.Context, run Runner, opts Options) Report {
	trunk := strings.TrimSpace(opts.Trunk)
	if trunk == "" {
		trunk = DefaultTrunk
	}
	window := opts.LiveWindow
	if window <= 0 {
		window = DefaultLiveWindow
	}
	now := opts.Now
	if now.IsZero() {
		now = time.Now()
	}

	rep := Report{Lock: diagnoseLock(opts.RepoRoot)}
	rep.LockResidue = diagnoseLockResidue(opts.RepoRoot, residueThreshold(window, opts.ResidueMinAge), now)
	rep.RefLocks = diagnoseRefLocks(opts.RepoRoot, opts.RefLock, now)
	if opts.LocksOnly {
		// Locks-only: skip both whole-tree walks. Leaving Worktrees empty is also what
		// keeps Sweep's prune loop from having anything to remove (see Options.LocksOnly).
		return rep
	}
	rep.Worktrees = diagnoseWorktrees(ctx, run, opts.RepoRoot, trunk, window, now, opts.ProcessAlive)
	rep.WIP = diagnoseWIP(ctx, run, opts.RepoRoot, window, now, opts.WIP)
	rep.ScratchHygiene = diagnoseScratchHygiene(opts.RepoRoot)
	return rep
}

// Sweep performs the safe reclaim: reap a stale commit lock and (when apply) remove every
// prunable worktree. With apply=false it is a no-op that returns what WOULD be done (the
// same as reading the Report). It returns the actions taken (or planned).
func Sweep(ctx context.Context, run Runner, opts Options, apply bool) (Report, []string) {
	rep := Diagnose(ctx, run, opts)
	var actions []string

	if rep.Lock.Stale {
		if apply {
			if safecommit.ReapStaleLock(rep.Lock.Path) {
				actions = append(actions, "reaped stale commit lock (dead PID "+strconv.Itoa(rep.Lock.HolderPID)+")")
			}
		} else {
			actions = append(actions, "would reap stale commit lock (dead PID "+strconv.Itoa(rep.Lock.HolderPID)+")")
		}
	}

	for _, f := range rep.LockResidue {
		if !f.Sweepable {
			continue
		}
		if apply {
			if err := os.Remove(f.Path); err == nil {
				actions = append(actions, "swept orphan lock residue "+f.Path)
			}
		} else {
			actions = append(actions, "would sweep orphan lock residue "+f.Path)
		}
	}

	// Orphaned ref locks (packed-refs.lock / AUTO_MERGE.lock) and the loose-ref pressure
	// they cause. Report-only unless apply; the reap bar is a frozen mtime PLUS the
	// two-sample advancing witness, never age alone (see reflocks.go's header).
	actions = append(actions, sweepRefLocks(rep.RefLocks, apply)...)

	// Tracked separately from len(actions) so a report-only advisory (e.g. loose-ref
	// pressure) can never make the sweep fire a `git worktree prune` it did not earn.
	prunedWorktree := false
	for _, w := range rep.Worktrees {
		if !w.Prunable {
			continue
		}
		kind := "merged worktree"
		if w.IsWorker {
			kind = "orphan worker worktree"
		}
		if w.Archive {
			// This lightweight Go sweep has no archive writer. Preserve dirty crashed-
			// worker output and direct automation to the archive-capable doctor instead
			// of silently destroying it with `git worktree remove --force`.
			actions = append(actions, "archive required before pruning "+kind+" "+w.Path)
			continue
		}
		if apply {
			if _, _, err := run(ctx, opts.RepoRoot, "worktree", "remove", "--force", w.Path); err == nil {
				actions = append(actions, "pruned "+kind+" "+w.Path)
				prunedWorktree = true
			}
		} else {
			actions = append(actions, "would prune "+kind+" "+w.Path)
		}
	}
	if apply && prunedWorktree {
		_, _, _ = run(ctx, opts.RepoRoot, "worktree", "prune")
	}
	return rep, actions
}

func diagnoseLock(repoRoot string) LockState {
	path := filepath.Join(repoRoot, ".git", "fak-commit.lock")
	p := safecommit.ProbeLock(path)
	return LockState{
		Path:      path,
		Present:   p.Exists,
		HolderPID: p.HolderPID,
		Stale:     p.Stale,
	}
}

// StaleAsideSuffix is the infix a lock-recovery step stamps onto a git lock file it moves
// ASIDE instead of deleting: `<name>.lock` + StaleAsideSuffix + `<UTC stamp>`, e.g.
// `fak-commit.lock.stale-20260716-044444`. It is exported, and paired with StaleAsideName
// below, so a WRITER and this reaper share ONE spelling.
//
// That pairing is the point. This matcher shipped globbing `*.lock.orphan*` — a name
// nothing in this repository has ever written. The reaper and the residue were never
// connected by anything but a hand-copied literal, so the drift was invisible: the doctor
// reported a clean `.git` while seven `fak-commit.lock.stale-*` files and a 1.16 MB
// `index.lock.stale-20260716-044445` sat there for thirteen days, untouchable by
// `fak tree-doctor --apply` (#5335). Deriving both ends from these constants is what stops
// that class of drift from recurring.
const StaleAsideSuffix = ".stale-"

// StaleAsideStampLayout is the UTC time layout in a stale-aside name (`20060102-150405`),
// matching the residue the fleet has actually produced.
const StaleAsideStampLayout = "20060102-150405"

// StaleAsideName returns the path a recovery step must rename lockPath to when it moves a
// lock aside rather than removing it, so the result is guaranteed to be swept later by
// diagnoseLockResidue. Callers that build the name by hand are exactly how the matcher
// drifted away from the residue in the first place.
func StaleAsideName(lockPath string, at time.Time) string {
	return lockPath + StaleAsideSuffix + at.UTC().Format(StaleAsideStampLayout)
}

// lockResidueGlobs match the renamed-aside lock files a recovery mechanism leaves in the
// git common dir. Each `*` never crosses a path separator, so only files directly in
// `.git` are considered — the same depth git's own top-level locks live at. Every pattern
// keeps the same safety invariant: it requires something AFTER `.lock`, so an ACTIVE lock
// name (`index.lock`, `packed-refs.lock`) can never match.
//
//   - the StaleAsideSuffix family is what this tree actually accumulates;
//   - `*.lock.orphan*` is the legacy spelling this matcher shipped with. It is kept so an
//     older tree still cleans up, but it has never matched anything this repo writes.
var lockResidueGlobs = []string{
	"*.lock" + StaleAsideSuffix + "*",
	"*.lock.orphan*",
}

// DefaultResidueMinAge is the floor under which lock residue is NEVER swept, whatever
// LiveWindow says. A rename-aside is instantaneous, so residue this young may belong to a
// recovery still in flight — and unlike a worktree, residue costs only bytes to keep. The
// floor is deliberately larger than DefaultLiveWindow so the residue sweep is strictly the
// more conservative of the two.
const DefaultResidueMinAge = 30 * time.Minute

// residueThreshold is the age a residue file must reach before it may be swept: the larger
// of the caller's live window and the residue floor, so neither knob can make the sweep
// more aggressive than the other allows. A NEGATIVE ResidueMinAge disables the floor (the
// test seam for exercising the age gate without a 30-minute fixture).
func residueThreshold(window, minAge time.Duration) time.Duration {
	if minAge == 0 {
		minAge = DefaultResidueMinAge
	}
	if minAge > window {
		return minAge
	}
	return window
}

// diagnoseLockResidue lists renamed-aside lock residue in <repoRoot>/.git, marking as
// Sweepable each file aged past the residue threshold. It never removes anything — Sweep
// does, and only for the Sweepable ones. A missing/unglobbable dir yields an empty result
// (the fail-safe read). Results are de-duplicated (a name can satisfy more than one
// pattern) and sorted, so the report and the actions are stable run to run.
func diagnoseLockResidue(repoRoot string, threshold time.Duration, now time.Time) []LockResidueState {
	gitDir := filepath.Join(repoRoot, ".git")
	seen := map[string]bool{}
	var paths []string
	for _, glob := range lockResidueGlobs {
		matches, err := filepath.Glob(filepath.Join(gitDir, glob))
		if err != nil {
			continue // a malformed pattern must not blind the other families
		}
		for _, p := range matches {
			if !seen[p] {
				seen[p] = true
				paths = append(paths, p)
			}
		}
	}
	sort.Strings(paths)

	var out []LockResidueState
	for _, path := range paths {
		info, serr := os.Stat(path)
		if serr != nil || info.IsDir() {
			continue
		}
		age := now.Sub(info.ModTime())
		var ageSec int64
		if age > 0 {
			ageSec = int64(age / time.Second)
		}
		out = append(out, LockResidueState{
			Path:       path,
			AgeSeconds: ageSec,
			Sweepable:  age >= threshold,
		})
	}
	return out
}

func diagnoseWorktrees(ctx context.Context, run Runner, repoRoot, trunk string, window time.Duration, now time.Time, processAlive func(int) bool) []WorktreeState {
	out, _, err := run(ctx, repoRoot, "worktree", "list", "--porcelain")
	if err != nil {
		return nil
	}
	var states []WorktreeState
	for _, wt := range parseWorktreeList(out) {
		s := WorktreeState{Path: wt.path, Head: wt.head}
		if samelyPath(wt.path, repoRoot) {
			s.IsMain = true
			s.Keep = "main checkout"
			states = append(states, s)
			continue
		}
		s.IsWorker = isWorkerWorktree(wt.path)
		s.IsSelfUpdate = selfinstall.IsSelfUpdateBuildWorktree(wt.path)
		// Merged? HEAD an ancestor of trunk => its commits are already on the trunk.
		if _, code, aerr := run(ctx, wt.path, "merge-base", "--is-ancestor", "HEAD", trunk); aerr == nil && code == 0 {
			s.Merged = true
		}
		// Dirty count (informational; does not gate the prune — a merged worktree's only
		// uncommitted content is by definition not on the trunk and is scratch/stale).
		if porc, _, derr := run(ctx, wt.path, "status", "--porcelain"); derr == nil {
			s.DirtyN = countLines(porc)
		}
		// Live? a recent touch or a live stamped owner means an active session.
		if recentlyTouched(wt.path, window, now) {
			s.Live = true
		}
		if ownerLive, known := workerworktree.OwnerProcessLive(wt.path, processAlive); known && ownerLive {
			s.Live = true
			s.Keep = "live (owner process)"
		}
		switch {
		case s.IsSelfUpdate:
			// Self-update checkout collection has stronger ownership and liveness gates
			// than this generic age heuristic. Keeping the lifecycle exclusive prevents
			// a slow build from losing its source after DefaultLiveWindow expires.
			s.Keep = "self-update lifecycle (owner-aware GC)"
		case s.Live:
			// Touched within the window => an active session (or a live worker),
			// never pruned even if merged/marker. Checked first so a live worker
			// worktree is kept exactly like any other live tree.
			if s.Keep == "" {
				s.Keep = "live (touched within window)"
			}
		case s.IsWorker:
			// A fak-worker-wt-* worktree is throwaway editing space, but a crashed
			// worker can leave its only useful diff here. Mark dirty trees for an
			// archive-before-remove action; the CLI refuses to remove them otherwise.
			s.Archive = s.DirtyN > 0
			s.Prunable = true
		case !s.Merged:
			s.Keep = "not merged into " + trunk
		default:
			s.Prunable = true
		}
		states = append(states, s)
	}
	return states
}

type wtEntry struct{ path, head string }

// parseWorktreeList parses `git worktree list --porcelain` into (path, head) entries.
func parseWorktreeList(out string) []wtEntry {
	var entries []wtEntry
	var cur wtEntry
	flush := func() {
		if cur.path != "" {
			entries = append(entries, cur)
		}
		cur = wtEntry{}
	}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimRight(line, "\r")
		switch {
		case strings.HasPrefix(line, "worktree "):
			flush()
			cur.path = strings.TrimSpace(strings.TrimPrefix(line, "worktree "))
		case strings.HasPrefix(line, "HEAD "):
			cur.head = strings.TrimSpace(strings.TrimPrefix(line, "HEAD "))
		case line == "":
			flush()
		}
	}
	flush()
	return entries
}

func countLines(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	return len(strings.Split(s, "\n"))
}

// recentlyTouched reports whether any file under dir was modified within window of now.
// It walks at most a bounded number of entries so a huge worktree cannot stall the check;
// any recent mtime short-circuits to true.
func recentlyTouched(dir string, window time.Duration, now time.Time) bool {
	cutoff := now.Add(-window)
	const maxEntries = 5000
	seen := 0
	found := false
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || found {
			return nil
		}
		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		seen++
		if seen > maxEntries {
			return filepath.SkipAll
		}
		if d.IsDir() {
			return nil
		}
		if info, ierr := d.Info(); ierr == nil && info.ModTime().After(cutoff) {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

// samelyPath compares two filesystem paths after cleaning, case-insensitively (Windows
// paths from git can differ in case/separators from the configured RepoRoot).
func samelyPath(a, b string) bool {
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}
