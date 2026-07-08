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
// the things that are provably safe — a lockfile whose holder PID is dead; a worktree whose
// HEAD is an ancestor of the trunk (already merged) AND has not been touched recently; and an
// orphaned per-worker isolation worktree (its dir carries WorkerWorktreeMarker,
// `fak-worker-wt-*`) that is not live — disposable editing space whose only durable output
// already landed on the trunk, so a crashed worker's leaked tree is reclaimed even when its
// in-worktree commit left HEAD off the trunk. Everything else is reported, never removed.
//
// Like safecommit, every git effect goes through an injected Runner so the whole decision
// tree is testable with no git and no repo. Diagnose is read-only; Sweep performs the
// reclaim only when apply is true.
package treedoctor

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
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

// WorktreeState classifies one worktree for the prune decision.
type WorktreeState struct {
	Path     string `json:"path"`
	Head     string `json:"head,omitempty"`
	IsMain   bool   `json:"is_main"`             // the RepoRoot itself — never a prune candidate
	IsWorker bool   `json:"is_worker,omitempty"` // carries WorkerWorktreeMarker — disposable, reap when not live
	Merged   bool   `json:"merged"`              // HEAD is an ancestor of Trunk (commits already on the trunk)
	Live     bool   `json:"live"`                // touched within LiveWindow — an active session, keep
	DirtyN   int    `json:"dirty_n"`             // count of uncommitted entries (informational)
	Prunable bool   `json:"prunable"`            // safe to remove: (Merged || IsWorker) && !Live && !IsMain
	Keep     string `json:"keep,omitempty"`
}

// Report is the full read-only diagnosis.
type Report struct {
	Lock      LockState       `json:"lock"`
	Worktrees []WorktreeState `json:"worktrees"`
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
	rep.Worktrees = diagnoseWorktrees(ctx, run, opts.RepoRoot, trunk, window, now)
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

	for _, w := range rep.Worktrees {
		if !w.Prunable {
			continue
		}
		kind := "merged worktree"
		if w.IsWorker {
			kind = "orphan worker worktree"
		}
		if apply {
			if _, _, err := run(ctx, opts.RepoRoot, "worktree", "remove", "--force", w.Path); err == nil {
				actions = append(actions, "pruned "+kind+" "+w.Path)
			}
		} else {
			actions = append(actions, "would prune "+kind+" "+w.Path)
		}
	}
	if apply && len(actions) > 0 {
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

func diagnoseWorktrees(ctx context.Context, run Runner, repoRoot, trunk string, window time.Duration, now time.Time) []WorktreeState {
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
		// Merged? HEAD an ancestor of trunk => its commits are already on the trunk.
		if _, code, aerr := run(ctx, wt.path, "merge-base", "--is-ancestor", "HEAD", trunk); aerr == nil && code == 0 {
			s.Merged = true
		}
		// Dirty count (informational; does not gate the prune — a merged worktree's only
		// uncommitted content is by definition not on the trunk and is scratch/stale).
		if porc, _, derr := run(ctx, wt.path, "status", "--porcelain"); derr == nil {
			s.DirtyN = countLines(porc)
		}
		// Live? touched recently => an active session, never prune.
		if recentlyTouched(wt.path, window, now) {
			s.Live = true
		}
		switch {
		case s.Live:
			// Touched within the window => an active session (or a live worker),
			// never pruned even if merged/marker. Checked first so a live worker
			// worktree is kept exactly like any other live tree.
			s.Keep = "live (touched within window)"
		case s.IsWorker:
			// A fak-worker-wt-* worktree is throwaway editing space: its only durable
			// output already landed on the trunk via land_worktree_diff. An orphan
			// (worker crashed, or the host died mid-wave) leaks its tree + in-worktree
			// GOCACHE with no scheduled reclaimer — and if the worker committed
			// in-worktree before dying its HEAD is NOT an ancestor of the trunk, so the
			// merged-only rule below would keep it forever. Reap it regardless of merge
			// status; isWorkerWorktree is the load-bearing guardrail.
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
