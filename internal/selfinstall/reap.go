package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// buildDirPrefix is the fixed name prefix of every self-update build worktree. The build
// directory is "<buildDirPrefix><pid>" (BuildDirName), so the prefix both MARKS a directory
// as self-update's own throwaway origin checkout and ENCODES the pid of the process that
// created it — which is exactly what makes dead-owner reaping provably safe.
const buildDirPrefix = "fak-selfupdate-build-"

// BuildDirName returns the per-process build-worktree directory name for pid. cmdSelfUpdate
// joins this onto os.TempDir() to place the worktree; ReapStaleBuilds parses the pid back
// out of it. Single-sourcing the name here keeps the creator and the reaper from ever
// drifting apart (a drift would silently turn the reaper into a no-op).
func BuildDirName(pid int) string { return buildDirPrefix + strconv.Itoa(pid) }

// pidFromBuildDir extracts the pid encoded in a build-worktree directory's basename, or
// (0,false) when base is not a "<buildDirPrefix><pid>" name with a positive numeric pid.
func pidFromBuildDir(base string) (int, bool) {
	rest, ok := strings.CutPrefix(base, buildDirPrefix)
	if !ok {
		return 0, false
	}
	pid, err := strconv.Atoi(rest)
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

// ReapStaleBuilds removes self-update build worktrees left behind by PRIOR runs whose owning
// process is gone. Self-update builds from a fresh detached origin/main worktree named
// "<buildDirPrefix><pid>" and removes it in a deferred cleanup — but a tick that is KILLED
// before the defer runs (common on a saturated shared box hitting a wall-clock timeout)
// leaks the worktree, and nothing else ever reaps it. Calling this at the START of every
// self-update makes the leak self-healing: each run buries the corpses of the runs the OS
// killed before it.
//
// It is safe to run blind:
//   - it only ever touches directories whose basename is "<buildDirPrefix><pid>" — a name
//     no other tool produces — so a peer's worktree or the main checkout is never a target;
//   - it reaps one ONLY when alive(pid) is false (the owner is dead) and pid != selfPID
//     (never our own in-progress build), so a concurrently-building self-update is left be;
//   - such a worktree only ever holds a pristine detached origin checkout (self-update never
//     commits in it), so there is no peer work to lose even in the worst case.
//
// Every git effect goes through the injected Runner and liveness through the injected alive
// predicate, so the whole decision tree is testable with no git and no real processes. It
// returns the paths it reaped (empty when there was nothing to do).
func ReapStaleBuilds(ctx context.Context, run Runner, repoRoot string, selfPID int, alive func(int) bool) []string {
	out, ok := run(ctx, repoRoot, "git", "worktree", "list", "--porcelain")
	if !ok {
		return nil
	}
	var reaped []string
	for _, path := range worktreePaths(out) {
		pid, ok := pidFromBuildDir(filepath.Base(filepath.Clean(path)))
		if !ok || pid == selfPID || alive(pid) {
			continue
		}
		// Prefer git's own removal (it drops the admin entry too); if it fails because the
		// directory is already half-gone, fall back to a direct delete so the `git worktree
		// prune` below can then reap the dangling admin entry. Only count a path once it is
		// actually gone, so the returned list is an honest record of what was removed.
		removed := false
		if _, gok := run(ctx, repoRoot, "git", "worktree", "remove", "--force", path); gok {
			removed = true
		} else if err := os.RemoveAll(path); err == nil {
			removed = true
		}
		if removed {
			reaped = append(reaped, path)
		}
	}
	if len(reaped) > 0 {
		_, _ = run(ctx, repoRoot, "git", "worktree", "prune")
	}
	return reaped
}

// worktreePaths returns the worktree paths from `git worktree list --porcelain` output (the
// value after each "worktree " header line).
func worktreePaths(porcelain string) []string {
	var paths []string
	for _, line := range strings.Split(porcelain, "\n") {
		line = strings.TrimRight(line, "\r")
		if p, ok := strings.CutPrefix(line, "worktree "); ok {
			paths = append(paths, strings.TrimSpace(p))
		}
	}
	return paths
}
