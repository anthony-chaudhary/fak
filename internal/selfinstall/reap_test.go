package selfinstall

import (
	"context"
	"strings"
	"testing"
)

// reapRunner serves a fixed `worktree list --porcelain` body and records every git argv it
// is asked to run, so a test can assert exactly which worktrees were removed (and that prune
// ran only when something was reaped). All commands "succeed" (ok=true).
type reapRunner struct {
	list string
	ran  [][]string
}

func (r *reapRunner) run(_ context.Context, _, name string, args ...string) (string, bool) {
	r.ran = append(r.ran, append([]string{name}, args...))
	if name == "git" && len(args) >= 2 && args[0] == "worktree" && args[1] == "list" {
		return r.list, true
	}
	return "", true
}

// removedPaths returns the targets of every `git worktree remove --force <path>` issued.
func (r *reapRunner) removedPaths() []string {
	var out []string
	for _, c := range r.ran {
		if len(c) == 5 && c[0] == "git" && c[1] == "worktree" && c[2] == "remove" && c[3] == "--force" {
			out = append(out, c[4])
		}
	}
	return out
}

func (r *reapRunner) prunedOnce() bool {
	n := 0
	for _, c := range r.ran {
		if len(c) == 3 && c[0] == "git" && c[1] == "worktree" && c[2] == "prune" {
			n++
		}
	}
	return n == 1
}

// porcelain builds a `git worktree list --porcelain` body from a set of worktree paths.
func porcelain(paths ...string) string {
	var b strings.Builder
	for i, p := range paths {
		if i > 0 {
			b.WriteString("\n")
		}
		b.WriteString("worktree " + p + "\nHEAD deadbeef\ndetached\n")
	}
	return b.String()
}

func TestReapStaleBuildsRemovesOnlyDeadPrefixedWorktrees(t *testing.T) {
	const selfPID = 1000
	// alive: only pid 2000 (a concurrently-building peer self-update) is live.
	alive := func(pid int) bool { return pid == 2000 }

	r := &reapRunner{list: porcelain(
		"C:/work/fak", // main checkout — no prefix, ignored
		"C:/Users/u/AppData/Local/Temp/peer-worktree-x",           // a peer worktree — no prefix, ignored
		"C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-3000", // dead owner -> REAP
		"C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-2000", // live owner -> keep
		"C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-1000", // our own pid -> keep
		"C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-4000", // dead owner -> REAP
	)}

	reaped := ReapStaleBuilds(context.Background(), r.run, "C:/work/fak", selfPID, alive)

	got := strings.Join(reaped, ",")
	want := "C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-3000,C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-4000"
	if got != want {
		t.Fatalf("reaped = %q, want %q", got, want)
	}
	if rp := strings.Join(r.removedPaths(), ","); rp != want {
		t.Fatalf("removed = %q, want exactly the two dead build worktrees %q", rp, want)
	}
	if !r.prunedOnce() {
		t.Fatalf("expected exactly one `git worktree prune` after reaping; ran %v", r.ran)
	}
}

func TestReapStaleBuildsNoopWhenNothingStale(t *testing.T) {
	const selfPID = 1000
	alive := func(int) bool { return true } // every owner still alive

	r := &reapRunner{list: porcelain(
		"C:/work/fak",
		"C:/Users/u/AppData/Local/Temp/fak-selfupdate-build-2000",
	)}
	reaped := ReapStaleBuilds(context.Background(), r.run, "C:/work/fak", selfPID, alive)
	if len(reaped) != 0 {
		t.Fatalf("reaped %v, want none", reaped)
	}
	if len(r.removedPaths()) != 0 {
		t.Fatalf("removed %v, want none", r.removedPaths())
	}
	// Prune must NOT run when nothing was reaped (no dangling entries to clean).
	for _, c := range r.ran {
		if len(c) >= 2 && c[1] == "prune" {
			t.Fatalf("prune ran with nothing reaped; ran %v", r.ran)
		}
	}
}

func TestReapStaleBuildsReturnsNilWhenListFails(t *testing.T) {
	failList := func(_ context.Context, _, name string, args ...string) (string, bool) {
		return "git not available", false
	}
	if got := ReapStaleBuilds(context.Background(), failList, "/repo", 1, func(int) bool { return false }); got != nil {
		t.Fatalf("reaped %v, want nil when `worktree list` fails", got)
	}
}

func TestBuildDirNameRoundTrips(t *testing.T) {
	for _, pid := range []int{1, 42, 67260} {
		name := BuildDirName(pid)
		if !strings.HasPrefix(name, buildDirPrefix) {
			t.Fatalf("BuildDirName(%d)=%q lacks prefix %q", pid, name, buildDirPrefix)
		}
		got, ok := pidFromBuildDir(name)
		if !ok || got != pid {
			t.Fatalf("pidFromBuildDir(%q) = (%d,%v), want (%d,true)", name, got, ok, pid)
		}
	}
	// Non-matching names yield (0,false).
	for _, bad := range []string{"fak-build", "peer-worktree", "fak-selfupdate-build-", "fak-selfupdate-build-abc", "fak-selfupdate-build--1"} {
		if pid, ok := pidFromBuildDir(bad); ok {
			t.Fatalf("pidFromBuildDir(%q) = (%d,true), want not-ok", bad, pid)
		}
	}
}
