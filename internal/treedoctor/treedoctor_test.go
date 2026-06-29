package treedoctor

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeGit answers git invocations from a keyed reply table and records every argv. ancestor
// holds, keyed by worktree dir, whether `merge-base --is-ancestor HEAD <trunk>` returns 0.
type fakeGit struct {
	listOut  string
	ancestor map[string]bool // dir => merged
	dirty    map[string]string
	calls    [][]string
}

func (f *fakeGit) run(_ context.Context, dir string, args ...string) (string, int, error) {
	f.calls = append(f.calls, append([]string{dir}, args...))
	switch {
	case len(args) >= 2 && args[0] == "worktree" && args[1] == "list":
		return f.listOut, 0, nil
	case len(args) >= 3 && args[0] == "merge-base" && args[1] == "--is-ancestor":
		if f.ancestor[dir] {
			return "", 0, nil
		}
		return "", 1, nil
	case len(args) >= 2 && args[0] == "status" && args[1] == "--porcelain":
		return f.dirty[dir], 0, nil
	case len(args) >= 2 && args[0] == "worktree" && (args[1] == "remove" || args[1] == "prune"):
		return "", 0, nil
	}
	return "", 0, nil
}

func listPorcelain(entries ...[2]string) string {
	var b strings.Builder
	for _, e := range entries {
		b.WriteString("worktree " + e[0] + "\nHEAD " + e[1] + "\n\n")
	}
	return b.String()
}

func TestDiagnoseClassifiesWorktrees(t *testing.T) {
	main := t.TempDir()
	mergedDir := filepath.Join(main, "wt-merged")     // merged, not live => prunable
	liveDir := filepath.Join(main, "wt-live")         // merged but freshly touched => keep
	unmergedDir := filepath.Join(main, "wt-unmerged") // not merged => keep
	for _, d := range []string{mergedDir, liveDir, unmergedDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// Make liveDir look freshly touched; the others old.
	old := time.Now().Add(-time.Hour)
	writeAt(t, filepath.Join(mergedDir, "f"), old)
	writeAt(t, filepath.Join(unmergedDir, "f"), old)
	writeAt(t, filepath.Join(liveDir, "f"), time.Now())

	f := &fakeGit{
		listOut: listPorcelain(
			[2]string{main, "aaaa"},
			[2]string{mergedDir, "bbbb"},
			[2]string{liveDir, "cccc"},
			[2]string{unmergedDir, "dddd"},
		),
		ancestor: map[string]bool{mergedDir: true, liveDir: true, unmergedDir: false},
	}

	rep := Diagnose(context.Background(), f.run, Options{RepoRoot: main, Now: time.Now()})

	byPath := map[string]WorktreeState{}
	for _, w := range rep.Worktrees {
		byPath[w.Path] = w
	}
	if !byPath[main].IsMain || byPath[main].Prunable {
		t.Fatalf("main: %+v", byPath[main])
	}
	if !byPath[mergedDir].Prunable {
		t.Fatalf("merged+old should be prunable: %+v", byPath[mergedDir])
	}
	if byPath[liveDir].Prunable || !byPath[liveDir].Live {
		t.Fatalf("merged+live must be KEPT: %+v", byPath[liveDir])
	}
	if byPath[unmergedDir].Prunable {
		t.Fatalf("unmerged must be KEPT: %+v", byPath[unmergedDir])
	}
	if n := len(rep.PrunableWorktrees()); n != 1 {
		t.Fatalf("prunable count = %d, want 1", n)
	}
}

func TestDiagnoseDetectsStaleLock(t *testing.T) {
	main := t.TempDir()
	if err := os.MkdirAll(filepath.Join(main, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Seed a stale lock (dead PID).
	dead := deadPID(t)
	if err := os.WriteFile(filepath.Join(main, ".git", "fak-commit.lock"), []byte(strconv.Itoa(dead)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := &fakeGit{listOut: listPorcelain([2]string{main, "aaaa"})}
	rep := Diagnose(context.Background(), f.run, Options{RepoRoot: main})
	if !rep.StaleLockWedged() || rep.Lock.HolderPID != dead {
		t.Fatalf("stale lock not detected: %+v", rep.Lock)
	}
}

func TestSweepDryRunMakesNoChanges(t *testing.T) {
	main := t.TempDir()
	if err := os.MkdirAll(filepath.Join(main, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	lockPath := filepath.Join(main, ".git", "fak-commit.lock")
	if err := os.WriteFile(lockPath, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := &fakeGit{listOut: listPorcelain([2]string{main, "aaaa"})}

	_, actions := Sweep(context.Background(), f.run, Options{RepoRoot: main}, false)
	if len(actions) == 0 || !strings.Contains(actions[0], "would reap") {
		t.Fatalf("dry-run actions = %v, want a 'would reap' plan", actions)
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("dry-run removed the lock: %v", err)
	}
	// No worktree remove/prune should have been issued in dry-run.
	for _, c := range f.calls {
		if len(c) >= 3 && c[1] == "worktree" && (c[2] == "remove" || c[2] == "prune") {
			t.Fatalf("dry-run issued a mutating git call: %v", c)
		}
	}
}

func TestSweepApplyReapsStaleLock(t *testing.T) {
	main := t.TempDir()
	if err := os.MkdirAll(filepath.Join(main, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	lockPath := filepath.Join(main, ".git", "fak-commit.lock")
	if err := os.WriteFile(lockPath, []byte(strconv.Itoa(deadPID(t))+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	f := &fakeGit{listOut: listPorcelain([2]string{main, "aaaa"})}

	_, actions := Sweep(context.Background(), f.run, Options{RepoRoot: main}, true)
	if len(actions) == 0 || !strings.Contains(actions[0], "reaped stale commit lock") {
		t.Fatalf("apply actions = %v, want a 'reaped' result", actions)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("apply did not remove the stale lock: %v", err)
	}
}

// deadPID returns a pid that is no longer running, by spawning the test binary on a
// non-matching run filter (so it exits ~immediately) and reaping it.
func deadPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=NoSuchTestZZZ")
	if err := cmd.Start(); err != nil {
		t.Fatalf("spawn helper: %v", err)
	}
	pid := cmd.Process.Pid
	_ = cmd.Wait()
	return pid
}

func writeAt(t *testing.T, path string, mod time.Time) {
	t.Helper()
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
}
