package treedoctor

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
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

func TestSweepPreservesDirtyOrphanWorkerUntilArchived(t *testing.T) {
	main := t.TempDir()
	now := time.Now()
	worker := filepath.Join(t.TempDir(), "fak-worker-wt-tools-dirty")
	file := filepath.Join(worker, "unfinished.go")
	if err := os.MkdirAll(worker, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("package unfinished\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeAt(t, file, now.Add(-time.Hour))
	git := &fakeGit{
		listOut:  listPorcelain([2]string{main, "aaa"}, [2]string{worker, "bbb"}),
		ancestor: map[string]bool{worker: false},
		dirty:    map[string]string{worker: "?? unfinished.go\n"},
	}

	rep, actions := Sweep(context.Background(), git.run, Options{RepoRoot: main, Now: now}, true)

	if len(rep.Worktrees) != 2 || !rep.Worktrees[1].Archive {
		t.Fatalf("dirty worker must require archive: %+v", rep.Worktrees)
	}
	if got := strings.Join(actions, "\n"); !strings.Contains(got, "archive required before pruning") {
		t.Fatalf("missing archive-required action: %v", actions)
	}
	for _, call := range git.calls {
		if len(call) >= 3 && call[1] == "worktree" && call[2] == "remove" {
			t.Fatalf("dirty worker was removed without archive: %v", call)
		}
	}
}

// TestSweepWorkerWorktree is the #3179 witness: an orphaned fak-worker-wt-* worktree
// (worker crashed / host died mid-wave, so its in-worktree commit left HEAD OFF the trunk)
// is reaped by the sweep even though it is NOT merged, while a live marker worktree and a
// non-marker worktree are both left untouched. The marker check is the load-bearing
// guardrail: only OUR disposable worker trees get the relaxed (merge-status-agnostic) reap.
func TestSweepWorkerWorktree(t *testing.T) {
	main := t.TempDir()
	orphan := filepath.Join(main, "fak-worker-wt-cmd-deadbeef00")     // marker, crashed, unmerged => reap
	liveWorker := filepath.Join(main, "fak-worker-wt-cmd-abc1234567") // marker, freshly touched => keep
	scratch := filepath.Join(main, "wt-scratch")                      // non-marker, unmerged => keep
	for _, d := range []string{orphan, liveWorker, scratch} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	old := time.Now().Add(-time.Hour)
	writeAt(t, filepath.Join(orphan, "f"), old)
	writeAt(t, filepath.Join(scratch, "f"), old)
	writeAt(t, filepath.Join(liveWorker, "f"), time.Now())

	// ancestor is empty => NONE of the three is an ancestor of the trunk (all "unmerged").
	// So the merged-only rule alone would keep all three; only the marker rule reaps the orphan.
	f := &fakeGit{
		listOut: listPorcelain(
			[2]string{main, "aaaa"},
			[2]string{orphan, "bbbb"},
			[2]string{liveWorker, "cccc"},
			[2]string{scratch, "dddd"},
		),
		ancestor: map[string]bool{},
	}

	rep := Diagnose(context.Background(), f.run, Options{RepoRoot: main, Now: time.Now()})
	byPath := map[string]WorktreeState{}
	for _, w := range rep.Worktrees {
		byPath[w.Path] = w
	}
	if !byPath[orphan].IsWorker || !byPath[orphan].Prunable {
		t.Fatalf("orphan marker worktree must be prunable even when unmerged: %+v", byPath[orphan])
	}
	if byPath[liveWorker].Prunable || !byPath[liveWorker].Live {
		t.Fatalf("live worker worktree must be KEPT: %+v", byPath[liveWorker])
	}
	if byPath[scratch].Prunable || byPath[scratch].IsWorker {
		t.Fatalf("non-marker worktree must be KEPT: %+v", byPath[scratch])
	}
	if n := len(rep.PrunableWorktrees()); n != 1 {
		t.Fatalf("prunable count = %d, want exactly 1 (the orphan)", n)
	}

	// Apply the sweep: ONLY the orphan is force-removed, and `git worktree prune` follows.
	_, actions := Sweep(context.Background(), f.run, Options{RepoRoot: main, Now: time.Now()}, true)
	var removed []string
	prunedAfter := false
	for _, c := range f.calls {
		if len(c) >= 5 && c[1] == "worktree" && c[2] == "remove" && c[3] == "--force" {
			removed = append(removed, c[4])
		}
		if len(c) >= 3 && c[1] == "worktree" && c[2] == "prune" {
			prunedAfter = true
		}
	}
	if len(removed) != 1 || removed[0] != orphan {
		t.Fatalf("sweep force-removed %v, want exactly [%s]", removed, orphan)
	}
	if !prunedAfter {
		t.Fatalf("sweep did not run `git worktree prune` after reaping the orphan")
	}
	if joined := strings.Join(actions, "\n"); !strings.Contains(joined, orphan) {
		t.Fatalf("sweep actions %v do not name the reaped orphan", actions)
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

func TestDiagnoseClassifiesLockResidue(t *testing.T) {
	main := t.TempDir()
	gitDir := filepath.Join(main, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	// Two renamed-aside orphan residue files (one aged, one fresh) alongside a genuine
	// ACTIVE lock name — which ends in exactly `.lock` and must NEVER be seen as residue.
	writeAt(t, filepath.Join(gitDir, "HEAD.lock.orphan-recovered-06344"), now.Add(-time.Hour))
	writeAt(t, filepath.Join(gitDir, "packed-refs.lock.orphan-1437010601"), now)
	writeAt(t, filepath.Join(gitDir, "index.lock"), now.Add(-time.Hour))

	rep := Diagnose(context.Background(), (&fakeGit{}).run, Options{RepoRoot: main, Now: now})

	sweepable := map[string]bool{}
	for _, r := range rep.LockResidue {
		sweepable[filepath.Base(r.Path)] = r.Sweepable
	}
	if _, seen := sweepable["index.lock"]; seen {
		t.Fatalf("active lock index.lock misclassified as orphan residue: %+v", rep.LockResidue)
	}
	if sw, ok := sweepable["HEAD.lock.orphan-recovered-06344"]; !ok || !sw {
		t.Fatalf("aged residue not marked sweepable: %+v", rep.LockResidue)
	}
	if fr, ok := sweepable["packed-refs.lock.orphan-1437010601"]; !ok || fr {
		t.Fatalf("fresh residue should be reported but NOT sweepable: %+v", rep.LockResidue)
	}
	if n := len(rep.SweepableLockResidue()); n != 1 {
		t.Fatalf("SweepableLockResidue() = %d, want 1", n)
	}
}

func TestSweepApplyRemovesAgedLockResidueOnly(t *testing.T) {
	main := t.TempDir()
	gitDir := filepath.Join(main, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	aged := filepath.Join(gitDir, "HEAD.lock.orphan-recovered-1")
	fresh := filepath.Join(gitDir, "config.lock.orphan-2")
	writeAt(t, aged, now.Add(-time.Hour))
	writeAt(t, fresh, now)

	_, actions := Sweep(context.Background(), (&fakeGit{}).run, Options{RepoRoot: main, Now: now}, true)

	if _, err := os.Stat(aged); !os.IsNotExist(err) {
		t.Fatalf("aged residue not removed: err=%v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh residue wrongly removed: %v", err)
	}
	if joined := strings.Join(actions, "\n"); !strings.Contains(joined, "swept orphan lock residue") {
		t.Fatalf("no sweep action recorded: %v", actions)
	}
}

func TestSweepDryRunKeepsLockResidue(t *testing.T) {
	main := t.TempDir()
	gitDir := filepath.Join(main, ".git")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	aged := filepath.Join(gitDir, "HEAD.lock.orphan-recovered-9")
	writeAt(t, aged, now.Add(-time.Hour))

	_, actions := Sweep(context.Background(), (&fakeGit{}).run, Options{RepoRoot: main, Now: now}, false)

	if _, err := os.Stat(aged); err != nil {
		t.Fatalf("dry-run removed residue: %v", err)
	}
	if joined := strings.Join(actions, "\n"); !strings.Contains(joined, "would sweep orphan lock residue") {
		t.Fatalf("dry-run did not plan a sweep: %v", actions)
	}
}

func TestDiagnoseKeepsStaleWorkerWorktreeWithLiveOwner(t *testing.T) {
	now := time.Date(2026, 8, 17, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	wt := filepath.Join(filepath.Dir(root), workerworktree.WorktreeMarker+"-live-owner")
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	old := now.Add(-2 * DefaultLiveWindow)
	if err := os.Chtimes(wt, old, old); err != nil {
		t.Fatal(err)
	}
	stamp := workerworktree.OwnerStamp{Schema: "fak-worker-worktree-owner/1", PID: 222, LeaseID: "lease", CreatedAt: old}
	raw, err := json.Marshal(stamp)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(workerworktree.OwnerStampPath(wt)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(workerworktree.OwnerStampPath(wt), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if live, known := workerworktree.OwnerProcessLive(wt, func(pid int) bool { return pid == 222 }); !known || !live {
		t.Fatalf("owner stamp not readable: live=%v known=%v path=%s", live, known, workerworktree.OwnerStampPath(wt))
	}
	git := &fakeGit{listOut: listPorcelain([2]string{root, "abc"}, [2]string{wt, "abc"}), ancestor: map[string]bool{wt: true}, dirty: map[string]string{}}
	report := Diagnose(context.Background(), git.run, Options{RepoRoot: root, Now: now, ProcessAlive: func(pid int) bool { return pid == 222 }})
	if len(report.Worktrees) != 2 || !report.Worktrees[1].Live || report.Worktrees[1].Keep != "live (owner process)" || report.Worktrees[1].Prunable {
		t.Fatalf("worker report = %+v, want live owner protection", report.Worktrees)
	}
}
