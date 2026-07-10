package treedoctor

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// realGit runs an actual git command for the sweep-scratch tests, which must exercise the real
// `git clean -X` behavior end-to-end (a fake cannot prove that -X spares tracked/untracked WIP).
// It mirrors the cmd/fak gitRunner contract: combined output, exit code, error only when git
// could not be executed. The context carries the test's deadline, so the exec is bounded.
func realGit(ctx context.Context, dir string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(out), ee.ExitCode(), nil
	}
	return "", -1, err
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustGit(t *testing.T, ctx context.Context, dir string, args ...string) {
	t.Helper()
	out, code, err := realGit(ctx, dir, args...)
	if err != nil || code != 0 {
		t.Fatalf("git %v failed (code=%d err=%v): %s", args, code, err, out)
	}
}

// TestSweepScratchReapsOnlyGitignored is the #3211 witness: `git clean -Xdf` removes ONLY the
// gitignored scratch file, while a tracked file AND a real (non-ignored) untracked WIP file both
// SURVIVE — the -X ignored-only guarantee. The dry-run pass removes nothing but still LISTS the
// scratch path, so an operator can preview before reaping.
func TestSweepScratchReapsOnlyGitignored(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	ctx := context.Background()
	repo := t.TempDir()
	mustGit(t, ctx, repo, "init")

	// (a) a TRACKED file — staged into the index, so `git clean` must never touch it.
	tracked := filepath.Join(repo, "tracked.go")
	mustWrite(t, tracked, "package x\n")
	// A .gitignore declaring the scratch as disposable: the scratch file and a whole scratch dir.
	gitignore := filepath.Join(repo, ".gitignore")
	mustWrite(t, gitignore, "scratch.txt\n/_scratch/\n")
	mustGit(t, ctx, repo, "add", "--", "tracked.go", ".gitignore")

	// (b) a REAL untracked, NON-ignored file — an in-flight WIP the sweep must SPARE.
	untrackedWIP := filepath.Join(repo, "untracked_wip.go")
	mustWrite(t, untrackedWIP, "package x\n")

	// (c) gitignored scratch — a loose file AND a file inside an ignored dir (proves -d recursion).
	scratch := filepath.Join(repo, "scratch.txt")
	mustWrite(t, scratch, "junk\n")
	scratchDirFile := filepath.Join(repo, "_scratch", "note.txt")
	mustWrite(t, scratchDirFile, "more junk\n")

	// --- dry-run: nothing deleted, but the scratch is listed. ---
	dry, err := SweepScratch(ctx, realGit, repo, true)
	if err != nil {
		t.Fatalf("dry-run SweepScratch: %v", err)
	}
	if !dry.DryRun {
		t.Fatalf("dry.DryRun = false, want true")
	}
	if !containsPath(dry.Removed, "scratch.txt") {
		t.Fatalf("dry-run did not list the gitignored scratch: %v", dry.Removed)
	}
	if containsPath(dry.Removed, "tracked.go") || containsPath(dry.Removed, "untracked_wip.go") {
		t.Fatalf("dry-run listed a tracked/real-untracked file: %v", dry.Removed)
	}
	// Dry-run removed nothing on disk.
	for _, p := range []string{tracked, gitignore, untrackedWIP, scratch, scratchDirFile} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("dry-run deleted %s: %v", p, err)
		}
	}

	// --- real sweep: only the gitignored scratch is reaped. ---
	got, err := SweepScratch(ctx, realGit, repo, false)
	if err != nil {
		t.Fatalf("SweepScratch: %v", err)
	}
	if got.DryRun {
		t.Fatalf("real sweep reported DryRun=true")
	}
	if !containsPath(got.Removed, "scratch.txt") {
		t.Fatalf("real sweep did not report reaping scratch.txt: %v", got.Removed)
	}
	if containsPath(got.Removed, "tracked.go") || containsPath(got.Removed, "untracked_wip.go") {
		t.Fatalf("real sweep reported reaping a tracked/real-untracked file: %v", got.Removed)
	}
	// (a) tracked and (b) real-untracked WIP SURVIVE.
	if _, err := os.Stat(tracked); err != nil {
		t.Fatalf("tracked file was reaped: %v", err)
	}
	if _, err := os.Stat(untrackedWIP); err != nil {
		t.Fatalf("real untracked WIP file was reaped: %v", err)
	}
	// (c) the gitignored scratch is GONE (both the loose file and the ignored dir).
	if _, err := os.Stat(scratch); !os.IsNotExist(err) {
		t.Fatalf("gitignored scratch survived the reap: err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(repo, "_scratch")); !os.IsNotExist(err) {
		t.Fatalf("gitignored scratch dir survived the reap: err=%v", err)
	}
}

// containsPath reports whether want is in paths, matching git's forward-slash output against a
// possibly OS-separated expectation (git clean emits "_scratch/" and forward slashes on Windows).
func containsPath(paths []string, want string) bool {
	for _, p := range paths {
		p = filepath.ToSlash(p)
		if p == want || p == want+"/" {
			return true
		}
	}
	return false
}
