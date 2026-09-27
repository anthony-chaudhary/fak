package safecommit

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// Env guards for the killed-commit witness. The pre-commit hook re-execs THIS test binary
// as a hook child that blocks the way the real 2-3 minute boundary hooks do; the guard
// selects that role in the re-exec'd binary and names the file it publishes its pid to.
// realRunner passes os.Environ() down to git, and git passes it to the hook, so the test
// publishes both through t.Setenv.
const (
	killedCommitSleeperEnv     = "FAK_SAFECOMMIT_KILLED_COMMIT_SLEEPER"
	killedCommitSleeperPIDFile = "FAK_SAFECOMMIT_KILLED_COMMIT_SLEEPER_PID"
)

// TestHelperKilledCommitHookSleeper is the hook's child: a real OS process that outlives
// any reasonable deadline unless the tree kill reaches it. Not a test — the env guard
// turns the re-exec'd binary into the sleeper.
func TestHelperKilledCommitHookSleeper(t *testing.T) {
	if os.Getenv(killedCommitSleeperEnv) != "1" {
		return
	}
	pidFile := os.Getenv(killedCommitSleeperPIDFile)
	if pidFile == "" {
		os.Exit(3)
	}
	tmp := pidFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		os.Exit(4)
	}
	if err := os.Rename(tmp, pidFile); err != nil {
		os.Exit(5)
	}
	time.Sleep(5 * time.Minute)
	os.Exit(0)
}

// TestRealRunnerKilledCommitReapsHookTreeAndRollsBackLocks reproduces the 2026-09-26
// COMMIT_STALLED wedge end to end with real git: a partial `git commit -- <path>` sits in
// a pre-commit hook whose child blocks, and the commit's context is cancelled (what the
// post-validation deadline does). Before the fix the cancel killed git.exe alone: the hook
// child kept running and holding the output pipes, and git left .git/index.lock plus
// .git/next-index-<pid>.lock behind for every peer to trip on. The witness is the OS
// process table (the hook child is dead) and the git dir (both locks are gone, HEAD did
// not move, and the next commit lands).
func TestRealRunnerKilledCommitReapsHookTreeAndRollsBackLocks(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("no real tree-kill oracle for GOOS=%s", runtime.GOOS)
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	events := captureReapEvents(t)

	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	tempRepoGit(t, root, "init", "-q", repo)
	configureTestGitRepo(t, repo, "fak test", "fak-test@example.invalid")
	hooksDir := filepath.Join(repo, ".git", "hooks")
	// Pin the hooks dir locally so a host-wide core.hooksPath cannot bypass the hook.
	tempRepoGit(t, repo, "config", "core.hooksPath", filepath.ToSlash(hooksDir))
	writeTempRepoFile(t, filepath.Join(repo, "f.txt"), "one\n")
	tempRepoGit(t, repo, "add", "--", "f.txt")
	tempRepoGit(t, repo, "commit", "-qm", "seed")
	head0 := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	writeTempRepoFile(t, filepath.Join(repo, "f.txt"), "two\n")

	hook := filepath.Join(hooksDir, "pre-commit")
	writeTempRepoFile(t, hook, "#!/bin/sh\n\""+filepath.ToSlash(self)+"\" '-test.run=^TestHelperKilledCommitHookSleeper$'\n")
	if err := os.Chmod(hook, 0o755); err != nil {
		t.Fatal(err)
	}
	pidFile := filepath.Join(root, "sleeper.pid")
	t.Setenv(killedCommitSleeperEnv, "1")
	t.Setenv(killedCommitSleeperPIDFile, pidFile)

	type ran struct {
		out  string
		code int
		err  error
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan ran, 1)
	go func() {
		out, code, rerr := realRunner(ctx, repo, "commit", "-m", "killed mid-hook", "--", "f.txt")
		done <- ran{out, code, rerr}
	}()

	sleeper := 0
	deadline := time.Now().Add(90 * time.Second)
	for sleeper <= 0 && time.Now().Before(deadline) {
		select {
		case r := <-done:
			t.Fatalf("git returned before the hook child started (code=%d err=%v): %s", r.code, r.err, r.out)
		case <-time.After(50 * time.Millisecond):
		}
		if raw, rerr := os.ReadFile(pidFile); rerr == nil {
			sleeper, _ = strconv.Atoi(strings.TrimSpace(string(raw)))
		}
	}
	if sleeper <= 0 {
		t.Fatalf("hook child never published its pid to %s", pidFile)
	}
	t.Cleanup(func() {
		if processAlive(sleeper) {
			_, _ = procguard.KillPID(sleeper)
		}
	})

	// Precondition: git is inside the hook holding BOTH partial-commit locks, so the cancel
	// below exercises exactly the orphan shape the incident left.
	gitDir := filepath.Join(repo, ".git")
	if !lockFileExists(filepath.Join(gitDir, "index.lock")) {
		t.Fatal("precondition: index.lock not held while the pre-commit hook runs")
	}
	if nexts, _ := filepath.Glob(filepath.Join(gitDir, NextIndexLockGlob)); len(nexts) != 1 {
		t.Fatalf("precondition: want exactly one next-index-<pid>.lock while the hook runs, got %v", nexts)
	}

	cancel() // the commit deadline firing mid-hook

	var r ran
	select {
	case r = <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("realRunner did not return after cancel: the hook child still holds git's output pipes")
	}
	if r.err == nil && r.code == 0 {
		t.Fatalf("a commit killed mid-hook reported success: %s", r.out)
	}

	reaped := false
	for end := time.Now().Add(30 * time.Second); time.Now().Before(end); time.Sleep(50 * time.Millisecond) {
		if !processAlive(sleeper) {
			reaped = true
			break
		}
	}
	if !reaped {
		t.Fatalf("hook child pid %d survived the commit cancel: only git was killed, its hook tree was orphaned", sleeper)
	}
	if lockFileExists(filepath.Join(gitDir, "index.lock")) {
		t.Fatalf("killed commit left .git/index.lock behind (events: %v)", *events)
	}
	if nexts, _ := filepath.Glob(filepath.Join(gitDir, NextIndexLockGlob)); len(nexts) != 0 {
		t.Fatalf("killed commit left next-index residue %v (events: %v)", nexts, *events)
	}
	if got := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD")); got != head0 {
		t.Fatalf("a killed commit moved HEAD %s -> %s", head0, got)
	}
	// Which party cleared the locks is a race the fix deliberately tolerates: the tree reaper
	// kills descendants first, so git sometimes sees its hook die and runs its own rollback
	// before it is killed; otherwise the runner's post-kill rollback clears them. Either way no
	// lock may survive, which the checks above pin. Record which path this run took.
	if strings.Contains(strings.Join(*events, "\n"), "INDEX_LOCK_ROLLED_BACK") {
		t.Logf("locks cleared by the runner's post-kill rollback: %v", *events)
	} else {
		t.Logf("locks cleared by git's own rollback before the kill reached it (events: %v)", *events)
	}

	// The lane is usable again: with the slow hook gone, the same partial commit lands.
	if err := os.Remove(hook); err != nil {
		t.Fatal(err)
	}
	tempRepoGit(t, repo, "commit", "-qm", "after rollback", "--", "f.txt")
	if got := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD")); got == head0 {
		t.Fatal("the follow-up commit did not land after the rollback")
	}
}
