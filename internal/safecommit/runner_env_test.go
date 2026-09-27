package safecommit

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"slices"
	"testing"
)

// The commit hot path stays drained only because every git invocation safecommit makes
// carries GIT_OPTIONAL_LOCKS=0 — so a read probe (`git status`/`diff`/`rev-parse`) never
// opportunistically takes .git/index.lock and collides with a concurrent writer on a busy
// shared tree. That single env line is the whole "drained by default" property of the read
// side; a careless edit dropping it silently re-opens the burst-time index.lock contention
// class that once wedged the shared-trunk commit lane. These guards pin the invariant at
// the newGitCmd seam without ever spawning git.

// TestNewGitCmdAlwaysDisablesOptionalLocks: GIT_OPTIONAL_LOCKS=0 rides EVERY invocation —
// read probe or write — never just some of them.
func TestNewGitCmdAlwaysDisablesOptionalLocks(t *testing.T) {
	cases := [][]string{
		{"status", "--porcelain"},
		{"rev-parse", "--absolute-git-dir"},
		{"symbolic-ref", "--short", "HEAD"},
		{"diff", "--cached", "--name-only"},
		{"add", "-A"},
		{"commit", "-m", "x"},
		{"push"},
		nil, // a bare `git` with no args must still be pinned
	}
	for _, args := range cases {
		cmd := newGitCmd(context.Background(), "", args...)
		if !slices.Contains(cmd.Env, "GIT_OPTIONAL_LOCKS=0") {
			t.Errorf("newGitCmd(%v): GIT_OPTIONAL_LOCKS=0 missing from env — read probes could take .git/index.lock", args)
		}
	}
}

// TestNewGitCmdVettedMarkerOnlyOnCommit: the BARE_COMMIT_SWEEP handshake
// (FAK_SAFECOMMIT_VETTED=1, issue #3615) rides `git commit` and nothing else. If it leaked
// onto a raw read/push it would defeat the gate; if it were dropped from commit, safecommit's
// own vetted commit would be re-flagged as an unvetted bare sweep.
func TestNewGitCmdVettedMarkerOnlyOnCommit(t *testing.T) {
	commit := newGitCmd(context.Background(), "", "commit", "-m", "x")
	if !slices.Contains(commit.Env, "FAK_SAFECOMMIT_VETTED=1") {
		t.Error("git commit: FAK_SAFECOMMIT_VETTED=1 missing — the BARE_COMMIT_SWEEP handshake was dropped")
	}
	for _, args := range [][]string{{"status"}, {"push"}, {"add", "-A"}, {"diff"}, nil} {
		cmd := newGitCmd(context.Background(), "", args...)
		if slices.Contains(cmd.Env, "FAK_SAFECOMMIT_VETTED=1") {
			t.Errorf("newGitCmd(%v): FAK_SAFECOMMIT_VETTED=1 leaked onto a non-commit invocation", args)
		}
	}
}

// TestNewGitCmdDirWiring: dir is applied when set and left empty otherwise (an empty dir
// means "run in the process cwd", never a spurious chdir).
func TestNewGitCmdDirWiring(t *testing.T) {
	if got := newGitCmd(context.Background(), "/some/dir", "status").Dir; got != "/some/dir" {
		t.Errorf("Dir = %q, want /some/dir", got)
	}
	if got := newGitCmd(context.Background(), "", "status").Dir; got != "" {
		t.Errorf("empty dir must stay empty, got %q", got)
	}
}

// TestNewGitCmdTreeCancel: a cancelled git invocation reaps git's whole descendant tree
// (hooks and whatever they launched), not git alone, and Wait's pipe drain is bounded. A
// bare CommandContext kill left the 2026-09-26 pre-commit hook tree running and holding
// the output pipes after the commit deadline fired.
func TestNewGitCmdTreeCancel(t *testing.T) {
	var reaped []int
	orig := killGitTree
	killGitTree = func(pid int) (bool, string) {
		reaped = append(reaped, pid)
		return true, "stub: tree reaped"
	}
	t.Cleanup(func() { killGitTree = orig })

	for _, args := range [][]string{{"commit", "-m", "x", "--", "a"}, {"push"}, {"status"}} {
		cmd := newGitCmd(context.Background(), "", args...)
		if cmd.Cancel == nil {
			t.Fatalf("newGitCmd(%v): Cancel is nil — a deadline would kill git alone and orphan its hook tree", args)
		}
		if cmd.WaitDelay != gitWaitDelay {
			t.Fatalf("newGitCmd(%v): WaitDelay = %s, want %s — a surviving descendant could hang Wait on the pipes", args, cmd.WaitDelay, gitWaitDelay)
		}
		self, err := os.FindProcess(os.Getpid())
		if err != nil {
			t.Fatal(err)
		}
		cmd.Process = self // the stubbed reaper never signals it
		if err := cmd.Cancel(); err != nil {
			t.Fatalf("newGitCmd(%v): Cancel returned %v after a successful tree reap", args, err)
		}
	}
	if len(reaped) != 3 || reaped[0] != os.Getpid() {
		t.Fatalf("Cancel must hand git's pid to the tree reaper each time, got %v", reaped)
	}

	// A deadline that fires after Wait already collected git must not reap anything (the
	// pid may be recycled) and must answer os.ErrProcessDone, so exec does not report a
	// commit that landed as a context error.
	done := exec.Command(os.Args[0], "-test.run=^$")
	if err := done.Run(); err != nil {
		t.Fatalf("start an exited stand-in process: %v", err)
	}
	cmd := newGitCmd(context.Background(), "", "commit", "-m", "x")
	cmd.Process = done.Process
	reaped = nil
	if err := cmd.Cancel(); !errors.Is(err, os.ErrProcessDone) {
		t.Fatalf("Cancel on an already-collected process = %v, want os.ErrProcessDone", err)
	}
	if len(reaped) != 0 {
		t.Fatalf("Cancel walked a finished pid %v — it may belong to an unrelated process by now", reaped)
	}
}

// TestNewGitCmdEmptyStdin: background git subprocesses must have an empty Stdin
// so they never block or hang waiting for terminal input (e.g. prompts or credentials).
func TestNewGitCmdEmptyStdin(t *testing.T) {
	cmd := newGitCmd(context.Background(), "", "status")
	if cmd.Stdin == nil {
		t.Fatal("cmd.Stdin must be non-nil to prevent hanging on stdin")
	}
	b, err := io.ReadAll(cmd.Stdin)
	if err != nil || len(b) != 0 {
		t.Errorf("cmd.Stdin must be empty reader, got %d bytes (err: %v)", len(b), err)
	}
}
