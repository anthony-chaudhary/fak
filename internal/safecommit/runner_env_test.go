package safecommit

import (
	"context"
	"io"
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

