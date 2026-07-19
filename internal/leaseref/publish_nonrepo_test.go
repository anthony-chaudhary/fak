package leaseref

// publish_nonrepo_test.go pins the fix for the non-repo-cwd publish bug: a serve whose
// process cwd is NOT a git repo (a dedicated GPU serve box running from /home/USER) makes
// `git hash-object -w` exit 128 on EVERY durable-session refresh, which used to surface as
// ~75 "hash-object exited 128" log lines / 5 min AND never published the cross-machine
// side ref. The fix probes the object DB ONCE and disables the publisher cleanly. These
// tests exercise the REAL-git path (NewInDir), the only path the probe applies to.

import (
	"bytes"
	"context"
	"log"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// gitAvailable reports whether the real git binary can be executed — the two real-git
// tests below skip cleanly when it cannot (the package's fake-runner tests still cover the
// algorithm without git).
func gitAvailable(t *testing.T) bool {
	t.Helper()
	_, err := exec.LookPath("git")
	return err == nil
}

// dirHasGitDir reports whether `git rev-parse --git-dir` succeeds in dir — the same
// predicate the fix probes. Used only to SKIP the no-repo test if the tempdir happens to
// sit inside a parent repo (so the "no object DB" precondition genuinely holds).
func dirHasGitDir(dir string) bool {
	cmd := exec.Command("git", "rev-parse", "--git-dir")
	cmd.Dir = dir
	return cmd.Run() == nil
}

// TestPublishSessionNonRepoDisablesCleanly proves case (a): a real-git store whose dir is
// NOT a git repo publishes with NO error and NO per-refresh spam — the publisher disables
// itself once. Before the fix each call returned `leaseref: hash-object exited 128`.
func TestPublishSessionNonRepoDisablesCleanly(t *testing.T) {
	if !gitAvailable(t) {
		t.Skip("git not available; no-object-DB probe path needs the real git binary")
	}
	dir := t.TempDir()
	if dirHasGitDir(dir) {
		t.Skip("tempdir resolves a parent git dir; cannot exercise the no-object-DB path here")
	}

	// Capture the package's one-time setup log so we can prove it fires AT MOST ONCE (the
	// anti-spam property), then restore the default output.
	var buf bytes.Buffer
	oldOut, oldFlags := log.Writer(), log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(oldOut); log.SetFlags(oldFlags) })

	s := NewInDir(dir)
	ctx := context.Background()
	d := SessionDescriptor{ID: "sess-nonrepo", Host: "gpu-box", PCBState: "RUNNING", UpdatedAt: 1000, TTLSecs: 300}

	// Refresh several times, exactly as a durable serve does — none may error.
	const refreshes = 4
	for i := 0; i < refreshes; i++ {
		ref, err := s.PublishSession(ctx, d)
		if err != nil {
			t.Fatalf("PublishSession refresh %d in non-repo dir: got error %v, want nil (publisher must disable cleanly)", i, err)
		}
		if ref != d.Ref() {
			t.Fatalf("PublishSession refresh %d ref = %q, want %q", i, ref, d.Ref())
		}
	}

	// Nothing was actually published (no object DB to write into) and reading is not an
	// error either — absence is the honest post-state.
	if _, ok, err := s.GetSession(ctx, "sess-nonrepo"); err != nil || ok {
		t.Fatalf("GetSession after non-repo publish: ok=%v err=%v, want (false, nil) — nothing should have been written", ok, err)
	}

	// The disabling notice logged AT MOST ONCE across all refreshes — the whole point of
	// the fix (was ~75 error lines / 5 min).
	n := strings.Count(buf.String(), "session side-ref publishing disabled")
	if n != 1 {
		t.Fatalf("setup notice logged %d times across %d refreshes, want exactly 1 (no per-refresh spam):\n%s", n, refreshes, buf.String())
	}
}

// TestPublishSessionRealRepoPreserved proves case (b): in a genuine git repo the existing
// publish behavior is UNCHANGED — the descriptor blob is written and reads back. This is
// the happy path the fix must not disturb.
func TestPublishSessionRealRepoPreserved(t *testing.T) {
	if !gitAvailable(t) {
		t.Skip("git not available; real-repo publish path needs the real git binary")
	}
	dir := t.TempDir()
	if out, err := gitInit(dir); err != nil {
		t.Fatalf("git init %s: %v (%s)", dir, err, out)
	}

	s := NewInDir(dir)
	ctx := context.Background()
	d := SessionDescriptor{ID: "sess-realrepo", Host: "dev-box", PCBState: "RUNNING", UpdatedAt: 2000, TTLSecs: 300}

	ref, err := s.PublishSession(ctx, d)
	if err != nil {
		t.Fatalf("PublishSession in real repo: %v", err)
	}
	if ref != d.Ref() {
		t.Fatalf("PublishSession ref = %q, want %q", ref, d.Ref())
	}

	got, ok, err := s.GetSession(ctx, "sess-realrepo")
	if err != nil || !ok {
		t.Fatalf("GetSession after real publish: ok=%v err=%v, want a descriptor", ok, err)
	}
	if got.Host != "dev-box" || got.PCBState != "RUNNING" || got.TTLSecs != 300 {
		t.Fatalf("GetSession returned %+v, want the published descriptor", got)
	}
}

// gitInit initializes a bare-minimum git repo in dir. hash-object -w and update-ref need
// only an object DB, not a committer identity, so no user.name/email config is required.
func gitInit(dir string) (string, error) {
	cmd := exec.Command("git", "init")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_TERMINAL_PROMPT=0")
	out, err := cmd.CombinedOutput()
	return string(out), err
}
