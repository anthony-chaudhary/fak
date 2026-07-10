package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// wipTestRepo inits a throwaway git repo with one committed tracked file (built
// with plumbing so the test never touches the caller's commit-signing config) and
// returns the repo dir plus the tracked file path.
func wipTestRepo(t *testing.T) (string, string) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "wip@test.local"},
		{"config", "user.name", "wip test"},
	} {
		if _, err := gitWipOut(ctx, dir, nil, args...); err != nil {
			t.Fatalf("git %v: %v", args, err)
		}
	}
	file := filepath.Join(dir, "note.txt")
	if err := os.WriteFile(file, []byte("base line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", "note.txt"); err != nil {
		t.Fatal(err)
	}
	commit, err := wipPlumbBaseCommit(ctx, dir, "base")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", commit); err != nil {
		t.Fatal(err)
	}
	return dir, file
}

// TestWipSelfcheckPasses wires the runnable spine proof into `go test`.
func TestWipSelfcheckPasses(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runWipSelfcheck(&out, &errb, nil); rc != 0 {
		t.Fatalf("wip selfcheck rc=%d\nstdout: %s\nstderr: %s", rc, out.String(), errb.String())
	}
}

// TestWipCheckpointRestoreByteIdentical is the DONE-condition end to end: a tracked
// delta survives a destructive `git checkout -- .` and restores byte-for-byte.
func TestWipCheckpointRestoreByteIdentical(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)

	dirty := []byte("base line\nan uncommitted edit\n")
	if err := os.WriteFile(file, dirty, 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := wipCheckpoint(ctx, dir, "sess1", true, 1000)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if res.Clean {
		t.Fatal("checkpoint reported clean despite a delta")
	}
	if res.Object == "" || res.StartSHA == "" {
		t.Fatalf("checkpoint result missing object/start_sha: %+v", res)
	}

	// Destroy the working-tree delta.
	if _, err := gitWipOut(ctx, dir, nil, "checkout", "--", "."); err != nil {
		t.Fatalf("checkout: %v", err)
	}
	if b, _ := os.ReadFile(file); string(b) == string(dirty) {
		t.Fatal("checkout did not wipe the delta (test precondition)")
	}

	// Restore must reproduce the delta byte-identical.
	if rc, err := wipRestore(ctx, dir, "sess1", true, io.Discard); err != nil || rc != 0 {
		t.Fatalf("restore rc=%d err=%v", rc, err)
	}
	got, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, dirty) {
		t.Fatalf("restored bytes mismatch:\n got  %q\n want %q", got, dirty)
	}
}

// TestWipStatusListsCheckpoint asserts status folds a live checkpoint into a
// deterministic report keyed on the session.
func TestWipStatusListsCheckpoint(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("base line\nedit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 500); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	report, err := wipStatus(ctx, dir, 560)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if report.Count != 1 {
		t.Fatalf("count=%d, want 1", report.Count)
	}
	s := report.Sessions[0]
	if s.Session != "sessA" {
		t.Errorf("session=%q, want sessA", s.Session)
	}
	if s.AgeSeconds != 60 { // 560 - 500
		t.Errorf("age=%d, want 60", s.AgeSeconds)
	}
	if len(s.Leaves) == 0 {
		t.Error("expected at least one dirty leaf recorded")
	}
}

// TestWipCheckpointCleanTree reports Clean and writes no ref when there is no delta.
func TestWipCheckpointCleanTree(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	res, err := wipCheckpoint(ctx, dir, "sessClean", true, 1)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if !res.Clean {
		t.Fatal("expected Clean=true on an unmodified tree")
	}
	report, err := wipStatus(ctx, dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	if report.Count != 0 {
		t.Fatalf("clean checkpoint wrote a ref: count=%d", report.Count)
	}
}

// TestWipRestoreMissingSession errors (rc 1) for a session with no checkpoint.
func TestWipRestoreMissingSession(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)
	rc, err := wipRestore(ctx, dir, "ghost", false, io.Discard)
	if err == nil || rc == 0 {
		t.Fatalf("expected error for missing session, got rc=%d err=%v", rc, err)
	}
}
