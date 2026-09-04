package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/wipref"
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

// TestWipCheckpointUntrackedRoundTrip is the #4336 DONE condition: a tree whose
// ONLY change is a new untracked file must checkpoint (not report clean, ref
// written), and after `git clean -fd` wipes it, restore --apply must reproduce
// the file byte-identical. Ignored untracked files must stay out of the snapshot.
func TestWipCheckpointUntrackedRoundTrip(t *testing.T) {
	ctx := context.Background()
	dir, _ := wipTestRepo(t)

	fresh := filepath.Join(dir, "newleaf.go.txt")
	body := []byte("package newleaf // brand-new WIP, never git-added\n")
	if err := os.WriteFile(fresh, body, 0o644); err != nil {
		t.Fatal(err)
	}
	// An ignored sibling must NOT be swept into the checkpoint (.gitignore boundary).
	if err := os.WriteFile(filepath.Join(dir, ".gitignore"), []byte("junk.bin\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "junk.bin"), []byte("ignored artifact"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := wipCheckpoint(ctx, dir, "sessU", true, 1000)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if res.Clean {
		t.Fatal("checkpoint reported clean: a new-file-only WIP was dropped under a false all-clear")
	}
	if res.Object == "" {
		t.Fatalf("checkpoint wrote no ref for a pure-untracked delta: %+v", res)
	}
	names, err := gitWipOut(ctx, dir, nil, "diff", "--name-only", res.Object+"^1", res.Object)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(names, "newleaf.go.txt") {
		t.Fatalf("checkpoint tree missing the untracked file, got: %q", names)
	}
	if strings.Contains(names, "junk.bin") {
		t.Fatalf("checkpoint swept in a .gitignore'd path: %q", names)
	}

	// Destroy the untracked WIP the way a `git clean -fd` would.
	if _, err := gitWipOut(ctx, dir, nil, "clean", "-fd"); err != nil {
		t.Fatalf("git clean: %v", err)
	}
	if _, err := os.Stat(fresh); !os.IsNotExist(err) {
		t.Fatal("git clean did not remove the untracked file (test precondition)")
	}

	if rc, err := wipRestore(ctx, dir, "sessU", true, io.Discard); err != nil || rc != 0 {
		t.Fatalf("restore rc=%d err=%v", rc, err)
	}
	got, err := os.ReadFile(fresh)
	if err != nil {
		t.Fatalf("restored file unreadable: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("restored bytes mismatch:\n got  %q\n want %q", got, body)
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

func TestWipAutoCheckpointDebouncesUnchangedTree(t *testing.T) {
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errout bytes.Buffer
	args := []string{"-C", dir, "--session", "autosess", "--reason", "stop", "--json"}
	if code := runWipAutoCheckpoint(&out, &errout, args); code != 0 {
		t.Fatalf("first code=%d err=%s", code, errout.String())
	}
	firstBytes, err := gitWipOut(context.Background(), dir, nil, "rev-parse", wipref.SessionRef("autosess"))
	if err != nil {
		t.Fatal(err)
	}
	first := strings.TrimSpace(string(firstBytes))
	out.Reset()
	errout.Reset()
	if code := runWipAutoCheckpoint(&out, &errout, args); code != 0 {
		t.Fatalf("second code=%d err=%s", code, errout.String())
	}
	secondBytes, err := gitWipOut(context.Background(), dir, nil, "rev-parse", wipref.SessionRef("autosess"))
	if err != nil {
		t.Fatal(err)
	}
	second := strings.TrimSpace(string(secondBytes))
	if second != first {
		t.Fatalf("unchanged tree minted duplicate ref: %s -> %s", first, second)
	}
}

func TestWipReapPersistsHistoryAfterDeletingCheckpointRef(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("landed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	checkpoint, err := wipCheckpoint(ctx, dir, "reaped-history", true, 1_700_000_000)
	if err != nil || checkpoint.Object == "" {
		t.Fatalf("checkpoint=%+v err=%v", checkpoint, err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "add", "note.txt"); err != nil {
		t.Fatal(err)
	}
	commit, err := wipPlumbBaseCommit(ctx, dir, "land checkpoint")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", "HEAD", commit); err != nil {
		t.Fatal(err)
	}

	result, err := wipReap(ctx, dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Reaped) != 1 {
		t.Fatalf("reap=%+v", result)
	}
	if _, err := gitWipOut(ctx, dir, nil, "show-ref", "--verify", checkpoint.Ref); err == nil {
		t.Fatalf("checkpoint ref %s still exists", checkpoint.Ref)
	}
	receipts, err := wiplifecycle.List(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(receipts) != 1 || receipts[0].Kind != "checkpoint-reap" || receipts[0].FinishedAt == "" {
		t.Fatalf("post-ref lifecycle history missing: %#v", receipts)
	}
}

func TestWipReapPurgesStaleOrphanedCheckpoints(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("unlanded work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	checkpoint, err := wipCheckpoint(ctx, dir, "stale-session", true, time.Now().Unix())
	if err != nil || checkpoint.Object == "" {
		t.Fatalf("checkpoint=%+v err=%v", checkpoint, err)
	}

	// Give a small elapsed window so commit creation is older than 50ms
	time.Sleep(100 * time.Millisecond)

	// First verify dry-run reports it but does not delete
	var out bytes.Buffer
	var errOut bytes.Buffer
	code := runWipReap(&out, &errOut, []string{"-C", dir, "--max-age", "50ms", "--dry-run", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "stale-session") || !strings.Contains(out.String(), "stale orphaned checkpoint") {
		t.Fatalf("dry-run missing expected verdict: %s", out.String())
	}
	if _, err := gitWipOut(ctx, dir, nil, "show-ref", "--verify", checkpoint.Ref); err != nil {
		t.Fatalf("dry-run should not have deleted ref %s", checkpoint.Ref)
	}

	// Live reap should delete it
	out.Reset()
	errOut.Reset()
	code = runWipReap(&out, &errOut, []string{"-C", dir, "--max-age", "50ms", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "stale-session") {
		t.Fatalf("reap output missing stale-session: %s", out.String())
	}
	if _, err := gitWipOut(ctx, dir, nil, "show-ref", "--verify", checkpoint.Ref); err == nil {
		t.Fatalf("live reap should have deleted ref %s", checkpoint.Ref)
	}
}
