package main

// wip_scope_test.go — the CAPTURE-scope vs CLAIM-scope contract of `fak wip` (#5539).
//
// A checkpoint is minted by staging the ENTIRE working tree into a throwaway index
// (`read-tree HEAD` + `add -A`, wip.go), so on a shared tree the object filed at
// refs/fak/wip/<session> holds every concurrently-dirty peer's edits too. That width is
// DELIBERATE — it is what makes the snapshot lossless for crash recovery — but it means
// the ref's session key names the CAPTURER, never the AUTHOR. These tests pin both halves:
// capture stays tree-wide, and the one verb that turns the snapshot into an irreversible
// act (`wip land`, which commits) refuses to commit an unattributable tree-wide snapshot
// unless the caller declares the paths it owns.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeWipFile is the fixture's file writer: it fails the test rather than returning an
// error, so a case body reads as the scenario it describes.
func writeWipFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

// TestWipCheckpointCapturesPeerEditsTreeWide pins the DECIDED invariant (#5539): a
// checkpoint's captured tree is the whole working tree, peers' in-flight edits included.
// Narrowing the CAPTURE would make a crashed session's un-declared edits unrecoverable,
// which is strictly worse than storing too much — so this width is a property, not a bug.
// What must not follow from it is an unqualified claim that the ref is the session's work.
func TestWipCheckpointCapturesPeerEditsTreeWide(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	writeWipFile(t, dir, "mine.txt", "session A's own edit\n")
	writeWipFile(t, dir, "peer.txt", "a concurrent peer's edit\n")

	res, err := wipCheckpoint(ctx, dir, "sessA", true, 1000)
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	files, err := wipDeltaFiles(ctx, dir, res.Object)
	if err != nil {
		t.Fatalf("delta files: %v", err)
	}
	joined := strings.Join(files, ",")
	if !strings.Contains(joined, "mine.txt") || !strings.Contains(joined, "peer.txt") {
		t.Fatalf("checkpoint delta = %v, want BOTH the session's own file and the peer's (capture is tree-wide by design)", files)
	}
}

// TestWipLandRefusesTreeWideSnapshotWithPeerCheckpoints is the #5539 done-condition: with
// another session holding a live checkpoint — the repo's own evidence that this tree is
// shared — landing an UNDECLARED snapshot would commit that peer's file under this
// session's name. That is the `git add -A` sweep the repo forbids, laundered through a
// session key, so land refuses (exit 3, TREE_WIDE_SNAPSHOT) and commits nothing.
func TestWipLandRefusesTreeWideSnapshotWithPeerCheckpoints(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	// A peer checkpoints first: its ref is the live evidence of a concurrent session.
	writeWipFile(t, dir, "peer.txt", "a concurrent peer's edit\n")
	if _, err := wipCheckpoint(ctx, dir, "peersess", true, 1000); err != nil {
		t.Fatalf("peer checkpoint: %v", err)
	}
	writeWipFile(t, dir, "mine.txt", "session A's own edit\n")
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 1001); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	res, code, err := wipLand(ctx, dir, "sessA", "", false)
	if code != 3 || err == nil {
		t.Fatalf("expected a TREE_WIDE_SNAPSHOT refusal (rc=3, err), got rc=%d err=%v res=%+v", code, err, res)
	}
	if res.Reason != "TREE_WIDE_SNAPSHOT" || res.Committed {
		t.Fatalf("expected TREE_WIDE_SNAPSHOT and no commit, got %+v", res)
	}
	// Nothing landed: the peer's file is still only in the working tree.
	if _, err := gitWipOut(ctx, dir, nil, "cat-file", "-e", "HEAD:peer.txt"); err == nil {
		t.Fatal("a refused land committed the peer's file anyway")
	}
}

// TestWipLandDeclaredPathsCommitOnlyThoseFiles proves the remedy the refusal names: an
// explicit --path declaration (the same discipline `fak commit --path` enforces) narrows
// BOTH the pathspec committed and the patch materialized, so the peer's file is left
// entirely alone even though the snapshot object contains it.
func TestWipLandDeclaredPathsCommitOnlyThoseFiles(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	writeWipFile(t, dir, "peer.txt", "a concurrent peer's edit\n")
	if _, err := wipCheckpoint(ctx, dir, "peersess", true, 1000); err != nil {
		t.Fatalf("peer checkpoint: %v", err)
	}
	writeWipFile(t, dir, "mine.txt", "session A's own edit\n")
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 1001); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	res, code, err := wipLandWith(ctx, dir, "sessA", wipLandOptions{Paths: []string{"mine.txt"}})
	if err != nil || code != 0 {
		t.Fatalf("wipLand rc=%d err=%v res=%+v", code, err, res)
	}
	if len(res.Files) != 1 || res.Files[0] != "mine.txt" {
		t.Fatalf("Files = %v, want exactly [mine.txt]", res.Files)
	}
	if len(res.Excluded) != 1 || res.Excluded[0] != "peer.txt" {
		t.Fatalf("Excluded = %v, want exactly [peer.txt]", res.Excluded)
	}
	if _, err := gitWipOut(ctx, dir, nil, "cat-file", "-e", "HEAD:mine.txt"); err != nil {
		t.Fatalf("the declared file did not land: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "cat-file", "-e", "HEAD:peer.txt"); err == nil {
		t.Fatal("the peer's file landed under this session's commit — the sweep #5539 exists to stop")
	}
}

// TestWipCheckpointStampedScopeNarrowsLand proves the durable half of the remedy: a scope
// declared AT CAPTURE (`fak wip checkpoint --path`) is carried in the stamp, so a LATER
// process — a fleet host recovering a crashed session, which cannot know what the dead
// session owned — lands only what that session claimed, with no flags of its own.
func TestWipCheckpointStampedScopeNarrowsLand(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	writeWipFile(t, dir, "peer.txt", "a concurrent peer's edit\n")
	if _, err := wipCheckpoint(ctx, dir, "peersess", true, 1000); err != nil {
		t.Fatalf("peer checkpoint: %v", err)
	}
	writeWipFile(t, dir, "mine.txt", "session A's own edit\n")
	res, err := wipCheckpointScoped(ctx, dir, "sessA", true, 1001, []string{"mine.txt"})
	if err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if len(res.Scope) != 1 || res.Scope[0] != "mine.txt" {
		t.Fatalf("Scope = %v, want [mine.txt]", res.Scope)
	}
	// The stamp — not just the in-process result — carries the claim.
	rec, err := wipRecordAt(ctx, dir, "refs/fak/wip/sessA", res.Object)
	if err != nil {
		t.Fatalf("read stamp: %v", err)
	}
	if len(rec.Stamp.Scope) != 1 || rec.Stamp.Scope[0] != "mine.txt" {
		t.Fatalf("stamped Scope = %v, want [mine.txt]", rec.Stamp.Scope)
	}

	landed, code, err := wipLand(ctx, dir, "sessA", "", false)
	if err != nil || code != 0 {
		t.Fatalf("wipLand rc=%d err=%v res=%+v", code, err, landed)
	}
	if len(landed.Files) != 1 || landed.Files[0] != "mine.txt" {
		t.Fatalf("Files = %v, want exactly [mine.txt] from the stamped scope", landed.Files)
	}
	if _, err := gitWipOut(ctx, dir, nil, "cat-file", "-e", "HEAD:peer.txt"); err == nil {
		t.Fatal("the peer's file landed despite a narrower stamped scope")
	}
}

// TestWipLandAllAcceptsTreeWideSnapshot keeps the honest escape open: an operator who
// genuinely wants the whole snapshot says so, and gets today's behaviour. The point of
// #5539 is that the sweep must be DECLARED, not that it must be impossible.
func TestWipLandAllAcceptsTreeWideSnapshot(t *testing.T) {
	ctx := context.Background()
	dir, _ := landTestRepo(t)

	writeWipFile(t, dir, "peer.txt", "a concurrent peer's edit\n")
	if _, err := wipCheckpoint(ctx, dir, "peersess", true, 1000); err != nil {
		t.Fatalf("peer checkpoint: %v", err)
	}
	writeWipFile(t, dir, "mine.txt", "session A's own edit\n")
	if _, err := wipCheckpoint(ctx, dir, "sessA", true, 1001); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	res, code, err := wipLandWith(ctx, dir, "sessA", wipLandOptions{All: true})
	if err != nil || code != 0 {
		t.Fatalf("wipLand --all rc=%d err=%v res=%+v", code, err, res)
	}
	if len(res.Files) != 2 {
		t.Fatalf("Files = %v, want both files under an explicit --all", res.Files)
	}
}
