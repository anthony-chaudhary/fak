package safesync

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func TestParkAheadBehindUniqueRetainedIdenticalSuppressed(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "main")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "top\nmiddle\nbottom\n")
	writeFile(t, filepath.Join(origin, "b.txt"), "b initial\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "base")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "clone", origin, clone)
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	// Upstream commit on origin modifies a.txt: adds "incoming"
	writeFile(t, filepath.Join(origin, "a.txt"), "top\nincoming\nmiddle\nbottom\n")
	git(t, origin, "add", "a.txt")
	git(t, origin, "commit", "-m", "upstream update")
	targetSHA := revString(t, origin, "HEAD")

	// Fetch in clone so origin/main is updated
	git(t, clone, "fetch", "origin")

	// Make local commit in clone so clone is ahead of base
	writeFile(t, filepath.Join(clone, "local.txt"), "local commit\n")
	git(t, clone, "add", "local.txt")
	git(t, clone, "commit", "-m", "local commit ahead")
	baseHead := revString(t, clone, "HEAD")

	// Working tree state in clone:
	// 1. a.txt has both incoming hunk AND unique hunk
	writeFile(t, filepath.Join(clone, "a.txt"), "top\nincoming\nmiddle\nbottom\nunique\n")
	// 2. Unrelated dirty path b.txt
	writeFile(t, filepath.Join(clone, "b.txt"), "b dirty uncommitted\n")
	// 3. Index sentinel
	writeFile(t, filepath.Join(clone, "staged.txt"), "staged sentinel\n")
	git(t, clone, "add", "staged.txt")

	// --- Step 1: Dry run preview ---
	dryOpts := ParkOptions{
		Repo:      clone,
		Session:   "sess-test-1",
		Paths:     []string{"a.txt"},
		TargetRef: "origin/main",
		Apply:     false,
	}

	dryRec, err := Park(ctx, dryOpts)
	if err != nil {
		t.Fatalf("dry run Park err: %v", err)
	}
	if dryRec.Status != ParkStatusDryRun {
		t.Fatalf("dry run status = %q, want %q", dryRec.Status, ParkStatusDryRun)
	}
	if !dryRec.OK {
		t.Fatalf("dry run OK = false, reason: %s", dryRec.Reason)
	}
	if len(dryRec.Effects) != 1 || dryRec.Effects[0].Classification != EffectCleanReapply {
		t.Fatalf("dry run effects = %+v, want clean_reapply", dryRec.Effects)
	}

	// Verify dry run did NOT mutate disk
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "top\nincoming\nmiddle\nbottom\nunique\n" {
		t.Fatalf("dry run mutated a.txt: %q", got)
	}
	if got := readFile(t, filepath.Join(clone, "b.txt")); got != "b dirty uncommitted\n" {
		t.Fatalf("dry run mutated b.txt: %q", got)
	}
	if got := revString(t, clone, "HEAD"); got != baseHead {
		t.Fatalf("dry run moved HEAD: got %s, want %s", got, baseHead)
	}

	// --- Step 2: Apply in-place integration and reapply ---
	applyOpts := ParkOptions{
		Repo:      clone,
		Session:   "sess-test-1",
		Paths:     []string{"a.txt"},
		TargetRef: "origin/main",
		Apply:     true,
	}

	applyRec, err := Park(ctx, applyOpts)
	if err != nil {
		t.Fatalf("apply Park err: %v", err)
	}
	if !applyRec.OK {
		t.Fatalf("apply Park OK = false: reason=%s, detail=%s", applyRec.Reason, applyRec.Detail)
	}
	if applyRec.Status != ParkStatusRestored {
		t.Fatalf("apply status = %q, want %q", applyRec.Status, ParkStatusRestored)
	}
	if applyRec.CheckpointRef != "refs/fak/wip/sess-test-1/park" {
		t.Fatalf("checkpoint ref = %q, want refs/fak/wip/sess-test-1/park", applyRec.CheckpointRef)
	}

	// Checkpoint ref must exist and be valid commit
	if got := revString(t, clone, "refs/fak/wip/sess-test-1/park"); got == "" {
		t.Fatal("checkpoint ref was not created in git")
	}

	// Verify a.txt content: unique effect retained, upstream-identical hunk suppressed
	wantA := "top\nincoming\nmiddle\nbottom\nunique\n"
	if gotA := readFile(t, filepath.Join(clone, "a.txt")); gotA != wantA {
		t.Fatalf("a.txt = %q, want %q", gotA, wantA)
	}

	// Verify b.txt untouched byte-for-byte
	if gotB := readFile(t, filepath.Join(clone, "b.txt")); gotB != "b dirty uncommitted\n" {
		t.Fatalf("b.txt modified: %q, want byte-for-byte original dirty content", gotB)
	}

	// Verify index sentinel preserved in index
	stagedOutput := gitOutput(t, clone, "diff", "--cached", "--name-only")
	if !strings.Contains(stagedOutput, "staged.txt") {
		t.Fatalf("index sentinel staged.txt missing from index: %q", stagedOutput)
	}
	if gotStaged := gitOutput(t, clone, "show", ":staged.txt"); gotStaged != "staged sentinel\n" {
		t.Fatalf("staged.txt index content = %q, want 'staged sentinel\\n'", gotStaged)
	}

	// Verify ahead + behind integration: NewHEAD must contain both baseHead and targetSHA
	newHead := revString(t, clone, "HEAD")
	if newHead == baseHead || newHead == targetSHA {
		t.Fatalf("newHead = %s, expected a merge commit incorporating both ahead and behind commits", newHead)
	}
	mergeParents := strings.Fields(gitOutput(t, clone, "log", "-1", "--format=%P", newHead))
	if len(mergeParents) < 2 {
		t.Fatalf("mergeParents = %v, expected at least 2 parents for ahead+behind merge commit", mergeParents)
	}
}

func TestParkRefusalOnUnownedCollidingPaths(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "main")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "a1\n")
	writeFile(t, filepath.Join(origin, "d.txt"), "d1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "base")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "clone", origin, clone)
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	// Upstream modifies both a.txt and d.txt
	writeFile(t, filepath.Join(origin, "a.txt"), "a2\n")
	writeFile(t, filepath.Join(origin, "d.txt"), "d2\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "upstream update")
	git(t, clone, "fetch", "origin")

	// In working tree, both a.txt and d.txt are dirty
	writeFile(t, filepath.Join(clone, "a.txt"), "a local\n")
	writeFile(t, filepath.Join(clone, "d.txt"), "d local unowned\n")

	// Caller only authorized a.txt, so d.txt is an unowned colliding dirty path!
	opts := ParkOptions{
		Repo:      clone,
		Session:   "sess-unowned",
		Paths:     []string{"a.txt"},
		TargetRef: "origin/main",
		Apply:     true,
	}

	rec, err := Park(ctx, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rec.OK {
		t.Fatal("expected refusal on unowned colliding path d.txt, got OK=true")
	}
	if rec.Status != ParkStatusConflict {
		t.Fatalf("status = %q, want %q", rec.Status, ParkStatusConflict)
	}
	if rec.Reason != ReasonDirtyWriteOverlap {
		t.Fatalf("reason = %q, want %q", rec.Reason, ReasonDirtyWriteOverlap)
	}
	if !strings.Contains(rec.Detail, "d.txt") {
		t.Fatalf("detail %q does not mention unowned colliding path d.txt", rec.Detail)
	}

	// Working tree must remain untouched
	if got := readFile(t, filepath.Join(clone, "d.txt")); got != "d local unowned\n" {
		t.Fatalf("d.txt modified: %q", got)
	}
}

func TestParkRefusalOnConflictingChanges(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "main")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "base line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "base")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "clone", origin, clone)
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	// Upstream changes a.txt
	writeFile(t, filepath.Join(origin, "a.txt"), "upstream conflicting line\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "upstream change")
	git(t, clone, "fetch", "origin")

	// Local dirty file changes a.txt incompatibly
	writeFile(t, filepath.Join(clone, "a.txt"), "local incompatible line\n")

	opts := ParkOptions{
		Repo:      clone,
		Session:   "sess-conflict",
		Paths:     []string{"a.txt"},
		TargetRef: "origin/main",
		Apply:     false,
	}

	rec, err := Park(ctx, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if rec.OK {
		t.Fatal("expected conflict refusal, got OK=true")
	}
	if rec.Status != ParkStatusConflict {
		t.Fatalf("status = %q, want %q", rec.Status, ParkStatusConflict)
	}
	if len(rec.Effects) != 1 || rec.Effects[0].Classification != EffectConflict {
		t.Fatalf("effects = %+v, want conflict", rec.Effects)
	}
}

func TestParkUpstreamIdenticalSuppressed(t *testing.T) {
	ctx := context.Background()
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	mkdir(t, origin)
	git(t, origin, "init", "-b", "main")
	git(t, origin, "config", "user.name", "test")
	git(t, origin, "config", "user.email", "test@example.com")
	writeFile(t, filepath.Join(origin, "a.txt"), "v1\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "base")

	clone := filepath.Join(tmp, "clone")
	git(t, tmp, "clone", origin, clone)
	git(t, clone, "config", "user.name", "test")
	git(t, clone, "config", "user.email", "test@example.com")

	// Upstream changes a.txt to v2
	writeFile(t, filepath.Join(origin, "a.txt"), "v2\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "v2")
	git(t, clone, "fetch", "origin")

	// Local dirty file already has v2 (upstream identical)
	writeFile(t, filepath.Join(clone, "a.txt"), "v2\n")

	opts := ParkOptions{
		Repo:      clone,
		Session:   "sess-ident",
		Paths:     []string{"a.txt"},
		TargetRef: "origin/main",
		Apply:     true,
	}

	rec, err := Park(ctx, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !rec.OK {
		t.Fatalf("expected OK=true, got reason: %s", rec.Reason)
	}
	if rec.Status != ParkStatusIntegrated {
		t.Fatalf("status = %q, want %q", rec.Status, ParkStatusIntegrated)
	}
	if len(rec.Effects) != 1 || rec.Effects[0].Classification != EffectUpstreamIdentical {
		t.Fatalf("effects = %+v, want upstream_identical", rec.Effects)
	}
	if got := readFile(t, filepath.Join(clone, "a.txt")); got != "v2\n" {
		t.Fatalf("a.txt = %q, want 'v2\\n'", got)
	}
}
