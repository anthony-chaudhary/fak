package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSyntheticTreeTransplantDisjoint(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	// Create initial commit with base file
	writeFile(t, filepath.Join(repo, "base.txt"), "base content\n")
	git(t, repo, "add", "base.txt")
	git(t, repo, "commit", "-m", "initial base commit")

	// Create target commit on a branch modifying file A
	git(t, repo, "checkout", "-b", "feature")
	writeFile(t, filepath.Join(repo, "a.txt"), "file A content\n")
	git(t, repo, "add", "a.txt")
	git(t, repo, "commit", "-m", "feature adds file A")
	targetSHA := revString(t, repo, "feature")

	// Advance main with a commit modifying file B (disjoint from file A)
	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "b.txt"), "file B content\n")
	git(t, repo, "add", "b.txt")
	git(t, repo, "commit", "-m", "main adds file B")
	headSHA := revString(t, repo, "main")

	// Add a dirty uncommitted modification in file C in the working tree
	dirtyContent := "dirty uncommitted file C\n"
	writeFile(t, filepath.Join(repo, "c.txt"), dirtyContent)

	// Call TransplantDisjointTree
	ctx := context.Background()
	newCommitSHA, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
	if err != nil {
		t.Fatalf("TransplantDisjointTree failed: %v", err)
	}
	if newCommitSHA == "" {
		t.Fatalf("expected non-empty newCommitSHA")
	}

	// 1) Branch advances to the new merge commit.
	curMainSHA := revString(t, repo, "main")
	if curMainSHA != newCommitSHA {
		t.Fatalf("main branch at %s, want new merge commit %s", curMainSHA, newCommitSHA)
	}
	curHeadSHA := revString(t, repo, "HEAD")
	if curHeadSHA != newCommitSHA {
		t.Fatalf("HEAD at %s, want new merge commit %s", curHeadSHA, newCommitSHA)
	}

	// 2) Merge commit has two parents (main previous HEAD and target commit).
	parentsRaw := gitOutput(t, repo, "rev-list", "--parents", "-n", "1", newCommitSHA)
	parts := strings.Fields(parentsRaw)
	if len(parts) != 3 {
		t.Fatalf("expected 2 parents for merge commit, got output: %q (fields=%v)", parentsRaw, parts)
	}
	if parts[1] != headSHA {
		t.Errorf("parent 1 = %s, want headSHA %s", parts[1], headSHA)
	}
	if parts[2] != targetSHA {
		t.Errorf("parent 2 = %s, want targetSHA %s", parts[2], targetSHA)
	}

	// 3) Uncommitted dirty file C remains byte-identical and un-clobbered in working tree.
	gotC := readFile(t, filepath.Join(repo, "c.txt"))
	if gotC != dirtyContent {
		t.Fatalf("c.txt modified! got %q, want %q", gotC, dirtyContent)
	}

	// 4) Synthetic merge tree contains both file A and file B changes.
	treeA := strings.TrimSpace(gitOutput(t, repo, "show", newCommitSHA+":a.txt"))
	if treeA != "file A content" {
		t.Errorf("a.txt in merge tree = %q, want %q", treeA, "file A content")
	}
	treeB := strings.TrimSpace(gitOutput(t, repo, "show", newCommitSHA+":b.txt"))
	if treeB != "file B content" {
		t.Errorf("b.txt in merge tree = %q, want %q", treeB, "file B content")
	}
	treeBase := strings.TrimSpace(gitOutput(t, repo, "show", newCommitSHA+":base.txt"))
	if treeBase != "base content" {
		t.Errorf("base.txt in merge tree = %q, want %q", treeBase, "base content")
	}

	// 5) Incoming disjoint file A must exist in the working directory (checked out).
	gotA := readFile(t, filepath.Join(repo, "a.txt"))
	if gotA != "file A content\n" {
		t.Errorf("a.txt in working tree = %q, want %q", gotA, "file A content\n")
	}
}

func TestRouteReconciliationDisjointTransplantWithDirtyWorkingTree(t *testing.T) {
	origin, clone := setupTestOriginAndClone(t)

	// Advance origin with a commit on file A
	writeFile(t, filepath.Join(origin, "a_remote.txt"), "remote A content\n")
	git(t, origin, "add", "a_remote.txt")
	git(t, origin, "commit", "-m", "remote adds a_remote.txt")

	// Advance clone with a commit on file B (disjoint)
	writeFile(t, filepath.Join(clone, "b_local.txt"), "local B content\n")
	git(t, clone, "add", "b_local.txt")
	git(t, clone, "commit", "-m", "local adds b_local.txt")

	// Add dirty uncommitted file C in clone working tree
	dirtyContent := "dirty worktree file C untouched\n"
	writeFile(t, filepath.Join(clone, "c_dirty.txt"), dirtyContent)

	// Run RouteReconciliation with Apply: true
	ctx := context.Background()
	opts := ReconcileOptions{
		Repo:   clone,
		Remote: "origin",
		Branch: "work",
		Goal:   "integrate",
		Apply:  true,
		Fetch:  true,
	}

	res, err := RouteReconciliation(ctx, opts)
	if err != nil {
		t.Fatalf("RouteReconciliation failed: %v", err)
	}

	if res.Route != RouteDisjointIntegrate {
		t.Fatalf("expected RouteDisjointIntegrate, got %q", res.Route)
	}
	if !res.OK {
		t.Fatalf("expected res.OK to be true, got false (reason=%s)", res.Reason)
	}
	if !res.Applied {
		t.Fatalf("expected res.Applied to be true, got false")
	}
	if res.Status != StateSynchronized {
		t.Errorf("expected res.Status = %q, got %q", StateSynchronized, res.Status)
	}
	if res.AppliedCommit == "" {
		t.Fatalf("expected non-empty res.AppliedCommit")
	}
	if res.Execution == nil || !res.Execution.Success {
		t.Fatalf("expected successful execution receipt, got %+v", res.Execution)
	}

	// Verify working tree dirty file was NOT clobbered
	gotDirty := readFile(t, filepath.Join(clone, "c_dirty.txt"))
	if gotDirty != dirtyContent {
		t.Fatalf("dirty file c_dirty.txt modified! got %q, want %q", gotDirty, dirtyContent)
	}

	// Verify synthetic merge tree contains changes from both sides
	showRemote := strings.TrimSpace(gitOutput(t, clone, "show", res.AppliedCommit+":a_remote.txt"))
	if showRemote != "remote A content" {
		t.Errorf("a_remote.txt = %q, want %q", showRemote, "remote A content")
	}
	showLocal := strings.TrimSpace(gitOutput(t, clone, "show", res.AppliedCommit+":b_local.txt"))
	if showLocal != "local B content" {
		t.Errorf("b_local.txt = %q, want %q", showLocal, "local B content")
	}

	// Verify incoming remote file is checked out to the working directory
	gotRemote := readFile(t, filepath.Join(clone, "a_remote.txt"))
	if gotRemote != "remote A content\n" {
		t.Errorf("a_remote.txt in working tree = %q, want %q", gotRemote, "remote A content\n")
	}
}

func TestSyntheticTreeTransplantDetachedHEAD(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(repo, "base.txt"), "base\n")
	git(t, repo, "add", "base.txt")
	git(t, repo, "commit", "-m", "base")

	git(t, repo, "checkout", "-b", "feature")
	writeFile(t, filepath.Join(repo, "a.txt"), "a\n")
	git(t, repo, "add", "a.txt")
	git(t, repo, "commit", "-m", "feature")
	targetSHA := revString(t, repo, "feature")

	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "b.txt"), "b\n")
	git(t, repo, "add", "b.txt")
	git(t, repo, "commit", "-m", "main")
	headSHA := revString(t, repo, "main")

	// Detach HEAD at main
	git(t, repo, "checkout", "--detach", headSHA)

	ctx := context.Background()
	newCommitSHA, err := TransplantDisjointTree(ctx, repo, "", headSHA, targetSHA, "refs/heads/feature")
	if err != nil {
		t.Fatalf("TransplantDisjointTree on detached HEAD failed: %v", err)
	}

	curHeadSHA := revString(t, repo, "HEAD")
	if curHeadSHA != newCommitSHA {
		t.Fatalf("detached HEAD at %s, want %s", curHeadSHA, newCommitSHA)
	}
}

func TestSyntheticTreeTransplantConflictError(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(repo, "conflict.txt"), "base line\n")
	git(t, repo, "add", "conflict.txt")
	git(t, repo, "commit", "-m", "base")

	git(t, repo, "checkout", "-b", "feature")
	writeFile(t, filepath.Join(repo, "conflict.txt"), "feature conflicting line\n")
	git(t, repo, "add", "conflict.txt")
	git(t, repo, "commit", "-m", "feature")
	targetSHA := revString(t, repo, "feature")

	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "conflict.txt"), "main conflicting line\n")
	git(t, repo, "add", "conflict.txt")
	git(t, repo, "commit", "-m", "main")
	headSHA := revString(t, repo, "main")

	ctx := context.Background()
	_, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
	if err == nil {
		t.Fatalf("expected conflict error from TransplantDisjointTree, got nil")
	}
}

func TestSyntheticTreeTransplantCheckoutIncomingPaths(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	writeFile(t, filepath.Join(repo, "base.txt"), "base\n")
	writeFile(t, filepath.Join(repo, "remote_mod.txt"), "remote base\n")
	writeFile(t, filepath.Join(repo, "local_mod.txt"), "local base\n")
	git(t, repo, "add", "base.txt", "remote_mod.txt", "local_mod.txt")
	git(t, repo, "commit", "-m", "base commit")

	// Target commit adds pkg/nested.txt and modifies remote_mod.txt
	git(t, repo, "checkout", "-b", "feature")
	if err := os.MkdirAll(filepath.Join(repo, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(repo, "pkg", "nested.txt"), "incoming nested\n")
	writeFile(t, filepath.Join(repo, "remote_mod.txt"), "remote modified\n")
	git(t, repo, "add", "pkg/nested.txt", "remote_mod.txt")
	git(t, repo, "commit", "-m", "feature commit")
	targetSHA := revString(t, repo, "feature")

	// Main modifies local_mod.txt (disjoint from feature)
	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "local_mod.txt"), "local modified commit\n")
	git(t, repo, "add", "local_mod.txt")
	git(t, repo, "commit", "-m", "main commit")
	headSHA := revString(t, repo, "main")

	// Dirty uncommitted edits in working tree
	writeFile(t, filepath.Join(repo, "local_mod.txt"), "local modified dirty\n")
	writeFile(t, filepath.Join(repo, "dirty.txt"), "dirty untracked\n")

	ctx := context.Background()
	newCommitSHA, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
	if err != nil {
		t.Fatalf("TransplantDisjointTree failed: %v", err)
	}
	if newCommitSHA == "" {
		t.Fatalf("expected non-empty newCommitSHA")
	}

	// 1. Incoming added file exists in working tree
	if got := readFile(t, filepath.Join(repo, "pkg", "nested.txt")); got != "incoming nested\n" {
		t.Errorf("nested.txt = %q, want %q", got, "incoming nested\n")
	}

	// 2. Incoming modified file updated in working tree
	if got := readFile(t, filepath.Join(repo, "remote_mod.txt")); got != "remote modified\n" {
		t.Errorf("remote_mod.txt = %q, want %q", got, "remote modified\n")
	}

	// 3. Local dirty uncommitted edits preserved (NOT clobbered)
	if got := readFile(t, filepath.Join(repo, "local_mod.txt")); got != "local modified dirty\n" {
		t.Errorf("local_mod.txt = %q, want %q", got, "local modified dirty\n")
	}

	// 4. Dirty untracked file preserved
	if got := readFile(t, filepath.Join(repo, "dirty.txt")); got != "dirty untracked\n" {
		t.Errorf("dirty.txt = %q, want %q", got, "dirty untracked\n")
	}

	// 5. git status does NOT have any phantom deletions
	status := gitOutput(t, repo, "status", "--porcelain")
	for _, line := range strings.Split(strings.TrimSpace(status), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "D ") || strings.HasPrefix(line, " D") {
			t.Errorf("unexpected deletion in git status: %q\nfull status:\n%s", line, status)
		}
	}
}
