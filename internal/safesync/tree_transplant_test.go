package safesync

import (
	"context"
	"fmt"
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

func TestSyntheticTreeTransplantSpacesAndUnicode(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "core.quotepath", "true")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	// Base commit
	writeFile(t, filepath.Join(repo, "base.txt"), "base\n")
	git(t, repo, "add", "base.txt")
	git(t, repo, "commit", "-m", "base commit")

	// Target commit adds files with spaces and unicode characters
	git(t, repo, "checkout", "-b", "feature")
	spaceDir := filepath.Join(repo, "folder with spaces")
	mkdir(t, spaceDir)
	spaceFile := filepath.Join(spaceDir, "file with spaces.txt")
	writeFile(t, spaceFile, "content in file with spaces\n")

	unicodeDir := filepath.Join(repo, "unicode_dir")
	mkdir(t, unicodeDir)
	unicodeFile := filepath.Join(unicodeDir, "données_été.txt")
	writeFile(t, unicodeFile, "données été utf8 content\n")

	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "feature commit with spaces and unicode")
	targetSHA := revString(t, repo, "feature")

	// Main branch makes a disjoint commit
	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "local_main.txt"), "local main content\n")
	git(t, repo, "add", "local_main.txt")
	git(t, repo, "commit", "-m", "main commit")
	headSHA := revString(t, repo, "main")

	ctx := context.Background()
	newCommitSHA, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
	if err != nil {
		t.Fatalf("TransplantDisjointTree failed: %v", err)
	}
	if newCommitSHA == "" {
		t.Fatalf("expected non-empty newCommitSHA")
	}

	// Verify the file with spaces exists in working tree with expected content
	if got := readFile(t, spaceFile); got != "content in file with spaces\n" {
		t.Errorf("space file content = %q, want %q", got, "content in file with spaces\n")
	}

	// Verify the file with unicode exists in working tree with expected content
	if got := readFile(t, unicodeFile); got != "données été utf8 content\n" {
		t.Errorf("unicode file content = %q, want %q", got, "données été utf8 content\n")
	}

	// Verify git status is clean
	status := strings.TrimSpace(gitOutput(t, repo, "status", "--porcelain"))
	if status != "" {
		t.Errorf("expected clean git status, got:\n%s", status)
	}
}

func TestTransplantRenames(t *testing.T) {
	t.Run("direct_transplant_basic_rename", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "core.autocrlf", "false")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		// Base commit with metal_mtp.go
		mtpContent := "package compute\n\n// Metal tokens and completions\nfunc MTP() int {\n\treturn 42\n}\n\nfunc Version() string {\n\treturn \"1.0\"\n}\n"
		writeFile(t, filepath.Join(repo, "metal_mtp.go"), mtpContent)
		git(t, repo, "add", "metal_mtp.go")
		git(t, repo, "commit", "-m", "base: add metal_mtp.go")

		// Remote branch renames metal_mtp.go -> chat_completions.go
		git(t, repo, "checkout", "-b", "feature")
		git(t, repo, "mv", "metal_mtp.go", "chat_completions.go")
		chatContent := "package compute\n\n// Metal tokens and completions\nfunc MTP() int {\n\treturn 42\n}\n\nfunc Version() string {\n\treturn \"1.1\"\n}\n"
		writeFile(t, filepath.Join(repo, "chat_completions.go"), chatContent)
		git(t, repo, "add", "chat_completions.go")
		git(t, repo, "commit", "-m", "feature: rename metal_mtp.go to chat_completions.go")
		targetSHA := revString(t, repo, "feature")

		// Main branch adds disjoint file local.go
		git(t, repo, "checkout", "main")
		localContent := "package compute\n\nfunc LocalWorker() {}\n"
		writeFile(t, filepath.Join(repo, "local.go"), localContent)
		git(t, repo, "add", "local.go")
		git(t, repo, "commit", "-m", "main: add local.go")
		headSHA := revString(t, repo, "main")

		ctx := context.Background()
		newCommitSHA, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
		if err != nil {
			t.Fatalf("TransplantDisjointTree failed: %v", err)
		}
		if newCommitSHA == "" {
			t.Fatal("expected non-empty newCommitSHA")
		}

		// 1. Old file metal_mtp.go must NOT exist on disk
		if _, err := os.Stat(filepath.Join(repo, "metal_mtp.go")); !os.IsNotExist(err) {
			t.Errorf("metal_mtp.go still exists on disk after transplant!")
		}

		// 2. New file chat_completions.go must exist on disk with correct content
		gotChat := readFile(t, filepath.Join(repo, "chat_completions.go"))
		if gotChat != chatContent {
			t.Errorf("chat_completions.go content = %q, want %q", gotChat, chatContent)
		}

		// 3. Local disjoint file local.go must remain intact
		gotLocal := readFile(t, filepath.Join(repo, "local.go"))
		if gotLocal != localContent {
			t.Errorf("local.go content = %q, want %q", gotLocal, localContent)
		}

		// 4. Git status should be completely clean (no phantom untracked or missing files)
		status := strings.TrimSpace(gitOutput(t, repo, "status", "--porcelain"))
		if status != "" {
			t.Errorf("git status after rename transplant is not clean:\n%s", status)
		}
	})

	t.Run("direct_transplant_dirty_source_preserved", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "core.autocrlf", "false")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		writeFile(t, filepath.Join(repo, "dirty_source.go"), "package compute\n// initial\n")
		git(t, repo, "add", "dirty_source.go")
		git(t, repo, "commit", "-m", "base")

		git(t, repo, "checkout", "-b", "feature")
		git(t, repo, "mv", "dirty_source.go", "dirty_target.go")
		git(t, repo, "commit", "-m", "feature rename")
		targetSHA := revString(t, repo, "feature")

		git(t, repo, "checkout", "main")
		writeFile(t, filepath.Join(repo, "other.go"), "package compute\n// other\n")
		git(t, repo, "add", "other.go")
		git(t, repo, "commit", "-m", "main add other")
		headSHA := revString(t, repo, "main")

		// Add uncommitted modification to dirty_source.go
		dirtyContent := "package compute\n// uncommitted local edits\n"
		writeFile(t, filepath.Join(repo, "dirty_source.go"), dirtyContent)

		ctx := context.Background()
		_, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
		if err != nil {
			t.Fatalf("TransplantDisjointTree failed: %v", err)
		}

		// Dirty file must NOT be clobbered
		gotDirty := readFile(t, filepath.Join(repo, "dirty_source.go"))
		if gotDirty != dirtyContent {
			t.Errorf("dirty_source.go was clobbered: got %q, want %q", gotDirty, dirtyContent)
		}

		// Target file should still be checked out
		if _, err := os.Stat(filepath.Join(repo, "dirty_target.go")); os.IsNotExist(err) {
			t.Errorf("dirty_target.go was not checked out!")
		}
	})

	t.Run("route_reconciliation_incoming_rename", func(t *testing.T) {
		origin, clone := setupTestOriginAndClone(t)

		// Create base file metal_mtp.go in origin and push/pull to clone
		mtpContent := "package compute\n\n// Metal tokens and completions\nfunc MTP() int {\n\treturn 42\n}\n\nfunc Version() string {\n\treturn \"1.0\"\n}\n"
		writeFile(t, filepath.Join(origin, "metal_mtp.go"), mtpContent)
		git(t, origin, "add", "metal_mtp.go")
		git(t, origin, "commit", "-m", "add metal_mtp.go")
		git(t, clone, "pull", "origin", "work")

		// Remote renames metal_mtp.go -> chat_completions.go
		git(t, origin, "mv", "metal_mtp.go", "chat_completions.go")
		chatContent := "package compute\n\n// Metal tokens and completions\nfunc MTP() int {\n\treturn 42\n}\n\nfunc Version() string {\n\treturn \"1.1\"\n}\n"
		writeFile(t, filepath.Join(origin, "chat_completions.go"), chatContent)
		git(t, origin, "add", "chat_completions.go")
		git(t, origin, "commit", "-m", "remote rename metal_mtp.go to chat_completions.go")

		// Clone makes disjoint commit
		localContent := "package compute\n\nfunc Local() {}\n"
		writeFile(t, filepath.Join(clone, "local.go"), localContent)
		git(t, clone, "add", "local.go")
		git(t, clone, "commit", "-m", "clone adds local.go")

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
			t.Fatalf("route = %q, want RouteDisjointIntegrate", res.Route)
		}
		if !res.OK || !res.Applied {
			t.Fatalf("expected OK and Applied, got OK=%v Applied=%v", res.OK, res.Applied)
		}

		// Verify metal_mtp.go is gone from clone working directory
		if _, err := os.Stat(filepath.Join(clone, "metal_mtp.go")); !os.IsNotExist(err) {
			t.Errorf("metal_mtp.go still exists on disk in clone!")
		}

		// Verify chat_completions.go is present in clone working directory
		gotChat := readFile(t, filepath.Join(clone, "chat_completions.go"))
		if gotChat != chatContent {
			t.Errorf("chat_completions.go = %q, want %q", gotChat, chatContent)
		}

		// Verify local.go is present in clone working directory
		gotLocal := readFile(t, filepath.Join(clone, "local.go"))
		if gotLocal != localContent {
			t.Errorf("local.go = %q, want %q", gotLocal, localContent)
		}

		// Verify clean status in clone
		status := strings.TrimSpace(gitOutput(t, clone, "status", "--porcelain"))
		if status != "" {
			t.Errorf("clone git status not clean:\n%s", status)
		}
	})

	t.Run("nested_and_multiple_renames", func(t *testing.T) {
		repo := t.TempDir()
		git(t, repo, "init", "-b", "main")
		git(t, repo, "config", "core.autocrlf", "false")
		git(t, repo, "config", "user.name", "test")
		git(t, repo, "config", "user.email", "test@example.com")

		subFile1 := filepath.Join(repo, "pkg", "sub1", "old_service.go")
		subFile2 := filepath.Join(repo, "pkg", "sub2", "old_helper.go")
		_ = os.MkdirAll(filepath.Dir(subFile1), 0755)
		_ = os.MkdirAll(filepath.Dir(subFile2), 0755)

		c1 := "package sub1\n\nfunc S1() string {\n\treturn \"service 1\"\n}\n"
		c2 := "package sub2\n\nfunc H2() string {\n\treturn \"helper 2\"\n}\n"
		writeFile(t, subFile1, c1)
		writeFile(t, subFile2, c2)
		git(t, repo, "add", ".")
		git(t, repo, "commit", "-m", "base: add sub packages")

		// Feature branch renames both:
		// 1. pkg/sub1/old_service.go -> pkg/sub1/new_service.go
		// 2. pkg/sub2/old_helper.go -> pkg/renamed_dir/new_helper.go
		git(t, repo, "checkout", "-b", "feature")
		newSub1 := filepath.Join(repo, "pkg", "sub1", "new_service.go")
		git(t, repo, "mv", filepath.Join("pkg", "sub1", "old_service.go"), filepath.Join("pkg", "sub1", "new_service.go"))
		_ = os.MkdirAll(filepath.Join(repo, "pkg", "renamed_dir"), 0755)
		git(t, repo, "mv", filepath.Join("pkg", "sub2", "old_helper.go"), filepath.Join("pkg", "renamed_dir", "new_helper.go"))
		git(t, repo, "commit", "-m", "feature: rename both files")
		targetSHA := revString(t, repo, "feature")

		// Main branch makes a disjoint commit
		git(t, repo, "checkout", "main")
		writeFile(t, filepath.Join(repo, "root_local.go"), "package main\n\nfunc Local() {}\n")
		git(t, repo, "add", "root_local.go")
		git(t, repo, "commit", "-m", "main: disjoint root_local.go")
		headSHA := revString(t, repo, "main")

		ctx := context.Background()
		_, err := TransplantDisjointTree(ctx, repo, "main", headSHA, targetSHA, "refs/heads/feature")
		if err != nil {
			t.Fatalf("TransplantDisjointTree failed: %v", err)
		}

		// Verify old paths gone
		if _, err := os.Stat(subFile1); !os.IsNotExist(err) {
			t.Errorf("old_service.go still exists!")
		}
		if _, err := os.Stat(subFile2); !os.IsNotExist(err) {
			t.Errorf("old_helper.go still exists!")
		}

		// Verify new paths exist
		if _, err := os.Stat(newSub1); os.IsNotExist(err) {
			t.Errorf("new_service.go missing!")
		}
		if _, err := os.Stat(filepath.Join(repo, "pkg", "renamed_dir", "new_helper.go")); os.IsNotExist(err) {
			t.Errorf("new_helper.go missing!")
		}

		// Verify status clean
		status := strings.TrimSpace(gitOutput(t, repo, "status", "--porcelain"))
		if status != "" {
			t.Errorf("git status not clean:\n%s", status)
		}
	})
}

func TestParseRenameSummary(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected []RenamePath
	}{
		{
			name:  "simple summary",
			input: " rename metal_mtp.go => chat_completions.go (100%)\n",
			expected: []RenamePath{
				{OldPath: "metal_mtp.go", NewPath: "chat_completions.go"},
			},
		},
		{
			name:  "common directory curly braces",
			input: " rename internal/compute/{metal_mtp.go => chat_completions.go} (95%)\n",
			expected: []RenamePath{
				{OldPath: "internal/compute/metal_mtp.go", NewPath: "internal/compute/chat_completions.go"},
			},
		},
		{
			name:  "directory rename curly braces",
			input: " rename {pkg1 => pkg2}/service.go (100%)\n",
			expected: []RenamePath{
				{OldPath: "pkg1/service.go", NewPath: "pkg2/service.go"},
			},
		},
		{
			name:  "different directory and filename",
			input: " rename dir1/foo.go => dir2/bar.go (98%)\n",
			expected: []RenamePath{
				{OldPath: "dir1/foo.go", NewPath: "dir2/bar.go"},
			},
		},
		{
			name:  "name-status format fallback",
			input: "R100\tinternal/compute/metal_mtp.go\tinternal/compute/chat_completions.go\n",
			expected: []RenamePath{
				{OldPath: "internal/compute/metal_mtp.go", NewPath: "internal/compute/chat_completions.go"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			actual := ParseRenameSummary(tc.input)
			if len(actual) != len(tc.expected) {
				t.Fatalf("got %d renames, want %d: %+v", len(actual), len(tc.expected), actual)
			}
			for i := range actual {
				if actual[i].OldPath != tc.expected[i].OldPath || actual[i].NewPath != tc.expected[i].NewPath {
					t.Errorf("[%d] got %+v, want %+v", i, actual[i], tc.expected[i])
				}
			}
		})
	}
}

func TestSyntheticTreeTransplantBatchCheckout(t *testing.T) {
	repo := t.TempDir()
	git(t, repo, "init", "-b", "main")
	git(t, repo, "config", "core.autocrlf", "false")
	git(t, repo, "config", "user.name", "test")
	git(t, repo, "config", "user.email", "test@example.com")

	// Base commit
	writeFile(t, filepath.Join(repo, "base.txt"), "base\n")
	git(t, repo, "add", "base.txt")
	git(t, repo, "commit", "-m", "base commit")

	// Feature branch adds 250 files
	git(t, repo, "checkout", "-b", "feature")
	const totalFiles = 250
	for i := 0; i < totalFiles; i++ {
		name := fmt.Sprintf("file_%03d.txt", i)
		writeFile(t, filepath.Join(repo, name), fmt.Sprintf("content %d\n", i))
	}
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "feature commit with 250 files")
	targetSHA := revString(t, repo, "feature")

	// Main branch adds disjoint file local.txt
	git(t, repo, "checkout", "main")
	writeFile(t, filepath.Join(repo, "local.txt"), "local content\n")
	git(t, repo, "add", "local.txt")
	git(t, repo, "commit", "-m", "main commit")
	headSHA := revString(t, repo, "main")

	// Intercept runner calls to record checkout invocations
	var checkoutBatches [][]string
	runner := func(ctx context.Context, dir string, args ...string) RunResult {
		if len(args) >= 3 && args[0] == "checkout" && args[2] == "--" {
			paths := append([]string(nil), args[3:]...)
			checkoutBatches = append(checkoutBatches, paths)
		}
		return RealRunner(ctx, dir, args...)
	}

	ctx := context.Background()
	newCommitSHA, err := TransplantDisjointTreeWithRunner(ctx, runner, repo, "main", headSHA, targetSHA, "refs/heads/feature")
	if err != nil {
		t.Fatalf("TransplantDisjointTreeWithRunner failed: %v", err)
	}
	if newCommitSHA == "" {
		t.Fatalf("expected non-empty newCommitSHA")
	}

	// Verify batching: 250 files with batch size 100 => 3 batches (100, 100, 50)
	if len(checkoutBatches) != 3 {
		t.Fatalf("expected 3 checkout batches, got %d", len(checkoutBatches))
	}
	if len(checkoutBatches[0]) != 100 {
		t.Errorf("batch 0 len = %d, want 100", len(checkoutBatches[0]))
	}
	if len(checkoutBatches[1]) != 100 {
		t.Errorf("batch 1 len = %d, want 100", len(checkoutBatches[1]))
	}
	if len(checkoutBatches[2]) != 50 {
		t.Errorf("batch 2 len = %d, want 50", len(checkoutBatches[2]))
	}

	// Verify each batch does not exceed batch size 100
	for idx, batch := range checkoutBatches {
		if len(batch) > 100 {
			t.Errorf("batch %d exceeded max batch size 100: got %d", idx, len(batch))
		}
	}

	// Verify all 250 files actually exist on disk in the working tree
	for i := 0; i < totalFiles; i++ {
		name := fmt.Sprintf("file_%03d.txt", i)
		got := readFile(t, filepath.Join(repo, name))
		want := fmt.Sprintf("content %d\n", i)
		if got != want {
			t.Fatalf("file %s = %q, want %q", name, got, want)
		}
	}
}

func TestTransplantDisjointTreeBatchedCheckout(t *testing.T) {
	TestSyntheticTreeTransplantBatchCheckout(t)
}
