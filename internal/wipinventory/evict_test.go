package wipinventory

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func initTestRepo(t *testing.T) (string, func(args ...string) string) {
	t.Helper()
	repo := t.TempDir()

	runGit := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=tester",
			"GIT_AUTHOR_EMAIL=tester@fak.local",
			"GIT_COMMITTER_NAME=tester",
			"GIT_COMMITTER_EMAIL=tester@fak.local",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s failed: %v\n%s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}

	runGit("init", "-q", "-b", "main")
	runGit("config", "user.name", "tester")
	runGit("config", "user.email", "tester@fak.local")

	basePath := filepath.Join(repo, "base.txt")
	if err := os.WriteFile(basePath, []byte("initial baseline commit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("add", "base.txt")
	runGit("commit", "-q", "-m", "initial baseline")

	return repo, runGit
}

func TestEvictOrphansPreservesAndCleans(t *testing.T) {
	repo, runGit := initTestRepo(t)
	ctx := context.Background()
	runner := GitRunner{}

	orphan1Rel := "orphan-a.txt"
	orphan1Bytes := []byte("orphan file A content with newlines\r\nand special bytes: \x00\x01\x02\xff\n")
	if err := os.WriteFile(filepath.Join(repo, orphan1Rel), orphan1Bytes, 0o644); err != nil {
		t.Fatal(err)
	}

	orphan2Rel := filepath.ToSlash(filepath.Join("nested", "deep", "orphan-b.bin"))
	orphan2Full := filepath.Join(repo, filepath.FromSlash(orphan2Rel))
	if err := os.MkdirAll(filepath.Dir(orphan2Full), 0o755); err != nil {
		t.Fatal(err)
	}
	orphan2Bytes := []byte("binary payload 9876543210\xaa\xbb\xcc\xdd")
	if err := os.WriteFile(orphan2Full, orphan2Bytes, 0o644); err != nil {
		t.Fatal(err)
	}

	sessionID := "test-session-11238"
	qref, err := EvictOrphans(ctx, repo, runner, EvictOptions{
		SessionID: sessionID,
		Reason:    "test evict-to-ref archive",
	})
	if err != nil {
		t.Fatalf("EvictOrphans failed: %v", err)
	}
	if qref == nil {
		t.Fatalf("expected non-nil QuarantineRef")
	}

	// 1. Verify QuarantineRef metadata
	expectedPrefix := QuarantineNamespace + sessionID + "/"
	if !strings.HasPrefix(qref.Ref, expectedPrefix) {
		t.Errorf("ref %q does not have prefix %s", qref.Ref, expectedPrefix)
	}
	if len(qref.SHA) != 40 {
		t.Errorf("expected 40-char SHA, got %q", qref.SHA)
	}
	if qref.Count != 2 {
		t.Errorf("expected count 2, got %d", qref.Count)
	}
	expectedByteTotal := int64(len(orphan1Bytes) + len(orphan2Bytes))
	if qref.ByteTotal != expectedByteTotal {
		t.Errorf("expected %d bytes, got %d", expectedByteTotal, qref.ByteTotal)
	}
	if len(qref.Files) != 2 || qref.Files[0] != orphan2Rel || qref.Files[1] != orphan1Rel {
		t.Errorf("expected sorted files [%s, %s], got %v", orphan2Rel, orphan1Rel, qref.Files)
	}

	// 2. Verify git ref exists and points to the commit SHA
	resolvedRef := runGit("rev-parse", qref.Ref)
	if resolvedRef != qref.SHA {
		t.Errorf("resolved ref %s = %s, want %s", qref.Ref, resolvedRef, qref.SHA)
	}

	// 3. Verify orphan files are removed from the working tree
	if _, err := os.Stat(filepath.Join(repo, orphan1Rel)); !os.IsNotExist(err) {
		t.Errorf("expected orphan1 to be removed from working tree, err: %v", err)
	}
	if _, err := os.Stat(orphan2Full); !os.IsNotExist(err) {
		t.Errorf("expected orphan2 to be removed from working tree, err: %v", err)
	}

	// 4. Verify working tree is clean of those files
	statusOut := runGit("status", "--porcelain")
	if strings.Contains(statusOut, orphan1Rel) || strings.Contains(statusOut, "orphan-b.bin") {
		t.Fatalf("working tree still shows orphan files in git status:\n%s", statusOut)
	}

	// 5. Verify byte-for-byte recovery from quarantine ref
	if err := RestoreQuarantine(ctx, repo, runner, qref.Ref); err != nil {
		t.Fatalf("RestoreQuarantine failed: %v", err)
	}

	recovered1, err := os.ReadFile(filepath.Join(repo, orphan1Rel))
	if err != nil {
		t.Fatalf("read recovered orphan1: %v", err)
	}
	if !bytes.Equal(recovered1, orphan1Bytes) {
		t.Fatalf("recovered orphan1 byte mismatch:\ngot  %q\nwant %q", recovered1, orphan1Bytes)
	}

	recovered2, err := os.ReadFile(orphan2Full)
	if err != nil {
		t.Fatalf("read recovered orphan2: %v", err)
	}
	if !bytes.Equal(recovered2, orphan2Bytes) {
		t.Fatalf("recovered orphan2 byte mismatch:\ngot  %q\nwant %q", recovered2, orphan2Bytes)
	}
}

func TestEvictOrphansDryRun(t *testing.T) {
	repo, _ := initTestRepo(t)
	ctx := context.Background()
	runner := GitRunner{}

	filePath := filepath.Join(repo, "dryrun-orphan.txt")
	content := []byte("dry run content")
	if err := os.WriteFile(filePath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	qref, err := EvictOrphans(ctx, repo, runner, EvictOptions{
		DryRun: true,
	})
	if err != nil {
		t.Fatalf("dry run EvictOrphans failed: %v", err)
	}
	if qref == nil || qref.Count != 1 {
		t.Fatalf("expected 1 file in dry run, got: %+v", qref)
	}

	// In dry run, file must still exist on disk
	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("dry run must not remove files from working tree: %v", err)
	}

	// In dry run, ref must not be updated
	if _, err := runner.Run(repo, "rev-parse", "--verify", qref.Ref); err == nil {
		t.Fatalf("dry run must not mint ref %s in git", qref.Ref)
	}
}

func TestEvictOrphansExplicitTargets(t *testing.T) {
	repo, runGit := initTestRepo(t)
	ctx := context.Background()
	runner := GitRunner{}

	fileKeep := filepath.Join(repo, "keep.txt")
	if err := os.WriteFile(fileKeep, []byte("keep this"), 0o644); err != nil {
		t.Fatal(err)
	}
	fileEvict := filepath.Join(repo, "evict.txt")
	if err := os.WriteFile(fileEvict, []byte("evict this"), 0o644); err != nil {
		t.Fatal(err)
	}

	qref, err := EvictOrphans(ctx, repo, runner, EvictOptions{
		Targets: []string{"evict.txt"},
	})
	if err != nil {
		t.Fatalf("EvictOrphans failed: %v", err)
	}
	if qref == nil || qref.Count != 1 || qref.Files[0] != "evict.txt" {
		t.Fatalf("unexpected qref: %+v", qref)
	}

	// fileEvict must be gone, fileKeep must remain
	if _, err := os.Stat(fileEvict); !os.IsNotExist(err) {
		t.Errorf("evict.txt was not removed")
	}
	if _, err := os.Stat(fileKeep); err != nil {
		t.Errorf("keep.txt should still exist: %v", err)
	}

	status := runGit("status", "--porcelain")
	if !strings.Contains(status, "keep.txt") || strings.Contains(status, "evict.txt") {
		t.Errorf("git status unexpected: %s", status)
	}
}

func TestEvictOrphansCleanTree(t *testing.T) {
	repo, _ := initTestRepo(t)
	ctx := context.Background()
	runner := GitRunner{}

	qref, err := EvictOrphans(ctx, repo, runner)
	if err != nil {
		t.Fatalf("expected nil error on clean tree: %v", err)
	}
	if qref != nil {
		t.Fatalf("expected nil qref on clean tree, got %+v", qref)
	}
}

func TestPurgeOrphansAlias(t *testing.T) {
	repo, _ := initTestRepo(t)
	ctx := context.Background()
	runner := GitRunner{}

	orphanPath := filepath.Join(repo, "purge-target.txt")
	if err := os.WriteFile(orphanPath, []byte("purge me"), 0o644); err != nil {
		t.Fatal(err)
	}

	qref, err := PurgeOrphans(ctx, repo, runner, EvictOptions{
		Targets: []string{"purge-target.txt"},
	})
	if err != nil {
		t.Fatalf("PurgeOrphans failed: %v", err)
	}
	if qref == nil || qref.Count != 1 {
		t.Fatalf("unexpected qref: %+v", qref)
	}
	if _, err := os.Stat(orphanPath); !os.IsNotExist(err) {
		t.Fatalf("purge-target.txt should have been removed")
	}
}
