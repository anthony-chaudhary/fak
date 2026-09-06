package safecommit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTurnCommit_Success(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)

	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	initialSHA := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if initialSHA == "" {
		t.Fatal("expected non-empty initial SHA")
	}

	newFilePath := filepath.Join(repo, "feature.txt")
	writeTempRepoFile(t, newFilePath, "feature work\n")

	res, err := CommitTurn(repo, []string{"feature.txt"}, "feat(core): turn commit", "")
	if err != nil {
		t.Fatalf("unexpected infrastructure error: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil TurnCommitResult")
	}
	if !res.Committed {
		t.Errorf("expected Committed=true, got %v", res.Committed)
	}
	if res.RolledBack {
		t.Errorf("expected RolledBack=false, got %v", res.RolledBack)
	}
	if res.Error != nil {
		t.Errorf("expected Error=nil, got %v", res.Error)
	}
	if res.CommitSHA == "" || res.CommitSHA == initialSHA {
		t.Errorf("expected new commit SHA, got %q", res.CommitSHA)
	}

	currentHead := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if currentHead != res.CommitSHA {
		t.Errorf("HEAD mismatch: got %q, want %q", currentHead, res.CommitSHA)
	}

	logMsg := strings.TrimSpace(tempRepoGit(t, repo, "log", "-1", "--format=%s"))
	if logMsg != "feat(core): turn commit" {
		t.Errorf("commit message mismatch: got %q, want %q", logMsg, "feat(core): turn commit")
	}

	status := strings.TrimSpace(tempRepoGit(t, repo, "status", "--porcelain"))
	if status != "" {
		t.Errorf("expected clean working tree after commit, got %q", status)
	}
}

func TestTurnCommit_SuccessWithPassingTest(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)

	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "initial\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "updated\n")

	res, err := CommitTurn(repo, []string{"seed.txt"}, "feat: update seed", "exit 0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Committed || res.RolledBack || res.Error != nil {
		t.Fatalf("unexpected result: %+v", res)
	}

	content, err := os.ReadFile(filepath.Join(repo, "seed.txt"))
	if err != nil {
		t.Fatalf("failed reading file: %v", err)
	}
	if string(content) != "updated\n" {
		t.Errorf("expected updated content, got %q", string(content))
	}
}

func TestTurnCommit_RollbackOnTestFailure(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)

	seedFile := filepath.Join(repo, "seed.txt")
	writeTempRepoFile(t, seedFile, "original seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial commit")

	initialSHA := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if initialSHA == "" {
		t.Fatal("expected non-empty initial SHA")
	}

	// Modify tracked file and create a new untracked file
	writeTempRepoFile(t, seedFile, "broken modifications\n")
	badFile := filepath.Join(repo, "bad.txt")
	writeTempRepoFile(t, badFile, "should be deleted on rollback\n")

	res, err := CommitTurn(repo, []string{"seed.txt", "bad.txt"}, "feat: doomed turn", "exit 1")
	if err != nil {
		t.Fatalf("unexpected infrastructure error: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil TurnCommitResult")
	}
	if res.Committed {
		t.Errorf("expected Committed=false, got %v", res.Committed)
	}
	if !res.RolledBack {
		t.Errorf("expected RolledBack=true, got %v", res.RolledBack)
	}
	if res.Error == nil {
		t.Errorf("expected non-nil Error, got nil")
	}

	// Verify working tree restored to pre-turn commit
	headAfter := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if headAfter != initialSHA {
		t.Errorf("HEAD moved: got %q, want %q", headAfter, initialSHA)
	}

	seedContent, err := os.ReadFile(seedFile)
	if err != nil {
		t.Fatalf("failed reading seed file: %v", err)
	}
	if string(seedContent) != "original seed content\n" {
		t.Errorf("seed file content not rolled back: got %q", string(seedContent))
	}

	if _, err := os.Stat(badFile); !os.IsNotExist(err) {
		t.Errorf("bad.txt was not deleted by rollback, err=%v", err)
	}

	status := strings.TrimSpace(tempRepoGit(t, repo, "status", "--porcelain"))
	if status != "" {
		t.Errorf("expected clean working tree after rollback, got %q", status)
	}
}

func TestTurnCommit_UnbornBranch_Rollback(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)

	initFile := filepath.Join(repo, "init.txt")
	writeTempRepoFile(t, initFile, "first file\n")

	res, err := CommitTurn(repo, []string{"init.txt"}, "feat: first commit", "exit 1")
	if err != nil {
		t.Fatalf("unexpected infrastructure error: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil TurnCommitResult")
	}
	if res.Committed || !res.RolledBack || res.Error == nil {
		t.Fatalf("unexpected result: %+v", res)
	}

	if _, err := os.Stat(initFile); !os.IsNotExist(err) {
		t.Errorf("init.txt was not deleted during rollback in unborn branch")
	}

	status := strings.TrimSpace(tempRepoGit(t, repo, "status", "--porcelain"))
	if status != "" {
		t.Errorf("expected clean status after rollback, got %q", status)
	}
}

func TestTurnCommit_PerTurnCommitter(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)

	writeTempRepoFile(t, filepath.Join(repo, "a.txt"), "alpha\n")
	tempRepoGit(t, repo, "add", "a.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial")

	committer := NewPerTurnCommitter(repo, "exit 0")

	writeTempRepoFile(t, filepath.Join(repo, "b.txt"), "beta\n")
	res, err := committer.CommitTurn([]string{"b.txt"}, "feat: add b")
	if err != nil {
		t.Fatalf("committer.CommitTurn failed: %v", err)
	}
	if !res.Committed || res.RolledBack {
		t.Fatalf("unexpected committer result: %+v", res)
	}

	// Change test command to failing command
	committer.TestCmd = "exit 1"
	writeTempRepoFile(t, filepath.Join(repo, "c.txt"), "gamma\n")
	resFail, err := committer.Commit([]string{"c.txt"}, "feat: should roll back")
	if err != nil {
		t.Fatalf("committer.Commit unexpected error: %v", err)
	}
	if resFail.Committed || !resFail.RolledBack || resFail.Error == nil {
		t.Fatalf("unexpected failing committer result: %+v", resFail)
	}

	if _, err := os.Stat(filepath.Join(repo, "c.txt")); !os.IsNotExist(err) {
		t.Errorf("c.txt still exists after rollback")
	}
}

func TestTurnCommit_InvalidRepo(t *testing.T) {
	nonRepo := t.TempDir()
	res, err := CommitTurn(nonRepo, []string{"file.txt"}, "feat: invalid", "")
	if err == nil {
		t.Fatal("expected error on non-git repository")
	}
	if res != nil {
		t.Errorf("expected nil result on invalid repo, got %+v", res)
	}
}
