package mergepreview

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPreviewDivergedSameTreeReportsEmptyNetDiff(t *testing.T) {
	repo := mergePreviewRepo(t)
	gitMP(t, repo, "checkout", "-q", "-b", "left")
	writeMP(t, filepath.Join(repo, "f.txt"), "same\n")
	gitMP(t, repo, "commit", "-am", "same-left", "-q")
	gitMP(t, repo, "checkout", "-q", "main")
	gitMP(t, repo, "checkout", "-q", "-b", "right")
	writeMP(t, filepath.Join(repo, "f.txt"), "same\n")
	gitMP(t, repo, "commit", "-am", "same-right", "-q")
	gitMP(t, repo, "checkout", "-q", "left")

	before := statusMP(t, repo)
	got, err := Preview(context.Background(), repo, "right", RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcome != OutcomeEmptyNetDiff || !got.CachedDiffEmpty {
		t.Fatalf("preview = %+v, want empty net diff", got)
	}
	if len(got.ChangedFiles) != 0 || len(got.Conflicts) != 0 {
		t.Fatalf("unexpected paths: %+v", got)
	}
	if after := statusMP(t, repo); after != before {
		t.Fatalf("preview mutated worktree/index: before %q after %q", before, after)
	}
}

func TestPreviewConflictNamesFiles(t *testing.T) {
	repo := mergePreviewRepo(t)
	gitMP(t, repo, "checkout", "-q", "-b", "left")
	writeMP(t, filepath.Join(repo, "f.txt"), "left\n")
	gitMP(t, repo, "commit", "-am", "left", "-q")
	gitMP(t, repo, "checkout", "-q", "main")
	gitMP(t, repo, "checkout", "-q", "-b", "right")
	writeMP(t, filepath.Join(repo, "f.txt"), "right\n")
	gitMP(t, repo, "commit", "-am", "right", "-q")
	gitMP(t, repo, "checkout", "-q", "left")

	got, err := Preview(context.Background(), repo, "right", RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcome != OutcomeConflicts || got.CachedDiffEmpty {
		t.Fatalf("preview = %+v, want conflicts", got)
	}
	if strings.Join(got.Conflicts, ",") != "f.txt" {
		t.Fatalf("conflicts = %+v, want f.txt", got.Conflicts)
	}
	if status := statusMP(t, repo); status != "" {
		t.Fatalf("preview left dirty status: %q", status)
	}
}

func TestPreviewCleanMergeListsChangedFiles(t *testing.T) {
	repo := mergePreviewRepo(t)
	gitMP(t, repo, "checkout", "-q", "-b", "left")
	writeMP(t, filepath.Join(repo, "left.txt"), "left\n")
	gitMP(t, repo, "add", "left.txt")
	gitMP(t, repo, "commit", "-m", "left", "-q")
	gitMP(t, repo, "checkout", "-q", "main")
	gitMP(t, repo, "checkout", "-q", "-b", "right")
	writeMP(t, filepath.Join(repo, "right.txt"), "right\n")
	gitMP(t, repo, "add", "right.txt")
	gitMP(t, repo, "commit", "-m", "right", "-q")
	gitMP(t, repo, "checkout", "-q", "left")

	got, err := Preview(context.Background(), repo, "right", RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	if got.Outcome != OutcomeCleanMerge || got.CachedDiffEmpty {
		t.Fatalf("preview = %+v, want clean merge with changed files", got)
	}
	if strings.Join(got.ChangedFiles, ",") != "right.txt" {
		t.Fatalf("changed files = %+v, want right.txt", got.ChangedFiles)
	}
}

func mergePreviewRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	gitMP(t, repo, "init", "-q", "-b", "main")
	gitMP(t, repo, "config", "core.autocrlf", "false")
	gitMP(t, repo, "config", "user.name", "test")
	gitMP(t, repo, "config", "user.email", "test@example.com")
	writeMP(t, filepath.Join(repo, "f.txt"), "base\n")
	gitMP(t, repo, "add", ".")
	gitMP(t, repo, "commit", "-m", "base", "-q")
	return repo
}

func gitMP(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func writeMP(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func statusMP(t *testing.T, repo string) string {
	t.Helper()
	cmd := exec.Command("git", "status", "--short")
	cmd.Dir = repo
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git status: %v", err)
	}
	return string(out)
}
