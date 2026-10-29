package safecommit

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// TestRefusedCommitRestoresIndexItStaged reproduces the loaded-host wedge with real git: the
// pre-commit hook refuses the commit the way a timed-out COMMITTED_RED build check does, after
// CommitWith already staged the requested pathspec. The refusal must leave the index as it
// found it (the requested paths unstaged again, a pre-existing staged entry untouched), so the
// retry is not refused PRESTAGED_PATH_OVERLAP on the command's own staging.
func TestRefusedCommitRestoresIndexItStaged(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	initTempRepo(t, repo)
	hooksDir := filepath.Join(repo, ".git", "hooks")
	tempRepoGit(t, repo, "config", "core.hooksPath", filepath.ToSlash(hooksDir))
	writeTempRepoFile(t, filepath.Join(repo, "f.txt"), "one\n")
	writeTempRepoFile(t, filepath.Join(repo, "peer.txt"), "peer one\n")
	tempRepoGit(t, repo, "add", "--", "f.txt", "peer.txt")
	tempRepoGit(t, repo, "commit", "-qm", "seed")
	head0 := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))

	// A peer's staged entry that is not part of this commit, plus this commit's own edits
	// (one modified file, one new file).
	writeTempRepoFile(t, filepath.Join(repo, "peer.txt"), "peer two\n")
	tempRepoGit(t, repo, "add", "--", "peer.txt")
	writeTempRepoFile(t, filepath.Join(repo, "f.txt"), "two\n")
	writeTempRepoFile(t, filepath.Join(repo, "new.txt"), "new\n")

	hook := filepath.Join(hooksDir, "pre-commit")
	writeTempRepoFile(t, hook, "#!/bin/sh\necho 'COMMITTED_RED: build-check skipped-timeout' >&2\nexit 1\n")
	if err := os.Chmod(hook, 0o755); err != nil {
		t.Fatal(err)
	}

	opts := Options{Dir: repo, Paths: []string{"f.txt", "new.txt"}, Message: "fix: refused by hook (fak safecommit)"}
	res, err := Commit(context.Background(), opts)
	if err != nil {
		t.Fatalf("Commit: %v", err)
	}
	if res.Reason != ReasonHookRefused || res.Committed {
		t.Fatalf("want an uncommitted %s refusal, got reason=%q committed=%t", ReasonHookRefused, res.Reason, res.Committed)
	}
	if got := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD")); got != head0 {
		t.Fatalf("a refused commit moved HEAD %s -> %s", head0, got)
	}
	if got, want := stagedNames(t, repo), []string{"peer.txt"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("index after refusal: staged=%v, want only the pre-existing %v", got, want)
	}
	if want := []string{"f.txt", "new.txt"}; !reflect.DeepEqual(res.IndexRestored, want) {
		t.Fatalf("IndexRestored=%v, want %v", res.IndexRestored, want)
	}

	// The retry, with the gate now passing, is admitted and commits exactly the pathspec.
	if err := os.Remove(hook); err != nil {
		t.Fatal(err)
	}
	res, err = Commit(context.Background(), opts)
	if err != nil {
		t.Fatalf("retry Commit: %v", err)
	}
	if res.Reason != "" || !res.Committed {
		t.Fatalf("retry after restore: reason=%q committed=%t", res.Reason, res.Committed)
	}
	if got, want := stagedNames(t, repo), []string{"peer.txt"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("index after retry: staged=%v, want the peer entry still staged %v", got, want)
	}
}

func stagedNames(t *testing.T, repo string) []string {
	t.Helper()
	var names []string
	for _, line := range strings.Split(tempRepoGit(t, repo, "diff", "--cached", "--name-only"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			names = append(names, line)
		}
	}
	return names
}
