package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWipRemoteDrainBareRemoteKeepsUnlandedAndReportIsReadOnly(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	repo := filepath.Join(root, "repo")
	gitDrainTest(t, root, "init", "--bare", remote)
	gitDrainTest(t, root, "init", repo)
	gitDrainTest(t, repo, "config", "user.name", "Test")
	gitDrainTest(t, repo, "config", "user.email", "test@example.invalid")
	os.WriteFile(filepath.Join(repo, "a.txt"), []byte("base\n"), 0644)
	gitDrainTest(t, repo, "add", "a.txt")
	gitDrainTest(t, repo, "commit", "-m", "base")
	gitDrainTest(t, repo, "branch", "-M", "main")
	gitDrainTest(t, repo, "remote", "add", "origin", remote)
	gitDrainTest(t, repo, "push", "-u", "origin", "main")
	gitDrainTest(t, remote, "symbolic-ref", "HEAD", "refs/heads/main")
	base := strings.TrimSpace(gitDrainTest(t, repo, "rev-parse", "HEAD"))
	os.WriteFile(filepath.Join(repo, "a.txt"), []byte("unlanded\n"), 0644)
	gitDrainTest(t, repo, "add", "a.txt")
	tree := strings.TrimSpace(gitDrainTest(t, repo, "write-tree"))
	candidate := strings.TrimSpace(gitDrainInputTest(t, repo, "checkpoint\n", "commit-tree", tree, "-p", base))
	gitDrainTest(t, repo, "update-ref", "refs/fak/wip/own", candidate)
	gitDrainTest(t, repo, "push", "origin", "refs/fak/wip/own:refs/fak/wip/own")
	before := gitDrainTest(t, root, "ls-remote", remote, "refs/fak/wip/*")
	report, err := wipRemoteDrain(context.Background(), repo, "origin", false, false)
	if err != nil {
		t.Fatal(err)
	}
	after := gitDrainTest(t, root, "ls-remote", remote, "refs/fak/wip/*")
	if before != after || report.Applied || len(report.Deleted) != 0 {
		t.Fatalf("report mutated remote: before=%q after=%q result=%+v", before, after, report)
	}
	applied, err := wipRemoteDrain(context.Background(), repo, "origin", true, false)
	if err != nil {
		t.Fatal(err)
	}
	after = gitDrainTest(t, root, "ls-remote", remote, "refs/fak/wip/*")
	if after == "" || len(applied.Deleted) != 0 || len(applied.Candidates) != 1 || applied.Candidates[0].DeltaContained {
		t.Fatalf("unlanded checkpoint deleted: %q %+v", after, applied)
	}
}

func gitDrainTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	o, e := c.CombinedOutput()
	if e != nil {
		t.Fatalf("git %v: %v: %s", args, e, o)
	}
	return string(o)
}
func gitDrainInputTest(t *testing.T, dir, input string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	c.Stdin = strings.NewReader(input)
	o, e := c.CombinedOutput()
	if e != nil {
		t.Fatalf("git %v: %v: %s", args, e, o)
	}
	return string(o)
}
