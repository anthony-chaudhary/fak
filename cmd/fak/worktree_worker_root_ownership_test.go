package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// landRootFixture builds one throwaway repository with a single commit and a
// detached lane checkout of it, and returns (repo, lane, head). When laneEdit is
// true the lane also carries one committed change trunk does not have.
func landRootFixture(t *testing.T, name string, laneEdit bool) (string, string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	parent := t.TempDir()
	repo := filepath.Join(parent, name)
	lane := filepath.Join(parent, name+"-lane")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git -C %s %s: %v: %s", dir, strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git(repo, "init", "-q", "-b", "main")
	git(repo, "config", "user.email", "t@t")
	git(repo, "config", "user.name", "t")
	git(repo, "config", "commit.gpgsign", "false")
	git(repo, "config", "core.hooksPath", "")
	if err := os.WriteFile(filepath.Join(repo, "a.txt"), []byte("one\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git(repo, "add", ".")
	git(repo, "commit", "-qm", "init")
	head := git(repo, "rev-parse", "HEAD")
	git(repo, "worktree", "add", "--detach", lane, head)
	if laneEdit {
		if err := os.WriteFile(filepath.Join(lane, "a.txt"), []byte("one\nlane\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		git(lane, "add", "a.txt")
		git(lane, "commit", "-qm", "docs(lane): a lane-only edit")
	}
	return repo, lane, head
}

// fak-test:runtime medium est=30s lane=default
func TestWorkerLandRefusesRootOfAnotherRepository(t *testing.T) {
	repoA, _, headA := landRootFixture(t, "repo-a", false)
	_, laneB, _ := landRootFixture(t, "repo-b", true)

	var out, errb bytes.Buffer
	res, code := runWorktreeWorkerLand(&out, &errb, []string{
		"--root", repoA,
		"--worktree", laneB,
		"--paths", "a.txt",
	})
	if code == 0 || res.OK || res.Code != "LAND_ROOT_WORKTREE_MISMATCH" {
		t.Fatalf("want a LAND_ROOT_WORKTREE_MISMATCH refusal, got res=%+v code=%d stderr=%s", res, code, errb.String())
	}
	if !res.Preserved {
		t.Fatal("a cross-repository land refusal must preserve the lane")
	}
	if got := gitHead(t, repoA); got != headA {
		t.Fatalf("repo A trunk moved to %s during a lane-B land (was %s): one repository's lane advanced another's trunk", got, headA)
	}
}

// fak-test:runtime medium est=45s lane=default
func TestWorkerLandDerivesTrunkFromTheWorktreeNotTheCwd(t *testing.T) {
	repoA, _, headA := landRootFixture(t, "repo-a", false)
	repoB, laneB, headB := landRootFixture(t, "repo-b", true)

	// The caller stands in repository A while landing a lane of repository B and
	// names no --root: the trunk under test must be B's, never A's.
	t.Chdir(repoA)
	var out, errb bytes.Buffer
	res, _ := runWorktreeWorkerLand(&out, &errb, []string{
		"--worktree", laneB,
		"--base-sha", headB,
		"--verify", "off",
		"--paths", "a.txt",
	})
	if res.Code == "LAND_ROOT_WORKTREE_MISMATCH" {
		t.Fatalf("an omitted --root must be derived from the worktree, not compared against the cwd: %+v", res)
	}
	if got := gitHead(t, repoA); got != headA {
		t.Fatalf("cwd repository A trunk moved to %s (was %s): the land acted on the caller's cwd, not the lane's repository", got, headA)
	}
	if !res.OK || !res.Committed {
		t.Fatalf("landing a lane of B from A's directory should advance B: %+v (stderr %s)", res, errb.String())
	}
	if got := gitHead(t, repoB); got == headB {
		t.Fatalf("land reported a commit against B (%+v) but B's trunk is unchanged at %s", res, headB)
	}
}

func gitHead(t *testing.T, dir string) string {
	t.Helper()
	cmd := exec.Command("git", "-C", dir, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("rev-parse HEAD in %s: %v", dir, err)
	}
	return strings.TrimSpace(string(out))
}
