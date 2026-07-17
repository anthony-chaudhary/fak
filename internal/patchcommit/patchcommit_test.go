package patchcommit

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCommitOwnsOneHunkAndPreservesPeerState(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "shared.txt", "one\ntwo\nthree\nfour\nfive\n")
	write(t, repo, "peer.txt", "base\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")

	// Owned hunk changes line 2. Capture exactly that hunk, then restore its
	// preimage before introducing a disjoint peer hunk in the same file.
	write(t, repo, "shared.txt", "one\nTWO\nthree\nfour\nfive\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--unified=0", "--", "shared.txt"))
	write(t, repo, "shared.txt", "one\nTWO\nthree\nfour\nFIVE\n")
	write(t, repo, "peer.txt", "peer staged\n")
	git(t, repo, "add", "peer.txt")
	peerBefore := git(t, repo, "diff", "--cached", "--binary")

	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"shared.txt"}, Message: "fix(test): commit owned hunk", Signoff: true})
	if err != nil || res.Reason != "" || res.SHA == "" {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
	if got := git(t, repo, "show", "--format=", "--unified=0", res.SHA); !strings.Contains(got, "+TWO") || strings.Contains(got, "+FIVE") {
		t.Fatalf("commit did not contain exactly owned hunk:\n%s", got)
	}
	if got := git(t, repo, "diff", "--unified=0", "--", "shared.txt"); !strings.Contains(got, "+FIVE") || strings.Contains(got, "+TWO") {
		t.Fatalf("peer hunk not preserved as dirty:\n%s", got)
	}
	if got := git(t, repo, "diff", "--cached", "--binary"); got != peerBefore {
		t.Fatalf("peer staging changed:\nbefore=%s\nafter=%s", peerBefore, got)
	}
}

func TestCommitRefusesPatchAfterOverlappingEdit(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "f.txt", "a\nb\nc\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	write(t, repo, "f.txt", "a\nB\nc\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--unified=0", "--", "f.txt"))
	write(t, repo, "f.txt", "a\nPEER\nc\n")
	before := git(t, repo, "rev-parse", "HEAD")
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"f.txt"}, Message: "fix(test): owned"})
	if err != nil || res.Reason != ReasonPatchInvalid {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
	if got := git(t, repo, "rev-parse", "HEAD"); got != before {
		t.Fatalf("HEAD moved: %s -> %s", before, got)
	}
}

func TestCommitAddsOwnedFile(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "base.txt", "base\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	write(t, repo, "new.txt", "owned\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", "diff --git a/new.txt b/new.txt\nnew file mode 100644\n--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+owned\n")
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"new.txt"}, Message: "feat(test): add owned"})
	if err != nil || res.Reason != "" {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
	if got := git(t, repo, "show", "HEAD:new.txt"); got != "owned\n" {
		t.Fatalf("committed file = %q", got)
	}
}

func TestCommitRefusesPathOutsideAllowlist(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "a.txt", "a\n")
	write(t, repo, "b.txt", "b\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	write(t, repo, "a.txt", "A\n")
	write(t, repo, "b.txt", "B\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--unified=0", "--", "a.txt", "b.txt"))
	before := git(t, repo, "rev-parse", "HEAD")
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"a.txt"}, Message: "fix(test): owned"})
	if err != nil || res.Reason != ReasonPatchPaths {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
	if got := git(t, repo, "rev-parse", "HEAD"); got != before {
		t.Fatalf("HEAD moved: %s -> %s", before, got)
	}
}

func TestCommitRefusesPrestagedRequestedPath(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "f.txt", "a\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	write(t, repo, "f.txt", "A\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--unified=0", "--", "f.txt"))
	git(t, repo, "add", "f.txt")
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"f.txt"}, Message: "fix(test): owned"})
	if err != nil || res.Reason != ReasonPatchStaged {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
}

func TestCommitRefusesModeChange(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("executable-bit fixture is not represented by the Windows worktree")
	}
	repo := newRepo(t)
	write(t, repo, "f.sh", "echo ok\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	git(t, repo, "update-index", "--chmod=+x", "f.sh")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--cached", "--binary", "HEAD"))
	git(t, repo, "reset", "-q", "HEAD", "--", "f.sh")
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"f.sh"}, Message: "fix(test): owned"})
	if err != nil || res.Reason != ReasonPatchInvalid {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
}

func TestCommitRefusesHeadRace(t *testing.T) {
	repo := newRepo(t)
	write(t, repo, "f.txt", "a\nb\n")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "-m", "base")
	write(t, repo, "f.txt", "A\nb\n")
	patch := filepath.Join(repo, "owned.patch")
	write(t, repo, "owned.patch", git(t, repo, "diff", "--unified=0", "--", "f.txt"))
	res, err := Commit(context.Background(), Options{Dir: repo, PatchFile: patch, Paths: []string{"f.txt"}, Message: "fix(test): owned", BeforeCAS: func() {
		write(t, repo, "race.txt", "race\n")
		git(t, repo, "add", "race.txt")
		git(t, repo, "commit", "-m", "racer")
	}})
	if err != nil || res.Reason != ReasonHeadRace {
		t.Fatalf("Commit = %#v, %v", res, err)
	}
	if got := strings.TrimSpace(git(t, repo, "log", "-1", "--format=%s")); got != "racer" {
		t.Fatalf("winner was overwritten: %q", got)
	}
}

func newRepo(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	git(t, d, "init", "-b", "main")
	git(t, d, "config", "user.name", "test")
	git(t, d, "config", "user.email", "test@example.test")
	return d
}
func write(t *testing.T, root, name, body string) {
	t.Helper()
	p := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}
func git(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	c.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1")
	b, e := c.CombinedOutput()
	if e != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), e, b)
	}
	return string(b)
}
