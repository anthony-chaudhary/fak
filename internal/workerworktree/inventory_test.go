package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestInventoryLandReadyIsReadOnlyAndExact(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	runGitTest(t, root, "init", repo)
	runGitTest(t, repo, "config", "user.name", "Test")
	runGitTest(t, repo, "config", "user.email", "test-at-example.invalid")
	if err := os.WriteFile(filepath.Join(repo, "owned.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer-base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitTest(t, repo, "add", "owned.txt", "peer.txt")
	runGitTest(t, repo, "commit", "-m", "base")
	base := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	prep := Prepare(repo, "test", "5994", base, filepath.Join(root, "workers"), nil)
	if !prep.OK {
		t.Fatalf("prepare = %+v", prep)
	}
	t.Cleanup(func() { _ = Reap(repo, prep.Path, nil) })
	if err := SaveIntent(prep.Path, base, "feat(test): land intent (#5994) (fak test)", []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(prep.Path, "owned.txt"), []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer-wip\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	beforeHead := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	beforeIndex := runGitTest(t, repo, "write-tree")
	beforePeer, _ := os.ReadFile(filepath.Join(repo, "peer.txt"))
	rows, err := Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %+v", rows)
	}
	want := []string{"fak", "worktree", "worker", "land", "--worktree", prep.Path, "--base-sha", base, "--msg-file", messagePath(prep.Path), "--paths", "owned.txt"}
	if rows[0].State != "LAND_READY" || !reflect.DeepEqual(rows[0].DirtyPaths, []string{"owned.txt"}) || !reflect.DeepEqual(rows[0].LandArgv, want) {
		t.Fatalf("row = %+v; want argv %q", rows[0], want)
	}
	if got := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD")); got != beforeHead {
		t.Fatalf("HEAD mutated: %s -> %s", beforeHead, got)
	}
	if got := runGitTest(t, repo, "write-tree"); got != beforeIndex {
		t.Fatalf("index mutated: %q -> %q", beforeIndex, got)
	}
	afterPeer, _ := os.ReadFile(filepath.Join(repo, "peer.txt"))
	if !reflect.DeepEqual(afterPeer, beforePeer) {
		t.Fatalf("peer bytes mutated")
	}
}

func TestInventoryCleanAndAmbiguousDoNotOverclaim(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	runGitTest(t, root, "init", repo)
	runGitTest(t, repo, "config", "user.name", "Test")
	runGitTest(t, repo, "config", "user.email", "test-at-example.invalid")
	_ = os.WriteFile(filepath.Join(repo, "a.txt"), []byte("a\n"), 0o644)
	_ = os.WriteFile(filepath.Join(repo, "b.txt"), []byte("b\n"), 0o644)
	runGitTest(t, repo, "add", ".")
	runGitTest(t, repo, "commit", "-m", "base")
	base := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	prep := Prepare(repo, "test", "ambiguous", base, filepath.Join(root, "workers"), nil)
	if !prep.OK {
		t.Fatalf("prepare = %+v", prep)
	}
	t.Cleanup(func() { _ = Reap(repo, prep.Path, nil) })
	if err := SaveIntent(prep.Path, base, "message", []string{"a.txt"}); err != nil {
		t.Fatal(err)
	}
	rows, err := Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].State != "CLEAN" || rows[0].NeedsOperator {
		t.Fatalf("clean row = %+v", rows)
	}
	_ = os.WriteFile(filepath.Join(prep.Path, "b.txt"), []byte("changed\n"), 0o644)
	rows, err = Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].State != "NEEDS_OPERATOR" || !rows[0].NeedsOperator || len(rows[0].LandArgv) != 0 {
		t.Fatalf("ambiguous row = %+v", rows)
	}
}
