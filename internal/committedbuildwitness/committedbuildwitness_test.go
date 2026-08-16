package committedbuildwitness

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestFreshRequiresExactFreshSuccessfulHead(t *testing.T) {
	repo := t.TempDir()
	runGit(t, repo, "init")
	now := time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC)

	if Fresh(repo, "head-a", now) {
		t.Fatal("missing witness reported fresh")
	}
	Record(repo, "head-a", "ci-preflight", now)
	if !Fresh(repo, "head-a", now.Add(time.Minute)) {
		t.Fatal("fresh exact-head witness was not reusable")
	}
	if Fresh(repo, "head-b", now.Add(time.Minute)) {
		t.Fatal("witness leaked across heads")
	}
	if Fresh(repo, "head-a", now.Add(TTL+time.Second)) {
		t.Fatal("stale witness was reusable")
	}
}

func TestFreshRejectsMalformedReceipt(t *testing.T) {
	repo := t.TempDir()
	runGit(t, repo, "init")
	path := witnessPath(repo, "head-a")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(`{"schema":"wrong"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if Fresh(repo, "head-a", time.Now()) {
		t.Fatal("malformed witness was reusable")
	}
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}
