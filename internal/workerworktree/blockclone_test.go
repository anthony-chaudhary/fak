package workerworktree

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestBlockCloneProbeFallbackIsRecorded(t *testing.T) {
	backend := blockClone{
		probe: func(string) error { return errors.New("unsupported volume") },
		clone: func(string, string) error { t.Fatal("clone called after failed probe"); return nil },
	}
	repo, base := testBackendRepo(t)
	res := PrepareWithBackend(repo, "workerworktree", "fallback", base, t.TempDir(), nil, backend)
	if !res.OK {
		t.Fatalf("prepare fallback: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })
	if res.Backend != gitWorktreeBackendName {
		t.Fatalf("backend=%q want %q", res.Backend, gitWorktreeBackendName)
	}
	if !strings.Contains(res.Detail, "unsupported volume") {
		t.Fatalf("detail does not record fallback: %q", res.Detail)
	}
}

func TestBlockCloneMaterializationPreservesLandPatch(t *testing.T) {
	repo, base := testBackendRepo(t)
	backend := blockClone{
		probe: func(string) error { return nil },
		clone: copyFileForBlockCloneTest,
	}
	res := PrepareWithBackend(repo, "workerworktree", "clone", base, t.TempDir(), nil, backend)
	if !res.OK {
		t.Fatalf("materialize: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })
	if res.Backend != blockCloneBackendName {
		t.Fatalf("backend=%q", res.Backend)
	}
	if err := os.WriteFile(filepath.Join(res.Path, "known.txt"), []byte("after\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch := runBlockCloneGitTest(t, res.Path, "diff", base, "--", "known.txt")
	if !strings.Contains(patch, "-before") || !strings.Contains(patch, "+after") {
		t.Fatalf("patch lost worker edit:\n%s", patch)
	}
}

func testBackendRepo(t *testing.T) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	runBlockCloneGitTest(t, repo, "init", "-q", "-b", "main")
	runBlockCloneGitTest(t, repo, "config", "user.email", "backend@test")
	runBlockCloneGitTest(t, repo, "config", "user.name", "backend")
	if err := os.WriteFile(filepath.Join(repo, "known.txt"), []byte("before\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runBlockCloneGitTest(t, repo, "add", "known.txt")
	runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "base")
	return repo, strings.TrimSpace(runBlockCloneGitTest(t, repo, "rev-parse", "HEAD"))
}

func runBlockCloneGitTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func copyFileForBlockCloneTest(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	info, err := in.Stat()
	if err != nil {
		return err
	}
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, info.Mode().Perm())
	if err != nil {
		return err
	}
	_, copyErr := out.ReadFrom(in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}
