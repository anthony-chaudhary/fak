package workerworktree

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestBlockCloneNativeProbeAndMaterialize(t *testing.T) {
	repo, base := testBackendRepo(t)
	targetRoot := t.TempDir()
	probeErr := probeBlockClone(targetRoot)

	backend := newBlockCloneBackend()
	start := time.Now()
	res := PrepareWithBackend(repo, "workerworktree", "native-probe-test", base, targetRoot, nil, backend)
	elapsed := time.Since(start)

	if !res.OK {
		t.Fatalf("prepare with native backend: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if probeErr != nil {
		// On non-CoW filesystems, backend must have fallen back to git-worktree.
		if res.Backend != gitWorktreeBackendName {
			t.Fatalf("expected fallback backend %q, got %q", gitWorktreeBackendName, res.Backend)
		}
		if !strings.Contains(res.Detail, "fell back to git-worktree") {
			t.Fatalf("detail does not mention fallback: %q", res.Detail)
		}
		t.Logf("probe failed as expected on non-CoW filesystem (%v); fell back to git-worktree in %v", probeErr, elapsed)
		return
	}

	// Probe succeeded, so block-clone backend should have materialized.
	if res.Backend != blockCloneBackendName {
		t.Fatalf("expected backend %q, got %q (detail: %s)", blockCloneBackendName, res.Backend, res.Detail)
	}
	t.Logf("native block-clone materialized in %v", elapsed)

	// Verify file content matches base commit.
	wtKnown := filepath.Join(res.Path, "known.txt")
	content, err := os.ReadFile(wtKnown)
	if err != nil {
		t.Fatalf("read cloned file: %v", err)
	}
	if string(content) != "before\n" {
		t.Fatalf("cloned content = %q, want %q", string(content), "before\n")
	}

	// Verify CoW isolation: edit worktree file and assert source repo is unmodified.
	if err := os.WriteFile(wtKnown, []byte("after\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	repoContent, err := os.ReadFile(filepath.Join(repo, "known.txt"))
	if err != nil {
		t.Fatalf("read repo file: %v", err)
	}
	if string(repoContent) != "before\n" {
		t.Fatalf("repo file mutated by worktree write: got %q want %q", string(repoContent), "before\n")
	}

	patch := runBlockCloneGitTest(t, res.Path, "diff", base, "--", "known.txt")
	if !strings.Contains(patch, "-before") || !strings.Contains(patch, "+after") {
		t.Fatalf("patch lost worker edit:\n%s", patch)
	}
}

func TestBlockCloneDirectFileClone(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.bin")
	dst := filepath.Join(dir, "dst.bin")

	// 8KB test payload
	data := make([]byte, 8192)
	for i := range data {
		data[i] = byte(i % 251)
	}
	if err := os.WriteFile(src, data, 0o644); err != nil {
		t.Fatal(err)
	}

	err := cloneFileBlocks(src, dst)
	if err != nil {
		// If platform/filesystem doesn't support block cloning, ensure dst is not left around.
		if _, statErr := os.Stat(dst); !os.IsNotExist(statErr) {
			t.Fatalf("cloneFileBlocks failed (%v) but left destination behind", err)
		}
		t.Logf("cloneFileBlocks unsupported on this volume/platform: %v", err)
		return
	}

	// Verify dst content matches src.
	cloned, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read cloned file: %v", err)
	}
	if len(cloned) != len(data) {
		t.Fatalf("cloned length = %d, want %d", len(cloned), len(data))
	}
	for i := range cloned {
		if cloned[i] != data[i] {
			t.Fatalf("cloned byte mismatch at %d: %x != %x", i, cloned[i], data[i])
		}
	}

	// Verify CoW isolation: mutating dst does not alter src.
	cloned[0] ^= 0xff
	if err := os.WriteFile(dst, cloned, 0o644); err != nil {
		t.Fatal(err)
	}
	srcCheck, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if srcCheck[0] != data[0] {
		t.Fatalf("src mutated after dst write: got %x want %x", srcCheck[0], data[0])
	}

	// Cloning onto an existing dst must fail (EEXIST on Darwin, O_EXCL on Linux/Windows).
	if dupErr := cloneFileBlocks(src, dst); dupErr == nil {
		t.Fatal("cloneFileBlocks onto existing dst succeeded, expected error")
	}
}

func TestBlockCloneFastCheckoutFallbackOnModifiedBlob(t *testing.T) {
	repo, base := testBackendRepo(t)
	targetRoot := t.TempDir()
	if err := probeBlockClone(targetRoot); err != nil {
		t.Skipf("skipping on non-block-clone volume: %v", err)
	}

	// Dirty the repo's working copy of known.txt so gitBlobSHA1(src) != objectID.
	if err := os.WriteFile(filepath.Join(repo, "known.txt"), []byte("dirty-in-repo\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	backend := newBlockCloneBackend()
	res := PrepareWithBackend(repo, "workerworktree", "dirty-fallback", base, targetRoot, nil, backend)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if res.Backend != blockCloneBackendName {
		t.Fatalf("expected backend %q, got %q", blockCloneBackendName, res.Backend)
	}

	// The materialized worktree should contain the clean base version "before\n" via git checkout fallback.
	wtContent, err := os.ReadFile(filepath.Join(res.Path, "known.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(wtContent) != "before\n" {
		t.Fatalf("worktree has dirty content %q, want clean base %q", string(wtContent), "before\n")
	}
}
