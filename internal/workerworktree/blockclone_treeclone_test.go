package workerworktree

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// treeCloneTestFixture builds a real git repo with a nested, multi-file tracked
// tree: nested directories, several regular files (one empty, one large-ish),
// an executable (mode 100755) and a symlink. It returns the repo root and the
// base commit SHA. Every top-level tracked entry is either a tree or a blob and
// all of it lives under the committed base.
func treeCloneTestFixture(t *testing.T) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	// The warm pool would let Prepare lease an idle member and bypass the
	// materialization we are witnessing; force the pre-pool create path.
	t.Setenv(PoolCapEnv, "0")

	repo := t.TempDir()
	runBlockCloneGitTest(t, repo, "init", "-q", "-b", "main")
	runBlockCloneGitTest(t, repo, "config", "user.email", "treeclone@test")
	runBlockCloneGitTest(t, repo, "config", "user.name", "treeclone")

	mustWrite := func(rel string, data []byte, mode os.FileMode) {
		t.Helper()
		full := filepath.Join(repo, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, data, mode); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(full, mode); err != nil {
			t.Fatal(err)
		}
	}

	// Top-level blob.
	mustWrite("top.txt", []byte("top\n"), 0o644)
	// Top-level tree with nested subdirectories and files.
	mustWrite("dir/alpha.txt", []byte("alpha\n"), 0o644)
	mustWrite("dir/beta.txt", bytes.Repeat([]byte("beta-line\n"), 1024), 0o644)
	mustWrite("dir/nested/gamma.txt", []byte("gamma\n"), 0o644)
	// Executable file (mode 100755).
	mustWrite("dir/run.sh", []byte("#!/bin/sh\necho hi\n"), 0o755)
	// Empty file (size 0).
	mustWrite("dir/empty.txt", nil, 0o644)
	// Symlink tracked by git.
	if err := os.Symlink("alpha.txt", filepath.Join(repo, "dir", "link")); err != nil {
		t.Fatal(err)
	}

	runBlockCloneGitTest(t, repo, "add", "-A")
	runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(runBlockCloneGitTest(t, repo, "rev-parse", "HEAD"))
	return repo, base
}

// cloneTreeCopyTest is a deterministic copy-based recursive tree clone used to
// inject the fast path on any volume (does NOT depend on APFS). It mirrors the
// real CloneTree contract: directories recreated, regular files byte-copied with
// their permission bits, symlinks recreated, non-regular nodes skipped.
func cloneTreeCopyTest(src, dst string) error {
	info, err := os.Lstat(src)
	if err != nil {
		return err
	}
	switch {
	case info.IsDir():
		if err := os.Mkdir(dst, info.Mode().Perm()); err != nil && !os.IsExist(err) {
			return err
		}
		entries, err := os.ReadDir(src)
		if err != nil {
			return err
		}
		for _, e := range entries {
			if err := cloneTreeCopyTest(filepath.Join(src, e.Name()), filepath.Join(dst, e.Name())); err != nil {
				return err
			}
		}
		return nil
	case info.Mode()&os.ModeSymlink != 0:
		target, err := os.Readlink(src)
		if err != nil {
			return err
		}
		return os.Symlink(target, dst)
	case info.Mode().IsRegular():
		return copyFileForBlockCloneTest(src, dst)
	default:
		return nil
	}
}

func treeCloneFastBackend() blockClone {
	return blockClone{
		probe:     func(string) error { return nil },
		clone:     copyFileForBlockCloneTest,
		treeClone: cloneTreeCopyTest,
	}
}

// TestTreeCloneFastPathCleanTreeContentAndIsolation witnesses the whole-tree
// fast path on a perfectly clean tree: it is TAKEN (counter delta), every
// tracked path is byte-identical, the executable bit, empty file and symlink
// survive, and mutating the worktree does not touch the source repo.
func TestTreeCloneFastPathCleanTreeContentAndIsolation(t *testing.T) {
	repo, base := treeCloneTestFixture(t)
	backend := treeCloneFastBackend()

	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-clean", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 1 {
		t.Fatalf("fast-path counter delta = %d, want 1 (res=%+v)", delta, res)
	}
	if res.Backend != blockCloneBackendName {
		t.Fatalf("backend = %q, want %q", res.Backend, blockCloneBackendName)
	}
	if !strings.Contains(res.Detail, "tree-clone fast path") {
		t.Fatalf("detail %q does not contain %q", res.Detail, "tree-clone fast path")
	}

	// Every tracked path exists with byte-identical content.
	_, listing := run(nil, repo, []string{"ls-tree", "-r", "-z", "--full-tree", base})
	for _, record := range strings.Split(listing, "\x00") {
		if record == "" {
			continue
		}
		tab := strings.IndexByte(record, '\t')
		if tab < 0 {
			t.Fatalf("bad ls-tree record: %q", record)
		}
		fields := strings.Fields(record[:tab])
		if len(fields) != 3 {
			t.Fatalf("bad ls-tree fields: %q", record)
		}
		mode, typ, rel := fields[0], fields[1], record[tab+1:]
		if typ != "blob" {
			continue
		}
		srcPath := filepath.Join(repo, filepath.FromSlash(rel))
		dstPath := filepath.Join(res.Path, filepath.FromSlash(rel))
		srcInfo, err := os.Lstat(srcPath)
		if err != nil {
			t.Fatalf("stat source %s: %v", rel, err)
		}
		if srcInfo.Mode()&os.ModeSymlink != 0 {
			want, err := os.Readlink(srcPath)
			if err != nil {
				t.Fatal(err)
			}
			got, err := os.Readlink(dstPath)
			if err != nil {
				t.Fatalf("materialized %s is not a symlink: %v", rel, err)
			}
			if got != want {
				t.Fatalf("symlink %s target = %q, want %q", rel, got, want)
			}
			continue
		}
		srcData, err := os.ReadFile(srcPath)
		if err != nil {
			t.Fatalf("read source %s: %v", rel, err)
		}
		dstData, err := os.ReadFile(dstPath)
		if err != nil {
			t.Fatalf("read materialized %s: %v", rel, err)
		}
		if !bytes.Equal(srcData, dstData) {
			t.Fatalf("content mismatch for %s: got %d bytes want %d bytes", rel, len(dstData), len(srcData))
		}
		if mode == "100755" {
			info, err := os.Stat(dstPath)
			if err != nil {
				t.Fatal(err)
			}
			if info.Mode().Perm() != 0o755 {
				t.Fatalf("executable %s perm = %o, want 755", rel, info.Mode().Perm())
			}
		}
	}

	// Empty file exists with size 0.
	if info, err := os.Stat(filepath.Join(res.Path, "dir", "empty.txt")); err != nil {
		t.Fatalf("empty file missing: %v", err)
	} else if info.Size() != 0 {
		t.Fatalf("empty file size = %d, want 0", info.Size())
	}
	// Symlink is still a symlink.
	if info, err := os.Lstat(filepath.Join(res.Path, "dir", "link")); err != nil {
		t.Fatalf("symlink missing: %v", err)
	} else if info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("dir/link mode = %v, want symlink", info.Mode())
	}

	// CoW/copy isolation: mutate the worktree, source stays put.
	target := filepath.Join(res.Path, "dir", "alpha.txt")
	if err := os.WriteFile(target, []byte("mutated\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(repo, "dir", "alpha.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "alpha\n" {
		t.Fatalf("source mutated to %q, want %q", got, "alpha\n")
	}
}

// TestTreeCloneFastPathDirtyTreeDeclines witnesses the core correctness gate:
// with an uncommitted modification to a tracked file the fast path must NOT be
// taken, and the materialized content must be the COMMITTED base, not the dirty
// working-tree bytes.
func TestTreeCloneFastPathDirtyTreeDeclines(t *testing.T) {
	repo, base := treeCloneTestFixture(t)

	// Dirty a tracked file in the source working tree (uncommitted).
	if err := os.WriteFile(filepath.Join(repo, "dir", "alpha.txt"), []byte("dirty-working-tree\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-dirty", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 for dirty tree (res=%+v)", delta, res)
	}
	if res.Backend != blockCloneBackendName {
		t.Fatalf("backend = %q, want %q (per-file loop is still block-clone)", res.Backend, blockCloneBackendName)
	}
	got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "alpha\n" {
		t.Fatalf("materialized content = %q, want committed base %q (dirty working-tree leaked)", got, "alpha\n")
	}
}

// TestTreeCloneFastPathUntrackedTreeDeclines witnesses that an untracked file
// also declines the fast path and never leaks into the materialized worktree.
func TestTreeCloneFastPathUntrackedTreeDeclines(t *testing.T) {
	repo, base := treeCloneTestFixture(t)
	if err := os.WriteFile(filepath.Join(repo, "untracked-leak.txt"), []byte("secret\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-untracked", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 for untracked file (res=%+v)", delta, res)
	}
	if _, err := os.Lstat(filepath.Join(res.Path, "untracked-leak.txt")); !os.IsNotExist(err) {
		t.Fatalf("untracked file leaked into worktree (stat err=%v)", err)
	}
}

// TestTreeCloneFastPathErrorFallsBack witnesses fail-open: a treeClone error on
// a clean tree must decline the fast path, not fail the materialization.
func TestTreeCloneFastPathErrorFallsBack(t *testing.T) {
	repo, base := treeCloneTestFixture(t)
	backend := blockClone{
		probe:     func(string) error { return nil },
		clone:     copyFileForBlockCloneTest,
		treeClone: func(src, dst string) error { return errors.New("boom") },
	}

	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-failopen", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare after treeClone error should still succeed: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 after treeClone error (res=%+v)", delta, res)
	}
	if res.Backend != blockCloneBackendName {
		t.Fatalf("backend = %q, want %q", res.Backend, blockCloneBackendName)
	}
	// Content still correct: the fallback per-file loop filled the tree.
	if got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "alpha\n" {
		t.Fatalf("content = %q, want %q", got, "alpha\n")
	}
	if got, err := os.ReadFile(filepath.Join(res.Path, "top.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "top\n" {
		t.Fatalf("content = %q, want %q", got, "top\n")
	}
}

// TestTreeCloneProductionBackendEndToEnd exercises the production backend on the
// real host. If the volume has no CoW probe support it skips (non-APFS CI).
func TestTreeCloneProductionBackendEndToEnd(t *testing.T) {
	repo, base := treeCloneTestFixture(t)
	targetRoot := t.TempDir()
	if err := probeBlockClone(targetRoot); err != nil {
		t.Skipf("skipping on non-block-clone volume: %v", err)
	}

	backend := newBlockCloneBackend()
	before := TreeCloneFastPathCount()
	start := time.Now()
	res := PrepareWithBackend(repo, "workerworktree", "tree-prod", base, targetRoot, nil, backend)
	elapsed := time.Since(start)
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if res.Backend != blockCloneBackendName {
		t.Fatalf("backend = %q, want %q", res.Backend, blockCloneBackendName)
	}
	if got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "alpha\n" {
		t.Fatalf("content = %q, want %q", got, "alpha\n")
	}
	if delta := TreeCloneFastPathCount() - before; delta != 1 {
		t.Fatalf("fast-path counter delta = %d, want 1 on a clean supported-volume fixture (res=%+v)", delta, res)
	}
	t.Logf("production block-clone materialization took %v (fast-path delta=%d)", elapsed, TreeCloneFastPathCount()-before)

	reaped := ReapWithBackend(repo, res.Path, nil, backend)
	if !reaped.OK || !reaped.Removed {
		t.Fatalf("reap did not remove worktree: %+v", reaped)
	}
	if _, err := os.Stat(res.Path); !os.IsNotExist(err) {
		t.Fatalf("worktree still present after reap (stat err=%v)", err)
	}
}

// TestTreeCloneFastPathDeclinesOnIgnoredFileInTrackedTree is a regression for
// the OLD gate's unsoundness: a gitignored file inside a tracked top-level
// directory leaves `git status --porcelain --untracked-files=all` EMPTY, yet the
// whole-tree clone would copy the ignored file verbatim into the worktree. The
// hardened gate adds `--ignored` and must decline.
func TestTreeCloneFastPathDeclinesOnIgnoredFileInTrackedTree(t *testing.T) {
	repo, base := treeCloneTestFixture(t)

	// Ignore dir/secret-ignored.txt via a committed .gitignore, then create the
	// ignored file: untracked-and-ignored, invisible to plain status.
	if err := os.WriteFile(filepath.Join(repo, ".gitignore"), []byte("dir/secret-ignored.txt\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runBlockCloneGitTest(t, repo, "add", ".gitignore")
	runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "ignore secret")
	if err := os.WriteFile(filepath.Join(repo, "dir", "secret-ignored.txt"), []byte("SECRET-IGNORED\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Sanity: plain status (no --ignored) is empty, which the OLD gate mistook
	// for "clean enough to clone verbatim".
	status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
	if status != "" {
		t.Fatalf("plain status not empty: %q", status)
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-ignored", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 for ignored file in tracked tree (res=%+v)", delta, res)
	}
	if _, err := os.Lstat(filepath.Join(res.Path, "dir", "secret-ignored.txt")); !os.IsNotExist(err) {
		t.Fatalf("ignored file leaked into materialized worktree (stat err=%v)", err)
	}
	// Tracked content still correct.
	if got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "alpha\n" {
		t.Fatalf("content = %q, want %q", got, "alpha\n")
	}
}

// TestTreeCloneFastPathDeclinesOnContentFilter is the exact counterexample to a
// status-only gate: with a registered `filter` attribute and a no-op
// clean/smudge driver, `git status --porcelain` reads CLEAN while the working
// bytes may differ from the committed blob. The hardened gate's check-attr
// probe must decline.
func TestTreeCloneFastPathDeclinesOnContentFilter(t *testing.T) {
	repo, base := treeCloneTestFixture(t)

	if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt filter=mangle\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Register a trivial passthrough filter so the attribute is active (not
	// treated as unspecified); the driver itself does not transform bytes.
	runBlockCloneGitTest(t, repo, "config", "filter.mangle.clean", "cat")
	runBlockCloneGitTest(t, repo, "config", "filter.mangle.smudge", "cat")
	runBlockCloneGitTest(t, repo, "add", ".gitattributes")
	runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "add filter attr")
	// Re-clean the filtered file so the index/worktree agree -> status clean.
	// The passthrough filter is byte-identical, so force a re-stat/re-clean
	// via `git add --renormalize` (a second content commit would be empty).
	runBlockCloneGitTest(t, repo, "add", "--renormalize", "dir/alpha.txt")

	status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
	if status != "" {
		t.Fatalf("plain status not empty: %q", status)
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-filter", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 despite clean status (res=%+v)", delta, res)
	}
	if got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt")); err != nil {
		t.Fatal(err)
	} else if string(got) != "alpha\n" {
		t.Fatalf("materialized content = %q, want committed base %q", got, "alpha\n")
	}
}

// TestTreeCloneFastPathDeclinesOnAssumeUnchangedDirt is the third status-only
// counterexample: `git update-index --assume-unchanged` hides an uncommitted
// modification from `git status`, so the OLD gate would clone the dirty bytes.
// The hardened gate's `git ls-files -v` probe must decline.
func TestTreeCloneFastPathDeclinesOnAssumeUnchangedDirt(t *testing.T) {
	repo, base := treeCloneTestFixture(t)

	runBlockCloneGitTest(t, repo, "update-index", "--assume-unchanged", "dir/alpha.txt")
	t.Cleanup(func() {
		exec.Command("git", "-C", repo, "update-index", "--no-assume-unchanged", "dir/alpha.txt").Run()
	})
	if err := os.WriteFile(filepath.Join(repo, "dir", "alpha.txt"), []byte("hidden-dirty\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Sanity: dirt is hidden from plain status.
	status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
	if status != "" {
		t.Fatalf("plain status not empty despite hidden dirt: %q", status)
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-assume", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 for assume-unchanged dirt (res=%+v)", delta, res)
	}
	got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "alpha\n" {
		t.Fatalf("materialized content = %q, want committed base %q (hidden dirt leaked)", got, "alpha\n")
	}
}

// TestTreeCloneCleanGateCounterexamples documents the treeCloneClean contract at
// the unit level: clean fixture -> true; each of the three status-invisible
// hazards -> false.
func TestTreeCloneCleanGateCounterexamples(t *testing.T) {
	t.Run("clean", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if !treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(clean fixture) = false, want true")
		}
	})
	t.Run("ignored", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if err := os.WriteFile(filepath.Join(repo, ".gitignore"), []byte("dir/secret-ignored.txt\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runBlockCloneGitTest(t, repo, "add", ".gitignore")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "ignore")
		if err := os.WriteFile(filepath.Join(repo, "dir", "secret-ignored.txt"), []byte("SECRET-IGNORED\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(ignored file) = true, want false")
		}
	})
	t.Run("assume_unchanged", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		runBlockCloneGitTest(t, repo, "update-index", "--assume-unchanged", "dir/alpha.txt")
		t.Cleanup(func() {
			exec.Command("git", "-C", repo, "update-index", "--no-assume-unchanged", "dir/alpha.txt").Run()
		})
		if err := os.WriteFile(filepath.Join(repo, "dir", "alpha.txt"), []byte("hidden-dirty\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(assume-unchanged dirt) = true, want false")
		}
	})
	t.Run("filter_attr", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt filter=mangle\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runBlockCloneGitTest(t, repo, "config", "filter.mangle.clean", "cat")
		runBlockCloneGitTest(t, repo, "config", "filter.mangle.smudge", "cat")
		runBlockCloneGitTest(t, repo, "add", ".gitattributes")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "attr")
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(filter attribute) = true, want false")
		}
	})
	t.Run("eol_attr_crlf", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt eol=crlf\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runBlockCloneGitTest(t, repo, "add", ".gitattributes")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "eol attr")
		// Renormalize so the index/worktree agree and plain status is clean.
		runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite eol=crlf divergence: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(eol=crlf attribute) = true, want false")
		}
	})
	t.Run("text_attr_auto", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt text=auto\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runBlockCloneGitTest(t, repo, "add", ".gitattributes")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "text attr")
		runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite text=auto divergence: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(text=auto attribute) = true, want false")
		}
	})
	t.Run("working_tree_encoding_attr", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt working-tree-encoding=UTF-16LE\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runBlockCloneGitTest(t, repo, "add", ".gitattributes")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "wte attr")
		// Committing the renormalized index is what makes plain status clean:
		// git stores the UTF-16LE-encoded blob while re-checking the worktree
		// out to its decoded (LF) form, so status reads clean yet the worktree
		// bytes differ from the committed blob.
		runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")
		runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "wte renormalize")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite working-tree-encoding divergence: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(working-tree-encoding attribute) = true, want false")
		}
	})
	t.Run("core_autocrlf_true", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		runBlockCloneGitTest(t, repo, "config", "core.autocrlf", "true")
		runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite core.autocrlf=true divergence: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(core.autocrlf=true) = true, want false")
		}
	})
	t.Run("core_autocrlf_input", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		runBlockCloneGitTest(t, repo, "config", "core.autocrlf", "input")
		runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite core.autocrlf=input divergence: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(core.autocrlf=input) = true, want false")
		}
	})
	t.Run("core_eol_lf", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		runBlockCloneGitTest(t, repo, "config", "core.eol", "lf")
		status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
		if status != "" {
			t.Fatalf("plain status not empty despite core.eol=lf: %q", status)
		}
		if treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(core.eol=lf) = true, want false")
		}
	})
	t.Run("core_autocrlf_false", func(t *testing.T) {
		repo, _ := treeCloneTestFixture(t)
		runBlockCloneGitTest(t, repo, "config", "core.autocrlf", "false")
		// An explicit benign value must NOT false-decline the fast path.
		if !treeCloneClean(repo, nil) {
			t.Fatalf("treeCloneClean(core.autocrlf=false) = false, want true (explicit benign value must not decline)")
		}
	})
}

// TestTreeCloneFastPathDeclinesOnEolAttribute is the end-to-end counterpart to
// the unit gate case: `dir/alpha.txt eol=crlf` plus a CRLF working tree that
// renormalizes to a CLEAN status. The working bytes differ from the committed
// blob (LF), so cloning verbatim would materialize CRLF content that is not the
// base object. The hardened gate must decline (fast-path delta 0) and the
// per-file fallback must materialize the committed LF base bytes.
func TestTreeCloneFastPathDeclinesOnEolAttribute(t *testing.T) {
	repo, base := treeCloneTestFixture(t)

	// Attribute drives checkout of alpha.txt to CRLF.
	if err := os.WriteFile(filepath.Join(repo, ".gitattributes"), []byte("dir/alpha.txt eol=crlf\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Write the working file with CRLF content, then renormalize so index and
	// worktree agree -> plain status clean while worktree bytes != blob bytes.
	if err := os.WriteFile(filepath.Join(repo, "dir", "alpha.txt"), []byte("alpha\r\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runBlockCloneGitTest(t, repo, "add", ".gitattributes")
	runBlockCloneGitTest(t, repo, "commit", "-q", "-m", "eol attr")
	runBlockCloneGitTest(t, repo, "add", "--renormalize", ".")

	// Sanity: divergence hidden from plain status (no --ignored).
	status := strings.TrimSpace(runBlockCloneGitTest(t, repo, "status", "--porcelain=v1", "--untracked-files=all"))
	if status != "" {
		t.Fatalf("plain status not empty despite eol divergence: %q", status)
	}

	// The committed base blob for dir/alpha.txt is the original LF "alpha\n".
	_, blobOut := run(nil, repo, []string{"cat-file", "blob", base + ":dir/alpha.txt"})
	if blobOut != "alpha\n" {
		t.Fatalf("base blob = %q, want %q", blobOut, "alpha\n")
	}

	backend := treeCloneFastBackend()
	before := TreeCloneFastPathCount()
	res := PrepareWithBackend(repo, "workerworktree", "tree-eol", base, t.TempDir(), nil, backend)
	after := TreeCloneFastPathCount()
	if !res.OK {
		t.Fatalf("prepare: %+v", res)
	}
	t.Cleanup(func() { ReapWithBackend(repo, res.Path, nil, backend) })

	if delta := after - before; delta != 0 {
		t.Fatalf("fast-path counter delta = %d, want 0 despite clean status (res=%+v)", delta, res)
	}
	got, err := os.ReadFile(filepath.Join(res.Path, "dir", "alpha.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != blobOut {
		t.Fatalf("materialized content = %q, want committed base bytes %q (CRLF working bytes leaked)", got, blobOut)
	}
}
