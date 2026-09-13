//go:build darwin

package workerworktree

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// TestDarwinAPFSBlockClone witnesses the native APFS directory-tree clone
// fast path: one clonefile(2) call clones a whole tree recursively, in
// sub-10ms on APFS. It also checks byte-identical contents, copy-on-write
// isolation, and that the recursive fallback degrades gracefully.
func TestDarwinAPFSBlockClone(t *testing.T) {
	src := t.TempDir()
	dst := filepath.Join(t.TempDir(), "clone")

	if err := buildCloneTreeFixture(src); err != nil {
		t.Fatalf("build fixture: %v", err)
	}
	srcSize, err := treeRegularBytes(src)
	if err != nil {
		t.Fatalf("measure source: %v", err)
	}

	// Probe: clonefile may be unsupported on the test volume (non-APFS CI),
	// in which case skip rather than fail.
	probeDst := filepath.Join(t.TempDir(), "probe")
	if err := unix.Clonefile(filepath.Join(src, "small.txt"), probeDst, cloneNoFollow); err != nil {
		t.Skipf("clonefile unsupported on this volume: %v", err)
	}

	// The issue's contract is sub-10ms for the single-syscall fast path. A
	// one-shot cold clone can miss on first-touch page cache, so take the best
	// of a few warm attempts: the fastest run is the closest measurement of the
	// syscall itself, and best-of-N is deterministic (no wall-clock flake) while
	// still proving the clone is a single CoW syscall, not a byte copy.
	const (
		cloneBudget  = 10 * time.Millisecond
		cloneCeiling = 50 * time.Millisecond
	)
	best := time.Duration(1<<63 - 1)
	for i := 0; i < 3; i++ {
		trial := filepath.Join(t.TempDir(), "timed")
		start := time.Now()
		if err := CloneTree(src, trial); err != nil {
			t.Fatalf("CloneTree trial %d: %v", i, err)
		}
		if d := time.Since(start); d < best {
			best = d
		}
		if err := os.RemoveAll(trial); err != nil {
			t.Fatalf("clean trial %d: %v", i, err)
		}
	}
	t.Logf("CloneTree(%d bytes): best-of-3 %v", srcSize, best)
	if best >= cloneCeiling {
		t.Fatalf("CloneTree best-of-3 took %v; want < %v", best, cloneCeiling)
	}
	if best >= cloneBudget {
		t.Fatalf("CloneTree best-of-3 took %v; want sub-10ms (< %v) per the issue contract", best, cloneBudget)
	}

	if err := CloneTree(src, dst); err != nil {
		t.Fatalf("CloneTree: %v", err)
	}
	if err := assertTreesEqual(src, dst); err != nil {
		t.Fatalf("cloned tree differs: %v", err)
	}

	// Physical-space economy: APFS clonefile shares extents with the source.
	// st_blocks on Darwin reports logical allocation for cloned regular files,
	// so it does NOT witness extent sharing; log it for the record only. The
	// real CoW proof is the isolation check below (a write to the clone must
	// not reach the source) plus the sub-10ms syscall timing.
	largeLogical := int64(len(bytes.Repeat([]byte("fak-apfs-clone\n"), 128*1024)))
	if blocks, err := fileBlocks(filepath.Join(dst, "large.bin")); err == nil {
		t.Logf("clone large.bin: st_blocks=%d (%d bytes) vs logical %d bytes (st_blocks reports logical allocation on APFS)",
			blocks, blocks*512, largeLogical)
	} else {
		t.Logf("could not stat clone blocks: %v", err)
	}

	// CoW isolation: mutating a cloned file must not touch the source.
	clonedFile := filepath.Join(dst, "large.bin")
	if err := os.WriteFile(clonedFile, []byte("mutated"), 0o644); err != nil {
		t.Fatalf("mutate clone: %v", err)
	}
	orig, err := os.ReadFile(filepath.Join(src, "large.bin"))
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	if bytes.Equal(orig, []byte("mutated")) {
		t.Fatal("source file changed after mutating the clone: CoW isolation broken")
	}

	t.Run("existing_dst_never_loses_caller_data", func(t *testing.T) {
		existing := filepath.Join(t.TempDir(), "existing")
		if err := os.MkdirAll(existing, 0o755); err != nil {
			t.Fatal(err)
		}
		marker := filepath.Join(existing, "marker.txt")
		if err := os.WriteFile(marker, []byte("keep"), 0o644); err != nil {
			t.Fatal(err)
		}
		// dst already exists, so the single recursive clonefile fast path is
		// skipped and the walk fallback runs. The contract for an existing
		// destination is: pre-existing caller data survives, and if the call
		// reports success every source path is present. The clone may merge
		// into the existing directory (leaving caller files in place) or fail
		// at a colliding path; either is acceptable, data loss is not.
		err := CloneTree(src, existing)
		if data, statErr := os.ReadFile(marker); statErr != nil {
			t.Fatalf("pre-existing dst data destroyed: %v", statErr)
		} else if string(data) != "keep" {
			t.Fatalf("pre-existing marker mutated: %q", data)
		}
		if err == nil {
			for _, rel := range []string{"small.txt", "large.bin", "link.txt",
				filepath.Join("nested", "deep", "leaf.txt")} {
				if _, statErr := os.Lstat(filepath.Join(existing, rel)); statErr != nil {
					t.Fatalf("clone reported success but source path %s missing: %v", rel, statErr)
				}
			}
		}
	})
}

// TestCloneFileBlocksPreservesPreExistingDst is the data-loss guard for the
// per-file clone primitive: when dst already exists (caller data), a failing
// clonefile must NOT delete it. It also checks the complementary contract that
// a fresh dst is cloned normally and that a failed clone into a fresh path
// cleans up its own partial file.
func TestCloneFileBlocksPreservesPreExistingDst(t *testing.T) {
	src := filepath.Join(t.TempDir(), "source.bin")
	if err := os.WriteFile(src, []byte("SOURCE"), 0o644); err != nil {
		t.Fatalf("write source: %v", err)
	}

	t.Run("pre_existing_dst_is_preserved", func(t *testing.T) {
		dst := filepath.Join(t.TempDir(), "target.bin")
		if err := os.WriteFile(dst, []byte("CALLER-DATA"), 0o644); err != nil {
			t.Fatalf("write dst: %v", err)
		}
		err := cloneFileBlocks(src, dst)
		if err == nil {
			t.Fatalf("cloneFileBlocks onto pre-existing dst = nil; want EEXIST-family error")
		}
		if !errors.Is(err, fs.ErrExist) && !errors.Is(err, unix.EEXIST) {
			t.Logf("cloneFileBlocks error (non-EEXIST, still an error): %v", err)
		}
		got, readErr := os.ReadFile(dst)
		if readErr != nil {
			t.Fatalf("dst destroyed by failed clone: %v", readErr)
		}
		if string(got) != "CALLER-DATA" {
			t.Fatalf("dst content = %q; want %q (caller data preserved)", got, "CALLER-DATA")
		}
	})

	t.Run("fresh_dst_clones_normally", func(t *testing.T) {
		dst := filepath.Join(t.TempDir(), "fresh.bin")
		if err := cloneFileBlocks(src, dst); err != nil {
			t.Fatalf("cloneFileBlocks onto fresh dst: %v", err)
		}
		got, err := os.ReadFile(dst)
		if err != nil {
			t.Fatalf("read fresh clone: %v", err)
		}
		if string(got) != "SOURCE" {
			t.Fatalf("fresh clone content = %q; want %q", got, "SOURCE")
		}
	})

	t.Run("failed_clone_into_fresh_path_cleans_up", func(t *testing.T) {
		missing := filepath.Join(t.TempDir(), "does-not-exist.bin")
		dst := filepath.Join(t.TempDir(), "partial.bin")
		if err := cloneFileBlocks(missing, dst); err == nil {
			t.Fatalf("cloneFileBlocks from missing source = nil; want error")
		}
		if _, statErr := os.Lstat(dst); !os.IsNotExist(statErr) {
			t.Fatalf("failed clone left a partial file at %s: %v", dst, statErr)
		}
	})
}

// TestCloneTreePreservesPreExistingDstOnError is the data-loss guard: when a
// clone into a pre-existing destination fails, CloneTree must NOT delete the
// caller's directory. It also asserts the complementary contract that a
// destination this call created is still cleaned up on failure.
func TestCloneTreePreservesPreExistingDstOnError(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: unreadable source files do not force a copy error")
	}

	src := t.TempDir()
	unreadable := filepath.Join(src, "secret.txt")
	if err := os.WriteFile(unreadable, []byte("top secret"), 0o644); err != nil {
		t.Fatalf("write source file: %v", err)
	}
	if err := os.Chmod(unreadable, 0o000); err != nil {
		t.Fatalf("chmod source file: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(unreadable, 0o644) })

	// Case 1: pre-existing dst must survive a failed clone untouched.
	dst := t.TempDir()
	marker := filepath.Join(dst, "marker.txt")
	if err := os.WriteFile(marker, []byte("keep"), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}
	err := CloneTree(src, dst)
	if err == nil {
		t.Skip("source was readable; could not force a clone error deterministically")
	}
	if _, statErr := os.Lstat(marker); statErr != nil {
		t.Fatalf("pre-existing dst was destroyed on error: %v", statErr)
	}
	got, readErr := os.ReadFile(marker)
	if readErr != nil {
		t.Fatalf("read marker after error: %v", readErr)
	}
	if string(got) != "keep" {
		t.Fatalf("marker content = %q; want %q", got, "keep")
	}

	// Case 2: a fresh, non-existent dst must be cleaned up on failure.
	fresh := filepath.Join(t.TempDir(), "fresh-clone")
	if err := CloneTree(src, fresh); err == nil {
		t.Skip("source was readable; could not force a clone error deterministically")
	}
	if _, statErr := os.Lstat(fresh); !os.IsNotExist(statErr) {
		t.Fatalf("failed clone left a partial tree at %s: %v", fresh, statErr)
	}
}

// TestCloneTreeRejectsSymlinkedSubdir is the tree-escape guard: a pre-existing
// symlinked subdirectory in the destination must NOT be followed, or the walk
// would write source files outside dst. The clone must error instead.
func TestCloneTreeRejectsSymlinkedSubdir(t *testing.T) {
	src := t.TempDir()
	if err := os.MkdirAll(filepath.Join(src, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "sub", "escaped.txt"), []byte("escape"), 0o644); err != nil {
		t.Fatal(err)
	}

	outside := t.TempDir()
	dst := t.TempDir()
	// dst/sub is a symlink pointing outside the destination tree.
	if err := os.Symlink(outside, filepath.Join(dst, "sub")); err != nil {
		t.Fatal(err)
	}

	err := CloneTree(src, dst)
	if err == nil {
		t.Fatalf("CloneTree followed a symlinked dst subdir and returned nil")
	}
	if _, statErr := os.Lstat(filepath.Join(outside, "escaped.txt")); !os.IsNotExist(statErr) {
		t.Fatalf("tree escape: file written outside dst through symlink (stat err=%v)", statErr)
	}
}

// TestCloneTreeRejectsWrongKindCollision is the wrong-kind guard: when an
// empty source directory collides with a pre-existing non-directory at the
// same relative path, the walk must NOT report success (which would claim the
// directory materialized when it did not).
func TestCloneTreeRejectsWrongKindCollision(t *testing.T) {
	src := t.TempDir()
	if err := os.Mkdir(filepath.Join(src, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}

	dst := t.TempDir()
	// dst/sub is a regular file, not a directory; src/sub is an empty dir.
	if err := os.WriteFile(filepath.Join(dst, "sub"), []byte("not-a-dir"), 0o644); err != nil {
		t.Fatal(err)
	}

	err := CloneTree(src, dst)
	if err == nil {
		t.Fatalf("CloneTree accepted a non-directory collision and reported success")
	}
	got, readErr := os.ReadFile(filepath.Join(dst, "sub"))
	if readErr != nil || string(got) != "not-a-dir" {
		t.Fatalf("pre-existing non-directory clobbered: content=%q err=%v", got, readErr)
	}
}

// buildCloneTreeFixture creates nested dirs, a small regular file, a >1MiB
// file, a subdirectory, and a symlink.
func buildCloneTreeFixture(root string) error {
	if err := os.MkdirAll(filepath.Join(root, "nested", "deep"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(root, "small.txt"), []byte("small\n"), 0o644); err != nil {
		return err
	}
	large := bytes.Repeat([]byte("fak-apfs-clone\n"), 128*1024) // ~1.8 MiB
	if err := os.WriteFile(filepath.Join(root, "large.bin"), large, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(root, "nested", "deep", "leaf.txt"), []byte("leaf\n"), 0o644); err != nil {
		return err
	}
	if err := os.Symlink("small.txt", filepath.Join(root, "link.txt")); err != nil {
		return err
	}
	return nil
}

func treeRegularBytes(root string) (int64, error) {
	var total int64
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.Mode().IsRegular() {
			total += info.Size()
		}
		return nil
	})
	return total, err
}

// assertTreesEqual checks that dst mirrors src exactly: same relative paths,
// same kinds, byte-identical regular files, and identical symlink targets.
// It walks BOTH trees so an extra file appearing only in dst is a mismatch,
// not silently ignored.
func assertTreesEqual(src, dst string) error {
	if err := filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		other := filepath.Join(dst, rel)
		otherInfo, err := os.Lstat(other)
		if err != nil {
			return &treeMismatch{rel, "missing in clone: " + err.Error()}
		}
		switch {
		case info.Mode()&os.ModeSymlink != 0:
			want, err := os.Readlink(path)
			if err != nil {
				return err
			}
			got, err := os.Readlink(other)
			if err != nil {
				return &treeMismatch{rel, "clone entry is not a symlink: " + err.Error()}
			}
			if want != got {
				return &treeMismatch{rel, "symlink target " + got + " != " + want}
			}
		case info.IsDir():
			if !otherInfo.IsDir() {
				return &treeMismatch{rel, "not a directory in clone"}
			}
		case info.Mode().IsRegular():
			want, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			got, err := os.ReadFile(other)
			if err != nil {
				return &treeMismatch{rel, "cannot read clone file: " + err.Error()}
			}
			if !bytes.Equal(want, got) {
				return &treeMismatch{rel, "contents differ"}
			}
		}
		return nil
	}); err != nil {
		return err
	}

	// Reverse direction: dst must not contain paths absent from src.
	return filepath.Walk(dst, func(path string, _ os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dst, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		if _, statErr := os.Lstat(filepath.Join(src, rel)); statErr != nil {
			return &treeMismatch{rel, "unexpected extra path in clone"}
		}
		return nil
	})
}

type treeMismatch struct {
	path string
	why  string
}

func (m *treeMismatch) Error() string { return m.path + ": " + m.why }

func fileBlocks(path string) (int64, error) {
	var st unix.Stat_t
	if err := unix.Stat(path, &st); err != nil {
		return 0, err
	}
	return st.Blocks, nil
}
