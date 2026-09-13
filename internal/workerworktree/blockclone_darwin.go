//go:build darwin

package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// cloneNoFollow corresponds to CLONE_NOFOLLOW (0x0001) in <sys/clonefile.h>.
// It ensures symbolic links themselves are cloned rather than their targets.
const cloneNoFollow = 0x0001

func nativeIsolationBackend() IsolationBackend { return newBlockCloneBackend() }

func probeBlockClone(targetRoot string) error {
	if err := os.MkdirAll(targetRoot, 0o755); err != nil {
		return err
	}
	dir, err := os.MkdirTemp(targetRoot, ".fak-block-clone-probe-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "source")
	dst := filepath.Join(dir, "target")
	if err := os.WriteFile(src, make([]byte, 4096), 0o600); err != nil {
		return err
	}
	return cloneFileBlocks(src, dst)
}

func cloneFileBlocks(src, dst string) error {
	preExisting := false
	if _, statErr := os.Lstat(dst); statErr == nil {
		preExisting = true
	}
	if err := unix.Clonefile(src, dst, cloneNoFollow); err != nil {
		if !preExisting {
			_ = os.Remove(dst)
		}
		return fmt.Errorf("clonefile: %w", err)
	}
	return nil
}

// cloneTree clones the directory tree at src to dst.
//
// The fast path is a single clonefile(2) call, which clones the whole tree
// recursively on APFS in one syscall. If that fails (dst already exists, or
// the destination volume does not support cloning, e.g. exFAT), it falls back
// to a recursive walk that clones each regular file and recreates directories
// and symlinks. A per-file clone failure degrades to a plain byte copy. Only
// genuine I/O failures are returned; a non-clone-capable filesystem still
// succeeds via the byte-copy fallback.
//
// On error, a destination this call created is removed so callers are not left
// with a half-populated tree; a destination that already existed is left
// untouched so pre-existing caller data is never destroyed. Cloning into an
// existing tree therefore fails at a colliding regular file rather than
// replacing it.
func cloneTree(src, dst string) error {
	preExisting := false
	if _, err := os.Lstat(dst); err == nil {
		preExisting = true
	}
	if !preExisting {
		if err := unix.Clonefile(src, dst, cloneNoFollow); err == nil {
			return nil
		}
	}
	err := cloneTreeWalk(src, dst)
	if err != nil {
		if !preExisting {
			_ = os.RemoveAll(dst)
		}
		return err
	}
	return nil
}
