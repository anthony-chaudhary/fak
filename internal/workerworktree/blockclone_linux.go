//go:build linux

package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

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
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	info, err := srcFile.Stat()
	if err != nil {
		return err
	}

	dstFile, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, info.Mode().Perm())
	if err != nil {
		return err
	}

	cloneErr := unix.IoctlFileClone(int(dstFile.Fd()), int(srcFile.Fd()))
	closeErr := dstFile.Close()
	if cloneErr != nil {
		_ = os.Remove(dst)
		return fmt.Errorf("FICLONE: %w", cloneErr)
	}
	if closeErr != nil {
		_ = os.Remove(dst)
		return closeErr
	}
	return nil
}

// cloneTree clones the directory tree at src to dst. Linux has no single
// recursive FICLONE for directories, so it walks the tree and clones each
// regular file, degrading to a byte copy when cloning is unsupported (e.g.
// across filesystems). On error it removes a destination this call created; a
// destination that already existed is left untouched so pre-existing caller
// data is never destroyed.
func cloneTree(src, dst string) error {
	preExisting := false
	if _, err := os.Lstat(dst); err == nil {
		preExisting = true
	}
	if err := cloneTreeWalk(src, dst); err != nil {
		if !preExisting {
			_ = os.RemoveAll(dst)
		}
		return err
	}
	return nil
}
