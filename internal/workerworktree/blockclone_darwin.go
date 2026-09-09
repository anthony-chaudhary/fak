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
	if err := unix.Clonefile(src, dst, cloneNoFollow); err != nil {
		_ = os.Remove(dst)
		return fmt.Errorf("clonefile: %w", err)
	}
	return nil
}
