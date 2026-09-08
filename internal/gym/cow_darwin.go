//go:build darwin

package gym

import (
	"bytes"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	// SYS_clonefile is the Darwin clonefile(2) syscall trap number.
	SYS_clonefile = 427

	// CLONE_NOFOLLOW prevents following symbolic links; clones the link itself.
	CLONE_NOFOLLOW = 0x0001
	// CLONE_NOOWNERCOPY skips copying file owner UID/GID from source.
	CLONE_NOOWNERCOPY = 0x0002
	// CLONE_ACL copies access control lists from source.
	CLONE_ACL = 0x0004
)

// clonefile invokes Darwin clonefile(2) to create a copy-on-write extent clone.
func clonefile(src, dst string, flags uint32) error {
	err := unix.Clonefile(src, dst, int(flags))
	if err != nil && errors.Is(err, syscall.ENOSYS) {
		srcPtr, e1 := syscall.BytePtrFromString(src)
		if e1 != nil {
			return e1
		}
		dstPtr, e2 := syscall.BytePtrFromString(dst)
		if e2 != nil {
			return e2
		}
		_, _, errno := syscall.Syscall6(SYS_clonefile, uintptr(unsafe.Pointer(srcPtr)), uintptr(unsafe.Pointer(dstPtr)), uintptr(flags), 0, 0, 0)
		if errno != 0 {
			return errno
		}
		return nil
	}
	return err
}

// isAPFSCloneAvailable probes whether both paths reside on the same APFS volume and support clonefile(2).
func isAPFSCloneAvailable(srcPath, dstPath string) bool {
	absSrc, err := filepath.Abs(srcPath)
	if err != nil {
		return false
	}
	absDst, err := filepath.Abs(dstPath)
	if err != nil {
		return false
	}

	var srcStat, dstStat unix.Statfs_t
	if err := unix.Statfs(absSrc, &srcStat); err != nil {
		return false
	}

	probeDst := absDst
	if _, err := os.Stat(probeDst); os.IsNotExist(err) {
		probeDst = filepath.Dir(probeDst)
	}
	if err := unix.Statfs(probeDst, &dstStat); err != nil {
		return false
	}

	srcType := unix.ByteSliceToString(srcStat.Fstypename[:])
	dstType := unix.ByteSliceToString(dstStat.Fstypename[:])
	if !strings.EqualFold(srcType, "apfs") || !strings.EqualFold(dstType, "apfs") {
		return false
	}

	// APFS block cloning requires source and destination to reside on the same filesystem volume.
	if srcStat.Fsid != dstStat.Fsid {
		return false
	}

	// Active canary probe in destination directory
	targetDir := absDst
	if fi, err := os.Stat(targetDir); err != nil || !fi.IsDir() {
		targetDir = filepath.Dir(absDst)
	}
	canaryFile, err := os.CreateTemp(targetDir, ".canary_apfs_probe_*")
	if err != nil {
		return false
	}
	canarySrc := canaryFile.Name()
	_, _ = canaryFile.WriteString("apfs-probe")
	_ = canaryFile.Close()
	defer os.Remove(canarySrc)

	canaryDst := canarySrc + ".clone"
	defer os.Remove(canaryDst)

	if err := clonefile(canarySrc, canaryDst, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); err != nil {
		return false
	}

	return true
}

// cloneDirTree recursively clones all files and directories from src to dst using APFS clonefile(2).
func cloneDirTree(src, dst string) error {
	// Fast path: if dst does not exist, clone the entire directory tree via clonefile(2) in a single syscall
	if _, err := os.Lstat(dst); os.IsNotExist(err) {
		if err := clonefile(src, dst, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); err == nil {
			return nil
		}
	}

	// Recursive walk: clone directories and files
	return filepath.Walk(src, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(src, path)
		if err != nil || rel == "." {
			return nil
		}

		targetPath := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(targetPath, info.Mode())
		}

		// Skip non-standard device nodes, sockets, and named pipes
		if info.Mode()&(os.ModeDevice|os.ModeNamedPipe|os.ModeSocket) != 0 {
			return nil
		}

		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return err
		}

		if info.Mode()&os.ModeSymlink != 0 {
			if err := clonefile(path, targetPath, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); err == nil {
				return nil
			}
			linkTarget, err := os.Readlink(path)
			if err != nil {
				return err
			}
			return os.Symlink(linkTarget, targetPath)
		}

		// Regular file
		if err := clonefile(path, targetPath, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); err != nil {
			return copyFile(path, targetPath)
		}
		return nil
	})
}

// darwinAPFSOverlay implements CoWOverlay using Darwin's native APFS copy-on-write clonefile engine.
type darwinAPFSOverlay struct {
	mu            sync.Mutex
	lowerDir      string
	upperDir      string
	mergedDir     string
	tempBase      string
	trashSeq      uint64
	standbyMerged string
	standbyUpper  string
	wg            sync.WaitGroup
	destroyed     bool
}

// newDarwinAPFSOverlay initializes a new darwinAPFSOverlay over lowerDir.
func newDarwinAPFSOverlay(lowerDir, tempBase string) (*darwinAPFSOverlay, error) {
	if strings.TrimSpace(lowerDir) == "" {
		return nil, errors.New("lower directory path is required")
	}
	absLower, err := filepath.Abs(lowerDir)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve lower directory path: %w", err)
	}
	if info, err := os.Stat(absLower); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("lower directory does not exist or is not a directory: %s", absLower)
	}

	createdTempBase := false
	if tempBase == "" {
		tmpRoot := defaultTempDir()
		base, err := os.MkdirTemp(tmpRoot, "fak-gym-apfs-*")
		if err != nil {
			return nil, fmt.Errorf("failed to create temporary base directory: %w", err)
		}
		tempBase = base
		createdTempBase = true
	}

	if !isAPFSCloneAvailable(absLower, tempBase) {
		if createdTempBase {
			_ = os.RemoveAll(tempBase)
		}
		return nil, fmt.Errorf("APFS clonefile not supported between %s and %s", absLower, tempBase)
	}

	upperDir := filepath.Join(tempBase, "upper")
	mergedDir := filepath.Join(tempBase, "merged")

	if err := os.MkdirAll(upperDir, 0755); err != nil {
		if createdTempBase {
			_ = os.RemoveAll(tempBase)
		}
		return nil, fmt.Errorf("failed to create upper directory: %w", err)
	}

	// Populate initial merged directory via APFS clone
	if err := cloneDirTree(absLower, mergedDir); err != nil {
		if createdTempBase {
			_ = os.RemoveAll(tempBase)
		}
		return nil, fmt.Errorf("failed to clone initial merged directory: %w", err)
	}

	overlay := &darwinAPFSOverlay{
		lowerDir:  absLower,
		upperDir:  upperDir,
		mergedDir: mergedDir,
		tempBase:  tempBase,
	}

	// Pre-seed a standby generation for instantaneous sub-millisecond Reset()
	seq := atomic.AddUint64(&overlay.trashSeq, 1)
	standbyMerged := filepath.Join(tempBase, fmt.Sprintf("merged_standby_%d", seq))
	standbyUpper := filepath.Join(tempBase, fmt.Sprintf("upper_standby_%d", seq))
	if err := os.MkdirAll(standbyUpper, 0755); err == nil {
		if err := cloneDirTree(absLower, standbyMerged); err == nil {
			overlay.standbyMerged = standbyMerged
			overlay.standbyUpper = standbyUpper
		} else {
			_ = os.RemoveAll(standbyUpper)
		}
	}

	return overlay, nil
}

// LowerDir returns the read-only host trunk / base workspace directory path.
func (o *darwinAPFSOverlay) LowerDir() string {
	return o.lowerDir
}

// UpperDir returns the ephemeral writable directory path.
func (o *darwinAPFSOverlay) UpperDir() string {
	return o.upperDir
}

// MergedDir returns the unified overlay directory path.
func (o *darwinAPFSOverlay) MergedDir() string {
	return o.mergedDir
}

// Reset instantaneously wipes mutations in <5ms, restoring pristine lower state.
func (o *darwinAPFSOverlay) Reset() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	oldUpper := o.upperDir
	oldMerged := o.mergedDir

	var newUpper, newMerged string
	if o.standbyMerged != "" && o.standbyUpper != "" {
		newUpper = o.standbyUpper
		newMerged = o.standbyMerged
		o.standbyMerged = ""
		o.standbyUpper = ""
	} else {
		seq := atomic.AddUint64(&o.trashSeq, 1)
		newUpper = filepath.Join(o.tempBase, fmt.Sprintf("upper_%d", seq))
		newMerged = filepath.Join(o.tempBase, fmt.Sprintf("merged_%d", seq))
		if err := os.MkdirAll(newUpper, 0755); err != nil {
			return fmt.Errorf("failed to create reset upper directory: %w", err)
		}
		if err := cloneDirTree(o.lowerDir, newMerged); err != nil {
			return fmt.Errorf("failed to restore merged state from lower directory: %w", err)
		}
	}

	o.upperDir = newUpper
	o.mergedDir = newMerged

	o.wg.Add(1)
	go func(u, m string) {
		defer o.wg.Done()
		_ = os.RemoveAll(u)
		_ = os.RemoveAll(m)

		// Replenish standby clone in background if idle
		o.mu.Lock()
		if o.destroyed || o.standbyMerged != "" {
			o.mu.Unlock()
			return
		}
		seq := atomic.AddUint64(&o.trashSeq, 1)
		nextMerged := filepath.Join(o.tempBase, fmt.Sprintf("merged_standby_%d", seq))
		nextUpper := filepath.Join(o.tempBase, fmt.Sprintf("upper_standby_%d", seq))
		lower := o.lowerDir
		o.mu.Unlock()

		if err := os.MkdirAll(nextUpper, 0755); err == nil {
			if err := cloneDirTree(lower, nextMerged); err == nil {
				o.mu.Lock()
				if !o.destroyed && o.standbyMerged == "" {
					o.standbyMerged = nextMerged
					o.standbyUpper = nextUpper
				} else {
					_ = os.RemoveAll(nextUpper)
					_ = os.RemoveAll(nextMerged)
				}
				o.mu.Unlock()
			} else {
				_ = os.RemoveAll(nextUpper)
				_ = os.RemoveAll(nextMerged)
			}
		}
	}(oldUpper, oldMerged)

	return nil
}

// Reconcile scans changes between mergedDir and lowerDir, staging mutations and whiteouts in upperDir.
func (o *darwinAPFSOverlay) Reconcile() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.reconcileLocked()
}

func (o *darwinAPFSOverlay) reconcileLocked() error {
	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	// 1. Detect creations and modifications in mergedDir relative to lowerDir
	err := filepath.Walk(o.mergedDir, func(mergedPath string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(o.mergedDir, mergedPath)
		if err != nil || rel == "." {
			return nil
		}

		lowerPath := filepath.Join(o.lowerDir, rel)
		upperPath := filepath.Join(o.upperDir, rel)

		if info.IsDir() {
			return os.MkdirAll(upperPath, info.Mode())
		}

		// Check if regular file exists in lower and whether it differs
		lowerInfo, err := os.Stat(lowerPath)
		isModified := false
		if err != nil {
			isModified = true
		} else if lowerInfo.IsDir() || lowerInfo.Size() != info.Size() {
			isModified = true
		} else {
			b1, err1 := os.ReadFile(mergedPath)
			b2, err2 := os.ReadFile(lowerPath)
			if err1 != nil || err2 != nil || !bytes.Equal(b1, b2) {
				isModified = true
			}
		}

		if isModified {
			if err := os.MkdirAll(filepath.Dir(upperPath), 0755); err != nil {
				return err
			}
			_ = os.Remove(upperPath)
			if err := clonefile(mergedPath, upperPath, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); err != nil {
				if err := copyFile(mergedPath, upperPath); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		return err
	}

	// 2. Detect deletions in lowerDir that are absent in mergedDir (place whiteouts in upperDir)
	err = filepath.Walk(o.lowerDir, func(lowerPath string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(o.lowerDir, lowerPath)
		if err != nil || rel == "." {
			return nil
		}

		mergedPath := filepath.Join(o.mergedDir, rel)
		if _, err := os.Stat(mergedPath); errors.Is(err, os.ErrNotExist) {
			whName := WhiteoutPrefix + filepath.Base(rel)
			whPath := filepath.Join(o.upperDir, filepath.Dir(rel), whName)
			if err := os.MkdirAll(filepath.Dir(whPath), 0755); err != nil {
				return err
			}
			if err := os.WriteFile(whPath, []byte{}, 0644); err != nil {
				return err
			}
		}
		return nil
	})

	return err
}

// ModifiedArtifacts returns a sorted list of relative paths modified, created, or deleted in the overlay.
func (o *darwinAPFSOverlay) ModifiedArtifacts() []string {
	o.mu.Lock()
	defer o.mu.Unlock()

	_ = o.reconcileLocked()

	var list []string
	_ = filepath.Walk(o.upperDir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		rel, err := filepath.Rel(o.upperDir, p)
		if err != nil || rel == "." {
			return nil
		}
		if info.IsDir() {
			return nil
		}
		base := filepath.Base(rel)
		if strings.HasPrefix(base, WhiteoutPrefix) {
			deletedTarget := filepath.Join(filepath.Dir(rel), strings.TrimPrefix(base, WhiteoutPrefix))
			list = append(list, filepath.ToSlash(deletedTarget)+" (deleted)")
		} else {
			list = append(list, filepath.ToSlash(rel))
		}
		return nil
	})
	sort.Strings(list)
	return list
}

// Promote copies modified and added files from Upper to targetDir, respecting whiteouts.
func (o *darwinAPFSOverlay) Promote(targetDir string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	if err := o.reconcileLocked(); err != nil {
		return fmt.Errorf("failed to reconcile changes before promote: %w", err)
	}

	cleanTarget := filepath.Clean(targetDir)
	if err := os.MkdirAll(cleanTarget, 0755); err != nil {
		return fmt.Errorf("failed to ensure target directory exists: %w", err)
	}

	err := filepath.Walk(o.upperDir, func(upperPath string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(o.upperDir, upperPath)
		if err != nil || rel == "." {
			return nil
		}

		base := filepath.Base(rel)
		if strings.HasPrefix(base, WhiteoutPrefix) {
			// Whiteout deletion marker: remove corresponding target path
			targetFileName := strings.TrimPrefix(base, WhiteoutPrefix)
			targetFilePath := filepath.Join(cleanTarget, filepath.Dir(rel), targetFileName)
			_ = os.RemoveAll(targetFilePath)
			return nil
		}

		targetPath := filepath.Join(cleanTarget, rel)
		if info.IsDir() {
			return os.MkdirAll(targetPath, info.Mode())
		}

		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return err
		}

		// Fast path: atomic replace via temporary clone + rename
		tmpDst := filepath.Join(filepath.Dir(targetPath), fmt.Sprintf(".tmp_promote_%d_%d", time.Now().UnixNano(), rand.Int63()))
		if cloneErr := clonefile(upperPath, tmpDst, CLONE_NOFOLLOW|CLONE_NOOWNERCOPY); cloneErr == nil {
			if renameErr := os.Rename(tmpDst, targetPath); renameErr == nil {
				return nil
			}
			_ = os.Remove(tmpDst)
		}

		// Fallback to copyFile if clonefile or rename fails (e.g. cross-volume boundary)
		return copyFile(upperPath, targetPath)
	})

	// If lowerDir was updated by promote, invalidate standby clone
	if cleanTarget == o.lowerDir {
		if o.standbyMerged != "" {
			sMerged := o.standbyMerged
			sUpper := o.standbyUpper
			o.standbyMerged = ""
			o.standbyUpper = ""
			o.wg.Add(1)
			go func(sm, su string) {
				defer o.wg.Done()
				_ = os.RemoveAll(sm)
				_ = os.RemoveAll(su)
			}(sMerged, sUpper)
		}
	}

	return err
}

// Destroy unmounts/removes all temporary overlay directories and waits for asynchronous cleaners.
func (o *darwinAPFSOverlay) Destroy() error {
	o.mu.Lock()
	if o.destroyed {
		o.mu.Unlock()
		return nil
	}
	o.destroyed = true
	o.mu.Unlock()

	o.wg.Wait()
	return os.RemoveAll(o.tempBase)
}

// newOSOverlay creates a Darwin APFS clonefile / copy-on-write overlay with userspace fallback.
func newOSOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	// Try native Darwin APFS clonefile overlay first
	overlay, err := newDarwinAPFSOverlay(lowerDir, tempBase)
	if err == nil {
		return overlay, nil
	}
	// Transparently fall back to userspace overlay if APFS cloning is unsupported
	return newUserspaceOverlay(lowerDir, tempBase)
}
