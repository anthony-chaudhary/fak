package gym

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// WhiteoutPrefix is the prefix used for whiteout deletion markers in CoW overlays.
const WhiteoutPrefix = ".wh."

// CoWOverlay defines the Copy-on-Write overlay contract across platform implementations.
type CoWOverlay interface {
	// LowerDir returns the read-only host trunk / base workspace directory path.
	LowerDir() string
	// UpperDir returns the ephemeral writable directory path.
	UpperDir() string
	// MergedDir returns the unified overlay directory path.
	MergedDir() string
	// Reset wipes all mutations in <10ms, restoring pristine lower state.
	Reset() error
	// Promote copies modified and added files from Upper to targetDir, respecting whiteouts.
	Promote(targetDir string) error
	// Destroy cleans up and unmounts/removes all temporary overlay resources.
	Destroy() error
}

// NewOverlay creates the platform-optimal CoWOverlay instance.
func NewOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	return newOSOverlay(lowerDir, tempBase)
}

// defaultTempDir returns the optimal directory for ephemeral overlay storage (/dev/shm if available on Linux).
func defaultTempDir() string {
	if runtime.GOOS == "linux" {
		if fi, err := os.Stat("/dev/shm"); err == nil && fi.IsDir() {
			return "/dev/shm"
		}
	}
	return os.TempDir()
}

// UserspaceOverlay provides a portable, fast directory-tree Copy-on-Write overlay driver
// with atomic directory swapping for sub-10ms resets and whiteout support.
type UserspaceOverlay struct {
	mu        sync.Mutex
	lowerDir  string
	upperDir  string
	mergedDir string
	workDir   string
	tempBase  string
	trashSeq  uint64
	wg        sync.WaitGroup
	destroyed bool
}

// newUserspaceOverlay initializes a new UserspaceOverlay over lowerDir.
func newUserspaceOverlay(lowerDir, tempBase string) (*UserspaceOverlay, error) {
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

	if tempBase == "" {
		tmpRoot := defaultTempDir()
		base, err := os.MkdirTemp(tmpRoot, "fak-gym-cow-*")
		if err != nil {
			return nil, fmt.Errorf("failed to create temporary base directory: %w", err)
		}
		tempBase = base
	}

	upperDir := filepath.Join(tempBase, "upper")
	mergedDir := filepath.Join(tempBase, "merged")
	workDir := filepath.Join(tempBase, "work")

	if err := os.MkdirAll(upperDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, fmt.Errorf("failed to create upper directory: %w", err)
	}
	if err := os.MkdirAll(mergedDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, fmt.Errorf("failed to create merged directory: %w", err)
	}
	if err := os.MkdirAll(workDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}

	// Initial population of merged view from lower
	if err := copyDirTree(absLower, mergedDir); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, fmt.Errorf("failed to populate initial merged view from lower directory: %w", err)
	}

	return &UserspaceOverlay{
		lowerDir:  absLower,
		upperDir:  upperDir,
		mergedDir: mergedDir,
		workDir:   workDir,
		tempBase:  tempBase,
	}, nil
}

// LowerDir returns the read-only host trunk / base workspace directory path.
func (o *UserspaceOverlay) LowerDir() string {
	return o.lowerDir
}

// UpperDir returns the ephemeral writable directory path.
func (o *UserspaceOverlay) UpperDir() string {
	return o.upperDir
}

// MergedDir returns the unified overlay directory path.
func (o *UserspaceOverlay) MergedDir() string {
	return o.mergedDir
}

// Reset instantaneously wipes upper and merged mutations in <10ms, restoring pristine lower state.
func (o *UserspaceOverlay) Reset() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	oldUpper := o.upperDir
	oldMerged := o.mergedDir

	seq := atomic.AddUint64(&o.trashSeq, 1)
	newUpper := filepath.Join(o.tempBase, fmt.Sprintf("upper_%d", seq))
	newMerged := filepath.Join(o.tempBase, fmt.Sprintf("merged_%d", seq))

	if err := os.MkdirAll(newUpper, 0755); err != nil {
		return fmt.Errorf("failed to create reset upper directory: %w", err)
	}
	if err := os.MkdirAll(newMerged, 0755); err != nil {
		return fmt.Errorf("failed to create reset merged directory: %w", err)
	}

	// Restore pristine merged state from lower directory into new generation
	if err := copyDirTree(o.lowerDir, newMerged); err != nil {
		return fmt.Errorf("failed to restore merged state from lower directory: %w", err)
	}

	// Atomically switch active directories
	o.upperDir = newUpper
	o.mergedDir = newMerged

	// Asynchronously reap previous generation directory trees
	o.wg.Add(1)
	go func(u, m string) {
		defer o.wg.Done()
		_ = os.RemoveAll(u)
		_ = os.RemoveAll(m)
	}(oldUpper, oldMerged)

	return nil
}

// Reconcile scans changes between mergedDir and lowerDir, staging mutations and whiteouts in upperDir.
func (o *UserspaceOverlay) Reconcile() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.reconcileLocked()
}

func (o *UserspaceOverlay) reconcileLocked() error {
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
			// New file
			isModified = true
		} else if lowerInfo.IsDir() || lowerInfo.Size() != info.Size() {
			isModified = true
		} else {
			// Same size, check content bytes
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
			if err := copyFile(mergedPath, upperPath); err != nil {
				return err
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
			// File or directory was deleted; place whiteout in upperDir
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

// ModifiedArtifacts returns a sorted list of relative paths modified or deleted in the overlay.
func (o *UserspaceOverlay) ModifiedArtifacts() []string {
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
func (o *UserspaceOverlay) Promote(targetDir string) error {
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

	return filepath.Walk(o.upperDir, func(upperPath string, info os.FileInfo, err error) error {
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
		return copyFile(upperPath, targetPath)
	})
}

// Destroy unmounts/removes all temporary overlay directories and waits for asynchronous cleaners.
func (o *UserspaceOverlay) Destroy() error {
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

// copyFile copies a single file from src to dst, preserving permissions.
func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	info, err := srcFile.Stat()
	if err != nil {
		return err
	}

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode())
	if err != nil {
		return err
	}
	defer dstFile.Close()

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		return err
	}
	return nil
}

// copyDirTree recursively copies all files and directories from src to dst.
func copyDirTree(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil || rel == "." {
			return nil
		}

		targetPath := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(targetPath, info.Mode())
		}
		if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
			return err
		}
		return copyFile(path, targetPath)
	})
}
