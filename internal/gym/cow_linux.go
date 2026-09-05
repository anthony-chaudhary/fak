//go:build linux

package gym

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// linuxOverlayFS implements CoWOverlay using the native Linux OverlayFS kernel driver.
type linuxOverlayFS struct {
	mu        sync.Mutex
	lowerDir  string
	upperDir  string
	workDir   string
	mergedDir string
	tempBase  string
	trashSeq  uint64
	wg        sync.WaitGroup
	mounted   bool
	destroyed bool
}

func newLinuxOverlayFS(lowerDir, tempBase string) (*linuxOverlayFS, error) {
	if tempBase == "" {
		tmpRoot := defaultTempDir()
		base, err := os.MkdirTemp(tmpRoot, "fak-gym-linux-*")
		if err != nil {
			return nil, err
		}
		tempBase = base
	}

	upperDir := filepath.Join(tempBase, "upper")
	workDir := filepath.Join(tempBase, "work")
	mergedDir := filepath.Join(tempBase, "merged")

	if err := os.MkdirAll(upperDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, err
	}
	if err := os.MkdirAll(workDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, err
	}
	if err := os.MkdirAll(mergedDir, 0755); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, err
	}

	opts := fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", lowerDir, upperDir, workDir)
	if err := syscall.Mount("overlay", mergedDir, "overlay", 0, opts); err != nil {
		_ = os.RemoveAll(tempBase)
		return nil, fmt.Errorf("linux overlay mount failed: %w", err)
	}

	return &linuxOverlayFS{
		lowerDir:  lowerDir,
		upperDir:  upperDir,
		workDir:   workDir,
		mergedDir: mergedDir,
		tempBase:  tempBase,
		mounted:   true,
	}, nil
}

func (o *linuxOverlayFS) LowerDir() string {
	return o.lowerDir
}

func (o *linuxOverlayFS) UpperDir() string {
	return o.upperDir
}

func (o *linuxOverlayFS) MergedDir() string {
	return o.mergedDir
}

func (o *linuxOverlayFS) Reset() error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	if o.mounted {
		if err := syscall.Unmount(o.mergedDir, 0); err != nil {
			return fmt.Errorf("failed to unmount overlay for reset: %w", err)
		}
		o.mounted = false
	}

	seq := atomic.AddUint64(&o.trashSeq, 1)
	r := rand.Int63()
	trashUpper := filepath.Join(o.tempBase, fmt.Sprintf(".trash_u_%d_%d_%d", time.Now().UnixNano(), seq, r))
	trashWork := filepath.Join(o.tempBase, fmt.Sprintf(".trash_w_%d_%d_%d", time.Now().UnixNano(), seq, r))

	_ = os.Rename(o.upperDir, trashUpper)
	_ = os.MkdirAll(o.upperDir, 0755)
	_ = os.Rename(o.workDir, trashWork)
	_ = os.MkdirAll(o.workDir, 0755)

	o.wg.Add(1)
	go func(u, w string) {
		defer o.wg.Done()
		_ = os.RemoveAll(u)
		_ = os.RemoveAll(w)
	}(trashUpper, trashWork)

	opts := fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", o.lowerDir, o.upperDir, o.workDir)
	if err := syscall.Mount("overlay", o.mergedDir, "overlay", 0, opts); err != nil {
		return fmt.Errorf("failed to remount overlay after reset: %w", err)
	}
	o.mounted = true
	return nil
}

func (o *linuxOverlayFS) Promote(targetDir string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if o.destroyed {
		return errors.New("overlay has been destroyed")
	}

	cleanTarget := filepath.Clean(targetDir)
	if err := os.MkdirAll(cleanTarget, 0755); err != nil {
		return err
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
		isWhiteout := strings.HasPrefix(base, WhiteoutPrefix)
		if !isWhiteout && (info.Mode()&os.ModeDevice != 0) {
			isWhiteout = true
		}

		if isWhiteout {
			targetName := strings.TrimPrefix(base, WhiteoutPrefix)
			targetFilePath := filepath.Join(cleanTarget, filepath.Dir(rel), targetName)
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

func (o *linuxOverlayFS) Destroy() error {
	o.mu.Lock()
	if o.destroyed {
		o.mu.Unlock()
		return nil
	}
	o.destroyed = true
	if o.mounted {
		_ = syscall.Unmount(o.mergedDir, 0)
		o.mounted = false
	}
	o.mu.Unlock()

	o.wg.Wait()
	return os.RemoveAll(o.tempBase)
}

func newOSOverlay(lowerDir, tempBase string) (CoWOverlay, error) {
	// Native Linux OverlayFS mount helper if root/capabilities permit, with automatic fallback to userspace CoW overlay.
	overlay, err := newLinuxOverlayFS(lowerDir, tempBase)
	if err == nil {
		return overlay, nil
	}
	return newUserspaceOverlay(lowerDir, tempBase)
}
