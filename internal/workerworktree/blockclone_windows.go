//go:build windows

package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

const fsctlDuplicateExtentsToFile = 0x00098344

type duplicateExtentsData struct {
	FileHandle       syscall.Handle
	SourceFileOffset int64
	TargetFileOffset int64
	ByteCount        int64
}

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
	srcp, err := syscall.UTF16PtrFromString(src)
	if err != nil {
		return err
	}
	dstp, err := syscall.UTF16PtrFromString(dst)
	if err != nil {
		return err
	}
	source, err := syscall.CreateFile(srcp, syscall.GENERIC_READ, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(source)
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	target, err := syscall.CreateFile(dstp, syscall.GENERIC_READ|syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.CREATE_NEW, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(target)
	if err := syscall.Ftruncate(target, info.Size()); err != nil {
		return err
	}
	input := duplicateExtentsData{FileHandle: source, ByteCount: info.Size()}
	var returned uint32
	if err := syscall.DeviceIoControl(target, fsctlDuplicateExtentsToFile, (*byte)(unsafe.Pointer(&input)), uint32(unsafe.Sizeof(input)), nil, 0, &returned, nil); err != nil {
		return fmt.Errorf("FSCTL_DUPLICATE_EXTENTS_TO_FILE: %w", err)
	}
	return nil
}
