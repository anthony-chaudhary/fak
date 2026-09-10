//go:build linux

package strix

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Linux DRM ioctl for GEM object close: _IOW('d', 0x09, struct drm_gem_close) = 0x40086409
const drmIoctlGEMClose uintptr = 0x40086409

type drmGemClose struct {
	Handle uint32
	Pad    uint32
}

func platformMmap(f *os.File, size int64) ([]byte, uintptr, func() error, error) {
	if f == nil {
		return nil, 0, nil, errors.New("nil file descriptor")
	}
	if size <= 0 {
		return nil, 0, nil, ErrZeroLengthFile
	}

	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("unix.Mmap: %w", err)
	}

	// Hint sequential readahead and hugepage backing where possible
	_ = unix.Madvise(data, unix.MADV_WILLNEED)
	_ = unix.Madvise(data, unix.MADV_HUGEPAGE)

	baseAddr := uintptr(unsafe.Pointer(&data[0]))
	unmapFn := func() error {
		return unix.Munmap(data)
	}

	return data, baseAddr, unmapFn, nil
}

func platformOpenDRM(path string) (int, error) {
	fd, err := unix.Open(path, unix.O_RDWR|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, fmt.Errorf("open drm device %s: %w", path, err)
	}
	return fd, nil
}

func platformGEMUserptr(drmFd uintptr, addr uint64, size uint64, flags uint32) (uint32, error) {
	args := DRMAMDGPUGEMUserptrArgs{
		Addr:  addr,
		Size:  size,
		Flags: flags,
	}

	r1, _, errno := unix.Syscall(unix.SYS_IOCTL, drmFd, DRM_IOCTL_AMDGPU_GEM_USERPTR, uintptr(unsafe.Pointer(&args)))
	if errno != 0 {
		return 0, fmt.Errorf("ioctl DRM_IOCTL_AMDGPU_GEM_USERPTR failed: %w", errno)
	}
	if int(r1) < 0 {
		return 0, fmt.Errorf("ioctl DRM_IOCTL_AMDGPU_GEM_USERPTR returned error code %d", r1)
	}

	return args.Handle, nil
}

func closeDRMHandle(drmFd int, handle uint32) error {
	var errs []error
	if drmFd >= 0 && handle > 0 {
		closeArgs := drmGemClose{
			Handle: handle,
		}
		_, _, errno := unix.Syscall(unix.SYS_IOCTL, uintptr(drmFd), drmIoctlGEMClose, uintptr(unsafe.Pointer(&closeArgs)))
		if errno != 0 {
			errs = append(errs, fmt.Errorf("drm gem close handle %d: %w", handle, errno))
		}
	}
	if drmFd >= 0 {
		if err := unix.Close(drmFd); err != nil {
			errs = append(errs, fmt.Errorf("close drm fd %d: %w", drmFd, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("closeDRMHandle: %v", errs)
	}
	return nil
}
