//go:build linux

package strix

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

func applyPlatformMadvise(ptr unsafe.Pointer, size int64) error {
	if ptr == nil || size <= 0 {
		return nil
	}
	buf := unsafe.Slice((*byte)(ptr), size)

	// Prohibit unpinned copy/fork migration across Zen 5 cores.
	_ = unix.Madvise(buf, unix.MADV_DONTFORK)

	// Request 2MB hugepages on Zen 5 MMU.
	_ = unix.Madvise(buf, unix.MADV_HUGEPAGE)

	// Lock pages in DRAM to prevent TTM page eviction.
	_ = unix.Mlock(buf)

	return nil
}

func checkKFDAvailabilityOS(kfdPath string) (bool, error) {
	fi, err := os.Stat(kfdPath)
	if err != nil {
		return false, fmt.Errorf("kfd node %s: %w", kfdPath, err)
	}
	if fi.IsDir() {
		return false, fmt.Errorf("kfd path %s is a directory", kfdPath)
	}
	if kfdPath == DefaultKFDPath && (fi.Mode()&os.ModeDevice == 0) {
		return false, fmt.Errorf("kfd node %s is not a device file (mode: %v)", kfdPath, fi.Mode())
	}
	f, err := os.OpenFile(kfdPath, os.O_RDWR, 0)
	if err != nil {
		return false, fmt.Errorf("kfd node %s cannot be opened R/W: %w", kfdPath, err)
	}
	_ = f.Close()
	return true, nil
}

func kfdIoctlAllocMemOS(fd uintptr, args *KFDAllocMemArgs) error {
	if args == nil {
		return errors.New("nil KFDAllocMemArgs")
	}
	r1, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, AMDKFD_IOCTL_ALLOC_MEM_OF_GPU, uintptr(unsafe.Pointer(args)))
	if errno != 0 {
		return errno
	}
	if int(r1) < 0 {
		return fmt.Errorf("ioctl AMDKFD_IOCTL_ALLOC_MEM_OF_GPU returned %d", r1)
	}
	return nil
}

func kfdIoctlFreeMemOS(fd uintptr, handle uint64) error {
	args := KFDFreeMemArgs{Handle: handle}
	r1, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, AMDKFD_IOCTL_FREE_MEM, uintptr(unsafe.Pointer(&args)))
	if errno != 0 {
		return errno
	}
	if int(r1) < 0 {
		return fmt.Errorf("ioctl AMDKFD_IOCTL_FREE_MEM returned %d", r1)
	}
	return nil
}

func allocKFDUnifiedMemoryOS(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
	fd, err := unix.Open(DefaultKFDPath, unix.O_RDWR, 0)
	if err != nil {
		return 0, 0, err
	}
	defer unix.Close(fd)

	if err := kfdIoctlAllocMemOS(uintptr(fd), args); err != nil {
		return 0, 0, err
	}

	var hostPtr uintptr
	if args.VAAddr != 0 {
		hostPtr = uintptr(args.VAAddr)
	} else if args.MmapOffset != 0 {
		mmapBytes, err := unix.Mmap(fd, int64(args.MmapOffset), int(args.Size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
		if err != nil {
			return 0, 0, err
		}
		hostPtr = uintptr(unsafe.Pointer(&mmapBytes[0]))
		args.VAAddr = uint64(hostPtr)
		_ = applyPlatformMadvise(unsafe.Pointer(&mmapBytes[0]), int64(args.Size))
	} else {
		hostPtr = SimulatedUnifiedVABase
		args.VAAddr = uint64(hostPtr)
	}

	devPtr := hostPtr
	return hostPtr, devPtr, nil
}

func drmOpenOS(drmPath string) (int, error) {
	if drmPath == "" {
		drmPath = DefaultDRMRenderPath
	}
	return unix.Open(drmPath, unix.O_RDWR, 0)
}

func drmIoctlGEMCreateOS(fd uintptr, args *DRMAMDGPUGEMCreateArgs) error {
	if args == nil {
		return errors.New("nil DRMAMDGPUGEMCreateArgs")
	}
	r1, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, DRM_IOCTL_AMDGPU_GEM_CREATE, uintptr(unsafe.Pointer(args)))
	if errno != 0 {
		return errno
	}
	if int(r1) < 0 {
		return fmt.Errorf("ioctl DRM_IOCTL_AMDGPU_GEM_CREATE returned %d", r1)
	}
	return nil
}

func drmIoctlGEMMmapOS(fd uintptr, args *DRMAMDGPUGEMMmapArgs) error {
	if args == nil {
		return errors.New("nil DRMAMDGPUGEMMmapArgs")
	}
	r1, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, DRM_IOCTL_AMDGPU_GEM_MMAP, uintptr(unsafe.Pointer(args)))
	if errno != 0 {
		return errno
	}
	if int(r1) < 0 {
		return fmt.Errorf("ioctl DRM_IOCTL_AMDGPU_GEM_MMAP returned %d", r1)
	}
	return nil
}

func drmMmapOS(fd int, offset int64, size int) ([]byte, error) {
	return unix.Mmap(fd, offset, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
}

func drmMunmapOS(b []byte) error {
	return unix.Munmap(b)
}
