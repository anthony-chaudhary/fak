//go:build !linux

package strix

import (
	"errors"
	"fmt"
	"os"
	"unsafe"
)

func applyPlatformMadvise(ptr unsafe.Pointer, size int64) error {
	// Simulated no-op for non-Linux platforms (Windows/Darwin development environments).
	return nil
}

func checkKFDAvailabilityOS(kfdPath string) (bool, error) {
	// On non-Linux systems, check if file exists and can be opened R/W (for test mocks).
	fi, err := os.Stat(kfdPath)
	if err != nil {
		return false, fmt.Errorf("kfd path %s not found: %w", kfdPath, err)
	}
	if fi.IsDir() {
		return false, fmt.Errorf("kfd path %s is a directory", kfdPath)
	}
	f, err := os.OpenFile(kfdPath, os.O_RDWR, 0)
	if err != nil {
		return false, fmt.Errorf("kfd path %s cannot be opened R/W: %w", kfdPath, err)
	}
	_ = f.Close()
	return true, nil
}

func kfdIoctlAllocMemOS(fd uintptr, args *KFDAllocMemArgs) error {
	return errors.New("kfd ioctl not available on non-linux OS")
}

func kfdIoctlFreeMemOS(fd uintptr, handle uint64) error {
	return errors.New("kfd ioctl not available on non-linux OS")
}

func allocKFDUnifiedMemoryOS(args *KFDAllocMemArgs) (uintptr, uintptr, error) {
	return 0, 0, errors.New("kfd ioctl not available on non-linux OS")
}

func drmOpenOS(drmPath string) (int, error) {
	return -1, errors.New("drm open not available on non-linux OS")
}

func drmIoctlGEMCreateOS(fd uintptr, args *DRMAMDGPUGEMCreateArgs) error {
	return errors.New("drm gem create ioctl not available on non-linux OS")
}

func drmIoctlGEMMmapOS(fd uintptr, args *DRMAMDGPUGEMMmapArgs) error {
	return errors.New("drm gem mmap ioctl not available on non-linux OS")
}

func drmMmapOS(fd int, offset int64, size int) ([]byte, error) {
	return nil, errors.New("drm mmap not available on non-linux OS")
}

func drmMunmapOS(b []byte) error {
	return nil
}
