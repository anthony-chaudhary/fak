//go:build !linux

package strix

import (
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"
)

var (
	errDRMNotSupportedNonLinux = errors.New("drm gem userptr ioctl not supported on non-linux OS")
)

// platformMmap simulates page-aligned memory mapping on non-Linux platforms (Windows/Darwin).
func platformMmap(f *os.File, size int64) ([]byte, uintptr, func() error, error) {
	if f == nil {
		return nil, 0, nil, errors.New("nil file descriptor")
	}
	if size <= 0 {
		return nil, 0, nil, ErrZeroLengthFile
	}

	// Allocate with 4096-byte padding to guarantee 4KB page alignment
	padSize := size + int64(DefaultPageSize)
	rawBuf := make([]byte, padSize)
	rawPtr := uintptr(unsafe.Pointer(&rawBuf[0]))

	// Compute offset to next 4096-byte boundary
	alignedOffset := (uintptr(DefaultPageSize) - (rawPtr % uintptr(DefaultPageSize))) % uintptr(DefaultPageSize)
	alignedSlice := rawBuf[alignedOffset : alignedOffset+uintptr(size)]
	alignedPtr := uintptr(unsafe.Pointer(&alignedSlice[0]))

	// Read file contents into the aligned slice
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, 0, nil, fmt.Errorf("seek file: %w", err)
	}
	if _, err := io.ReadFull(f, alignedSlice); err != nil {
		return nil, 0, nil, fmt.Errorf("read full: %w", err)
	}

	unmapFn := func() error {
		rawBuf = nil
		return nil
	}

	return alignedSlice, alignedPtr, unmapFn, nil
}

func platformOpenDRM(path string) (int, error) {
	return -1, errDRMNotSupportedNonLinux
}

func platformGEMUserptr(drmFd uintptr, addr uint64, size uint64, flags uint32) (uint32, error) {
	return 0, errDRMNotSupportedNonLinux
}

func closeDRMHandle(drmFd int, handle uint32) error {
	return nil
}
