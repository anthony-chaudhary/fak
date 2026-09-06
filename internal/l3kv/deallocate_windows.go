//go:build windows

package l3kv

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

func deallocateFileRangeOS(file *os.File, offset, length int64) error {
	handle := windows.Handle(file.Fd())

	// Best-effort: mark the file as sparse so FSCTL_SET_ZERO_DATA deallocates
	// physical clusters rather than allocating and writing zeroes to disk.
	var bytesReturned uint32
	_ = windows.DeviceIoControl(
		handle,
		windows.FSCTL_SET_SPARSE,
		nil,
		0,
		nil,
		0,
		&bytesReturned,
		nil,
	)

	zeroInfo := windows.FileZeroDataInformation{
		FileOffset:      offset,
		BeyondFinalZero: offset + length,
	}

	err := windows.DeviceIoControl(
		handle,
		windows.FSCTL_SET_ZERO_DATA,
		(*byte)(unsafe.Pointer(&zeroInfo)),
		uint32(unsafe.Sizeof(zeroInfo)),
		nil,
		0,
		&bytesReturned,
		nil,
	)
	if err == nil {
		return nil
	}

	// Graceful fallback if DeviceIoControl / sparse zero data is unsupported
	// (e.g. non-NTFS volumes, virtual filesystems, or permissions).
	return fallbackDeallocate(file, offset, length)
}
