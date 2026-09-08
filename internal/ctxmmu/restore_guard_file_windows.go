//go:build windows

package ctxmmu

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	quarantineMovefileReplaceExisting = 0x00000001
	quarantineMovefileWriteThrough    = 0x00000008
)

var quarantineMoveFileExW = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

func syncQuarantineLedgerCreation(string) error {
	// appendBytesLocked has already FlushFileBuffers'd the newly created file.
	// NTFS has no portable directory-handle fsync; snapshot namespace commits use
	// MoveFileExW(MOVEFILE_WRITE_THROUGH) below.
	return nil
}

func replaceQuarantineLedgerFile(tmpName, path string) error {
	src, err := syscall.UTF16PtrFromString(tmpName)
	if err != nil {
		return fmt.Errorf("replace quarantine ledger snapshot: %w", err)
	}
	dst, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return fmt.Errorf("replace quarantine ledger snapshot: %w", err)
	}
	r, _, callErr := quarantineMoveFileExW.Call(
		uintptr(unsafe.Pointer(src)),
		uintptr(unsafe.Pointer(dst)),
		quarantineMovefileReplaceExisting|quarantineMovefileWriteThrough,
	)
	if r == 0 {
		return fmt.Errorf("replace quarantine ledger snapshot: %w", callErr)
	}
	return nil
}
