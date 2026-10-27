//go:build windows

package mcpinstall

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

const (
	platformGuard      = "windows-handle-deny-write-delete-sharing"
	platformDurability = "file-flush;new-file-move-write-through;partial-append-unknown"
)

var moveFileEx = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

// Sharing checks exclude existing ordinary write handles as well as new opens;
// denying delete sharing also excludes pathname removal/replacement while held.
// This is not protection against privileged raw-volume writes or preexisting
// writable memory mappings. No advisory-only install fallback exists.
func openGuardedConfig(path string) (*os.File, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, fmt.Errorf("mcpinstall: native config guard refused: %w", err)
	}
	return os.NewFile(uintptr(handle), path), nil
}

func configIdentity(f *os.File) (string, error) {
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &info); err != nil {
		return "", err
	}
	return fmt.Sprintf("windows:%08x:%08x%08x", info.VolumeSerialNumber, info.FileIndexHigh, info.FileIndexLow), nil
}

// No REPLACE_EXISTING flag: collisions preserve the destination. WRITE_THROUGH
// is the Windows namespace-publication barrier; the staged file is already synced.
func publishNew(tmp, path string) error {
	src, err := syscall.UTF16PtrFromString(tmp)
	if err != nil {
		return err
	}
	dst, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return err
	}
	const writeThrough = 0x8
	r, _, callErr := moveFileEx.Call(uintptr(unsafe.Pointer(src)), uintptr(unsafe.Pointer(dst)), writeThrough)
	if r == 0 {
		return fmt.Errorf("mcpinstall: immutable publication failed: %w", callErr)
	}
	return nil
}

func syncPublished(path string) error {
	// Matching bytes do not prove an existing namespace entry crossed our
	// WRITE_THROUGH barrier. Do not assert durability or replace that destination.
	return errors.New("mcpinstall: existing immutable publication requires explicit reconciliation: " + path)
}
