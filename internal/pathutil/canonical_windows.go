//go:build windows

package pathutil

import (
	"fmt"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

const fileReadAttributes = 0x80

var getFinalPathNameByHandleW = syscall.NewLazyDLL("kernel32.dll").NewProc("GetFinalPathNameByHandleW")

// CanonicalExisting resolves an existing filesystem path to its physical location,
// including through directory junctions and other Windows reparse aliases.
func CanonicalExisting(path string) (string, error) {
	path = filepath.Clean(path)
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return "", fmt.Errorf("encode path: %w", err)
	}
	handle, err := syscall.CreateFile(
		name,
		fileReadAttributes,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil,
		syscall.OPEN_EXISTING,
		syscall.FILE_FLAG_BACKUP_SEMANTICS,
		0,
	)
	if err != nil {
		return "", err
	}
	defer syscall.CloseHandle(handle)

	buf := make([]uint16, 32768)
	n, _, callErr := getFinalPathNameByHandleW.Call(
		uintptr(handle),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)),
		0,
	)
	if n == 0 {
		return "", fmt.Errorf("resolve final path: %w", callErr)
	}
	if n >= uintptr(len(buf)) {
		return "", fmt.Errorf("resolve final path: invalid path length %d", n)
	}
	resolved := syscall.UTF16ToString(buf[:n])
	if strings.HasPrefix(resolved, `\\?\UNC\`) {
		resolved = `\\` + strings.TrimPrefix(resolved, `\\?\UNC\`)
	} else {
		resolved = strings.TrimPrefix(resolved, `\\?\`)
	}
	return filepath.Clean(resolved), nil
}
