//go:build windows

package gateway

import (
	"fmt"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

// canonicalExistingWorkspacePath resolves Windows reparse points, including
// directory junctions. filepath.EvalSymlinks alone does not reliably expose a
// junction's final target on Windows, so lease admission identifies the path by
// an opened handle before comparing it with the trusted workspace root.
func canonicalExistingWorkspacePath(path string) (string, error) {
	path = filepath.Clean(path)
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", fmt.Errorf("encode path: %w", err)
	}
	handle, err := windows.CreateFile(
		name,
		windows.FILE_READ_ATTRIBUTES,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE,
		nil,
		windows.OPEN_EXISTING,
		windows.FILE_FLAG_BACKUP_SEMANTICS,
		0,
	)
	if err != nil {
		return "", err
	}
	defer windows.CloseHandle(handle)

	buf := make([]uint16, 32768)
	n, err := windows.GetFinalPathNameByHandle(handle, &buf[0], uint32(len(buf)), 0)
	if err != nil {
		return "", fmt.Errorf("resolve final path: %w", err)
	}
	if n == 0 || n >= uint32(len(buf)) {
		return "", fmt.Errorf("resolve final path: invalid path length %d", n)
	}
	resolved := windows.UTF16ToString(buf[:n])
	if strings.HasPrefix(resolved, `\\?\UNC\`) {
		resolved = `\\` + strings.TrimPrefix(resolved, `\\?\UNC\`)
	} else {
		resolved = strings.TrimPrefix(resolved, `\\?\`)
	}
	return filepath.Clean(resolved), nil
}
