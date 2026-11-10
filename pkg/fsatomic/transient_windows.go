//go:build windows

package fsatomic

import (
	"errors"

	"golang.org/x/sys/windows"
)

// isTransientRenameError reports the refusals a briefly held destination
// produces on Windows.
func isTransientRenameError(err error) bool {
	return errors.Is(err, windows.ERROR_ACCESS_DENIED) ||
		errors.Is(err, windows.ERROR_SHARING_VIOLATION) ||
		errors.Is(err, windows.ERROR_LOCK_VIOLATION)
}
