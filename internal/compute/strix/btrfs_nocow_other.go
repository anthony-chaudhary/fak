//go:build !linux

package strix

import (
	"errors"
)

var errIoctlUnavailable = errors.New("ioctl not supported on non-linux OS")

func sysIoctlGetFlags(fd uintptr) (int64, error) {
	return 0, errIoctlUnavailable
}

func sysIoctlSetFlags(fd uintptr, flags int64) error {
	return errIoctlUnavailable
}

func sysCheckBtrfs(path string) (bool, error) {
	return false, errIoctlUnavailable
}
