//go:build linux

package strix

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

func sysIoctlGetFlags(fd uintptr) (int64, error) {
	var flags int64
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, FSIocGetFlags, uintptr(unsafe.Pointer(&flags)))
	if errno != 0 {
		return 0, errno
	}
	return flags, nil
}

func sysIoctlSetFlags(fd uintptr, flags int64) error {
	_, _, errno := unix.Syscall(unix.SYS_IOCTL, fd, FSIocSetFlags, uintptr(unsafe.Pointer(&flags)))
	if errno != 0 {
		return errno
	}
	return nil
}

func sysCheckBtrfs(path string) (bool, error) {
	var fs unix.Statfs_t
	if err := unix.Statfs(path, &fs); err != nil {
		return false, err
	}
	return uint32(fs.Type) == BtrfsSuperMagic, nil
}
