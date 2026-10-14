//go:build !windows

package processlock

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

func openFile(root *os.Root, name string) (*os.File, error) {
	return root.OpenFile(name, os.O_CREATE|os.O_RDWR|unix.O_NONBLOCK|unix.O_NOFOLLOW, 0600)
}

func lockFile(f *os.File) error {
	err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB)
	if errors.Is(err, unix.EWOULDBLOCK) || errors.Is(err, unix.EAGAIN) {
		return ErrBusy
	}
	return err
}

func unlockFile(f *os.File) error {
	return unix.Flock(int(f.Fd()), unix.LOCK_UN)
}
