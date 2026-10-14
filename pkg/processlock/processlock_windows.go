//go:build windows

package processlock

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/windows"
)

func openFile(root *os.Root, name string) (*os.File, error) {
	f, err := root.OpenFile(name, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	var info windows.ByHandleFileInformation
	metadataErr := windows.GetFileInformationByHandle(windows.Handle(f.Fd()), &info)
	typ, typeErr := windows.GetFileType(windows.Handle(f.Fd()))
	if metadataErr != nil || typeErr != nil || typ != windows.FILE_TYPE_DISK || info.FileAttributes&(windows.FILE_ATTRIBUTE_REPARSE_POINT|windows.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		_ = f.Close()
		return nil, fmt.Errorf("unsafe lock file: metadata=%v type=%v", metadataErr, typeErr)
	}
	return f, nil
}

func lockFile(f *os.File) error {
	err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &windows.Overlapped{})
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) || errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
		return ErrBusy
	}
	return err
}

func unlockFile(f *os.File) error {
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, &windows.Overlapped{})
}
