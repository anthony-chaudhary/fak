//go:build windows

package tempartifact

import (
	"os"
	"syscall"
)

const fileAttributeReparsePoint = 0x400

func isReparsePoint(path string, info os.FileInfo) (bool, error) {
	if info.Mode()&os.ModeSymlink != 0 {
		return true, nil
	}
	pointer, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return false, err
	}
	attributes, err := syscall.GetFileAttributes(pointer)
	if err != nil {
		return false, err
	}
	return attributes&fileAttributeReparsePoint != 0, nil
}
