//go:build windows

package debtlane

import (
	"os"
	"syscall"
)

func lockFileExclusively(path string) func() {
	p, err := syscall.UTF16PtrFromString(path)
	if err == nil {
		h, err := syscall.CreateFile(p, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
		if err == nil {
			return func() {
				_ = syscall.CloseHandle(h)
			}
		}
	}
	_ = os.Chmod(path, 0000)
	return func() {
		_ = os.Chmod(path, 0644)
	}
}
