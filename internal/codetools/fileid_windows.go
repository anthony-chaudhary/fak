//go:build windows

package codetools

import (
	"fmt"
	"os"
	"syscall"
)

func fileIdentity(f *os.File, _ os.FileInfo) (string, error) {
	var info syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(syscall.Handle(f.Fd()), &info); err != nil {
		return "", err
	}
	return fmt.Sprintf("%08x:%08x%08x", info.VolumeSerialNumber, info.FileIndexHigh, info.FileIndexLow), nil
}
