//go:build windows

package main

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

type modelArtifactWindowsBasicInfo struct {
	creationTime   int64
	lastAccessTime int64
	lastWriteTime  int64
	changeTime     int64
	fileAttributes uint32
	padding        uint32
}

func modelArtifactFileIdentity(f *os.File, _ os.FileInfo) (string, int64, error) {
	handle := windows.Handle(f.Fd())
	var fileInfo windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &fileInfo); err != nil {
		return "", 0, err
	}
	var basicInfo modelArtifactWindowsBasicInfo
	if err := windows.GetFileInformationByHandleEx(
		handle,
		windows.FileBasicInfo,
		(*byte)(unsafe.Pointer(&basicInfo)),
		uint32(unsafe.Sizeof(basicInfo)),
	); err != nil {
		return "", 0, err
	}
	identity := fmt.Sprintf("%d:%d:%d", fileInfo.VolumeSerialNumber, fileInfo.FileIndexHigh, fileInfo.FileIndexLow)
	return identity, basicInfo.changeTime, nil
}
