//go:build windows

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func truncateServeArtifactFixture(path string, size int64) (err error) {
	file, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, file.Close()) }()

	var returned uint32
	if err := windows.DeviceIoControl(windows.Handle(file.Fd()), windows.FSCTL_SET_SPARSE, nil, 0, nil, 0, &returned, nil); err != nil {
		return fmt.Errorf("mark fixture sparse before truncating: %w", err)
	}
	return file.Truncate(size)
}

// fak-test:runtime fast est=100ms lane=default
func TestServeArtifactFixtureTruncateIsSparse(t *testing.T) {
	const size = 16 << 20
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, []byte("GGUF"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := truncateServeArtifactFixture(path, size); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	// FILE_STANDARD_INFO from GetFileInformationByHandleEx.
	var info struct {
		allocationSize int64
		endOfFile      int64
		numberOfLinks  uint32
		deletePending  byte
		directory      byte
	}
	if err := windows.GetFileInformationByHandleEx(windows.Handle(file.Fd()), windows.FileStandardInfo, (*byte)(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		t.Fatal(err)
	}
	if info.endOfFile != size || info.allocationSize >= size/2 {
		t.Fatalf("fixture size = %d, allocated = %d; want size %d with less than half allocated", info.endOfFile, info.allocationSize, size)
	}
	var header [4]byte
	if _, err := file.ReadAt(header[:], 0); err != nil || string(header[:]) != "GGUF" {
		t.Fatalf("fixture header = %q, err = %v", header, err)
	}
	t.Logf("fixture size %d bytes, allocated %d bytes", info.endOfFile, info.allocationSize)
}
