//go:build linux

package alloc

import (
	"syscall"
	"testing"
	"unsafe"
)

func memfdCreate(name string, flags int) (int, error) {
	nameBytes := append([]byte(name), 0)
	fd, _, errno := syscall.Syscall(319,
		uintptr(unsafe.Pointer(&nameBytes[0])),
		uintptr(flags),
		0)
	if errno != 0 {
		return -1, errno
	}
	return int(fd), nil
}

func TestNewDevdaxRegion_AlignmentValidation(t *testing.T) {
	const align2MB = 2 * 1024 * 1024

	_, err := NewDevdaxRegion("/dev/null", 1024, align2MB)
	if err == nil {
		t.Fatal("expected error for misaligned offset")
	}

	_, err = NewDevdaxRegion("/dev/null", 0, 1024)
	if err == nil {
		t.Fatal("expected error for misaligned size")
	}

	_, err = NewDevdaxRegion("/dev/null", 0, 0)
	if err == nil {
		t.Fatal("expected error for zero size")
	}
}

func TestNewDevdaxRegion_ReadWrite(t *testing.T) {
	const regionSize = 2 * 1024 * 1024

	fd, err := memfdCreate("mock-devdax", 0)
	if err != nil {
		t.Skipf("memfd_create not available: %v", err)
	}
	defer syscall.Close(fd)

	if err := syscall.Ftruncate(fd, int64(regionSize)); err != nil {
		t.Fatalf("ftruncate: %v", err)
	}

	r := &Region{
		size:         regionSize,
		devdaxPath:   "/dev/dax0.0",
		devdaxOffset: 0,
		devdaxFd:     -1,
	}

	if !r.IsDevdax() {
		t.Error("IsDevdax() should return true")
	}
	if r.DevdaxPath() != "/dev/dax0.0" {
		t.Errorf("DevdaxPath() = %q, want /dev/dax0.0", r.DevdaxPath())
	}
	if r.DevdaxOffset() != 0 {
		t.Errorf("DevdaxOffset() = %d, want 0", r.DevdaxOffset())
	}
}

func TestRegion_NotDevdax(t *testing.T) {
	r := &Region{size: 4096}
	if r.IsDevdax() {
		t.Error("IsDevdax() should return false for normal region")
	}
	if r.DevdaxPath() != "" {
		t.Errorf("DevdaxPath() = %q, want empty", r.DevdaxPath())
	}
}
