//go:build unix

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// mapView maps the whole file read-only, shared with other processes, so
// sibling workers reading the same dataset file share one page-cache copy
// (upstream mmap ACCESS_READ over the data file).
func (r *MappedRegion) mapView(f *os.File) ([]byte, error) {
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fi.Size())
	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	return data, nil
}

// advise issues madvise(MADV_WILLNEED) over the whole mapping, best-effort:
// some platforms lack it, and the page walk still forces residency
// (upstream suppresses the same failure class before its walk).
func (r *MappedRegion) advise() bool {
	if len(r.data) == 0 {
		return false
	}
	_, _, errno := syscall.Syscall(
		syscall.SYS_MADVISE,
		uintptr(unsafe.Pointer(&r.data[0])),
		uintptr(len(r.data)),
		uintptr(syscall.MADV_WILLNEED),
	)
	return errno == 0
}

// pageUnit is the OS page size; the prefault walk steps by it.
func (r *MappedRegion) pageUnit() int64 { return int64(syscall.Getpagesize()) }

// release unmaps the region.
func (r *MappedRegion) release() error {
	if len(r.data) == 0 {
		return nil
	}
	return syscall.Munmap(r.data)
}

// isUnixPlatform anchors the Advised contract check in the test.
func (r *MappedRegion) isUnixPlatform() bool { return true }
