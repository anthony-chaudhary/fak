//go:build linux

package ggufload

import (
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

func fileResidentBytes(f *os.File) (resident, size int64, err error) {
	st, err := f.Stat()
	if err != nil {
		return 0, 0, err
	}
	size = st.Size()
	if size == 0 {
		return 0, 0, nil
	}
	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return 0, 0, err
	}
	defer unix.Munmap(data)
	page := int64(unix.Getpagesize())
	vec := make([]byte, (size+page-1)/page)
	if _, _, errno := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), uintptr(unsafe.Pointer(&vec[0]))); errno != 0 {
		return 0, 0, errno
	}
	for i, v := range vec {
		if v&1 == 0 {
			continue
		}
		resident += min(page, size-int64(i)*page)
	}
	return resident, size, nil
}
