//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package model

import (
	"io"
	"os"
	"syscall"
)

// mmapOpen maps path read-only via syscall.Mmap (stdlib only — no x/sys dependency) and
// returns the mapped bytes plus a Closer that munmaps the region and closes the file. The
// returned slice aliases OS-managed, read-only memory: callers must only READ it and must
// not retain it past the Closer. On any error before a successful map it returns
// errMmapUnsupported (zero-size file) or the underlying syscall error, so openSafetensorsFile
// falls back to the os.Open + ReadAt path.
func mmapOpen(path string) ([]byte, io.Closer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	st, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, nil, err
	}
	size := st.Size()
	if size <= 0 || size != int64(int(size)) {
		// Empty file (mmap of length 0 fails) or a size that overflows int on a 32-bit
		// host — neither is mappable here; fall back to the portable reader.
		_ = f.Close()
		return nil, nil, errMmapUnsupported
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, nil, err
	}
	return data, closerFunc(func() error {
		merr := syscall.Munmap(data)
		ferr := f.Close()
		if merr != nil {
			return merr
		}
		return ferr
	}), nil
}
