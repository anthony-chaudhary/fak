//go:build linux

package ggufload

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// ReadAheadRange advises only the pages intersecting an exact upcoming payload
// range. The mapping and its checkpoint owner remain unchanged. Success means
// the kernel accepted MADV_WILLNEED, not that any I/O finished or overlapped.
// Other platforms do not advertise this optional reader capability.
func (m *mmapReaderAt) ReadAheadRange(off, size int64) error {
	return m.readAheadRange(off, size, unix.Madvise)
}

func (m *mmapReaderAt) readAheadRange(off, size int64, advise func([]byte, int) error) error {
	if m == nil || off < 0 || size < 0 || off > int64(len(m.data)) || size > int64(len(m.data))-off {
		return fmt.Errorf("gguf: mmap readahead offset %d size %d outside mapping", off, size)
	}
	if size == 0 {
		return nil
	}
	end := off + size // validated by subtraction before addition
	// openShardReader owns a whole-file, page-aligned mapping. The syscall
	// rounds its length up; keep the Go slice within the file even when EOF
	// falls inside the last mapped page. No disjoint expert spans are merged.
	start := off - off%int64(unix.Getpagesize())
	if err := advise(m.data[start:end:end], unix.MADV_WILLNEED); err != nil {
		return fmt.Errorf("gguf: mmap readahead offset %d size %d: %w", off, size, err)
	}
	return nil
}
