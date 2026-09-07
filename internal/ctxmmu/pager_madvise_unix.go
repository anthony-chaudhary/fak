//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package ctxmmu

import "golang.org/x/sys/unix"

// osMadviseRandom advises the kernel that the given mapped byte range will be accessed
// with random, non-sequential access patterns (MADV_RANDOM). This suppresses aggressive
// sequential readahead (e.g. 128 KiB per read) on auxiliary tables like 51B PLE engram tables.
func osMadviseRandom(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	return unix.Madvise(data, unix.MADV_RANDOM) == nil
}

// osMadviseWillneed advises the kernel that the specified sub-range [off, off+length)
// is needed in the near future (MADV_WILLNEED), initiating asynchronous page-in ahead
// of synchronous gather execution.
func osMadviseWillneed(data []byte, off, length int) bool {
	if length <= 0 || off < 0 || off >= len(data) {
		return false
	}
	end := off + length
	if end > len(data) {
		end = len(data)
	}
	if page := unix.Getpagesize(); page > 0 {
		off -= off % page
	}
	return unix.Madvise(data[off:end], unix.MADV_WILLNEED) == nil
}
