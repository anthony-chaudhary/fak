//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package model

import "golang.org/x/sys/unix"

// madviseWillneed issues MADV_WILLNEED over the [off, off+length) sub-range of a read-only
// memory-mapped region so the kernel starts faulting those pages into the page cache ahead of
// a synchronous read. madvise requires a page-aligned start address, so off is rounded DOWN to
// the page boundary (advising a few extra leading bytes is harmless). It is a pure performance
// hint: any error — an unmapped range, an unsupported backing store — is swallowed and
// reported as false, because a failed readahead never affects correctness. This goes through
// golang.org/x/sys/unix rather than stdlib syscall because syscall.Madvise is undefined on
// several tagged targets (notably darwin/arm64 and the newer linux arches), whereas
// unix.Madvise is defined uniformly across every platform in the build constraint above.
// Solaris/illumos are excluded by that constraint and fall to the no-op stub.
func madviseWillneed(data []byte, off, length int) bool {
	if length <= 0 || off < 0 || off >= len(data) {
		return false
	}
	end := off + length
	if end > len(data) {
		end = len(data)
	}
	if page := unix.Getpagesize(); page > 0 {
		off -= off % page // round the start down to a page boundary (madvise requires it)
	}
	if err := unix.Madvise(data[off:end], unix.MADV_WILLNEED); err != nil {
		return false
	}
	return true
}
