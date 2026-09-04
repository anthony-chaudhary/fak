package qwen4exp

import (
	"io"
)

const (
	// SectorSize is the standard 4KB sector size for NVMe and SSD Direct I/O.
	SectorSize = 4096
)

// DirectIOHandle represents an open file configured for Direct (unbuffered) I/O.
type DirectIOHandle interface {
	io.Closer
	// ReadAtAligned reads len(dest) bytes starting at fileOffset into dest.
	// In Direct I/O mode, dest must be sector-aligned in memory and length,
	// and fileOffset must be sector-aligned.
	ReadAtAligned(dest []byte, fileOffset int64) (int, error)
	// IsDirect reports whether unbuffered Direct I/O (bypassing OS page cache) is active.
	IsDirect() bool
	// Path reports the file path.
	Path() string
}

// PinnedMemory represents host memory locked against paging/swapping.
type PinnedMemory interface {
	Bytes() []byte
	Free() error
}

// AlignUp rounds n up to the nearest multiple of align (align must be a power of 2).
func AlignUp(n int64, align int64) int64 {
	return (n + align - 1) &^ (align - 1)
}

// AlignDown rounds n down to the nearest multiple of align (align must be a power of 2).
func AlignDown(n int64, align int64) int64 {
	return n &^ (align - 1)
}

// IsAligned checks if address is aligned to alignment (align must be a power of 2).
func IsAligned(ptr uintptr, align uintptr) bool {
	return ptr&(align-1) == 0
}
