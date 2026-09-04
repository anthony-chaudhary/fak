package compute

import (
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"
)

// disk_direct.go — Direct I/O and post-upload page cache eviction for large-model weight streaming
// (borrowed from wkljohn/ds4-strix-halo-tp-odinlink; Issue #10763).
//
// PROBLEM / CONTEXT:
// When streaming massive model weights (such as 150+ GiB MoE parameters) from NVMe SSDs into accelerator
// memory on unified-memory APUs (e.g. AMD Strix Halo, Apple Silicon) or memory-constrained workstations,
// standard buffered file I/O causes the operating system kernel page cache to fill all physical RAM.
// This triggers memory reclamation stalls, evicts the active KV cache, and trips the OS Out-Of-Memory (OOM) killer.
//
// DS4-STRIX-HALO DIRECT I/O & AGGRESSIVE EVICTION PATTERN:
// 1. Direct I/O (O_DIRECT): block-aligned pread() calls bypassing the OS page cache completely.
// 2. Aggressive Post-Upload Eviction: posix_fadvise(..., POSIX_FADV_DONTNEED) and posix_madvise(..., POSIX_MADV_DONTNEED)
//    immediately after DMA upload into accelerator/unified memory completes.

// DirectIOAlignment is the standard 4096-byte memory and block alignment boundary
// required for Direct I/O (O_DIRECT) disk operations.
const DirectIOAlignment = 4096

// AlignedBlockBuffer represents a memory buffer aligned to 4096-byte (DirectIOAlignment)
// boundaries, required for Direct I/O (O_DIRECT) block transfers bypassing the OS page cache.
type AlignedBlockBuffer struct {
	raw   []byte
	Block []byte
}

// NewAlignedBlockBuffer allocates an AlignedBlockBuffer of the given size in bytes,
// ensuring the underlying Block slice address is strictly aligned to DirectIOAlignment (4096 bytes).
func NewAlignedBlockBuffer(size int) *AlignedBlockBuffer {
	if size <= 0 {
		return &AlignedBlockBuffer{}
	}
	// Allocate extra DirectIOAlignment bytes to guarantee we can find an aligned sub-slice.
	raw := make([]byte, size+DirectIOAlignment)
	ptr := uintptr(unsafe.Pointer(&raw[0]))
	offset := 0
	if rem := ptr % uintptr(DirectIOAlignment); rem != 0 {
		offset = int(uintptr(DirectIOAlignment) - rem)
	}
	block := raw[offset : offset+size : offset+size]
	return &AlignedBlockBuffer{
		raw:   raw,
		Block: block,
	}
}

// Bytes returns the aligned byte slice.
func (b *AlignedBlockBuffer) Bytes() []byte {
	if b == nil {
		return nil
	}
	return b.Block
}

// Len returns the length of the aligned block in bytes.
func (b *AlignedBlockBuffer) Len() int {
	if b == nil {
		return 0
	}
	return len(b.Block)
}

// Cap returns the capacity of the aligned block in bytes.
func (b *AlignedBlockBuffer) Cap() int {
	if b == nil {
		return 0
	}
	return cap(b.Block)
}

// IsAligned returns true if the buffer's start address is an exact multiple of DirectIOAlignment.
func (b *AlignedBlockBuffer) IsAligned() bool {
	if b == nil || len(b.Block) == 0 {
		return true
	}
	return uintptr(unsafe.Pointer(&b.Block[0]))%uintptr(DirectIOAlignment) == 0
}

// Reset clears the aligned block bytes to zero.
func (b *AlignedBlockBuffer) Reset() {
	if b != nil && len(b.Block) > 0 {
		clear(b.Block)
	}
}

// SubBlock returns a sub-slice of Block from start to end.
func (b *AlignedBlockBuffer) SubBlock(start, end int) []byte {
	if b == nil || b.Block == nil {
		return nil
	}
	return b.Block[start:end]
}

// DropFilePages evicts pages from the operating system page cache for the given file descriptor
// range [offset, offset+length). It uses POSIX_FADV_DONTNEED on Linux, fcntl F_NOCACHE on Darwin,
// and a cross-platform safe fallback on Windows and unsupported platforms.
func DropFilePages(fd uintptr, offset, length int64) error {
	if fd == 0 || length <= 0 {
		return nil
	}
	if offset < 0 {
		offset = 0
	}
	return dropFilePagesOS(fd, offset, length)
}

// DirectIOReader is a reader supporting block-aligned reads with optional post-upload eviction.
type DirectIOReader struct {
	r         io.ReaderAt
	fd        uintptr
	size      int64
	offset    int64
	blockSize int
	evict     bool
	buf       *AlignedBlockBuffer
}

// NewDirectIOReader wraps an io.ReaderAt with block-aligned read support and optional post-upload
// page cache eviction. If r exposes an Fd() uintptr method (such as *os.File), its descriptor
// is used for page cache eviction.
func NewDirectIOReader(r io.ReaderAt, size int64, blockSize int, evict bool) *DirectIOReader {
	if blockSize <= 0 {
		blockSize = DirectIOAlignment
	} else if rem := blockSize % DirectIOAlignment; rem != 0 {
		blockSize += DirectIOAlignment - rem
	}
	var fd uintptr
	if f, ok := r.(interface{ Fd() uintptr }); ok {
		fd = f.Fd()
	}
	return &DirectIOReader{
		r:         r,
		fd:        fd,
		size:      size,
		blockSize: blockSize,
		evict:     evict,
		buf:       NewAlignedBlockBuffer(blockSize),
	}
}

// NewDirectIOReaderFromFile creates a DirectIOReader from an *os.File, querying its size
// and setting blockSize to DirectIOAlignment.
func NewDirectIOReaderFromFile(f *os.File, evict bool) (*DirectIOReader, error) {
	if f == nil {
		return nil, errors.New("file is nil")
	}
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat file: %w", err)
	}
	return NewDirectIOReader(f, stat.Size(), DirectIOAlignment, evict), nil
}

// ReadAt reads len(p) bytes from the underlying reader at offset off. If eviction is enabled
// and an underlying file descriptor is present, DropFilePages is invoked for the read range.
func (d *DirectIOReader) ReadAt(p []byte, off int64) (int, error) {
	if d.r == nil {
		return 0, errors.New("underlying reader is nil")
	}
	n, err := d.r.ReadAt(p, off)
	if n > 0 && d.evict && d.fd != 0 {
		_ = DropFilePages(d.fd, off, int64(n))
	}
	return n, err
}

// Read reads up to len(p) bytes sequentially from the current offset, advancing the offset.
func (d *DirectIOReader) Read(p []byte) (int, error) {
	if d.size > 0 && d.offset >= d.size {
		return 0, io.EOF
	}
	n, err := d.ReadAt(p, d.offset)
	d.offset += int64(n)
	return n, err
}

// Seek sets the offset for the next Read.
func (d *DirectIOReader) Seek(offset int64, whence int) (int64, error) {
	var next int64
	switch whence {
	case io.SeekStart:
		next = offset
	case io.SeekCurrent:
		next = d.offset + offset
	case io.SeekEnd:
		if d.size < 0 {
			return d.offset, errors.New("cannot seek from end: unknown size")
		}
		next = d.size + offset
	default:
		return d.offset, errors.New("invalid whence")
	}
	if next < 0 {
		return d.offset, errors.New("negative seek position")
	}
	d.offset = next
	return d.offset, nil
}

// ReadAlignedBlock reads length bytes at off into an aligned buffer, returning the aligned slice.
func (d *DirectIOReader) ReadAlignedBlock(off int64, length int) ([]byte, error) {
	if length <= 0 {
		return nil, nil
	}
	var b *AlignedBlockBuffer
	if length <= d.blockSize && d.buf != nil {
		b = d.buf
	} else {
		b = NewAlignedBlockBuffer(length)
	}
	target := b.Block[:length]
	n, err := d.ReadAt(target, off)
	if err != nil && err != io.EOF {
		return nil, err
	}
	return target[:n], nil
}

// Evict explicitly evicts the specified range from the OS page cache.
func (d *DirectIOReader) Evict(offset, length int64) error {
	if d.fd == 0 {
		return nil
	}
	return DropFilePages(d.fd, offset, length)
}

// Fd returns the underlying file descriptor, or 0 if not available.
func (d *DirectIOReader) Fd() uintptr {
	return d.fd
}

// Size returns the total size in bytes, or -1 if unknown.
func (d *DirectIOReader) Size() int64 {
	return d.size
}

// Offset returns the current sequential read offset.
func (d *DirectIOReader) Offset() int64 {
	return d.offset
}

// BlockSize returns the configured block size.
func (d *DirectIOReader) BlockSize() int {
	return d.blockSize
}

// IsEvictEnabled reports whether post-upload eviction is enabled.
func (d *DirectIOReader) IsEvictEnabled() bool {
	return d.evict
}

// SetEvict enables or disables post-upload eviction.
func (d *DirectIOReader) SetEvict(evict bool) {
	d.evict = evict
}

// Close closes the underlying reader if it implements io.Closer.
func (d *DirectIOReader) Close() error {
	if c, ok := d.r.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// Stream streams the reader's content using StreamWeightsWithEviction.
func (d *DirectIOReader) Stream(chunkSize int, onChunk func(chunk []byte) error) (int64, error) {
	return StreamWeightsWithEviction(d.r, d.size, chunkSize, onChunk, d.evict)
}

// StreamWeightsWithEviction streams weights from r of given size in chunkSize increments,
// invoking onChunk for each block. When evict is true and r has an underlying file descriptor,
// it aggressively drops file page cache entries immediately after each chunk is processed,
// preventing host page-cache pollution and avoiding OOM on unified-memory APUs or workstations.
func StreamWeightsWithEviction(
	r io.ReaderAt,
	size int64,
	chunkSize int,
	onChunk func(chunk []byte) error,
	evict bool,
) (int64, error) {
	if size <= 0 {
		return 0, nil
	}
	if r == nil {
		return 0, errors.New("reader is nil")
	}
	if onChunk == nil {
		return 0, errors.New("onChunk callback is nil")
	}
	if chunkSize <= 0 {
		chunkSize = 2 * 1024 * 1024 // 2MB default chunk size
	}
	if chunkSize < DirectIOAlignment {
		chunkSize = DirectIOAlignment
	} else if rem := chunkSize % DirectIOAlignment; rem != 0 {
		chunkSize += DirectIOAlignment - rem
	}

	var fd uintptr
	if f, ok := r.(interface{ Fd() uintptr }); ok {
		fd = f.Fd()
	}

	buf := NewAlignedBlockBuffer(chunkSize)
	var totalStreamed int64

	for offset := int64(0); offset < size; {
		toRead := chunkSize
		remaining := size - offset
		if int64(toRead) > remaining {
			toRead = int(remaining)
		}

		target := buf.Block[:toRead]
		n, err := r.ReadAt(target, offset)
		if err != nil && toRead%DirectIOAlignment != 0 {
			// Direct I/O file descriptors (O_DIRECT) fail unaligned pread calls with EINVAL.
			// Try reading an aligned block size into the aligned buffer.
			alignedLen := ((toRead + DirectIOAlignment - 1) / DirectIOAlignment) * DirectIOAlignment
			if alignedLen <= len(buf.Block) {
				nAligned, errAligned := r.ReadAt(buf.Block[:alignedLen], offset)
				if nAligned > 0 {
					n = nAligned
					if n > toRead {
						n = toRead
					}
					err = errAligned
				}
			}
		}

		if n > 0 {
			chunk := target[:n]
			if errCB := onChunk(chunk); errCB != nil {
				return totalStreamed, errCB
			}
			totalStreamed += int64(n)

			if evict && fd != 0 {
				_ = DropFilePages(fd, offset, int64(n))
			}
			offset += int64(n)
		}

		if err != nil {
			if err == io.EOF {
				break
			}
			return totalStreamed, err
		}
		if n == 0 {
			break
		}
	}

	return totalStreamed, nil
}
