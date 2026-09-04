//go:build !windows

package qwen4exp

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

type unixDirectIOHandle struct {
	fd       int
	file     *os.File
	isDirect bool
	path     string
}

// OpenDirectIO opens the file at path for unbuffered Direct I/O.
// On Linux, it attempts O_DIRECT (0x4000). If rejected (e.g. WSL mounts, tmpfs,
// macOS), it falls back to standard os.OpenFile while reporting IsDirect() == false.
func OpenDirectIO(path string) (DirectIOHandle, error) {
	const oDirect = 0x4000
	fd, err := syscall.Open(path, syscall.O_RDONLY|oDirect, 0)
	if err == nil {
		return &unixDirectIOHandle{
			fd:       fd,
			isDirect: true,
			path:     path,
		}, nil
	}

	// Fallback to standard os.OpenFile
	f, fallbackErr := os.OpenFile(path, os.O_RDONLY, 0)
	if fallbackErr != nil {
		return nil, fmt.Errorf("direct_io: open failed (direct: %v, fallback: %w)", err, fallbackErr)
	}
	return &unixDirectIOHandle{
		fd:       int(f.Fd()),
		file:     f,
		isDirect: false,
		path:     path,
	}, nil
}

func (h *unixDirectIOHandle) ReadAtAligned(dest []byte, fileOffset int64) (int, error) {
	if len(dest) == 0 {
		return 0, nil
	}
	if h.file != nil && !h.isDirect {
		return h.file.ReadAt(dest, fileOffset)
	}
	n, err := syscall.Pread(h.fd, dest, fileOffset)
	if err != nil {
		return n, fmt.Errorf("direct_io: pread at offset %d: %w", fileOffset, err)
	}
	return n, nil
}

func (h *unixDirectIOHandle) Close() error {
	if h.file != nil {
		return h.file.Close()
	}
	if h.fd >= 0 {
		err := syscall.Close(h.fd)
		h.fd = -1
		return err
	}
	return nil
}

func (h *unixDirectIOHandle) IsDirect() bool {
	return h.isDirect
}

func (h *unixDirectIOHandle) Path() string {
	return h.path
}

type unixPinnedMemory struct {
	buf []byte
}

func AllocPinnedHostMemory(size int) (PinnedMemory, error) {
	alignedSize := int(AlignUp(int64(size), SectorSize))
	data, err := syscall.Mmap(-1, 0, alignedSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err == nil {
		_ = syscall.Mlock(data)
		return &unixPinnedMemory{buf: data}, nil
	}

	return allocAlignedFallback(alignedSize, SectorSize), nil
}

func (p *unixPinnedMemory) Bytes() []byte {
	return p.buf
}

func (p *unixPinnedMemory) Free() error {
	if len(p.buf) > 0 {
		_ = syscall.Munlock(p.buf)
		err := syscall.Munmap(p.buf)
		p.buf = nil
		return err
	}
	return nil
}

type alignedGoPinnedMemory struct {
	backing []byte
	buf     []byte
}

func allocAlignedFallback(size int, align int) PinnedMemory {
	backing := make([]byte, size+align)
	addr := uintptr(unsafe.Pointer(&backing[0]))
	offset := int(uintptr(AlignUp(int64(addr), int64(align))) - addr)
	return &alignedGoPinnedMemory{
		backing: backing,
		buf:     backing[offset : offset+size],
	}
}

func (m *alignedGoPinnedMemory) Bytes() []byte {
	return m.buf
}

func (m *alignedGoPinnedMemory) Free() error {
	m.buf = nil
	m.backing = nil
	return nil
}
