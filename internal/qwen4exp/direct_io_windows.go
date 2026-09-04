//go:build windows

package qwen4exp

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

const (
	fileFlagNoBuffering  = 0x20000000
	fileFlagRandomAccess = 0x10000000
	memCommit            = 0x00001000
	memReserve           = 0x00002000
	memRelease           = 0x00008000
	pageReadWrite        = 0x04
)

var (
	kernel32          = syscall.NewLazyDLL("kernel32.dll")
	procVirtualAlloc  = kernel32.NewProc("VirtualAlloc")
	procVirtualLock   = kernel32.NewProc("VirtualLock")
	procVirtualUnlock = kernel32.NewProc("VirtualUnlock")
	procVirtualFree   = kernel32.NewProc("VirtualFree")
)

type windowsDirectIOHandle struct {
	handle   syscall.Handle
	file     *os.File
	isDirect bool
	path     string
}

// OpenDirectIO opens the file at path for unbuffered Direct I/O.
// If the underlying filesystem rejects FILE_FLAG_NO_BUFFERING, it falls back
// to standard os.OpenFile to preserve functionality while reporting IsDirect() == false.
func OpenDirectIO(path string) (DirectIOHandle, error) {
	pathPtr, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, fmt.Errorf("direct_io: utf16 path: %w", err)
	}

	flags := uint32(syscall.FILE_ATTRIBUTE_NORMAL | fileFlagNoBuffering | fileFlagRandomAccess)
	h, err := syscall.CreateFile(
		pathPtr,
		syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ,
		nil,
		syscall.OPEN_EXISTING,
		flags,
		0,
	)
	if err == nil {
		return &windowsDirectIOHandle{
			handle:   h,
			isDirect: true,
			path:     path,
		}, nil
	}

	// Fallback to standard os.OpenFile
	f, fallbackErr := os.OpenFile(path, os.O_RDONLY, 0)
	if fallbackErr != nil {
		return nil, fmt.Errorf("direct_io: open failed (direct: %v, fallback: %w)", err, fallbackErr)
	}
	return &windowsDirectIOHandle{
		handle:   syscall.Handle(f.Fd()),
		file:     f,
		isDirect: false,
		path:     path,
	}, nil
}

func (h *windowsDirectIOHandle) ReadAtAligned(dest []byte, fileOffset int64) (int, error) {
	if len(dest) == 0 {
		return 0, nil
	}
	if h.file != nil && !h.isDirect {
		return h.file.ReadAt(dest, fileOffset)
	}

	var overlapped syscall.Overlapped
	overlapped.Offset = uint32(fileOffset)
	overlapped.OffsetHigh = uint32(fileOffset >> 32)

	var bytesRead uint32
	err := syscall.ReadFile(h.handle, dest, &bytesRead, &overlapped)
	if err != nil && err != syscall.ERROR_HANDLE_EOF {
		return int(bytesRead), fmt.Errorf("direct_io: ReadFile at offset %d: %w", fileOffset, err)
	}
	return int(bytesRead), nil
}

func (h *windowsDirectIOHandle) Close() error {
	if h.file != nil {
		return h.file.Close()
	}
	if h.handle != syscall.InvalidHandle {
		err := syscall.CloseHandle(h.handle)
		h.handle = syscall.InvalidHandle
		return err
	}
	return nil
}

func (h *windowsDirectIOHandle) IsDirect() bool {
	return h.isDirect
}

func (h *windowsDirectIOHandle) Path() string {
	return h.path
}

type windowsPinnedMemory struct {
	addr uintptr
	size int
	buf  []byte
}

func AllocPinnedHostMemory(size int) (PinnedMemory, error) {
	alignedSize := int(AlignUp(int64(size), SectorSize))
	addr, _, _ := procVirtualAlloc.Call(
		0,
		uintptr(alignedSize),
		uintptr(memCommit|memReserve),
		uintptr(pageReadWrite),
	)
	if addr == 0 {
		return allocAlignedFallback(alignedSize, SectorSize), nil
	}

	procVirtualLock.Call(addr, uintptr(alignedSize))

	var slice []byte
	hdr := (*struct {
		Data uintptr
		Len  int
		Cap  int
	})(unsafe.Pointer(&slice))
	hdr.Data = addr
	hdr.Len = alignedSize
	hdr.Cap = alignedSize

	return &windowsPinnedMemory{
		addr: addr,
		size: alignedSize,
		buf:  slice,
	}, nil
}

func (p *windowsPinnedMemory) Bytes() []byte {
	return p.buf
}

func (p *windowsPinnedMemory) Free() error {
	if p.addr != 0 {
		procVirtualUnlock.Call(p.addr, uintptr(p.size))
		procVirtualFree.Call(p.addr, 0, uintptr(memRelease))
		p.addr = 0
		p.buf = nil
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
