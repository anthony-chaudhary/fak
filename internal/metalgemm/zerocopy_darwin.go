//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework CoreFoundation -framework Accelerate
#include <stdlib.h>
#include <stddef.h>

void*  mg_zerocopy_create_shared_buffer(void* ptr, size_t length);
int    mg_zerocopy_buffer_is_shared(void* handle);
size_t mg_zerocopy_buffer_length(void* handle);
void*  mg_zerocopy_buffer_contents(void* handle);
void   mg_zerocopy_buffer_free(void* handle);
int    mg_zerocopy_upload_span(const unsigned char* raw, size_t nbytes, size_t offset, int out, int in);
int    mg_q4k_upload(const unsigned char* raw, int out, int in);
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

var (
	// ErrMetalUnavailable indicates Metal hardware or runtime is not available.
	ErrMetalUnavailable = errors.New("metalgemm: Metal backend is unavailable")
	// ErrUnalignedMemory indicates pointer or size is not aligned to the system page size.
	ErrUnalignedMemory = errors.New("metalgemm: memory pointer or length is not page-aligned")
)

// ZeroCopyBuffer represents an Apple Silicon unified memory buffer mapped directly
// into Metal using MTLResourceStorageModeShared with newBufferWithBytesNoCopy.
type ZeroCopyBuffer struct {
	handle unsafe.Pointer
	ptr    unsafe.Pointer
	length int
}

// NewZeroCopyBuffer wraps a page-aligned memory span into an MTLBuffer with MTLResourceStorageModeShared.
// The caller must ensure that span remains valid and pinned/unmoved for the lifetime of the buffer.
func NewZeroCopyBuffer(span []byte) (*ZeroCopyBuffer, error) {
	if !Available() {
		return nil, ErrMetalUnavailable
	}
	if len(span) == 0 {
		return nil, errors.New("metalgemm: empty span for zero-copy buffer")
	}
	pageSize := os.Getpagesize()
	base := unsafe.Pointer(unsafe.SliceData(span))
	if uintptr(base)%uintptr(pageSize) != 0 || len(span)%pageSize != 0 {
		return nil, ErrUnalignedMemory
	}
	handle := C.mg_zerocopy_create_shared_buffer(base, C.size_t(len(span)))
	if handle == nil {
		return nil, fmt.Errorf("metalgemm: Metal rejected no-copy shared buffer for %d bytes", len(span))
	}
	buf := &ZeroCopyBuffer{
		handle: handle,
		ptr:    base,
		length: len(span),
	}
	runtime.SetFinalizer(buf, (*ZeroCopyBuffer).Free)
	return buf, nil
}

// IsShared reports whether the underlying Metal buffer was allocated with MTLResourceStorageModeShared.
func (b *ZeroCopyBuffer) IsShared() bool {
	if b == nil || b.handle == nil {
		return false
	}
	return C.mg_zerocopy_buffer_is_shared(b.handle) == 1
}

// Length returns the length of the buffer in bytes.
func (b *ZeroCopyBuffer) Length() int {
	if b == nil {
		return 0
	}
	return b.length
}

// Contents returns the CPU-accessible pointer to the buffer's contents.
func (b *ZeroCopyBuffer) Contents() unsafe.Pointer {
	if b == nil || b.handle == nil {
		return nil
	}
	return C.mg_zerocopy_buffer_contents(b.handle)
}

// Free explicitly releases the Metal buffer handle.
func (b *ZeroCopyBuffer) Free() {
	if b != nil && b.handle != nil {
		C.mg_zerocopy_buffer_free(b.handle)
		b.handle = nil
		runtime.SetFinalizer(b, nil)
	}
}

// CalculatePageAlignment computes the page-aligned base offset and length for a tensor
// starting at fileOffset with payloadBytes.
func CalculatePageAlignment(fileOffset int64, payloadBytes int) (pageOffset int, baseOffset int64, alignedLength int64) {
	page := int64(os.Getpagesize())
	if page <= 0 {
		page = 16384
	}
	pageOffset = int(fileOffset % page)
	baseOffset = fileOffset - int64(pageOffset)
	alignedLength = (int64(payloadBytes) + int64(pageOffset) + page - 1) & ^(page - 1)
	return pageOffset, baseOffset, alignedLength
}

// UploadZeroCopyQ4KTensor maps a GGUF Q4_K tensor at fileOffset directly into a Metal shared buffer
// over the memory-mapped file pages.
// It automatically detects page alignment: if page-aligned, it constructs a zero-copy
// MTLResourceStorageModeShared buffer via newBufferWithBytesNoCopy; if misaligned, it falls back
// to a copied resident buffer.
func UploadZeroCopyQ4KTensor(mapped []byte, fileOffset int64, out, in int) (*Q4KWeight, error) {
	if !Available() {
		return nil, ErrMetalUnavailable
	}
	if in <= 0 || in%256 != 0 || out <= 0 {
		return nil, fmt.Errorf("metalgemm: invalid dimensions out=%d, in=%d", out, in)
	}
	need := (in / 256) * out * 144
	if fileOffset < 0 || fileOffset+int64(need) > int64(len(mapped)) {
		return nil, fmt.Errorf("metalgemm: tensor range [%d, %d) out of mapped bounds %d", fileOffset, fileOffset+int64(need), len(mapped))
	}

	page := os.Getpagesize()
	base := unsafe.Pointer(unsafe.SliceData(mapped))
	baseAligned := uintptr(base)%uintptr(page) == 0
	offsetAligned := fileOffset%32 == 0

	if baseAligned && offsetAligned {
		spanLen := len(mapped) - len(mapped)%page
		if fileOffset+int64(need) <= int64(spanLen) {
			span := mapped[:spanLen:spanLen]
			wid := int(C.mg_zerocopy_upload_span((*C.uchar)(base), C.size_t(len(span)), C.size_t(fileOffset), C.int(out), C.int(in)))
			if wid >= 0 {
				return &Q4KWeight{id: C.int(wid), Out: out, In: in, noCopy: true}, nil
			}
		}
	}

	// P4 Operations: fallback to copied buffer if misaligned or span upload failed
	raw := mapped[fileOffset : fileOffset+int64(need)]
	wid := int(C.mg_q4k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in)))
	if wid < 0 {
		return nil, fmt.Errorf("metalgemm: fallback copy upload failed for tensor (%d x %d)", out, in)
	}
	runtime.KeepAlive(raw)
	return &Q4KWeight{id: C.int(wid), Out: out, In: in, noCopy: false}, nil
}

// IsMetalSharedBuffer reports whether a Q4KWeight is backed by an MTLBuffer allocated
// with MTLResourceStorageModeShared.
func (w *Q4KWeight) IsMetalSharedBuffer() bool {
	return w != nil && w.id >= 0 && w.noCopy && Available()
}
