//go:build !(darwin && arm64 && cgo)

package metalgemm

import (
	"errors"
	"os"
	"unsafe"
)

var (
	// ErrMetalUnavailable indicates Metal hardware or runtime is not available.
	ErrMetalUnavailable = errors.New("metalgemm: Metal backend is unavailable")
	// ErrUnalignedMemory indicates pointer or size is not aligned to the system page size.
	ErrUnalignedMemory = errors.New("metalgemm: memory pointer or length is not page-aligned")
)

// ZeroCopyBuffer represents an inert buffer handle in stub builds.
type ZeroCopyBuffer struct{}

// Q4KWeight is an inert Q4_K weight handle in stub builds. It preserves the
// public shape of the Metal-backed handle for portable callers while no Metal
// resource is present.
type Q4KWeight struct {
	Out, In int
}

// NewZeroCopyBuffer is a stub returning ErrMetalUnavailable.
func NewZeroCopyBuffer(span []byte) (*ZeroCopyBuffer, error) {
	return nil, ErrMetalUnavailable
}

// IsShared reports false in stub builds.
func (b *ZeroCopyBuffer) IsShared() bool {
	return false
}

// Length reports 0 in stub builds.
func (b *ZeroCopyBuffer) Length() int {
	return 0
}

// Contents returns nil in stub builds.
func (b *ZeroCopyBuffer) Contents() unsafe.Pointer {
	return nil
}

// Free is a no-op in stub builds.
func (b *ZeroCopyBuffer) Free() {}

// CalculatePageAlignment computes the page-aligned base offset and length for a tensor.
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

// UploadZeroCopyQ4KTensor returns ErrMetalUnavailable in stub builds.
func UploadZeroCopyQ4KTensor(mapped []byte, fileOffset int64, out, in int) (*Q4KWeight, error) {
	return nil, ErrMetalUnavailable
}

// IsMetalSharedBuffer reports false in stub builds.
func (w *Q4KWeight) IsMetalSharedBuffer() bool {
	return false
}
