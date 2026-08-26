//go:build !darwin

package model

import (
	"math"
	"os"
	"runtime"
	"sync"
)

// mappedQ4KSpanOwner preserves the platform-neutral API shape. Unsupported
// platforms never construct one; callers receive mappedQ4KSpanUnavailableError.
type mappedQ4KSpanOwner struct {
	file          *os.File
	mapped        []byte
	logical       []byte
	logicalOffset int64
	mappedOffset  int64
	closeOnce     sync.Once
	closeErr      error
}

func openMappedQ4KSpan(_ string, offset int64, length int) (*mappedQ4KSpanOwner, error) {
	if offset < 0 {
		return nil, &mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: -1, Reason: "negative offset"}
	}
	if length <= 0 {
		return nil, &mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: -1, Reason: "length must be positive"}
	}
	if int64(length) <= 0 || offset > math.MaxInt64-int64(length) {
		return nil, &mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: -1, Reason: "logical end overflows int64"}
	}
	return nil, &mappedQ4KSpanUnavailableError{GOOS: runtime.GOOS}
}

func (o *mappedQ4KSpanOwner) bytes() []byte {
	if o == nil {
		return nil
	}
	return o.logical
}

func (o *mappedQ4KSpanOwner) Close() error {
	if o == nil {
		return nil
	}
	o.closeOnce.Do(func() {
		o.logical = nil
		o.mapped = nil
	})
	return o.closeErr
}
