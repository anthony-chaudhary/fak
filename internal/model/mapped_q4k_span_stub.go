//go:build !darwin

package model

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
)

type mappedQ4KSpanRangeError struct {
	Offset   int64
	Length   int
	FileSize int64
	Reason   string
}

func (e *mappedQ4KSpanRangeError) Error() string {
	return fmt.Sprintf("model: invalid mapped Q4_K span offset=%d length=%d file_size=%d: %s", e.Offset, e.Length, e.FileSize, e.Reason)
}

type mappedQ4KSpanUnavailableError struct {
	GOOS string
}

func (e *mappedQ4KSpanUnavailableError) Error() string {
	return fmt.Sprintf("model: mapped Q4_K spans unavailable on %s", e.GOOS)
}

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
