//go:build darwin

package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"syscall"
)

// mappedQ4KSpanOwner retains both the file and its page-aligned mapping. bytes
// aliases read-only OS memory and remains valid only while this owner is open.
type mappedQ4KSpanOwner struct {
	file          *os.File
	mapped        []byte
	logical       []byte
	logicalOffset int64
	mappedOffset  int64
	closeOnce     sync.Once
	closeErr      error
}

// openMappedQ4KSpan maps exactly the pages covering [offset, offset+length).
// The returned logical slice is capacity-capped so consumers cannot reach the
// alignment padding. The owner, rather than the slice, carries lifetime.
func openMappedQ4KSpan(path string, offset int64, length int) (*mappedQ4KSpanOwner, error) {
	if err := validateMappedQ4KSpanNumbers(offset, length, -1); err != nil {
		return nil, err
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*mappedQ4KSpanOwner, error) {
		_ = f.Close()
		return nil, err
	}

	st, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	fileSize := st.Size()
	if err := validateMappedQ4KSpanNumbers(offset, length, fileSize); err != nil {
		return fail(err)
	}

	page := int64(os.Getpagesize())
	mappedOffset := offset - offset%page
	logicalEnd := offset + int64(length)
	mappedEnd, ok := mappedQ4KSpanAlignUp(logicalEnd, page)
	if !ok {
		return fail(&mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: fileSize, Reason: "page-aligned end overflows int64"})
	}
	mappedLength64 := mappedEnd - mappedOffset
	if mappedLength64 <= 0 || int64(int(mappedLength64)) != mappedLength64 {
		return fail(&mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: fileSize, Reason: "page-aligned span exceeds host int range"})
	}

	mapped, err := syscall.Mmap(int(f.Fd()), mappedOffset, int(mappedLength64), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return fail(fmt.Errorf("model: mmap Q4_K span: %w", err))
	}
	logicalStart := int(offset - mappedOffset)
	logicalEndIndex := logicalStart + length
	owner := &mappedQ4KSpanOwner{
		file:          f,
		mapped:        mapped,
		logical:       mapped[logicalStart:logicalEndIndex:logicalEndIndex],
		logicalOffset: offset,
		mappedOffset:  mappedOffset,
	}
	return owner, nil
}

func validateMappedQ4KSpanNumbers(offset int64, length int, fileSize int64) error {
	reason := ""
	switch {
	case offset < 0:
		reason = "negative offset"
	case length <= 0:
		reason = "length must be positive"
	case int64(length) <= 0 || offset > math.MaxInt64-int64(length):
		reason = "logical end overflows int64"
	case fileSize >= 0 && offset+int64(length) > fileSize:
		reason = "logical range exceeds file size"
	}
	if reason != "" {
		return &mappedQ4KSpanRangeError{Offset: offset, Length: length, FileSize: fileSize, Reason: reason}
	}
	return nil
}

func mappedQ4KSpanAlignUp(value, alignment int64) (int64, bool) {
	remainder := value % alignment
	if remainder == 0 {
		return value, true
	}
	delta := alignment - remainder
	if value > math.MaxInt64-delta {
		return 0, false
	}
	return value + delta, true
}

func (o *mappedQ4KSpanOwner) bytes() []byte {
	if o == nil {
		return nil
	}
	return o.logical
}

// Close invalidates the logical view, unmaps the pages, and closes the retained
// file exactly once. It still attempts the file close when munmap fails.
func (o *mappedQ4KSpanOwner) Close() error {
	if o == nil {
		return nil
	}
	o.closeOnce.Do(func() {
		mapped := o.mapped
		file := o.file
		o.logical = nil
		o.mapped = nil
		o.file = nil

		var unmapErr, closeErr error
		if mapped != nil {
			unmapErr = syscall.Munmap(mapped)
		}
		if file != nil {
			closeErr = file.Close()
		}
		o.closeErr = errors.Join(unmapErr, closeErr)
	})
	return o.closeErr
}
