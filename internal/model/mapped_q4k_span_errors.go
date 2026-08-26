package model

import "fmt"

type mappedQ4KSpanRangeError struct {
	Offset   int64
	Length   int
	FileSize int64
	Reason   string
}

func (e *mappedQ4KSpanRangeError) Error() string {
	return fmt.Sprintf("model: invalid mapped Q4_K span offset=%d length=%d file_size=%d: %s", e.Offset, e.Length, e.FileSize, e.Reason)
}

// mappedQ4KSpanUnavailableError is shared across platforms so package callers
// can use errors.As without platform-specific source.
type mappedQ4KSpanUnavailableError struct {
	GOOS string
}

func (e *mappedQ4KSpanUnavailableError) Error() string {
	return fmt.Sprintf("model: mapped Q4_K spans unavailable on %s", e.GOOS)
}
