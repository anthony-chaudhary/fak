// Package linefmt creates newline-terminated formatted-text writers.
package linefmt

import (
	"fmt"
	"io"
)

// Writer returns a printf-style function that appends one newline per call.
func Writer(dst io.Writer) func(format string, args ...any) {
	return func(format string, args ...any) { fmt.Fprintf(dst, format+"\n", args...) }
}
