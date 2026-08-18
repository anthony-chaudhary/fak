// Package demoassert records self-check failures for runnable demonstrations.
package demoassert

import (
	"fmt"
	"io"
)

// Recorder accumulates failures while allowing a demo to show its full proof.
type Recorder struct {
	Failed bool
	Writer io.Writer
}

// Fail records and prints one failure.
func (r *Recorder) Fail(format string, args ...any) {
	r.Failed = true
	fmt.Fprintf(r.Writer, "FAIL: "+format+"\n", args...)
}
