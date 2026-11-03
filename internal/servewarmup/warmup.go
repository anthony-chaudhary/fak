// Package servewarmup is the serve-side reporting seam of the gateway's backend
// warmup gate: the gateway owns readiness policy and the bounded wait, this
// package writes a failed or timed-out warmup to the operator's stderr.
package servewarmup

import (
	"context"
	"fmt"
	"io"
	"time"
)

// Run invokes the backend startup warmup once and reports a non-nil result to
// stderr as `fak serve: backend startup warmup failed after <d>; readiness held
// as warmup_failed: <err>`; a successful warmup stays quiet because the gateway's
// MarkWarmupComplete already logged the [READY] line. This restores the failure
// reporting 42603583d6's refactor dropped: without it a hung warmup holds
// readiness pending with no journal line, the undiagnosable state #13500 exists to end.
func Run(ctx context.Context, run func(context.Context) (time.Duration, error), stderr io.Writer) {
	elapsed, err := run(ctx)
	if err == nil {
		return
	}
	if stderr == nil {
		stderr = io.Discard
	}
	fmt.Fprintf(stderr, "fak serve: backend startup warmup failed after %s; readiness held as warmup_failed: %v\n", elapsed.Round(time.Millisecond), err)
}
