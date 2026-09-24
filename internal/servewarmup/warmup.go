// Package servewarmup is the serve-side boot-wiring half of the gateway's #3051
// backend warmup gate (internal/gateway/readiness_warmup.go). The gateway owns the
// readiness POLICY and the bounded wait (#13500); this package owns the small
// reporting seam so a failed or timed-out warmup is written to the operator's
// stderr instead of silently holding readiness warmup_pending forever.
package servewarmup

import (
	"context"
	"fmt"
	"io"
	"time"
)

// Run invokes the backend startup warmup once and reports a NON-nil result to
// stderr: an error (a backend failure, or the gateway's bounded
// gateway.ErrWarmupTimeout when the forward never returns, #13500) is surfaced as
// `fak serve: backend startup warmup failed after <d>; readiness remains pending:
// <err>`. A successful warmup is quiet — it is already reported by
// gateway.MarkWarmupComplete's [READY] line.
//
// This restores the failure reporting f3f056fb52 added and 42603583d6's refactor
// dropped when it moved the reporter into this helper but kept discarding the
// error. Without it a hung warmup holds readiness pending with NO journal line —
// the exact undiagnosable state #13500 exists to end.
func Run(ctx context.Context, run func(context.Context) (time.Duration, error), stderr io.Writer) {
	elapsed, err := run(ctx)
	if err == nil {
		return
	}
	if stderr == nil {
		stderr = io.Discard
	}
	fmt.Fprintf(stderr, "fak serve: backend startup warmup failed after %s; readiness remains pending: %v\n", elapsed.Round(time.Millisecond), err)
}
