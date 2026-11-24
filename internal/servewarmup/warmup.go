// Package servewarmup is the serve-side reporting seam of the gateway's backend
// warmup gate: the gateway owns readiness policy and the bounded wait, this
// package writes a failed or timed-out warmup to the operator's stderr.
package servewarmup

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"
)

// FailureModeEnv=hold keeps the serve up, readiness held as warmup_failed, after a
// terminal warmup failure; unset or any other value lets the failure end the process.
const FailureModeEnv = "FAK_SERVE_WARMUP_FAILURE"

// Run invokes the backend startup warmup once and reports a non-nil result to
// stderr as `fak serve: backend startup warmup failed after <d>; readiness held
// as warmup_failed: <err>`; a successful warmup stays quiet because the gateway's
// MarkWarmupComplete already logged the [READY] line. This restores the failure
// reporting 42603583d6's refactor dropped: without it a hung warmup holds
// readiness pending with no journal line, the undiagnosable state #13500 exists to end.
// It returns the warmup error so the caller can apply Fatal.
func Run(ctx context.Context, run func(context.Context) (time.Duration, error), stderr io.Writer) error {
	elapsed, err := run(ctx)
	if err == nil {
		return nil
	}
	if stderr == nil {
		stderr = io.Discard
	}
	fmt.Fprintf(stderr, "fak serve: backend startup warmup failed after %s; readiness held as warmup_failed: %v\n", elapsed.Round(time.Millisecond), err)
	return err
}

// Fatal reports whether a warmup result must end the serve. Nothing re-runs a failed
// warmup, so a serve left up would answer 503 forever; a supervisor restarts an exited
// one. A shutdown already in progress is not a warmup failure, and hold is the
// operator's explicit opt-out.
func Fatal(ctx context.Context, err error, getenv func(string) string) bool {
	if err == nil || ctx.Err() != nil {
		return false
	}
	return getenv == nil || !strings.EqualFold(strings.TrimSpace(getenv(FailureModeEnv)), "hold")
}
