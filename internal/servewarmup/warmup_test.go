package servewarmup

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// TestRunReportsFailureAndPendingReadiness pins #13500: a non-nil warmup result
// (a backend error, or the gateway's bounded ErrWarmupTimeout when a non-returning
// forward is cut off) MUST be reported to stderr — otherwise readiness holds
// warmup_pending with no journal line, the undiagnosable strix1 state.
func TestRunReportsFailureAndPendingReadiness(t *testing.T) {
	backendErr := errors.New("backend startup warmup timed out")
	const elapsed = 600 * time.Second
	var stderr bytes.Buffer

	Run(context.Background(), func(context.Context) (time.Duration, error) {
		return elapsed, backendErr
	}, &stderr)

	got := stderr.String()
	for _, want := range []string{
		"backend startup warmup failed after 10m0s",
		"readiness remains pending",
		backendErr.Error(),
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("stderr = %q, want %q", got, want)
		}
	}
}

// TestRunSuccessIsQuiet pins that a successful warmup writes nothing — completion
// is already reported by the gateway's [READY] line, so this seam must stay silent.
func TestRunSuccessIsQuiet(t *testing.T) {
	var stderr bytes.Buffer
	Run(context.Background(), func(context.Context) (time.Duration, error) {
		return 1500 * time.Millisecond, nil
	}, &stderr)
	if stderr.Len() != 0 {
		t.Fatalf("successful warmup wrote %q, want quiet", stderr.String())
	}
}

// TestRunNilWriterDoesNotPanic pins that a nil stderr (a host that passes none) is
// tolerated rather than panicking on the failure path.
func TestRunNilWriterDoesNotPanic(t *testing.T) {
	Run(context.Background(), func(context.Context) (time.Duration, error) {
		return time.Second, errors.New("boom")
	}, nil)
}
