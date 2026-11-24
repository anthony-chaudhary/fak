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
		"readiness held as warmup_failed",
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

// A failed warmup is terminal (nothing re-runs it), so it must end the serve instead of
// leaving /readyz at 503 forever; shutdown-in-progress and the hold opt-out are not fatal.
func TestFatalPolicy(t *testing.T) {
	boom := errors.New("boom")
	live := context.Background()
	gone, cancel := context.WithCancel(live)
	cancel()
	env := func(v string) func(string) string {
		return func(k string) string {
			if k == FailureModeEnv {
				return v
			}
			return ""
		}
	}
	for _, c := range []struct {
		name string
		ctx  context.Context
		err  error
		mode string
		want bool
	}{
		{"failure is fatal", live, boom, "", true},
		{"nil getenv is fatal", live, boom, "-", true},
		{"success", live, nil, "", false},
		{"shutdown in progress", gone, boom, "", false},
		{"hold opt-out", live, boom, " HOLD ", false},
	} {
		getenv := env(c.mode)
		if c.mode == "-" {
			getenv = nil
		}
		if got := Fatal(c.ctx, c.err, getenv); got != c.want {
			t.Fatalf("%s: Fatal = %v, want %v", c.name, got, c.want)
		}
	}
	if err := Run(live, func(context.Context) (time.Duration, error) { return 0, boom }, nil); !errors.Is(err, boom) {
		t.Fatalf("Run returned %v, want the warmup error", err)
	}
}
