package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// The route seam is deliberately restored as soon as the bounded call returns. A timed-out
// worker may still be unwinding, so the production wrapper must snapshot the seam before
// launch rather than reread this package global after the timeout (race witness for #6860).
func TestDispatchWaveRouteIssuesBoundedReturnsTypedTimeout(t *testing.T) {
	old := dispatchRouteIssues
	t.Cleanup(func() { dispatchRouteIssues = old })
	release := make(chan struct{})
	dispatchRouteIssues = func(string, io.Writer) (dispatchtick.RouterPayload, error) {
		<-release
		return dispatchtick.RouterPayload{}, nil
	}
	t.Cleanup(func() { close(release) })

	started := time.Now()
	_, err := dispatchWaveRouteIssuesBounded(t.TempDir(), io.Discard, 20*time.Millisecond)
	if err == nil || !strings.Contains(err.Error(), "issue-contract discovery timed out after 20ms") {
		t.Fatalf("error = %v, want typed issue-contract timeout", err)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("bounded discovery returned after %s", elapsed)
	}
}

func TestDispatchWavePreflightBudgetExceedsPlanningBudget(t *testing.T) {
	preflightBudget := 3 * dispatchWaveDependencyTimeout
	// The live preflight has supported probes with a 60-second ceiling. The outer wave
	// budget must not manufacture WAVE_EMPTY before those probes can return.
	if preflightBudget <= 60*time.Second {
		t.Fatalf("preflight timeout = %s, must cover supported 60s probes", preflightBudget)
	}
}

func TestDispatchWaveBoundedDependencyTimesOut(t *testing.T) {
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	_, err := dispatchWaveDependency(20*time.Millisecond, "dispatch preflight", func() (dispatchWavePreflightResult, error) {
		<-release
		return dispatchWavePreflightResult{}, nil
	})
	if err == nil || err.Error() != "dispatch preflight timed out after 20ms" {
		t.Fatalf("error = %v, want typed preflight timeout", err)
	}
	var dep *dispatchWaveDependencyError
	if !errors.As(err, &dep) || dep.Kind != "timeout" || dep.Dependency != "dispatch preflight" || !dep.Retryable || dep.Attempts != 1 {
		t.Fatalf("typed error = %#v, want retryable dispatch preflight timeout", dep)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want errors.Is(context.DeadlineExceeded)", err)
	}
}

func TestDispatchWaveBoundedDependencyPreservesUpstreamError(t *testing.T) {
	upstream := errors.New("upstream refused")
	_, err := dispatchWaveDependency(time.Second, "account allocation", func() (int, error) {
		return 0, upstream
	})
	if !errors.Is(err, upstream) {
		t.Fatalf("error = %v, want wrapped %v", err, upstream)
	}
	var dep *dispatchWaveDependencyError
	if !errors.As(err, &dep) || dep.Kind != "upstream" || dep.Dependency != "account allocation" || dep.Retryable {
		t.Fatalf("typed error = %#v, want non-retryable upstream allocation error", dep)
	}
}

func TestDispatchWaveBoundedDependencyReturnsCompletedValue(t *testing.T) {
	got, err := dispatchWaveDependency(time.Second, "account allocation", func() (int, error) {
		return 7, nil
	})
	if err != nil || got != 7 {
		t.Fatalf("got (%d, %v), want (7, nil)", got, err)
	}
}

func TestDispatchWaveDependencyRetryRecoversReadOnlyUpstreamFailure(t *testing.T) {
	attempts := 0
	got, err := dispatchWaveDependencyRetry(time.Second, "issue-contract discovery", 2, func(error) bool { return true }, func() (int, error) {
		attempts++
		if attempts == 1 {
			return 0, errors.New("temporary read failure")
		}
		return 7, nil
	})
	if err != nil || got != 7 || attempts != 2 {
		t.Fatalf("got (%d, %v), attempts=%d; want (7, nil), attempts=2", got, err, attempts)
	}
}

func TestDispatchWaveDependencyTimeoutIsNotRetriedWhileCallMayStillRun(t *testing.T) {
	release := make(chan struct{})
	finished := make(chan struct{})
	attempts := 0
	_, err := dispatchWaveDependencyRetry(20*time.Millisecond, "issue-contract discovery", 2, func(error) bool { return true }, func() (int, error) {
		defer close(finished)
		attempts++
		<-release
		return 0, nil
	})
	close(release)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("timed-out dependency helper did not stop after release")
	}
	if err == nil || attempts != 1 {
		t.Fatalf("error = %v, attempts=%d; want timeout without overlapping retry", err, attempts)
	}
}

func TestDispatchWaveOutcomeClassifiesDependencyFailures(t *testing.T) {
	for _, tc := range []struct {
		kind, verdict, action string
		retryable             bool
	}{
		{kind: "timeout", verdict: "WAVE_DEPENDENCY_TIMEOUT", action: "retryable_error", retryable: true},
		{kind: "upstream", verdict: "WAVE_DEPENDENCY_ERROR", action: "error"},
		{kind: "internal", verdict: "WAVE_INTERNAL_ERROR", action: "error"},
	} {
		rec := map[string]any{"failure_class": tc.kind, "retryable": tc.retryable}
		verdict, action := dispatchWaveOutcome(rec)
		if verdict != tc.verdict || action != tc.action {
			t.Fatalf("kind %s => %s/%s, want %s/%s", tc.kind, verdict, action, tc.verdict, tc.action)
		}
	}
}

func TestDispatchWaveRecordDependencyErrorPublishesMachineFields(t *testing.T) {
	rec := map[string]any{}
	dispatchWaveRecordDependencyError(rec, &dispatchWaveDependencyError{
		Dependency: "dispatch preflight", Kind: "timeout", Attempts: 1, Retryable: true, Timeout: 90 * time.Second,
	})
	if rec["failure_class"] != "timeout" || rec["dependency"] != "dispatch preflight" || rec["retryable"] != true || rec["retry_disposition"] != "safe_to_retry" || rec["attempts"] != 1 || rec["timeout_ms"] != int64(90000) {
		t.Fatalf("record = %#v", rec)
	}
}

func TestDispatchWaveDependencyRetryPropagatesFinalUpstreamCause(t *testing.T) {
	first := errors.New("first transient")
	final := errors.New("final upstream detail")
	attempts := 0
	_, err := dispatchWaveDependencyRetry(time.Second, "issue-contract discovery", 2, func(error) bool { return true }, func() (int, error) {
		attempts++
		if attempts == 1 {
			return 0, first
		}
		return 0, final
	})
	if !errors.Is(err, final) || errors.Is(err, first) {
		t.Fatalf("error = %v, want final upstream cause only", err)
	}
	var dep *dispatchWaveDependencyError
	if !errors.As(err, &dep) || dep.Attempts != 2 || !dep.Retryable || dep.Kind != "upstream" {
		t.Fatalf("typed error = %#v, want exhausted retryable upstream error", dep)
	}
	rec := map[string]any{}
	dispatchWaveRecordDependencyError(rec, err)
	if rec["cause"] != final.Error() || rec["attempts"] != 2 || rec["failure_class"] != "upstream" {
		t.Fatalf("record = %#v, want propagated final cause", rec)
	}
}
