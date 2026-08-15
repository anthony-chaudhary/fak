package main

import (
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

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
}

func TestDispatchWaveBoundedDependencyPreservesUpstreamError(t *testing.T) {
	upstream := errors.New("upstream refused")
	_, err := dispatchWaveDependency(time.Second, "account allocation", func() (int, error) {
		return 0, upstream
	})
	if !errors.Is(err, upstream) {
		t.Fatalf("error = %v, want %v", err, upstream)
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
