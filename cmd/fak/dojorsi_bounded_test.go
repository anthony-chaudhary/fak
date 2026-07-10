package main

import (
	"context"
	"errors"
	"os/exec"
	"testing"
	"time"
)

// TestRunBoundedDOSTimesOut proves the dos exec sites in dojorsi.go (the dos
// improve receipt observer and the admitDojoRSILane arbitrate) are bounded: a
// wedged child is killed at the deadline and the helper returns promptly with an
// error wrapping context.DeadlineExceeded, rather than blocking the loop for the
// child's full runtime. This is the missed-receipt / stuck-startup guard for
// issue #3474.
func TestRunBoundedDOSTimesOut(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not on PATH; timeout-bounding test needs a long-lived child")
	}
	start := time.Now()
	_, err := runBoundedDOS(50*time.Millisecond, "", "sleep", "30")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected a timeout error from a wedged dos child, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected an error wrapping context.DeadlineExceeded, got %v", err)
	}
	if elapsed > 10*time.Second {
		t.Fatalf("helper blocked %s; expected a prompt return after the 50ms deadline", elapsed)
	}
}

// TestRunBoundedDOSSucceeds proves the bounded wrapper does not corrupt the
// happy path: a fast child that finishes before the deadline returns its output
// with no error.
func TestRunBoundedDOSSucceeds(t *testing.T) {
	if _, err := exec.LookPath("echo"); err != nil {
		t.Skip("echo not on PATH")
	}
	out, err := runBoundedDOS(30*time.Second, "", "echo", "ok")
	if err != nil {
		t.Fatalf("fast child should succeed, got %v", err)
	}
	if len(out) == 0 {
		t.Fatal("expected output from echo, got none")
	}
}
