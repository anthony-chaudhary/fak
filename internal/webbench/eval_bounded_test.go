package webbench

import (
	"context"
	"errors"
	"os/exec"
	"testing"
	"time"
)

// TestRunBoundedHarnessTimesOut proves the harness exec in RunEval is bounded:
// a child that would run far past the deadline is killed and the helper returns
// promptly with an error wrapping context.DeadlineExceeded, rather than blocking
// the caller for the child's full runtime. This is the wedged-browser guard for
// issue #3474 — the harness drives real browsers over the network with no
// natural deadline, so it must degrade to a timeout verdict.
func TestRunBoundedHarnessTimesOut(t *testing.T) {
	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not on PATH; timeout-bounding test needs a long-lived child")
	}
	start := time.Now()
	_, err := runBoundedHarness(50*time.Millisecond, "sleep", "30")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("expected a timeout error from a wedged child, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected an error wrapping context.DeadlineExceeded, got %v", err)
	}
	if elapsed > 10*time.Second {
		t.Fatalf("helper blocked %s; expected it to return promptly after the 50ms deadline", elapsed)
	}
}

// TestRunBoundedHarnessSucceeds proves the bounded wrapper does not corrupt the
// happy path: a fast child that finishes before the deadline returns its output
// with no error.
func TestRunBoundedHarnessSucceeds(t *testing.T) {
	if _, err := exec.LookPath("echo"); err != nil {
		t.Skip("echo not on PATH")
	}
	out, err := runBoundedHarness(30*time.Second, "echo", "passed 3 / 4")
	if err != nil {
		t.Fatalf("fast child should succeed, got %v", err)
	}
	if len(out) == 0 {
		t.Fatal("expected output from echo, got none")
	}
}
