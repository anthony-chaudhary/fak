package main

import (
	"context"
	"testing"
	"time"
)

func TestSpineRunsTenThousandLogicalContexts(t *testing.T) {
	r, err := run(context.Background(), config{Contexts: 10000, Workers: 64, Delay: time.Microsecond, Selfcheck: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || r.Completed != 10000 || r.Failed != 0 {
		t.Fatalf("bad report: %+v", r)
	}
	if r.SharedBaseInstalls != 1 {
		t.Fatalf("base installs=%d, want 1", r.SharedBaseInstalls)
	}
	if r.TurnCount != 10000 {
		t.Fatalf("turn count=%d, want 10000", r.TurnCount)
	}
	if r.PeakInFlight < 2 || r.PeakInFlight > 64 {
		t.Fatalf("peak=%d, want 2..64", r.PeakInFlight)
	}
}

func TestSpineRejectsInvalidDimensions(t *testing.T) {
	if _, err := run(context.Background(), config{}); err == nil {
		t.Fatal("expected invalid-dimensions error")
	}
}

func TestRunContextTimeout(t *testing.T) {
	ctx, cancel := overallDeadline(context.Background(), 0)
	defer cancel()
	if _, ok := ctx.Deadline(); ok {
		t.Fatal("disabled run timeout unexpectedly installed a deadline")
	}
	ctx, cancel = overallDeadline(context.Background(), time.Minute)
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Fatal("positive run timeout did not install a deadline")
	}
}
