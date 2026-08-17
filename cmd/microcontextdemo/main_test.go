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
	if r.Resources.LogicalCPUs < 1 || r.Resources.WallMS <= 0 {
		t.Fatalf("missing resource dimensions: %+v", r.Resources)
	}
	if !r.Resources.CPUAvailable || r.Resources.CPUTotalMS <= 0 || r.Resources.CPUCoreEquivalent <= 0 {
		t.Fatalf("missing CPU accounting: %+v", r.Resources)
	}
	if r.Resources.HeapPeakBytes < r.Resources.HeapStartBytes || r.Resources.SysPeakBytes < r.Resources.SysStartBytes {
		t.Fatalf("invalid memory peaks: %+v", r.Resources)
	}
	if r.Resources.TotalAllocBytes == 0 || r.Resources.Mallocs == 0 || r.Resources.GoroutinesPeak < r.Resources.GoroutinesStart || r.Resources.Samples < 2 {
		t.Fatalf("incomplete runtime accounting: %+v", r.Resources)
	}
}

func TestSpineRejectsInvalidDimensions(t *testing.T) {
	r, err := run(context.Background(), config{})
	if err == nil {
		t.Fatal("expected invalid-dimensions error")
	}
	if r.Schema != "" || r.Resources.LogicalCPUs != 0 {
		t.Fatalf("error path returned partial report: %+v", r)
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
