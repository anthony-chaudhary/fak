package shard

import (
	"testing"
	"time"
)

func TestLatencyTrackerSmallSample(t *testing.T) {
	tr := &opLatencyTracker{}
	for i := 0; i < 5; i++ {
		tr.record(time.Duration(i) * time.Microsecond)
	}
	if p := tr.p99(); p != 0 {
		t.Errorf("expected 0 for small sample, got %v", p)
	}
}

func TestLatencyTrackerUniform(t *testing.T) {
	tr := &opLatencyTracker{}
	for i := 0; i < 100; i++ {
		tr.record(time.Duration(i+1) * time.Microsecond)
	}
	p := tr.p99()
	// 99th percentile of 1..100us should be ~99us or 100us
	if p < 95*time.Microsecond || p > 105*time.Microsecond {
		t.Errorf("expected ~99-100us, got %v", p)
	}
}

func TestLatencyTrackerSkewed(t *testing.T) {
	tr := &opLatencyTracker{}
	// 99 fast ops + 1 slow op
	for i := 0; i < 99; i++ {
		tr.record(10 * time.Microsecond)
	}
	tr.record(10 * time.Millisecond) // outlier
	p := tr.p99()
	// p99 should be the slow one
	if p < 1*time.Millisecond {
		t.Errorf("expected p99 >= 1ms (outlier), got %v", p)
	}
}

func TestLatencyTrackerWraparound(t *testing.T) {
	tr := &opLatencyTracker{}
	// Fill ring twice
	for i := 0; i < latencyRingSize*2; i++ {
		tr.record(100 * time.Microsecond)
	}
	p := tr.p99()
	if p != 100*time.Microsecond {
		t.Errorf("expected 100us, got %v", p)
	}
}

func TestLatencyTrackerReset(t *testing.T) {
	tr := &opLatencyTracker{}
	for i := 0; i < 50; i++ {
		tr.record(time.Duration(i) * time.Microsecond)
	}
	tr.reset()
	if p := tr.p99(); p != 0 {
		t.Errorf("expected 0 after reset, got %v", p)
	}
}

func TestLatencyTrackerP50(t *testing.T) {
	tr := &opLatencyTracker{}
	for i := 0; i < 100; i++ {
		tr.record(time.Duration(i+1) * time.Microsecond)
	}
	p := tr.p50()
	// 50th percentile of 1..100us should be ~50us
	if p < 45*time.Microsecond || p > 55*time.Microsecond {
		t.Errorf("expected ~50us, got %v", p)
	}
}

func TestLatencyTrackerPercentile(t *testing.T) {
	tr := &opLatencyTracker{}
	for i := 0; i < 200; i++ {
		tr.record(time.Duration(i+1) * time.Microsecond)
	}
	p75 := tr.percentile(75)
	// Ring wraps at 256 so only last 200 entries exist (1..200us).
	// 75th percentile should be ~150us
	if p75 < 140*time.Microsecond || p75 > 160*time.Microsecond {
		t.Errorf("expected ~150us for p75, got %v", p75)
	}
}

func TestOpLatencyTrackersRouting(t *testing.T) {
	lt := &opLatencyTrackers{}

	// Record GETs
	for i := 0; i < 20; i++ {
		lt.record(OpGet, 10*time.Microsecond)
	}
	// Record SETs
	for i := 0; i < 20; i++ {
		lt.record(OpSet, 20*time.Microsecond)
	}
	// Record EXISTS
	for i := 0; i < 20; i++ {
		lt.record(OpTest, 5*time.Microsecond)
	}

	// all tracker should have 60 samples
	if lt.all.count != 60 {
		t.Errorf("expected all.count=60, got %d", lt.all.count)
	}
	if lt.get.count != 20 {
		t.Errorf("expected get.count=20, got %d", lt.get.count)
	}
	if lt.set.count != 20 {
		t.Errorf("expected set.count=20, got %d", lt.set.count)
	}
	if lt.exists.count != 20 {
		t.Errorf("expected exists.count=20, got %d", lt.exists.count)
	}

	// OpMGet should route to get
	lt.record(OpMGet, 15*time.Microsecond)
	if lt.get.count != 21 {
		t.Errorf("expected get.count=21 after OpMGet, got %d", lt.get.count)
	}

	// OpMSet should route to set
	lt.record(OpMSet, 25*time.Microsecond)
	if lt.set.count != 21 {
		t.Errorf("expected set.count=21 after OpMSet, got %d", lt.set.count)
	}

	// OpDelete should only go to all
	lt.record(OpDelete, 3*time.Microsecond)
	if lt.all.count != 63 {
		t.Errorf("expected all.count=63, got %d", lt.all.count)
	}
	if lt.get.count != 21 || lt.set.count != 21 || lt.exists.count != 20 {
		t.Error("OpDelete should not route to get/set/exists")
	}
}
