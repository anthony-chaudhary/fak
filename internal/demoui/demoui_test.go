package demoui

import (
	"bytes"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Probe must report a usable machine surface: at least one core, at least one matmul
// worker, at least the reference backend, and a non-empty human summary. This is the
// witness that a demo always has something honest to render about the hardware.
func TestProbeReportsMachine(t *testing.T) {
	hw := Probe()
	if hw.LogicalCores < 1 {
		t.Fatalf("logical cores = %d, want >= 1", hw.LogicalCores)
	}
	if hw.Workers < 1 {
		t.Fatalf("workers = %d, want >= 1", hw.Workers)
	}
	if len(hw.Backends) == 0 {
		t.Fatal("no compute backends registered (cpu-ref should always be present)")
	}
	if hw.Summary == "" {
		t.Fatal("empty hardware summary")
	}
	// On a default (untagged) build the only backend is the reference floor, so there
	// is no accelerator and the summary must say so rather than imply a GPU.
	if hw.Accelerator == "" && !strings.Contains(hw.Summary, "CPU") {
		t.Fatalf("CPU-only summary should mention CPU, got %q", hw.Summary)
	}
}

// Beat must tick repeatedly while work runs: a 350ms job with a 100ms cadence has to
// produce at least two heartbeats, which is the property the demos rely on to keep the
// screen alive (~1×/s) during a long blocking phase.
func TestBeatTicksDuringWork(t *testing.T) {
	var ticks int32
	Beat(80*time.Millisecond,
		func(time.Duration) { atomic.AddInt32(&ticks, 1) },
		func() { time.Sleep(320 * time.Millisecond) },
	)
	if got := atomic.LoadInt32(&ticks); got < 2 {
		t.Fatalf("ticks = %d, want >= 2 over a 320ms job at 80ms cadence", got)
	}
}

// Beat must not return until work is finished, even with a fast cadence — a caller
// that reads work's result right after Beat returns depends on this barrier.
func TestBeatWaitsForWork(t *testing.T) {
	var finished int32
	Beat(5*time.Millisecond,
		func(time.Duration) {},
		func() { time.Sleep(60 * time.Millisecond); atomic.StoreInt32(&finished, 1) },
	)
	if atomic.LoadInt32(&finished) != 1 {
		t.Fatal("Beat returned before work() finished")
	}
}

// A zero cadence opts out of ticking but still runs and waits on work.
func TestBeatZeroCadenceNoTicks(t *testing.T) {
	var ticks, finished int32
	Beat(0,
		func(time.Duration) { atomic.AddInt32(&ticks, 1) },
		func() { time.Sleep(20 * time.Millisecond); atomic.StoreInt32(&finished, 1) },
	)
	if atomic.LoadInt32(&ticks) != 0 {
		t.Fatalf("ticks = %d, want 0 at zero cadence", atomic.LoadInt32(&ticks))
	}
	if atomic.LoadInt32(&finished) != 1 {
		t.Fatal("Beat with zero cadence did not wait on work()")
	}
}

// Spinner must animate then leave a clean line (stop is idempotent).
func TestSpinnerAnimatesAndClears(t *testing.T) {
	var buf bytes.Buffer
	stop := Spinner(&buf, "Loading model")
	time.Sleep(300 * time.Millisecond)
	stop()
	stop() // idempotent — second call must not panic or write garbage
	out := buf.String()
	if !strings.Contains(out, "Loading model") {
		t.Fatalf("spinner output missing label: %q", out)
	}
	if !strings.Contains(out, "\r") {
		t.Fatal("spinner never rewrote its line (no carriage return)")
	}
}
