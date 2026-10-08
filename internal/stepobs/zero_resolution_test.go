package stepobs

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestZeroResolutionRecognizedTimerDomainIsCountOnly(t *testing.T) {
	zero := New()
	zero.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", 0))
	snap := zero.Snapshot()
	if !snap.KernelObserved || snap.KernelEvents != 1 || len(snap.KernelLatency) != 1 {
		t.Fatalf("zero-resolution event was lost: observed=%t events=%d series=%d", snap.KernelObserved, snap.KernelEvents, len(snap.KernelLatency))
	}
	stat := snap.KernelLatency[0]
	if stat.TimerDomain != "host_monotonic" || stat.Overflow || stat.Count != 1 {
		t.Fatalf("zero-resolution attribution timer_domain=%q overflow=%t count=%d", stat.TimerDomain, stat.Overflow, stat.Count)
	}
	if stat.Measured || stat.TotalSeconds != 0 || stat.MeanSeconds != 0 || stat.P50Seconds != 0 || stat.P95Seconds != 0 || stat.MaxSeconds != 0 {
		t.Fatalf("zero-resolution measured=%t total=%g mean=%g p50=%g p95=%g max=%g", stat.Measured, stat.TotalSeconds, stat.MeanSeconds, stat.P50Seconds, stat.P95Seconds, stat.MaxSeconds)
	}

	positive := New()
	positive.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", time.Nanosecond))
	stat = positive.Snapshot().KernelLatency[0]
	if !stat.Measured || stat.Count != 1 || stat.TotalSeconds <= 0 || stat.MeanSeconds <= 0 || stat.MaxSeconds <= 0 {
		t.Fatalf("positive timer measured=%t count=%d total=%g mean=%g max=%g", stat.Measured, stat.Count, stat.TotalSeconds, stat.MeanSeconds, stat.MaxSeconds)
	}
}
