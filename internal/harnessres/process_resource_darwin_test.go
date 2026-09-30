package harnessres

import (
	"math"
	"os"
	"syscall"
	"testing"
)

// Check the public per-PID seam against the OS counter, not a presence-only
// fixture: each observation must fit between direct self getrusage readings.
// fak-test:runtime fast est=10ms lane=default
func TestReadProcessResourceDarwinSelfCPU(t *testing.T) {
	t.Parallel()
	directCPU := func() float64 {
		t.Helper()
		var usage syscall.Rusage
		if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
			t.Fatalf("getrusage self: %v", err)
		}
		return float64(usage.Utime.Sec+usage.Stime.Sec) + float64(usage.Utime.Usec+usage.Stime.Usec)/1e6
	}
	previous := 0.0
	for sample := 0; sample < 8; sample++ {
		before := directCPU()
		got, ok := ReadProcessResource(os.Getpid())
		after := directCPU()
		if !ok || !got.HaveCPU {
			t.Fatalf("self process CPU must be available on Darwin: ok=%v resource=%+v", ok, got)
		}
		if math.IsNaN(got.CPUSeconds) || math.IsInf(got.CPUSeconds, 0) || got.CPUSeconds < 0 {
			t.Fatalf("self CPU counter must be finite and nonnegative: %v", got.CPUSeconds)
		}
		// Getrusage has microsecond resolution; permit one tick of floating
		// conversion difference without accepting a stale or fabricated counter.
		if got.CPUSeconds < before-1e-6 || got.CPUSeconds > after+1e-6 {
			t.Fatalf("self CPU=%g outside direct getrusage bounds [%g,%g]", got.CPUSeconds, before, after)
		}
		if got.CPUSeconds < previous {
			t.Fatalf("self CPU counter decreased: previous=%g current=%g", previous, got.CPUSeconds)
		}
		previous = got.CPUSeconds
	}
	// Self support must not fabricate a per-PID reader for another process.
	for _, pid := range []int{os.Getppid(), 0, -1} {
		got, ok := ReadProcessResource(pid)
		if ok || got.HaveCPU {
			t.Errorf("foreign/invalid PID %d must remain unavailable: ok=%v resource=%+v", pid, ok, got)
		}
	}
}
