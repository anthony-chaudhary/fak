package stepobs

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=1s — pure in-memory aggregation, no I/O, no model load.

func kernelEvent(kernel, backend, timerDomain string, d time.Duration) computetrace.Event {
	return computetrace.Event{
		Operation:   "matmul",
		Phase:       "kernel",
		Kernel:      kernel,
		Backend:     backend,
		Device:      backend + ":0",
		DurationNS:  d.Nanoseconds(),
		TimerDomain: timerDomain,
	}
}

// promSamples returns every rendered sample line for a family, in render order. A name
// matches on a line boundary (end of line, or the label brace) so a family never
// captures a longer name that merely starts with it.
func promSamples(render, name string) []string {
	var out []string
	for _, line := range strings.Split(render, "\n") {
		if strings.HasPrefix(line, "#") {
			continue
		}
		if line == name || strings.HasPrefix(line, name+" ") || strings.HasPrefix(line, name+"{") {
			out = append(out, line)
		}
	}
	return out
}

func promValue(t *testing.T, render, sample string) string {
	t.Helper()
	for _, line := range strings.Split(render, "\n") {
		if line == sample || strings.HasPrefix(line, sample+" ") {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				t.Fatalf("malformed sample line %q", line)
			}
			return fields[1]
		}
	}
	t.Fatalf("sample %q absent from render:\n%s", sample, render)
	return ""
}

// TestKernelLabelCardinalityIsBounded drives MaxKernelKeys+32 distinct (kernel,
// backend, timer_domain) triples and pins the three things that make the bound real:
// the series set stops at the bound, every rejected event lands in the overflow key,
// and the capped bit latches.
func TestKernelLabelCardinalityIsBounded(t *testing.T) {
	r := New()
	total := MaxKernelKeys + 32
	for i := 0; i < total; i++ {
		r.ObserveKernel(kernelEvent("k"+strconv.Itoa(i), "cpu", "host_monotonic", time.Millisecond))
	}
	snap := r.Snapshot()
	if got := len(snap.KernelKeys); got != MaxKernelKeys+1 {
		t.Fatalf("kernel series count = %d, want the bound %d plus one overflow series", got, MaxKernelKeys)
	}
	if got := snap.KernelOverflow; got != 32 {
		t.Fatalf("kernel overflow events = %d, want 32", got)
	}
	if snap.KernelEvents != uint64(total) {
		t.Fatalf("kernel events = %d, want %d (overflow must still be counted)", snap.KernelEvents, total)
	}
	if !snap.KernelKeysCapped {
		t.Fatal("keys_capped = false after exceeding the key bound")
	}

	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricKernelKeysCapped); got != "1" {
		t.Fatalf("%s = %s, want 1", MetricKernelKeysCapped, got)
	}
	if got := promValue(t, render, MetricKernelEventOverflowTotal); got != "32" {
		t.Fatalf("%s = %s, want 32", MetricKernelEventOverflowTotal, got)
	}
	// The rejected events must not vanish: they are folded, with a real duration, into
	// one fixed overflow key whose labels cannot grow.
	overflowCalls := promValue(t, render, MetricKernelCalls+`{kernel="overflow",backend="overflow"}`)
	if overflowCalls != "32" {
		t.Fatalf("overflow call count = %s, want 32", overflowCalls)
	}
	if n := len(promSamples(render, MetricKernelCalls)); n != MaxKernelKeys+1 {
		t.Fatalf("rendered %s series = %d, want %d real plus one overflow", MetricKernelCalls, n, MaxKernelKeys)
	}
}

// TestKernelLabelsAreBoundedNotDropped pins that a benign single key NEVER triggers the
// overflow path: the bound must not turn ordinary traffic into folded data.
func TestKernelLabelsAreBoundedNotDropped(t *testing.T) {
	r := New()
	for i := 0; i < 64; i++ {
		r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", 2*time.Millisecond))
	}
	snap := r.Snapshot()
	if snap.KernelOverflow != 0 || snap.KernelKeysCapped {
		t.Fatalf("benign traffic folded: overflow=%d capped=%t", snap.KernelOverflow, snap.KernelKeysCapped)
	}
	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricKernelCalls+`{kernel="f32_matmul",backend="cpu"}`); got != "64" {
		t.Fatalf("calls = %s, want 64", got)
	}
	// The counter and the histogram must agree on the number of calls, or a board
	// reading "64 calls" next to "0 observations" is a lie.
	if got := promValue(t, render, `fak_engine_kernel_seconds_count{kernel="f32_matmul",backend="cpu",timer_domain="host_monotonic"}`); got != "64" {
		t.Fatalf("histogram count = %s, want 64", got)
	}
}

// TestEmptyKernelLabelsCollapseToAFixedToken pins that a producer which reports no
// kernel id gets one stable label, not a blank one that reads as a board gap — and that
// the token is a constant, so it cannot be used to smuggle cardinality.
func TestEmptyKernelLabelsCollapseToAFixedToken(t *testing.T) {
	r := New()
	r.ObserveKernel(kernelEvent("", "  ", "", time.Millisecond))
	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricKernelCalls+`{kernel="unknown",backend="unknown"}`); got != "1" {
		t.Fatalf("collapsed-label call count = %s, want 1", got)
	}
	if n := len(promSamples(render, MetricKernelCalls)); n != 1 {
		t.Fatalf("collapsed labels produced %d series, want exactly 1", n)
	}
	if !strings.Contains(render, `fak_engine_kernel_seconds_count{kernel="unknown",backend="unknown",timer_domain="unknown"} 1`) {
		t.Fatalf("histogram did not collapse the same way:\n%s", render)
	}
}

// TestHistogramBucketAccounting pins the shared seconds ladder: every observation lands
// in exactly one bucket, the +Inf bucket equals the sample count, and the cumulative
// buckets are non-decreasing and end at the count.
func TestHistogramBucketAccounting(t *testing.T) {
	r := New()
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", 3*time.Millisecond))
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", 90*time.Second))

	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()

	const prefix = `fak_engine_kernel_seconds_bucket{kernel="f32_matmul",backend="cpu",timer_domain="host_monotonic",le="`
	var prev uint64
	sawThree := false
	for _, bound := range secondsBuckets {
		v, err := strconv.ParseUint(promValue(t, render, prefix+formatFloat(bound)+`"}`), 10, 64)
		if err != nil {
			t.Fatalf("bucket %s: %v", formatFloat(bound), err)
		}
		if v < prev {
			t.Fatalf("cumulative bucket %s went backwards: %d after %d", formatFloat(bound), v, prev)
		}
		if v == 1 {
			sawThree = true
		}
		prev = v
	}
	if !sawThree {
		t.Fatal("the 3ms observation never entered any finite bucket")
	}
	if got := promValue(t, render, prefix+`+Inf"}`); got != "2" {
		t.Fatalf("+Inf bucket = %s, want 2", got)
	}
	if got := promValue(t, render, `fak_engine_kernel_seconds_sum{kernel="f32_matmul",backend="cpu",timer_domain="host_monotonic"}`); got != "90.003" {
		t.Fatalf("_sum = %s, want 90.003", got)
	}
}

// TestKernelPresenceBitTransitions pins the honest absent/present contract: 0 with the
// family rendered before any producer is attached, 1 once a real event has flowed, and
// back to a defensible 0 on a fresh recorder.
func TestKernelPresenceBitTransitions(t *testing.T) {
	r := New()
	var b strings.Builder
	r.WritePrometheus(&b)
	if got := promValue(t, b.String(), MetricKernelObserved); got != "0" {
		t.Fatalf("%s before any event = %s, want 0", MetricKernelObserved, got)
	}
	if promSamples(b.String(), MetricKernelObserved) == nil {
		t.Fatal("the observed family must be emitted even with no producer attached")
	}
	if !strings.Contains(b.String(), "# TYPE "+MetricKernelObserved+" gauge") {
		t.Fatal("the observed family must declare its TYPE")
	}

	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", time.Millisecond))
	b.Reset()
	r.WritePrometheus(&b)
	if got := promValue(t, b.String(), MetricKernelObserved); got != "1" {
		t.Fatalf("%s after one real event = %s, want 1", MetricKernelObserved, got)
	}

	// A fresh recorder is ABSENT again: the bit tracks real data, not process uptime.
	fresh := New()
	var fb strings.Builder
	fresh.WritePrometheus(&fb)
	if got := promValue(t, fb.String(), MetricKernelObserved); got != "0" {
		t.Fatalf("%s on an unfed recorder = %s, want 0", MetricKernelObserved, got)
	}
}

// TestNegativeDurationIsRecordedNotDropped pins that the counter and the histogram can
// never disagree about how many calls happened: a clamped-to-zero duration is still
// counted.
func TestNegativeDurationIsRecordedNotDropped(t *testing.T) {
	r := New()
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", -5*time.Millisecond))
	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricKernelCalls+`{kernel="f32_matmul",backend="cpu"}`); got != "1" {
		t.Fatalf("calls = %s, want 1", got)
	}
	if got := promValue(t, render, `fak_engine_kernel_seconds_count{kernel="f32_matmul",backend="cpu",timer_domain="host_monotonic"}`); got != "1" {
		t.Fatalf("count = %s, want 1", got)
	}
	if got := promValue(t, render, `fak_engine_kernel_seconds_sum{kernel="f32_matmul",backend="cpu",timer_domain="host_monotonic"}`); got != "0" {
		t.Fatalf("_sum = %s, want 0", got)
	}
}

// TestKernelLabelEscaping pins that a free-form kernel id containing Prometheus label
// metacharacters cannot break the exposition format.
func TestKernelLabelEscaping(t *testing.T) {
	r := New()
	r.ObserveKernel(kernelEvent(`we"ird\kernel`, "cpu", "host_monotonic", time.Millisecond))
	var b strings.Builder
	r.WritePrometheus(&b)
	if !strings.Contains(b.String(), `kernel="we\"ird\\kernel"`) {
		t.Fatalf("label was not escaped:\n%s", b.String())
	}
}
