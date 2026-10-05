package stepobs

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=1s — pure in-memory aggregation, no I/O, no model load.

func renderOf(t *testing.T, r *Recorder) string {
	t.Helper()
	var b strings.Builder
	r.WritePrometheus(&b)
	return b.String()
}

// renderedFamilies returns the family names the render declares via # TYPE, mapped to
// their declared type. A name appearing twice is a hard failure: two bodies under one
// # TYPE line is how a scrape silently keeps only one of them.
func renderedFamilies(t *testing.T, render string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, line := range strings.Split(render, "\n") {
		if !strings.HasPrefix(line, "# TYPE ") {
			continue
		}
		fields := strings.Fields(strings.TrimPrefix(line, "# TYPE "))
		if len(fields) != 2 {
			t.Fatalf("malformed TYPE line %q", line)
		}
		if prev, dup := out[fields[0]]; dup {
			t.Fatalf("family %q declared twice (# TYPE ... %s and ... %s)", fields[0], prev, fields[1])
		}
		out[fields[0]] = fields[1]
	}
	return out
}

// TestEveryDeclaredFamilyIsRenderedExactlyOnce pins MetricFamilies against the render:
// every exported name is emitted, no name is emitted twice, and nothing is emitted that
// was not declared. The duplicate check is the one that matters for a Prometheus scrape.
func TestEveryDeclaredFamilyIsRenderedExactlyOnce(t *testing.T) {
	r := New()
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", time.Millisecond))
	r.ObservePlannerStep(StepKindStep, time.Millisecond)
	families := renderedFamilies(t, renderOf(t, r))

	if len(families) != len(MetricFamilies) {
		t.Fatalf("render declares %d families, MetricFamilies lists %d: %v", len(families), len(MetricFamilies), families)
	}
	declared := map[string]bool{}
	for _, name := range MetricFamilies {
		if declared[name] {
			t.Fatalf("MetricFamilies lists %q twice", name)
		}
		declared[name] = true
		if _, ok := families[name]; !ok {
			t.Fatalf("declared family %q is never rendered", name)
		}
	}
	for name := range families {
		if !declared[name] {
			t.Fatalf("render emits undeclared family %q", name)
		}
	}

	// The metric types are part of the contract a dashboard relies on.
	wantTypes := map[string]string{
		MetricKernelCalls:                  "counter",
		MetricKernelSeconds:                "histogram",
		MetricKernelObserved:               "gauge",
		MetricKernelKeysCapped:             "gauge",
		MetricKernelEventOverflowTotal:     "counter",
		MetricPlannerStepSeconds:           "histogram",
		MetricPlannerStepEvents:            "counter",
		MetricPlannerStepObserved:          "gauge",
		MetricPlannerStepKindOverflowTotal: "counter",
	}
	for name, typ := range wantTypes {
		if got := families[name]; got != typ {
			t.Fatalf("family %q declared as %q, want %q", name, got, typ)
		}
	}
}

// TestColdRenderIsHonestAbsentNotAFlatZero is the anti-fabrication contract: with no
// producer attached every family is present, and every observed bit is 0. A board must
// be able to tell "nothing is observing" from "the engine is idle".
func TestColdRenderIsHonestAbsentNotAFlatZero(t *testing.T) {
	render := renderOf(t, New())
	for _, name := range []string{MetricKernelObserved, MetricPlannerStepObserved} {
		if got := promValue(t, render, name); got != "0" {
			t.Fatalf("%s with no producer attached = %s, want 0", name, got)
		}
	}
	if got := promValue(t, render, MetricKernelKeysCapped); got != "0" {
		t.Fatalf("%s on a cold recorder = %s, want 0", MetricKernelKeysCapped, got)
	}
	// No kernel series may exist at all before a producer reports one: rendering a
	// fabricated zero kernel would be indistinguishable from real data.
	if n := len(promSamples(render, MetricKernelCalls)); n != 0 {
		t.Fatalf("cold render emitted %d %s series, want 0", n, MetricKernelCalls)
	}
	// The planner families DO render every kind: the vocabulary is closed, so their zero
	// is a real answer rather than an absent one.
	if n := len(promSamples(render, MetricPlannerStepEvents)); n != len(StepKinds) {
		t.Fatalf("cold render emitted %d planner-step series, want the whole closed vocabulary %d", n, len(StepKinds))
	}
	// And no family may claim presence in its HELP text while the bit says absent.
	if !strings.Contains(render, "# HELP "+MetricKernelObserved+" ") {
		t.Fatal("the kernel observed family carries no HELP line")
	}
}

// TestWarmRenderCarriesRealValuesAndTheObservedBit pins the other half of the contract:
// once fed, the bit is 1 and the recorded call/duration is legible in the render.
func TestWarmRenderCarriesRealValuesAndTheObservedBit(t *testing.T) {
	r := New()
	r.ObserveKernel(computetrace.Event{
		Operation:   "matmul",
		Kernel:      "q4_k_matmul",
		Backend:     "metal",
		TimerDomain: "metal_command_buffer",
		DurationNS:  int64(4 * time.Millisecond),
	})
	r.ObservePlannerStep(StepKindSeat, 2*time.Millisecond)

	render := renderOf(t, r)
	if got := promValue(t, render, MetricKernelObserved); got != "1" {
		t.Fatalf("%s after a real kernel event = %s, want 1", MetricKernelObserved, got)
	}
	if got := promValue(t, render, MetricPlannerStepObserved); got != "1" {
		t.Fatalf("%s after a real planner leg = %s, want 1", MetricPlannerStepObserved, got)
	}
	if got := promValue(t, render, MetricKernelCalls+`{kernel="q4_k_matmul",backend="metal"}`); got != "1" {
		t.Fatalf("kernel calls = %s, want 1", got)
	}
	if got := promValue(t, render, `fak_engine_kernel_seconds_sum{kernel="q4_k_matmul",backend="metal",timer_domain="metal_command_buffer"}`); got != "0.004" {
		t.Fatalf("kernel _sum = %s, want 0.004", got)
	}
	if got := promValue(t, render, MetricPlannerStepEvents+`{kind="seat"}`); got != "1" {
		t.Fatalf("seat events = %s, want 1", got)
	}
}

// TestRenderIsStableAcrossScrapes pins that the series order does not depend on map
// iteration: a Prometheus text diff between two scrapes must be empty at rest.
func TestRenderIsStableAcrossScrapes(t *testing.T) {
	r := New()
	for i := 0; i < 8; i++ {
		r.ObserveKernel(kernelEvent("k"+strconv.Itoa(i), "cpu", "host_monotonic", time.Millisecond))
		r.ObservePlannerStep(StepKinds[i%len(StepKinds)], time.Millisecond)
	}
	first := renderOf(t, r)
	for i := 0; i < 8; i++ {
		if got := renderOf(t, r); got != first {
			t.Fatalf("render changed between scrapes at rest (iteration %d)", i)
		}
	}
}

// TestNilRecorderIsASafeNoOp pins the unattached-producer contract: a nil recorder must
// not panic on the producer hot path, and must render nothing.
func TestNilRecorderIsASafeNoOp(t *testing.T) {
	var r *Recorder
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", time.Millisecond))
	r.ObservePlannerStep(StepKindStep, time.Millisecond)
	r.ObserveEngineStep("step", time.Millisecond)
	r.Attach()
	if r.Attached() {
		t.Fatal("a nil recorder reported itself attached")
	}
	var b strings.Builder
	r.WritePrometheus(&b)
	if b.Len() != 0 {
		t.Fatalf("nil recorder rendered %q, want nothing", b.String())
	}
	if snap := r.Snapshot(); snap.Schema != SnapshotSchema {
		t.Fatalf("nil snapshot schema = %q, want %q", snap.Schema, SnapshotSchema)
	}
}

// TestCompactLineNamesTheAbsentCase pins that the agent-facing line says ABSENT rather
// than reporting a confident empty engine.
func TestCompactLineNamesTheAbsentCase(t *testing.T) {
	if got := New().Snapshot().Compact(); !strings.Contains(got, "KERNEL absent") || !strings.Contains(got, "STEP absent") {
		t.Fatalf("cold compact line = %q, want both registries named absent", got)
	}
	r := New()
	r.ObserveKernel(kernelEvent("f32_matmul", "cpu", "host_monotonic", time.Millisecond))
	r.ObservePlannerStep(StepKindStep, time.Millisecond)
	got := r.Snapshot().Compact()
	for _, want := range []string{"KERNEL observed", "calls=1", "keys=1", "STEP observed", "step=1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("compact line %q missing %q", got, want)
		}
	}
}
