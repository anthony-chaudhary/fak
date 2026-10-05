package stepobs

import (
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:runtime fast est=1s — pure in-memory aggregation, no I/O, no model load.

// sampleLabels returns the label names of one rendered sample line, in the order they
// appear, or nil for a label-less sample.
func sampleLabels(t *testing.T, line string) []string {
	t.Helper()
	open := strings.Index(line, "{")
	if open < 0 {
		return nil
	}
	close := strings.LastIndex(line, "}")
	if close < open {
		t.Fatalf("unbalanced braces in sample line %q", line)
	}
	body := line[open+1 : close]
	var out []string
	for _, pair := range strings.Split(body, ",") {
		eq := strings.Index(pair, "=")
		if eq < 0 {
			t.Fatalf("label %q in %q has no value", pair, line)
		}
		out = append(out, pair[:eq])
	}
	return out
}

func wantLabels(t *testing.T, line string, want ...string) {
	t.Helper()
	got := sampleLabels(t, line)
	sort.Strings(got)
	sorted := append([]string(nil), want...)
	sort.Strings(sorted)
	if strings.Join(got, ",") != strings.Join(sorted, ",") {
		t.Fatalf("sample %q carries labels [%s], want exactly [%s]", line, strings.Join(got, ","), strings.Join(sorted, ","))
	}
}

// TestRenderedFamilyAndLabelContract is the closed contract a dashboard binds to. It
// pins every emitted family's declared TYPE, its exact label set, and the identity of
// the six contract names, so a dashboard panel cannot be written against a shape that
// silently drifts. The render is logged so a panel author can read the exact text.
func TestRenderedFamilyAndLabelContract(t *testing.T) {
	r := New()
	r.ObserveKernel(computetrace.Event{
		Operation: "matmul", Kernel: "f32_matmul", Backend: "cpu",
		TimerDomain: "host_monotonic", DurationNS: int64(1500 * time.Microsecond),
	})
	r.ObserveKernel(computetrace.Event{
		Operation: "matmul", Kernel: "q8_0_matmul", Backend: "metal",
		TimerDomain: "metal_command_buffer", DurationNS: int64(4 * time.Millisecond),
	})
	r.ObservePlannerStep(StepKindStep, 9*time.Millisecond)
	r.ObservePlannerStep(StepKindSeat, 2*time.Millisecond)

	render := renderOf(t, r)
	t.Logf("rendered /metrics block:\n%s", render)

	families := renderedFamilies(t, render)
	contract := map[string]string{
		MetricKernelCalls:         "counter",
		MetricKernelSeconds:       "histogram",
		MetricKernelObserved:      "gauge",
		MetricPlannerStepSeconds:  "histogram",
		MetricPlannerStepEvents:   "counter",
		MetricPlannerStepObserved: "gauge",
	}
	for name, typ := range contract {
		if got := families[name]; got != typ {
			t.Fatalf("contract family %q declared as %q, want %q", name, got, typ)
		}
		if !strings.Contains(render, "# HELP "+name+" ") {
			t.Fatalf("contract family %q carries no HELP line", name)
		}
	}

	// Exact label sets, per family. A drifting label set is the failure mode that turns
	// a dashboard panel into a silently empty band.
	for _, line := range promSamples(render, MetricKernelCalls) {
		wantLabels(t, line, "kernel", "backend")
	}
	for _, line := range promSamples(render, MetricKernelSeconds) {
		name, _, ok := strings.Cut(line, "{")
		if !ok {
			t.Fatalf("kernel histogram sample %q has no labels", line)
		}
		switch {
		case strings.HasSuffix(name, "_bucket"):
			wantLabels(t, line, "kernel", "backend", "timer_domain", "le")
		case strings.HasSuffix(name, "_sum"), strings.HasSuffix(name, "_count"):
			wantLabels(t, line, "kernel", "backend", "timer_domain")
		default:
			t.Fatalf("unexpected kernel histogram member %q", line)
		}
	}
	for _, line := range promSamples(render, MetricPlannerStepEvents) {
		wantLabels(t, line, "kind")
	}
	for _, line := range promSamples(render, MetricPlannerStepSeconds) {
		name, _, ok := strings.Cut(line, "{")
		if !ok {
			t.Fatalf("planner histogram sample %q has no labels", line)
		}
		switch {
		case strings.HasSuffix(name, "_bucket"):
			wantLabels(t, line, "kind", "le")
		case strings.HasSuffix(name, "_sum"), strings.HasSuffix(name, "_count"):
			wantLabels(t, line, "kind")
		default:
			t.Fatalf("unexpected planner histogram member %q", line)
		}
	}
	// The label-less scalar families must STAY label-less: a reader that writes
	// `fak_engine_kernel_observed{instance="$i"}` by hand silently gets nothing the day
	// the family grows a label.
	for _, name := range []string{
		MetricKernelObserved, MetricPlannerStepObserved, MetricKernelKeysCapped,
		MetricKernelEventOverflowTotal, MetricPlannerStepKindOverflowTotal,
	} {
		lines := promSamples(render, name)
		if len(lines) != 1 {
			t.Fatalf("scalar family %s rendered %d samples, want exactly 1: %v", name, len(lines), lines)
		}
		if strings.Contains(lines[0], "{") {
			t.Fatalf("scalar family %s rendered a labeled sample %q, want a label-less one", name, lines[0])
		}
	}
}
