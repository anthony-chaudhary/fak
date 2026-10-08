package stepobs

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=200ms lane=default
func TestSnapshotLatencySummariesMatchPrometheus(t *testing.T) {
	r := New()
	for _, d := range []time.Duration{time.Millisecond, 3 * time.Millisecond, 6 * time.Millisecond} {
		r.ObserveKernel(kernelEvent("reference", "cpu", "host_monotonic", d))
		for _, kind := range StepKinds {
			r.ObservePlannerStep(kind, d)
		}
	}
	s := r.Snapshot()
	if len(s.KernelLatency) != 1 || len(s.PlannerLatency) != 5 {
		t.Fatalf("summary cardinality kernel=%d planner=%d", len(s.KernelLatency), len(s.PlannerLatency))
	}
	k := s.KernelLatency[0]
	if k.Kernel != "reference" || k.Backend != "cpu" || k.TimerDomain != "host_monotonic" || !k.Measured || k.Overflow {
		t.Fatal("kernel attribution missing")
	}
	var b strings.Builder
	r.WritePrometheus(&b)
	check := func(stat LatencyStat, family, labels string) {
		t.Helper()
		if stat.Count != 3 || math.Abs(stat.TotalSeconds-.01) > 1e-12 || math.Abs(stat.MeanSeconds-.01/3) > 1e-12 || stat.MaxSeconds != .006 {
			t.Fatalf("latency arithmetic count=%d total=%g mean=%g max=%g", stat.Count, stat.TotalSeconds, stat.MeanSeconds, stat.MaxSeconds)
		}
		if stat.P50Seconds < .003 || stat.P50Seconds > stat.P95Seconds || stat.P95Seconds > stat.MaxSeconds {
			t.Fatalf("quantile bounds p50=%g p95=%g max=%g", stat.P50Seconds, stat.P95Seconds, stat.MaxSeconds)
		}
		n, err := strconv.ParseUint(promValue(t, b.String(), family+"_count{"+labels+"}"), 10, 64)
		if err != nil || n != stat.Count {
			t.Fatal("Prometheus count disagrees")
		}
		sum, err := strconv.ParseFloat(promValue(t, b.String(), family+"_sum{"+labels+"}"), 64)
		if err != nil || math.Abs(sum-stat.TotalSeconds) > 1e-12 {
			t.Fatal("Prometheus duration disagrees")
		}
	}
	check(k.LatencyStat, MetricKernelSeconds, `kernel="reference",backend="cpu",timer_domain="host_monotonic"`)
	for _, kind := range StepKinds {
		check(s.PlannerLatency[string(kind)], MetricPlannerStepSeconds, `kind="`+string(kind)+`"`)
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"kernel_latency", "planner_step_latency", "recorder_attached"} {
		if _, ok := wire[field]; !ok {
			t.Fatalf("missing JSON field %s", field)
		}
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestSnapshotLatencyBoundedCopiedAndUnavailable(t *testing.T) {
	r := New()
	cold := r.Snapshot()
	if cold.KernelObserved || cold.PlannerObserved || cold.RecorderAttached || len(cold.KernelLatency) != 0 || len(cold.PlannerLatency) != 5 {
		t.Fatal("cold recorder claims a producer")
	}
	for i := 0; i < MaxKernelKeys; i++ {
		r.ObserveKernel(kernelEvent("k"+strconv.Itoa(i), "cpu", "host_monotonic", time.Millisecond))
	}
	r.ObserveKernel(kernelEvent("host-overflow", "cpu", "host_monotonic", time.Millisecond))
	r.ObserveKernel(kernelEvent("device-overflow", "cuda", "cuda_event", 2*time.Millisecond))
	r.ObserveKernel(kernelEvent("unavailable-overflow", "vulkan", "vulkan_performance_query_unavailable", 0))
	r.ObservePlannerStep(StepKind("invented"), time.Second)
	s := r.Snapshot()
	if len(s.KernelLatency) != MaxKernelKeys+1 || s.KernelOverflow != 3 || s.PlannerKindOverflow != 1 || len(s.PlannerLatency) != 5 {
		t.Fatal("snapshot cardinality is unbounded or overflow lost")
	}
	last := s.KernelLatency[len(s.KernelLatency)-1]
	if !last.Overflow || last.Measured || last.Count != 3 || last.TotalSeconds != 0 || last.MeanSeconds != 0 || last.P50Seconds != 0 || last.P95Seconds != 0 || last.MaxSeconds != 0 {
		t.Fatal("mixed-clock overflow lost counts or claims measured latency")
	}
	s.KernelLatency[0].Kernel = "changed"
	s.KernelKeys[0] = "changed"
	s.PlannerLatency["step"] = LatencyStat{Count: 999}
	s.PlannerEvents["step"] = 999
	again := r.Snapshot()
	if again.KernelLatency[0].Kernel == "changed" || again.KernelKeys[0] == "changed" || again.PlannerLatency["step"].Count != 0 || again.PlannerEvents["step"] != 0 {
		t.Fatal("snapshot aliases recorder storage")
	}
	unavailable := New()
	unavailable.ObserveKernel(kernelEvent("gemm", "vulkan", "vulkan_performance_query_unavailable", 0))
	u := unavailable.Snapshot().KernelLatency[0]
	if u.Measured || u.Count != 1 || u.TotalSeconds != 0 || u.MaxSeconds != 0 {
		t.Fatal("unavailable timer claims measured latency")
	}
}
