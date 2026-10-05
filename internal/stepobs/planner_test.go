package stepobs

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// fak-test:runtime fast est=1s — pure in-memory aggregation, no I/O, no model load.

// TestPlannerStepFamiliesAlwaysCoverTheClosedVocabulary pins the board-shape promise:
// every kind in the closed vocabulary is rendered from a cold recorder, so "zero tool
// legs" reads as a real zero instead of a missing series.
func TestPlannerStepFamiliesAlwaysCoverTheClosedVocabulary(t *testing.T) {
	r := New()
	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()

	for _, k := range StepKinds {
		if got := promValue(t, render, MetricPlannerStepEvents+`{kind="`+string(k)+`"}`); got != "0" {
			t.Fatalf("events for kind %q on a cold recorder = %s, want 0", k, got)
		}
		if got := promValue(t, render, `fak_engine_planner_step_seconds_count{kind="`+string(k)+`"}`); got != "0" {
			t.Fatalf("histogram count for kind %q on a cold recorder = %s, want 0", k, got)
		}
	}
	if n := len(promSamples(render, MetricPlannerStepEvents)); n != len(StepKinds) {
		t.Fatalf("rendered %d planner-step event series, want %d", n, len(StepKinds))
	}
	if got := promValue(t, render, MetricPlannerStepObserved); got != "0" {
		t.Fatalf("%s on a cold recorder = %s, want 0 (producer ABSENT, not idle)", MetricPlannerStepObserved, got)
	}
	if got := promValue(t, render, MetricPlannerStepKindOverflowTotal); got != "0" {
		t.Fatalf("%s = %s, want 0", MetricPlannerStepKindOverflowTotal, got)
	}
}

// TestPlannerStepPresenceBitTransitions pins the 0 -> 1 -> 0 contract on a fresh
// recorder: the bit reports real data, not process uptime.
func TestPlannerStepPresenceBitTransitions(t *testing.T) {
	r := New()
	var b strings.Builder
	r.WritePrometheus(&b)
	if got := promValue(t, b.String(), MetricPlannerStepObserved); got != "0" {
		t.Fatalf("%s before any leg = %s, want 0", MetricPlannerStepObserved, got)
	}
	r.ObservePlannerStep(StepKindStep, 12*time.Millisecond)
	b.Reset()
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricPlannerStepObserved); got != "1" {
		t.Fatalf("%s after one real leg = %s, want 1", MetricPlannerStepObserved, got)
	}
	if got := promValue(t, render, MetricPlannerStepEvents+`{kind="step"}`); got != "1" {
		t.Fatalf("step events = %s, want 1", got)
	}
	// The other kinds must stay at a real zero: feeding one kind must not light the
	// whole board.
	if got := promValue(t, render, MetricPlannerStepEvents+`{kind="verdict"}`); got != "0" {
		t.Fatalf("verdict events = %s, want 0", got)
	}
	if got := promValue(t, render, `fak_engine_planner_step_seconds_sum{kind="step"}`); got != "0.012" {
		t.Fatalf("step _sum = %s, want 0.012", got)
	}
}

// TestPlannerStepUnknownKindOverflowsRatherThanWideningTheBoard pins that an
// out-of-vocabulary kind is counted and dropped, never turned into a new label value.
func TestPlannerStepUnknownKindOverflowsRatherThanWideningTheBoard(t *testing.T) {
	r := New()
	r.ObservePlannerStep(StepKindStep, time.Millisecond)
	r.ObservePlannerStep(StepKind("invented_"+strconv.Itoa(1)), 5*time.Millisecond)
	r.ObservePlannerStep(StepKind("another_"+strconv.Itoa(2)), 5*time.Millisecond)

	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricPlannerStepKindOverflowTotal); got != "2" {
		t.Fatalf("%s = %s, want 2", MetricPlannerStepKindOverflowTotal, got)
	}
	if n := len(promSamples(render, MetricPlannerStepEvents)); n != len(StepKinds) {
		t.Fatalf("out-of-vocabulary kinds widened the board to %d series, want %d", n, len(StepKinds))
	}
	if got := promValue(t, render, MetricPlannerStepEvents+`{kind="step"}`); got != "1" {
		t.Fatalf("step events = %s, want 1 (the dropped legs must not be folded in)", got)
	}
	if snap := r.Snapshot(); snap.PlannerObserved != true || snap.PlannerKindOverflow != 2 {
		t.Fatalf("snapshot = %+v", snap)
	}
}

// TestEnginePhaseMappingIsTotalOverThePhaseVocabulary pins the closed mapping in BOTH
// directions: every declared phase is either mapped to exactly one kind or explicitly
// unmapped, and every mapped kind is in the closed StepKind vocabulary.
func TestEnginePhaseMappingIsTotalOverThePhaseVocabulary(t *testing.T) {
	allowed := map[StepKind]bool{}
	for _, k := range StepKinds {
		allowed[k] = true
	}
	seen := 0
	for _, p := range enginestep.Phases {
		kind, ok := StepKindForEnginePhase(p)
		if !ok {
			// An unmapped phase must record nothing at all, not a default kind.
			r := New()
			r.ObserveEnginePhase(p, time.Millisecond)
			if snap := r.Snapshot(); snap.PlannerObserved {
				t.Fatalf("unmapped phase %q recorded a planner-step leg", p)
			}
			continue
		}
		if !allowed[kind] {
			t.Fatalf("phase %q maps to kind %q, which is outside the closed vocabulary", p, kind)
		}
		seen++
	}
	if seen == 0 {
		t.Fatal("no serving-loop phase maps to a planner-step kind; the seam is dead")
	}

	// The decode forward is the sub-planner detail PhaseDecode cannot name, so it is
	// pinned separately: PhaseDecode is the request's WHOLE loop.
	if kind, ok := StepKindForEnginePhase(enginestep.PhaseDecode); !ok || kind != StepKindStep {
		t.Fatalf("PhaseDecode maps to (%q, %t), want (%q, true)", kind, ok, StepKindStep)
	}
	if kind, ok := StepKindForEnginePhase(enginestep.PhaseDeviceWait); !ok || kind != StepKindSeat {
		t.Fatalf("PhaseDeviceWait maps to (%q, %t), want (%q, true)", kind, ok, StepKindSeat)
	}
	if kind, ok := StepKindForEnginePhase(enginestep.PhaseAdmissionWait); !ok || kind != StepKindAdmission {
		t.Fatalf("PhaseAdmissionWait maps to (%q, %t), want (%q, true)", kind, ok, StepKindAdmission)
	}
}

// TestObserveEngineStepAcceptsTheMicroSpanVocabulary pins that the microagent bridge — a
// plain string, because internal/stepobs must not import internal/metrics — lands in the
// same closed series set the serving loop writes, so both producers read on one family.
func TestObserveEngineStepAcceptsTheMicroSpanVocabulary(t *testing.T) {
	r := New()
	for _, k := range StepKinds {
		r.ObserveEngineStep(string(k), time.Millisecond)
	}
	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	for _, k := range StepKinds {
		if got := promValue(t, render, MetricPlannerStepEvents+`{kind="`+string(k)+`"}`); got != "1" {
			t.Fatalf("events for kind %q = %s, want 1", k, got)
		}
	}
	if got := promValue(t, render, MetricPlannerStepObserved); got != "1" {
		t.Fatalf("%s = %s, want 1", MetricPlannerStepObserved, got)
	}
	if got := promValue(t, render, MetricPlannerStepKindOverflowTotal); got != "0" {
		t.Fatalf("%s = %s, want 0", MetricPlannerStepKindOverflowTotal, got)
	}
}

// TestPlannerStepHistogramAccounting pins that the counter and the histogram never
// disagree, and that a duration lands in the ladder bucket it belongs to.
func TestPlannerStepHistogramAccounting(t *testing.T) {
	r := New()
	for i := 0; i < 5; i++ {
		r.ObservePlannerStep(StepKindSeat, 250*time.Microsecond)
	}
	r.ObservePlannerStep(StepKindSeat, 90*time.Second)

	var b strings.Builder
	r.WritePrometheus(&b)
	render := b.String()
	if got := promValue(t, render, MetricPlannerStepEvents+`{kind="seat"}`); got != "6" {
		t.Fatalf("seat events = %s, want 6", got)
	}
	if got := promValue(t, render, `fak_engine_planner_step_seconds_count{kind="seat"}`); got != "6" {
		t.Fatalf("seat histogram count = %s, want 6", got)
	}
	if got := promValue(t, render, `fak_engine_planner_step_seconds_sum{kind="seat"}`); got != "90.00125" {
		t.Fatalf("seat _sum = %s, want 90.00125", got)
	}
	// 250us sits in the 0.00025 bucket; the 90s sample sits above every finite bound, so
	// only +Inf may exceed the 5 that fit in a finite bucket.
	if got := promValue(t, render, `fak_engine_planner_step_seconds_bucket{kind="seat",le="0.00025"}`); got != "5" {
		t.Fatalf("0.00025 bucket = %s, want 5", got)
	}
	if got := promValue(t, render, `fak_engine_planner_step_seconds_bucket{kind="seat",le="60"}`); got != "5" {
		t.Fatalf("60 bucket = %s, want 5", got)
	}
	if got := promValue(t, render, `fak_engine_planner_step_seconds_bucket{kind="seat",le="+Inf"}`); got != "6" {
		t.Fatalf("+Inf bucket = %s, want 6", got)
	}
}
