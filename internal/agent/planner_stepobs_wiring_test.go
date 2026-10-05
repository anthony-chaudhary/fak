package agent

import (
	"bytes"
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/stepobs"
)

// planner_stepobs_wiring_test.go — the SOURCE half of the fak_engine_planner_step_*
// families, proven from the real planner entry point.
//
// The sink half already had a witness (internal/gateway/stepobs_wiring_test.go installs
// the observer and feeds it a hand-built recorder / a hand-built MicroTracer). That is
// the witness this ticket rules out: it proves the registry MOVES, not that a serving
// request CAN move it. Every case below drives InKernelPlanner.Complete — the entry
// gateway.New installs as a native server's planner — and reads the rendered /metrics
// text off the process recorder, so a family that is present-but-dark cannot pass.

// fak-test:runtime fast est=3s lane=default — one synthetic in-kernel turn on the CPU
// reference path plus in-memory Prometheus rendering; no model weights are loaded.

func newPlannerStepobsPlanner(t *testing.T, modelID string) *InKernelPlanner {
	t.Helper()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), modelID, false, nil, false)
	p.maxNew = 4
	p.batchDecode = false
	return p
}

func runPlannerStepobsTurn(t *testing.T, p *InKernelPlanner) {
	t.Helper()
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "planner step"}}, nil); err != nil {
		t.Fatal(err)
	}
}

func renderStepobs(t *testing.T, r *stepobs.Recorder) string {
	t.Helper()
	var buf bytes.Buffer
	r.WritePrometheus(&buf)
	return buf.String()
}

// plannerStepScalar returns the value of one label-less sample of a scalar family, and
// whether the family rendered such a sample at all. A missing sample is a distinct
// outcome from a rendered 0, so the two are never collapsed.
func plannerStepScalar(render, family string) (string, bool) {
	for _, line := range strings.Split(render, "\n") {
		rest, ok := strings.CutPrefix(line, family+" ")
		if !ok || strings.Contains(line, "{") {
			continue
		}
		return strings.TrimSpace(rest), true
	}
	return "", false
}

// plannerStepKind returns the counter value of one member of the closed kind vocabulary
// and whether it rendered. A kind outside the vocabulary renders nothing at all, so this
// is also the drift alarm: the mirrored string and the rendered label must agree.
func plannerStepKind(render string, kind stepobs.StepKind) (uint64, bool) {
	prefix := stepobs.MetricPlannerStepEvents + "{kind=" + strconv.Quote(string(kind)) + "} "
	for _, line := range strings.Split(render, "\n") {
		rest, ok := strings.CutPrefix(line, prefix)
		if !ok {
			continue
		}
		n, err := strconv.ParseUint(strings.TrimSpace(rest), 10, 64)
		if err != nil {
			return 0, false
		}
		return n, true
	}
	return 0, false
}

// TestRealPlannerTurnFeedsThePlannerStepFamily is the positive witness. Only the
// production attachment gateway.New installs (stepobs.Attach, reached from
// attachStepObservation) is in play; the registry is read as rendered Prometheus text
// off the process recorder, not off a recorder the test built.
func TestRealPlannerTurnFeedsThePlannerStepFamily(t *testing.T) {
	p := newPlannerStepobsPlanner(t, "stepobs-planner-positive")
	t.Cleanup(stepobs.Attach)
	stepobs.Attach()

	before := renderStepobs(t, stepobs.Default)
	beforeStep, ok := plannerStepKind(before, stepobs.StepKindStep)
	if !ok {
		t.Fatalf("%s has no kind=%q member; the closed vocabulary drifted", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep)
	}
	if _, ok := plannerStepScalar(before, stepobs.MetricPlannerStepObserved); !ok {
		t.Fatalf("%s rendered no presence bit on the attached recorder", stepobs.MetricPlannerStepObserved)
	}

	runPlannerStepobsTurn(t, p)

	after := renderStepobs(t, stepobs.Default)
	afterStep, ok := plannerStepKind(after, stepobs.StepKindStep)
	if !ok {
		t.Fatalf("%s has no kind=%q member after a real planner turn", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep)
	}
	if afterStep <= beforeStep {
		t.Fatalf("%s{kind=%q} = %d after a real planner turn, want > %d: the source half is not attached", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep, afterStep, beforeStep)
	}
	if got, _ := plannerStepScalar(after, stepobs.MetricPlannerStepObserved); got != "1" {
		t.Fatalf("%s = %q after a real planner turn, want 1", stepobs.MetricPlannerStepObserved, got)
	}
	if got, _ := plannerStepScalar(after, stepobs.MetricPlannerStepKindOverflowTotal); got != "0" {
		t.Fatalf("%s = %q after a real planner turn, want 0: a leg fell outside the closed vocabulary", stepobs.MetricPlannerStepKindOverflowTotal, got)
	}
	t.Logf("%s{kind=%q} %d -> %d and %s 0 -> 1 across one real planner turn", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep, beforeStep, afterStep, stepobs.MetricPlannerStepObserved)
}

// TestPlannerStepObservedIsLoadBearingOnTheAttachedSeam is the A/B that makes the
// attachment load-bearing rather than incidental: the SAME real planner turn moves the
// registry on an attached recorder and moves nothing at all once the producer seam is
// detached. With nothing attached the family must still render — present and dark — so
// absence keeps reading as absence instead of as a missing band or a confident "zero
// steps ran".
func TestPlannerStepObservedIsLoadBearingOnTheAttachedSeam(t *testing.T) {
	p := newPlannerStepobsPlanner(t, "stepobs-planner-ab")
	t.Cleanup(stepobs.Attach)

	stepobs.Attach()
	attachedBefore := stepobs.Default.Snapshot()
	runPlannerStepobsTurn(t, p)
	attachedAfter := stepobs.Default.Snapshot()
	if !attachedAfter.PlannerObserved {
		t.Fatal("planner_step_observed is false after a real planner turn on the attached recorder")
	}
	if got := attachedAfter.PlannerEvents[string(stepobs.StepKindStep)] - attachedBefore.PlannerEvents[string(stepobs.StepKindStep)]; got == 0 {
		t.Fatalf("%s{kind=%q} moved by 0 across one real planner turn", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep)
	}

	// enginestep.SetPhaseObserver is the only producer seam a planner turn reaches in
	// this binary — the decode-step and microagent-span seams are installed by
	// internal/gateway and are absent here — so detaching it removes the path rather
	// than one of several parallel ones.
	enginestep.SetPhaseObserver(nil)
	detachedBefore := stepobs.Default.Snapshot()
	runPlannerStepobsTurn(t, p)
	detachedAfter := stepobs.Default.Snapshot()
	for _, k := range stepobs.StepKinds {
		if got := detachedAfter.PlannerEvents[string(k)] - detachedBefore.PlannerEvents[string(k)]; got != 0 {
			t.Fatalf("%s{kind=%q} moved by %d with no producer attached; absence must read as absence", stepobs.MetricPlannerStepEvents, k, got)
		}
	}
	if detachedAfter.PlannerKindOverflow != detachedBefore.PlannerKindOverflow {
		t.Fatalf("kind overflow moved from %d to %d with no producer attached, want no movement", detachedBefore.PlannerKindOverflow, detachedAfter.PlannerKindOverflow)
	}

	render := renderStepobs(t, stepobs.New())
	if !strings.Contains(render, "# TYPE "+stepobs.MetricPlannerStepObserved+" ") {
		t.Fatalf("%s is missing from the render on an unattached recorder", stepobs.MetricPlannerStepObserved)
	}
	if got, ok := plannerStepScalar(render, stepobs.MetricPlannerStepObserved); !ok || got != "0" {
		t.Fatalf("%s = %q (rendered=%v) on an unattached recorder, want 0", stepobs.MetricPlannerStepObserved, got, ok)
	}
	for _, k := range stepobs.StepKinds {
		if got, ok := plannerStepKind(render, k); !ok || got != 0 {
			t.Fatalf("%s{kind=%q} = %d (rendered=%v) on an unattached recorder, want 0", stepobs.MetricPlannerStepEvents, k, got, ok)
		}
	}
}

// TestPlannerStepDarkLegsRenderZeroBesideALitStep is the per-kind honesty the closed
// vocabulary exists for. A real planner turn lights kind="step"; the legs that have no
// serving-path producer — a tool call the step made, and an adjudication verdict — must
// read an honest 0 next to it rather than borrow a neighbouring label or invent a
// sixth kind.
func TestPlannerStepDarkLegsRenderZeroBesideALitStep(t *testing.T) {
	t.Cleanup(stepobs.Attach)
	stepobs.Attach()
	p := newPlannerStepobsPlanner(t, "stepobs-planner-dark-legs")
	runPlannerStepobsTurn(t, p)

	render := renderStepobs(t, stepobs.Default)
	if got, ok := plannerStepKind(render, stepobs.StepKindStep); !ok || got == 0 {
		t.Fatalf("%s{kind=%q} = %d (rendered=%v) after a real planner turn, want >= 1", stepobs.MetricPlannerStepEvents, stepobs.StepKindStep, got, ok)
	}
	for _, k := range []stepobs.StepKind{stepobs.StepKindTool, stepobs.StepKindVerdict} {
		if got, ok := plannerStepKind(render, k); !ok || got != 0 {
			t.Fatalf("%s{kind=%q} = %d (rendered=%v): no serving-path producer records this leg yet, so it must read 0", stepobs.MetricPlannerStepEvents, k, got, ok)
		}
	}
	if got, _ := plannerStepScalar(render, stepobs.MetricPlannerStepKindOverflowTotal); got != "0" {
		t.Fatalf("%s = %q, want 0: a dark leg was folded into the vocabulary", stepobs.MetricPlannerStepKindOverflowTotal, got)
	}
}
