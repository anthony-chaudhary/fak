package taskmgr

import (
	"encoding/json"
	"testing"
	"time"
)

func TestSnapshotTracksStepResourceConceptAndETA(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	m := NewManager(
		WithClock(func() time.Time { return now }),
		WithSampler(fakeSampler(base)),
	)

	task, err := m.StartTask(TaskSpec{TaskID: "task_build", Title: "Build release", Total: 10, Unit: "phase"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	now = now.Add(1 * time.Second)
	step, err := task.StartStep(StepSpec{StepID: "step_tests", Title: "Run tests", Concept: "verify", Total: 4, Unit: "suite"})
	if err != nil {
		t.Fatalf("start step: %v", err)
	}
	now = now.Add(3 * time.Second)
	if err := task.SetProgress(2, 10, "phase"); err != nil {
		t.Fatalf("set task progress: %v", err)
	}
	if err := step.SetProgress(1, 4, "suite"); err != nil {
		t.Fatalf("set step progress: %v", err)
	}

	snap := m.Snapshot()
	if snap.Schema != SchemaSnapshot {
		t.Fatalf("schema = %q, want %q", snap.Schema, SchemaSnapshot)
	}
	if snap.UptimeSeconds != 4 || snap.ResourceDelta.WallSeconds != 4 {
		t.Fatalf("process runtime/delta = %.1f/%.1f, want 4/4", snap.UptimeSeconds, snap.ResourceDelta.WallSeconds)
	}
	if len(snap.Tasks) != 1 {
		t.Fatalf("tasks = %d, want 1", len(snap.Tasks))
	}

	gotTask := snap.Tasks[0]
	if gotTask.RuntimeSeconds != 4 {
		t.Fatalf("task runtime = %.1f, want 4", gotTask.RuntimeSeconds)
	}
	if gotTask.Progress.Percent == nil || *gotTask.Progress.Percent != 20 {
		t.Fatalf("task percent = %v, want 20", gotTask.Progress.Percent)
	}
	if gotTask.LivenessClass != LivenessLive || gotTask.BeatsSeen != 2 {
		t.Fatalf("task liveness/beats = %s/%d, want live/2", gotTask.LivenessClass, gotTask.BeatsSeen)
	}
	if gotTask.ETASeconds == nil || *gotTask.ETASeconds != 16 {
		t.Fatalf("task eta = %v, want 16", gotTask.ETASeconds)
	}
	wantETAAt := now.Add(16 * time.Second).UnixNano()
	if gotTask.ETAUnixNano == nil || *gotTask.ETAUnixNano != wantETAAt {
		t.Fatalf("task eta_at = %v, want %d", gotTask.ETAUnixNano, wantETAAt)
	}
	if gotTask.CurrentStep != "step_tests" {
		t.Fatalf("current step = %q, want step_tests", gotTask.CurrentStep)
	}
	if gotTask.Resource.Delta.CPUSeconds != 2 {
		t.Fatalf("task cpu delta = %.1f, want 2", gotTask.Resource.Delta.CPUSeconds)
	}

	if len(gotTask.Steps) != 1 {
		t.Fatalf("steps = %d, want 1", len(gotTask.Steps))
	}
	gotStep := gotTask.Steps[0]
	if gotStep.RuntimeSeconds != 3 {
		t.Fatalf("step runtime = %.1f, want 3", gotStep.RuntimeSeconds)
	}
	if gotStep.ETASeconds == nil || *gotStep.ETASeconds != 9 {
		t.Fatalf("step eta = %v, want 9", gotStep.ETASeconds)
	}
	if gotStep.Resource.Delta.CPUSeconds != 1.5 {
		t.Fatalf("step cpu delta = %.1f, want 1.5", gotStep.Resource.Delta.CPUSeconds)
	}
	if gotStep.LivenessClass != LivenessLive || gotStep.BeatsSeen != 1 {
		t.Fatalf("step liveness/beats = %s/%d, want live/1", gotStep.LivenessClass, gotStep.BeatsSeen)
	}
	if len(gotTask.Concepts) != 1 || gotTask.Concepts[0].Concept != "verify" || gotTask.Concepts[0].RuntimeSeconds != 3 {
		t.Fatalf("task concepts = %+v, want verify/3s", gotTask.Concepts)
	}
	if len(snap.Concepts) != 1 || snap.Concepts[0].Concept != "verify" || snap.Concepts[0].RunningSteps != 1 {
		t.Fatalf("snapshot concepts = %+v, want one running verify step", snap.Concepts)
	}
}

func TestBeatLivenessClassifiesLiveIdleAndStalled(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	m := NewManager(
		WithClock(func() time.Time { return now }),
		WithSampler(fakeSampler(base)),
		WithLivenessTimeout(2*time.Second),
	)
	task, err := m.StartTask(TaskSpec{TaskID: "task_loop", Total: 10})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	step, err := task.StartStep(StepSpec{StepID: "step_loop", Total: 10})
	if err != nil {
		t.Fatalf("start step: %v", err)
	}

	snap := m.Snapshot()
	if got := snap.Tasks[0].LivenessClass; got != LivenessIdle {
		t.Fatalf("new task liveness = %s, want idle", got)
	}

	now = now.Add(1 * time.Second)
	if err := step.Beat(); err != nil {
		t.Fatalf("beat step: %v", err)
	}
	snap = m.Snapshot()
	gotTask, gotStep := snap.Tasks[0], snap.Tasks[0].Steps[0]
	if gotTask.LivenessClass != LivenessLive || gotStep.LivenessClass != LivenessLive {
		t.Fatalf("recent beat liveness task/step = %s/%s, want live/live", gotTask.LivenessClass, gotStep.LivenessClass)
	}
	if gotTask.BeatsSeen != 1 || gotStep.BeatsSeen != 1 {
		t.Fatalf("beats task/step = %d/%d, want 1/1", gotTask.BeatsSeen, gotStep.BeatsSeen)
	}

	now = now.Add(3 * time.Second)
	snap = m.Snapshot()
	gotTask, gotStep = snap.Tasks[0], snap.Tasks[0].Steps[0]
	if gotTask.LivenessClass != LivenessStalled || gotStep.LivenessClass != LivenessStalled {
		t.Fatalf("stale beat liveness task/step = %s/%s, want stalled/stalled", gotTask.LivenessClass, gotStep.LivenessClass)
	}
	if gotTask.LastBeatAgeSeconds == nil || *gotTask.LastBeatAgeSeconds != 3 {
		t.Fatalf("task beat age = %v, want 3", gotTask.LastBeatAgeSeconds)
	}

	now = now.Add(1 * time.Second)
	if err := task.SetProgress(2, 10, "item"); err != nil {
		t.Fatalf("set task progress: %v", err)
	}
	snap = m.Snapshot()
	if got := snap.Tasks[0].LivenessClass; got != LivenessLive {
		t.Fatalf("progress refreshed liveness = %s, want live", got)
	}
	if got := snap.Tasks[0].BeatsSeen; got != 2 {
		t.Fatalf("task beats after progress = %d, want 2", got)
	}

	if err := task.Finish(); err != nil {
		t.Fatalf("finish task: %v", err)
	}
	snap = m.Snapshot()
	if got := snap.Tasks[0].LivenessClass; got != LivenessIdle {
		t.Fatalf("terminal liveness = %s, want idle", got)
	}
}

func TestETAIsUnknownWithoutMeasuredProgress(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	m := NewManager(WithClock(func() time.Time { return now }), WithSampler(fakeSampler(base)))
	task, err := m.StartTask(TaskSpec{TaskID: "task_wait", Total: 10})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	now = now.Add(5 * time.Second)
	if err := task.SetProgress(0, 10, "item"); err != nil {
		t.Fatalf("set progress: %v", err)
	}
	snap := m.Snapshot()
	if snap.Tasks[0].ETASeconds != nil || snap.Tasks[0].ETAUnixNano != nil {
		t.Fatalf("eta should be unknown without positive progress: %+v", snap.Tasks[0])
	}
}

func TestFinishFreezesRuntimeAndCompletesRunningStep(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	m := NewManager(WithClock(func() time.Time { return now }), WithSampler(fakeSampler(base)))
	task, err := m.StartTask(TaskSpec{TaskID: "task_ship", Total: 1, Unit: "job"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	if _, err := task.StartStep(StepSpec{StepID: "step_publish", Concept: "ship"}); err != nil {
		t.Fatalf("start step: %v", err)
	}
	now = now.Add(2 * time.Second)
	if err := task.Finish(); err != nil {
		t.Fatalf("finish task: %v", err)
	}
	now = now.Add(10 * time.Second)

	snap := m.Snapshot()
	got := snap.Tasks[0]
	if got.State != StateDone || got.RuntimeSeconds != 2 {
		t.Fatalf("task state/runtime = %s/%.1f, want done/2", got.State, got.RuntimeSeconds)
	}
	if got.ETASeconds != nil {
		t.Fatalf("finished task should not carry eta: %v", got.ETASeconds)
	}
	if len(got.Steps) != 1 || got.Steps[0].State != StateDone || got.Steps[0].RuntimeSeconds != 2 {
		t.Fatalf("running step was not completed with task: %+v", got.Steps)
	}
	if got.Resource.Delta.WallSeconds != 2 {
		t.Fatalf("resource delta after finish = %.1f, want frozen 2", got.Resource.Delta.WallSeconds)
	}
}

func TestDuplicateTaskAndStepIDsAreRefused(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_dupe"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	if _, err := m.StartTask(TaskSpec{TaskID: "task_dupe"}); err == nil {
		t.Fatalf("duplicate task id was accepted")
	}
	if _, err := task.StartStep(StepSpec{StepID: "step_dupe"}); err != nil {
		t.Fatalf("start step: %v", err)
	}
	if _, err := task.StartStep(StepSpec{StepID: "step_dupe"}); err == nil {
		t.Fatalf("duplicate step id was accepted")
	}
}

func TestSnapshotJSONRoundTrip(t *testing.T) {
	m := NewManager()
	if _, err := m.StartTask(TaskSpec{TaskID: "task_json", Total: 1}); err != nil {
		t.Fatalf("start task: %v", err)
	}
	b, err := json.Marshal(m.Snapshot())
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	var round Snapshot
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	if round.Schema != SchemaSnapshot || len(round.Tasks) != 1 || round.Tasks[0].TaskID != "task_json" {
		t.Fatalf("round trip snapshot = %+v", round)
	}
}

func fakeSampler(base time.Time) Sampler {
	return func(processStart, now time.Time) ResourceSample {
		elapsed := now.Sub(base).Seconds()
		return ResourceSample{
			TSUnixNano:     now.UnixNano(),
			WallSeconds:    now.Sub(processStart).Seconds(),
			CPUSeconds:     10 + elapsed/2,
			HeapAllocBytes: uint64(1000 + elapsed*100),
			HeapInuseBytes: uint64(2000 + elapsed*100),
			HeapSysBytes:   uint64(3000 + elapsed*100),
			SysBytes:       uint64(4000 + elapsed*100),
			Goroutines:     5 + int(elapsed),
		}
	}
}
