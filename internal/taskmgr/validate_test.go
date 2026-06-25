package taskmgr

import (
	"reflect"
	"testing"
	"time"
)

// validSnapshot builds a snapshot that ValidateSnapshot must accept: a running
// task with measurable progress and an ETA, carrying one running step whose ETA is
// (validly) omitted.
func validSnapshot() Snapshot {
	taskPct := 50.0
	stepPct := 25.0
	etaSeconds := 8.0
	etaAt := int64(1700000016000000000)
	return Snapshot{
		Schema:          SchemaSnapshot,
		ProcessID:       4321,
		GoOS:            "linux",
		GoArch:          "amd64",
		GoVersion:       "go1.x",
		StartedUnixNano: 1700000000000000000,
		TSUnixNano:      1700000004000000000,
		UptimeSeconds:   4,
		Resource:        ResourceSample{TSUnixNano: 1700000004000000000, WallSeconds: 4, CPUSeconds: 2, Goroutines: 6},
		ResourceDelta:   ResourceDelta{WallSeconds: 4, CPUSeconds: 1},
		Tasks: []TaskSnapshot{{
			TaskID:         "task_build",
			Title:          "Build release",
			State:          StateRunning,
			RuntimeSeconds: 4,
			Progress:       Progress{Done: 2, Total: 4, Unit: "phase", Percent: &taskPct},
			ETASeconds:     &etaSeconds,
			ETAUnixNano:    &etaAt,
			CurrentStep:    "step_tests",
			Resource: ResourceWindow{
				Start:   ResourceSample{WallSeconds: 0, CPUSeconds: 1, Goroutines: 5},
				Current: ResourceSample{WallSeconds: 4, CPUSeconds: 2, Goroutines: 6},
				Delta:   ResourceDelta{WallSeconds: 4, CPUSeconds: 1, Goroutines: 1},
			},
			Steps: []StepSnapshot{{
				StepID:         "step_tests",
				Concept:        "verify",
				State:          StateRunning,
				RuntimeSeconds: 3,
				Progress:       Progress{Done: 1, Total: 4, Unit: "suite", Percent: &stepPct},
				Resource: ResourceWindow{
					Start:   ResourceSample{WallSeconds: 1, CPUSeconds: 1, Goroutines: 5},
					Current: ResourceSample{WallSeconds: 4, CPUSeconds: 2, Goroutines: 6},
					Delta:   ResourceDelta{WallSeconds: 3, CPUSeconds: 1, Goroutines: 1},
				},
			}},
		}},
	}
}

func TestValidateSnapshotAcceptsValid(t *testing.T) {
	if err := ValidateSnapshot(validSnapshot()); err != nil {
		t.Fatalf("valid snapshot rejected: %v", err)
	}
}

// TestValidateSnapshotAcceptsTerminalWithoutETA proves a finished task with frozen
// runtime and no ETA is valid, and that omitting the ETA on a running step is fine.
func TestValidateSnapshotAcceptsTerminalWithoutETA(t *testing.T) {
	s := validSnapshot()
	done := 100.0
	s.Tasks[0].State = StateDone
	s.Tasks[0].Progress = Progress{Done: 4, Total: 4, Unit: "phase", Percent: &done}
	s.Tasks[0].ETASeconds = nil
	s.Tasks[0].ETAUnixNano = nil
	s.Tasks[0].Steps[0].State = StateDone
	if err := ValidateSnapshot(s); err != nil {
		t.Fatalf("terminal snapshot rejected: %v", err)
	}
}

// TestValidateSnapshotAcceptsOverrun encodes the explicit decision that progress
// overrun (done > total, percent > 100) is an honest over-budget signal, not an error.
func TestValidateSnapshotAcceptsOverrun(t *testing.T) {
	s := validSnapshot()
	over := 150.0
	s.Tasks[0].State = StateDone
	s.Tasks[0].Progress = Progress{Done: 6, Total: 4, Unit: "phase", Percent: &over}
	s.Tasks[0].ETASeconds = nil
	s.Tasks[0].ETAUnixNano = nil
	s.Tasks[0].Steps[0].State = StateDone
	if err := ValidateSnapshot(s); err != nil {
		t.Fatalf("overrun snapshot rejected: %v", err)
	}
}

func TestValidateSnapshotRejectsInvalid(t *testing.T) {
	bad := 999.0
	cases := []struct {
		name   string
		mutate func(*Snapshot)
	}{
		{"bad schema", func(s *Snapshot) { s.Schema = "fak.task-manager-snapshot.v0" }},
		{"negative uptime", func(s *Snapshot) { s.UptimeSeconds = -1 }},
		{"negative snapshot cpu sample", func(s *Snapshot) { s.Resource.CPUSeconds = -1 }},
		{"negative wall delta", func(s *Snapshot) { s.ResourceDelta.WallSeconds = -1 }},
		{"empty task id", func(s *Snapshot) { s.Tasks[0].TaskID = "" }},
		{"duplicate task id", func(s *Snapshot) {
			dup := s.Tasks[0]
			dup.ETASeconds = nil
			dup.ETAUnixNano = nil
			dup.State = StateDone
			s.Tasks = append(s.Tasks, dup)
		}},
		{"empty step id", func(s *Snapshot) { s.Tasks[0].Steps[0].StepID = "" }},
		{"duplicate step id", func(s *Snapshot) {
			s.Tasks[0].Steps = append(s.Tasks[0].Steps, s.Tasks[0].Steps[0])
		}},
		{"unknown task state", func(s *Snapshot) { s.Tasks[0].State = State("paused") }},
		{"unknown step state", func(s *Snapshot) { s.Tasks[0].Steps[0].State = State("zombie") }},
		{"negative task runtime", func(s *Snapshot) { s.Tasks[0].RuntimeSeconds = -2 }},
		{"negative step resource counter", func(s *Snapshot) { s.Tasks[0].Steps[0].Resource.Current.CPUSeconds = -1 }},
		{"negative progress done", func(s *Snapshot) { s.Tasks[0].Progress.Done = -1 }},
		{"negative progress total", func(s *Snapshot) { s.Tasks[0].Progress.Total = -1 }},
		{"percent without total", func(s *Snapshot) {
			s.Tasks[0].Progress = Progress{Done: 1, Total: 0, Percent: &bad}
			s.Tasks[0].ETASeconds = nil
			s.Tasks[0].ETAUnixNano = nil
		}},
		{"total without percent", func(s *Snapshot) {
			s.Tasks[0].Progress.Percent = nil
		}},
		{"inconsistent percent", func(s *Snapshot) { *s.Tasks[0].Progress.Percent = 99 }},
		{"eta on terminal task", func(s *Snapshot) { s.Tasks[0].State = StateDone }},
		{"one-sided eta seconds", func(s *Snapshot) { s.Tasks[0].ETAUnixNano = nil }},
		{"one-sided eta timestamp", func(s *Snapshot) { s.Tasks[0].ETASeconds = nil }},
		{"negative eta", func(s *Snapshot) { neg := -3.0; s.Tasks[0].ETASeconds = &neg }},
		{"eta without measurable progress", func(s *Snapshot) {
			s.Tasks[0].Progress = Progress{Done: 4, Total: 4, Unit: "phase", Percent: ptr(100.0)}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := validSnapshot()
			tc.mutate(&s)
			if err := ValidateSnapshot(s); err == nil {
				t.Fatalf("%s: expected validation error, got nil", tc.name)
			}
		})
	}
}

// TestValidateSnapshotIsReadOnly proves validation never mutates the snapshot it
// inspects, even on a rejecting input.
func TestValidateSnapshotIsReadOnly(t *testing.T) {
	s := validSnapshot()
	before := deepCopySnapshot(t, s)
	_ = ValidateSnapshot(s)
	if !reflect.DeepEqual(before, s) {
		t.Fatalf("ValidateSnapshot mutated its input")
	}
}

// TestValidateSnapshotMatchesManagerOutput proves a snapshot produced by the
// Manager's own constructor validates, closing the loop between producer and contract.
func TestValidateSnapshotMatchesManagerOutput(t *testing.T) {
	base := time.Unix(1700000000, 0)
	now := base
	m := NewManager(WithClock(func() time.Time { return now }), WithSampler(fakeSampler(base)))
	task, err := m.StartTask(TaskSpec{TaskID: "task_x", Total: 10, Unit: "phase"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	now = now.Add(1 * time.Second)
	step, err := task.StartStep(StepSpec{StepID: "step_x", Concept: "verify", Total: 4, Unit: "suite"})
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
	if err := ValidateSnapshot(m.Snapshot()); err != nil {
		t.Fatalf("manager snapshot failed validation: %v", err)
	}
	if err := task.Finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}
	if err := ValidateSnapshot(m.Snapshot()); err != nil {
		t.Fatalf("terminal manager snapshot failed validation: %v", err)
	}
}

func ptr(v float64) *float64 { return &v }

func deepCopySnapshot(t *testing.T, s Snapshot) Snapshot {
	t.Helper()
	out := s
	out.Tasks = make([]TaskSnapshot, len(s.Tasks))
	for i, task := range s.Tasks {
		ct := task
		if task.ETASeconds != nil {
			v := *task.ETASeconds
			ct.ETASeconds = &v
		}
		if task.ETAUnixNano != nil {
			v := *task.ETAUnixNano
			ct.ETAUnixNano = &v
		}
		if task.Progress.Percent != nil {
			v := *task.Progress.Percent
			ct.Progress.Percent = &v
		}
		ct.Steps = make([]StepSnapshot, len(task.Steps))
		for j, step := range task.Steps {
			cs := step
			if step.Progress.Percent != nil {
				v := *step.Progress.Percent
				cs.Progress.Percent = &v
			}
			ct.Steps[j] = cs
		}
		out.Tasks[i] = ct
	}
	return out
}
