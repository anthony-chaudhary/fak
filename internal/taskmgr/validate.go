package taskmgr

import (
	"fmt"
	"math"
)

// percentEpsilon is the tolerance used when checking that a snapshot's reported
// progress percent agrees with 100*done/total. The percent is float64 math, so an
// exact equality check would reject honestly-constructed snapshots over rounding.
const percentEpsilon = 1e-6

// ValidateSnapshot reports whether s satisfies the task-manager snapshot contract:
// the schema tag, unique non-empty task/step IDs, the closed state vocabulary,
// non-negative runtimes and resource counters, consistent progress fields, and the
// ETA presence/absence rule (an ETA may appear only on a running record with
// measurable progress, and the two ETA fields are present or absent together).
//
// ValidateSnapshot is read-only: it takes the snapshot by value and never repairs
// or mutates it. A nil return means valid; any defect is reported as the first
// error encountered, naming the offending task or step.
//
// Progress overrun (done > total) is allowed on purpose: it is an honest
// over-budget signal and yields a percent above 100. Validation therefore does not
// reject overrun; it only requires that the reported percent agree with the
// done/total it was derived from.
func ValidateSnapshot(s Snapshot) error {
	if s.Schema != SchemaSnapshot {
		return fmt.Errorf("taskmgr: snapshot schema = %q, want %q", s.Schema, SchemaSnapshot)
	}
	if s.UptimeSeconds < 0 {
		return fmt.Errorf("taskmgr: snapshot uptime is negative: %v", s.UptimeSeconds)
	}
	if err := validateSample("snapshot resource", s.Resource); err != nil {
		return err
	}
	// The process-level window runs from process start to now, so its wall/CPU
	// deltas accumulate and must not go backwards. Heap/sys/goroutine deltas are
	// genuinely signed (memory can be released) and are left unchecked here.
	if s.ResourceDelta.WallSeconds < 0 {
		return fmt.Errorf("taskmgr: snapshot wall delta is negative: %v", s.ResourceDelta.WallSeconds)
	}
	if s.ResourceDelta.CPUSeconds < 0 {
		return fmt.Errorf("taskmgr: snapshot cpu delta is negative: %v", s.ResourceDelta.CPUSeconds)
	}

	seenTasks := make(map[string]struct{}, len(s.Tasks))
	for i := range s.Tasks {
		task := s.Tasks[i]
		if task.TaskID == "" {
			return fmt.Errorf("taskmgr: task at index %d has an empty id", i)
		}
		if _, dup := seenTasks[task.TaskID]; dup {
			return fmt.Errorf("taskmgr: duplicate task id %q", task.TaskID)
		}
		seenTasks[task.TaskID] = struct{}{}

		if err := validateRecord("task "+task.TaskID, task.State, task.RuntimeSeconds, task.Progress, task.ETASeconds, task.ETAUnixNano, task.Resource); err != nil {
			return err
		}
		if err := validateWitness("task "+task.TaskID, task.Witness); err != nil {
			return err
		}

		seenSteps := make(map[string]struct{}, len(task.Steps))
		for j := range task.Steps {
			step := task.Steps[j]
			if step.StepID == "" {
				return fmt.Errorf("taskmgr: task %q step at index %d has an empty id", task.TaskID, j)
			}
			if _, dup := seenSteps[step.StepID]; dup {
				return fmt.Errorf("taskmgr: task %q has duplicate step id %q", task.TaskID, step.StepID)
			}
			seenSteps[step.StepID] = struct{}{}

			ctx := fmt.Sprintf("task %q step %q", task.TaskID, step.StepID)
			if err := validateRecord(ctx, step.State, step.RuntimeSeconds, step.Progress, step.ETASeconds, step.ETAUnixNano, step.Resource); err != nil {
				return err
			}
			if err := validateWitness(ctx, step.Witness); err != nil {
				return err
			}
		}
	}
	return nil
}

// validateRecord checks the fields a task snapshot and a step snapshot share: the
// state vocabulary, non-negative runtime, the resource window, progress, and ETA.
func validateRecord(ctx string, state State, runtime float64, p Progress, etaSeconds *float64, etaAt *int64, rw ResourceWindow) error {
	if !validState(state) {
		return fmt.Errorf("taskmgr: %s has unknown state %q", ctx, state)
	}
	if runtime < 0 {
		return fmt.Errorf("taskmgr: %s runtime is negative: %v", ctx, runtime)
	}
	if err := validateSample(ctx+" resource start", rw.Start); err != nil {
		return err
	}
	if err := validateSample(ctx+" resource current", rw.Current); err != nil {
		return err
	}
	if err := validateProgress(ctx, p); err != nil {
		return err
	}
	return validateETA(ctx, state, p, etaSeconds, etaAt)
}

// validateProgress enforces non-negative done/total and that the percent field is
// present exactly when total is positive and agrees with 100*done/total.
func validateProgress(ctx string, p Progress) error {
	if p.Done < 0 {
		return fmt.Errorf("taskmgr: %s progress done is negative: %v", ctx, p.Done)
	}
	if p.Total < 0 {
		return fmt.Errorf("taskmgr: %s progress total is negative: %v", ctx, p.Total)
	}
	if p.Total > 0 {
		if p.Percent == nil {
			return fmt.Errorf("taskmgr: %s has total but no percent", ctx)
		}
		want := 100 * p.Done / p.Total
		if math.Abs(*p.Percent-want) > percentEpsilon {
			return fmt.Errorf("taskmgr: %s percent = %v, want %v for %v/%v", ctx, *p.Percent, want, p.Done, p.Total)
		}
		return nil
	}
	if p.Percent != nil {
		return fmt.Errorf("taskmgr: %s reports percent %v without a positive total", ctx, *p.Percent)
	}
	return nil
}

// validateETA enforces the ETA presence/absence rule: the seconds and timestamp
// fields are present or absent together; a present ETA is non-negative, appears
// only on a running record, and only when progress is measurable (0 < done < total).
// An absent ETA is always valid: an unknown ETA is omitted, never guessed.
func validateETA(ctx string, state State, p Progress, etaSeconds *float64, etaAt *int64) error {
	if (etaSeconds == nil) != (etaAt == nil) {
		return fmt.Errorf("taskmgr: %s ETA seconds and timestamp must be present or absent together", ctx)
	}
	if etaSeconds == nil {
		return nil
	}
	if *etaSeconds < 0 {
		return fmt.Errorf("taskmgr: %s ETA is negative: %v", ctx, *etaSeconds)
	}
	if state != StateRunning {
		return fmt.Errorf("taskmgr: %s is %s but carries an ETA", ctx, state)
	}
	if !(p.Total > 0 && p.Done > 0 && p.Done < p.Total) {
		return fmt.Errorf("taskmgr: %s carries an ETA without measurable progress (%v/%v)", ctx, p.Done, p.Total)
	}
	return nil
}

// validateSample enforces that a resource counter sample holds non-negative wall
// time, CPU seconds, and goroutine count. The byte counters are unsigned and so
// cannot be negative.
func validateSample(ctx string, s ResourceSample) error {
	if s.WallSeconds < 0 {
		return fmt.Errorf("taskmgr: %s wall seconds is negative: %v", ctx, s.WallSeconds)
	}
	if s.CPUSeconds < 0 {
		return fmt.Errorf("taskmgr: %s cpu seconds is negative: %v", ctx, s.CPUSeconds)
	}
	if s.Goroutines < 0 {
		return fmt.Errorf("taskmgr: %s goroutine count is negative: %d", ctx, s.Goroutines)
	}
	return nil
}

// validState reports whether state is a member of the closed task/step state
// vocabulary.
func validState(state State) bool {
	switch state {
	case StateRunning, StateDone, StateFailed, StateCanceled:
		return true
	default:
		return false
	}
}
