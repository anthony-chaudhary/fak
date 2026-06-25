package taskmgr

import (
	"fmt"
	"os"
)

// VerifiedState is the witnessed-completion rung, kept deliberately separate from
// the claimed State a process reports about itself. A process may claim StateDone;
// only a Witness that reads the effect back from a source the process did not
// author can raise the record to VerifiedDone. The task manager must never treat
// its own completion string as proof.
type VerifiedState string

const (
	// VerifiedUnknown is the zero value: no witness has run, so the record carries
	// only the process's own claim. Every snapshot defaults to this, which is why
	// witness-free snapshots stay valid.
	VerifiedUnknown VerifiedState = ""
	// VerifiedDone means a witness confirmed the claimed effect from evidence.
	VerifiedDone VerifiedState = "verified_done"
	// VerifiedRefused means a witness ran and the evidence contradicted the claim.
	VerifiedRefused VerifiedState = "verified_refused"
	// VerifiedUnavailable means a witness was asked but could not read the effect
	// back (no network, a missing ref). The claim is neither confirmed nor refused;
	// it must not silently downgrade to claimed-done.
	VerifiedUnavailable VerifiedState = "verified_unavailable"
)

// EvidenceRef points a witness at an artifact it can read back: a file path, a git
// ref, a plan/phase. It is the claim's pointer to proof, not the proof itself.
type EvidenceRef struct {
	Kind string `json:"kind"`          // e.g. "path", "commit", "plan-phase"
	Ref  string `json:"ref,omitempty"` // e.g. a path, a sha, "PLAN/PHASE"
	Note string `json:"note,omitempty"`
}

// WitnessRecord is the evidence-backed verdict a Witness attaches to a task or
// step. It never replaces the claimed State; it sits beside it as a separate rung,
// so a snapshot can hold both "the process says done" and "a witness has/has not
// confirmed it".
type WitnessRecord struct {
	VerifiedState   VerifiedState `json:"verified_state"`
	Source          string        `json:"source,omitempty"`  // which witness produced this
	Verdict         string        `json:"verdict,omitempty"` // the witness's raw verdict text
	SHA             string        `json:"sha,omitempty"`
	Detail          string        `json:"detail,omitempty"`
	EvidenceRefs    []EvidenceRef `json:"evidence_refs,omitempty"`
	CheckedUnixNano int64         `json:"checked_unix_nano,omitempty"`
}

// Claim is what a Witness is asked to corroborate: the identity of the task/step,
// the state the process claims, and the artifacts it points at as proof.
type Claim struct {
	TaskID string
	StepID string // empty for a task-level claim
	State  State
	Refs   []EvidenceRef
}

// Witness reads an effect back from a source the reporting process did not author
// and returns an evidence-backed record. A Witness must never treat the claimed
// State as proof — that separation is the whole point of the rung. The interface
// is intentionally tiny so a host can bridge git/DOS evidence without this
// foundation-tier package importing DOS.
type Witness interface {
	WitnessClaim(Claim) WitnessRecord
}

// WitnessFunc adapts an ordinary function to the Witness interface.
type WitnessFunc func(Claim) WitnessRecord

// WitnessClaim calls the underlying function.
func (f WitnessFunc) WitnessClaim(c Claim) WitnessRecord { return f(c) }

// PathWitness corroborates a claim by checking that every "path" EvidenceRef
// exists on disk. It is a small, network-free example of an out-of-process
// witness: it reads the filesystem, not the process's own claim. Exists is
// injectable so tests need no real files; it defaults to an os.Stat probe.
type PathWitness struct {
	Exists func(string) bool
}

// WitnessClaim returns VerifiedDone when every referenced path exists,
// VerifiedRefused when one is missing, and VerifiedUnavailable when the claim
// carries no path evidence to read back.
func (w PathWitness) WitnessClaim(c Claim) WitnessRecord {
	exists := w.Exists
	if exists == nil {
		exists = func(p string) bool { _, err := os.Stat(p); return err == nil }
	}
	checked := 0
	var missing string
	for _, ref := range c.Refs {
		if ref.Kind != "path" {
			continue
		}
		checked++
		if !exists(ref.Ref) {
			if missing == "" {
				missing = ref.Ref
			}
		}
	}
	rec := WitnessRecord{Source: "path", EvidenceRefs: c.Refs}
	switch {
	case checked == 0:
		rec.VerifiedState = VerifiedUnavailable
		rec.Detail = "no path evidence to read back"
	case missing == "":
		rec.VerifiedState = VerifiedDone
		rec.Detail = "all referenced paths exist"
	default:
		rec.VerifiedState = VerifiedRefused
		rec.Verdict = "missing path evidence"
		rec.Detail = missing
	}
	return rec
}

// SetTaskWitness attaches a precomputed WitnessRecord to a task, leaving the
// task's claimed State untouched. It errors on an unknown task or a verified state
// outside the vocabulary.
func (m *Manager) SetTaskWitness(taskID string, rec WitnessRecord) error {
	if err := validateVerifiedState(rec.VerifiedState); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	task, ok := m.tasks[taskID]
	if !ok {
		return fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	task.witness = cloneWitness(&rec)
	return nil
}

// SetStepWitness attaches a precomputed WitnessRecord to a step, leaving the
// step's claimed State untouched.
func (m *Manager) SetStepWitness(taskID, stepID string, rec WitnessRecord) error {
	if err := validateVerifiedState(rec.VerifiedState); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	step, err := m.stepLocked(taskID, stepID)
	if err != nil {
		return err
	}
	step.witness = cloneWitness(&rec)
	return nil
}

// WitnessTask builds a Claim from the task's current claimed state plus refs, runs
// w against it (outside the lock, so a witness may do I/O), stores the resulting
// record, and returns it. The claimed State is never overwritten: a refused or
// unavailable witness leaves the claim standing and visible alongside the verdict.
func (m *Manager) WitnessTask(taskID string, w Witness, refs []EvidenceRef) (WitnessRecord, error) {
	if w == nil {
		return WitnessRecord{}, fmt.Errorf("taskmgr: nil witness")
	}
	m.mu.Lock()
	task, ok := m.tasks[taskID]
	if !ok {
		m.mu.Unlock()
		return WitnessRecord{}, fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	claim := Claim{TaskID: taskID, State: task.state, Refs: refs}
	m.mu.Unlock()

	rec := w.WitnessClaim(claim)
	if rec.CheckedUnixNano == 0 {
		rec.CheckedUnixNano = m.clock().UnixNano()
	}
	if err := m.SetTaskWitness(taskID, rec); err != nil {
		return WitnessRecord{}, err
	}
	return rec, nil
}

func validateVerifiedState(s VerifiedState) error {
	switch s {
	case VerifiedUnknown, VerifiedDone, VerifiedRefused, VerifiedUnavailable:
		return nil
	default:
		return fmt.Errorf("taskmgr: unknown verified state %q", s)
	}
}

// validateWitness checks an optional witness record attached to a task or step. A
// nil witness is valid: an unwitnessed record carries only the process's claim.
func validateWitness(ctx string, w *WitnessRecord) error {
	if w == nil {
		return nil
	}
	if err := validateVerifiedState(w.VerifiedState); err != nil {
		return fmt.Errorf("%s witness: %w", ctx, err)
	}
	return nil
}

func cloneWitness(w *WitnessRecord) *WitnessRecord {
	if w == nil {
		return nil
	}
	out := *w
	if len(w.EvidenceRefs) > 0 {
		out.EvidenceRefs = make([]EvidenceRef, len(w.EvidenceRefs))
		copy(out.EvidenceRefs, w.EvidenceRefs)
	} else {
		out.EvidenceRefs = nil
	}
	return &out
}
