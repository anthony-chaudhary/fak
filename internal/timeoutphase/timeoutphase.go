// Package timeoutphase is a pure classifier over one timed-out worker attempt: given the
// facts the caller observed (which lifecycle stage markers fired before the kill), decide
// WHICH stage the timeout actually happened in -- before the worker ever started, during its
// edit pass, during tests, during commit, or during push (#1793). It never inspects a live
// process or reads a clock; the caller supplies the observed stage facts as data, and this
// package folds them into a Row deterministically. Same Attempt in, same Row out.
//
// # The gap it closes
//
// cmd/dispatchworker/worker.go bounds an unattended worker with a wall-clock deadline
// (defaultTimeoutS) and, on context.DeadlineExceeded, reports one opaque fact: Timeout: true,
// ReturnCode: 124. It carries no notion of WHERE in the worker's lifecycle the deadline hit --
// a worker that timed out before it ever wrote a line of code looks identical, in that report,
// to one that timed out mid-push after doing all the real work. This package adds the missing
// breakdown: it takes the last-observed lifecycle marker(s) for one timed-out attempt and
// classifies the timeout into a closed five-phase vocabulary, so a later pass (a ledger, a
// report, a triage heuristic) can tell "never got going" apart from "died at the finish line".
//
// # Pure and total
//
// Classify takes an Attempt (observed facts only -- no process handle, no filesystem, no
// clock) and returns a Row. Every Attempt classifies to a defined Phase, including the edge
// case of no observed stage markers at all (PhaseUnknown) -- it never panics or errors.
package timeoutphase

// Phase is the closed lifecycle-stage vocabulary a worker timeout is classified into. It is
// deliberately a distinct type from internal/lifecycle.Phase: that package's Phase is the
// run-state skeleton shared by loopmgr/session (running/paused/draining/stopped), a different
// axis entirely from "which stage of one attempt's work was the timeout in".
type Phase string

const (
	// PhaseBeforeStartup: the timeout fired before the worker reached its first observed
	// lifecycle marker -- it never got past preflight/launch (e.g. account/cap wait, process
	// spawn, or the model backend itself never returned a first token).
	PhaseBeforeStartup Phase = "before_startup"
	// PhaseDuringEdit: the worker had started and was observed making edits (its last
	// observed marker was an edit-stage signal) when the timeout hit.
	PhaseDuringEdit Phase = "during_edit"
	// PhaseDuringTests: the worker had moved on to running tests when the timeout hit.
	PhaseDuringTests Phase = "during_tests"
	// PhaseDuringCommit: the worker was in its commit step (staging/committing) when the
	// timeout hit.
	PhaseDuringCommit Phase = "during_commit"
	// PhaseDuringPush: the worker had committed and was pushing (or already pushed and was
	// finishing up, e.g. a slow post-push verification) when the timeout hit.
	PhaseDuringPush Phase = "during_push"
	// PhaseUnknown: the attempt carries no observed stage markers at all (or the caller
	// supplied a stage this package does not recognize). A defined, non-crashing default --
	// never silently coerced to PhaseBeforeStartup, so an unclassifiable attempt stays visibly
	// unclassified rather than looking like a clean "never started" case.
	PhaseUnknown Phase = "unknown"
)

// Stage is the closed vocabulary of lifecycle markers a caller may report as observed for one
// attempt. It mirrors the five named worker stages (startup itself is the absence of any
// stage marker, so it has no constant of its own -- see Classify).
type Stage string

const (
	StageEdit   Stage = "edit"
	StageTest   Stage = "test"
	StageCommit Stage = "commit"
	StagePush   Stage = "push"
)

// stagePhase is the closed, total mapping from an observed Stage to the Phase a timeout in
// that stage classifies to.
var stagePhase = map[Stage]Phase{
	StageEdit:   PhaseDuringEdit,
	StageTest:   PhaseDuringTests,
	StageCommit: PhaseDuringCommit,
	StagePush:   PhaseDuringPush,
}

// Attempt is the caller-observed facts for one timed-out worker attempt. LastStage is the
// most recent lifecycle marker the caller saw fire before the kill; "" (or an unrecognized
// value) means no stage marker was ever observed, so the timeout classifies as
// PhaseBeforeStartup when Started is true, or PhaseUnknown when it is not even known whether
// the worker started.
type Attempt struct {
	// ID names the attempt (e.g. the issue number or run id) for the caller's own bookkeeping;
	// this package only copies it through onto the Row.
	ID string
	// Started reports whether the caller positively observed the worker process/session begin
	// (e.g. a launch log line, a PID recorded) -- distinct from LastStage being empty, which
	// only means no STAGE marker fired, not that startup itself is unconfirmed.
	Started bool
	// LastStage is the last lifecycle marker observed before the timeout. Zero value means
	// none was observed.
	LastStage Stage
	// FailureClass is an optional free-text-avoidance tag the caller already carries (e.g.
	// from an existing attempt-history ledger); this package does not interpret it, only
	// copies it through so a Row stays a complete audit record.
	FailureClass string
	// TimestampUnix is the caller-supplied clock reading for when the timeout was recorded.
	// This package never reads a clock itself.
	TimestampUnix int64
}

// Row is one attempt's classified timeout record.
type Row struct {
	ID            string `json:"id"`
	Phase         Phase  `json:"phase"`
	LastStage     Stage  `json:"last_stage,omitempty"`
	FailureClass  string `json:"failure_class,omitempty"`
	TimestampUnix int64  `json:"timestamp_unix"`
}

// Report aggregates classified rows plus a per-phase count, so a caller can render "N timeouts
// before startup, M during tests, ..." without re-walking Rows.
type Report struct {
	Rows       []Row         `json:"rows"`
	PhaseCount map[Phase]int `json:"phase_count"`
}

// Classify deterministically classifies one timed-out Attempt into a Row. It performs zero
// I/O and reads no clock -- every field of the result is a pure function of a.
//
// Classification order: an observed LastStage wins outright (the worker got at least that
// far); otherwise a positively-observed Started with no stage marker means the worker began
// but died before its first stage signal (PhaseBeforeStartup); otherwise -- Started is false
// and no stage was observed -- the attempt carries no usable evidence at all and classifies to
// PhaseUnknown rather than being guessed into PhaseBeforeStartup.
func Classify(a Attempt) Row {
	row := Row{
		ID:            a.ID,
		LastStage:     a.LastStage,
		FailureClass:  a.FailureClass,
		TimestampUnix: a.TimestampUnix,
	}
	if phase, ok := stagePhase[a.LastStage]; ok {
		row.Phase = phase
		return row
	}
	if a.Started {
		row.Phase = PhaseBeforeStartup
		return row
	}
	row.Phase = PhaseUnknown
	return row
}

// Record classifies every attempt and folds the rows into a Report, tallying PhaseCount.
func Record(attempts []Attempt) Report {
	rep := Report{
		Rows:       make([]Row, 0, len(attempts)),
		PhaseCount: make(map[Phase]int),
	}
	for _, a := range attempts {
		row := Classify(a)
		rep.Rows = append(rep.Rows, row)
		rep.PhaseCount[row.Phase]++
	}
	return rep
}
