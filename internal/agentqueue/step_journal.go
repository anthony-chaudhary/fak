// Adapted from dbos-inc/dbos-transact-golang, dbos/workflow.go (RunAsStep step
// memoization and CancelWorkflow), pinned at
// d516e2191d78673e2e5623a51164b0fec11e7cc9, MIT License, Copyright (c) 2024
// DBOS, Inc. Reimplemented over an append-only JSONL file instead of a
// Postgres system database.

package agentqueue

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// StepJournalSchema tags every step journal record so foreign files are refused.
const StepJournalSchema = "fak.agentqueue.step-journal.v1"

var (
	ErrRunCancelled          = errors.New("agentqueue: run cancelled")
	ErrStepNondeterministic  = errors.New("agentqueue: step does not match recorded journal")
	ErrStepJournalCorrupt    = errors.New("agentqueue: step journal corrupt")
	errStepJournalInvalidArg = errors.New("agentqueue: invalid step journal argument")
)

const (
	stepRecordKind   = "step"
	cancelRecordKind = "cancel"

	cancelBackoffBase = 10 * time.Millisecond
	cancelBackoffMax  = time.Second
)

// StepJournalOptions injects the clock and the cancel-retry wait.
type StepJournalOptions struct {
	Now  func() time.Time
	Wait func(ctx context.Context, d time.Duration) error
}

// StepJournal memoizes step outcomes per run so a resumed run skips completed steps.
type StepJournal struct {
	dir  string
	now  func() time.Time
	wait func(ctx context.Context, d time.Duration) error
	// appendLine is a seam for durable-write failures; nil uses appendStepJournalLine.
	appendLine func(path string, offset int64, line []byte) error

	mu   sync.Mutex
	runs map[string]*stepJournalRun
}

type stepJournalRun struct {
	path      string
	size      int64
	steps     map[int]stepJournalRecord
	cancelled bool
}

type stepJournalRecord struct {
	Schema string          `json:"schema"`
	RunID  string          `json:"run_id"`
	Kind   string          `json:"kind"`
	StepID int             `json:"step_id"`
	Name   string          `json:"name,omitempty"`
	Output json.RawMessage `json:"output,omitempty"`
	Failed bool            `json:"failed,omitempty"`
	Error  string          `json:"error,omitempty"`
	Reason string          `json:"reason,omitempty"`
	At     time.Time       `json:"at"`
}

// RecordedStepError is a step failure replayed from the journal.
type RecordedStepError struct {
	RunID   string
	StepID  int
	Name    string
	Message string
}

func (e *RecordedStepError) Error() string { return e.Message }

// OpenStepJournal returns a journal storing one JSONL file per run under dir.
func OpenStepJournal(dir string, opts StepJournalOptions) (*StepJournal, error) {
	if dir == "" {
		return nil, fmt.Errorf("%w: journal directory is required", errStepJournalInvalidArg)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("agentqueue: create step journal directory: %w", err)
	}
	j := &StepJournal{dir: dir, now: opts.Now, wait: opts.Wait, runs: map[string]*stepJournalRun{}}
	if j.now == nil {
		j.now = time.Now
	}
	if j.wait == nil {
		j.wait = waitStepJournal
	}
	return j, nil
}

func waitStepJournal(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// StepRun hands out deterministic step ids for one execution of a run; it is
// not safe for concurrent use because step order is the replay identity.
type StepRun struct {
	journal *StepJournal
	runID   string
	next    int
}

// Run starts a fresh execution of runID whose first step id is zero.
func (j *StepJournal) Run(runID string) (*StepRun, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if _, err := j.loadRunLocked(runID); err != nil {
		return nil, err
	}
	return &StepRun{journal: j, runID: runID}, nil
}

// RunStep returns the recorded outcome of the run's next step without calling
// fn, or calls fn and durably records its outcome before returning it.
func RunStep[T any](ctx context.Context, run *StepRun, name string, fn func(context.Context) (T, error)) (T, error) {
	var zero T
	if run == nil || fn == nil || name == "" {
		return zero, fmt.Errorf("%w: run, step name and step function are required", errStepJournalInvalidArg)
	}
	stepID := run.next
	run.next++
	j := run.journal

	recorded, found, err := j.lookupStep(run.runID, stepID)
	if err != nil {
		return zero, err
	}
	if found {
		if recorded.Name != name {
			return zero, fmt.Errorf("%w: run %q step %d recorded as %q, replayed as %q",
				ErrStepNondeterministic, run.runID, stepID, recorded.Name, name)
		}
		return decodeRecordedStep[T](recorded)
	}

	out, stepErr := fn(ctx)
	// An interrupted step leaves no record so a resume re-executes it.
	if ctx.Err() != nil {
		return out, fmt.Errorf("agentqueue: run %q step %d interrupted: %w", run.runID, stepID, errors.Join(stepErr, ctx.Err()))
	}
	rec := stepJournalRecord{Kind: stepRecordKind, StepID: stepID, Name: name}
	if stepErr != nil {
		rec.Failed = true
		rec.Error = stepErr.Error()
	} else {
		rec.Output, err = json.Marshal(out)
		if err != nil {
			return zero, fmt.Errorf("agentqueue: encode output of run %q step %d: %w", run.runID, stepID, err)
		}
	}
	prior, err := j.recordStep(run.runID, rec)
	if err != nil {
		return zero, err
	}
	// Another execution of the run recorded this step first; its outcome is
	// the durable one, so every execution observes the same value.
	if prior != nil {
		return decodeRecordedStep[T](*prior)
	}
	return out, stepErr
}

func decodeRecordedStep[T any](rec stepJournalRecord) (T, error) {
	var out T
	if rec.Failed {
		return out, &RecordedStepError{RunID: rec.RunID, StepID: rec.StepID, Name: rec.Name, Message: rec.Error}
	}
	if err := json.Unmarshal(rec.Output, &out); err != nil {
		var zero T
		return zero, fmt.Errorf("agentqueue: decode recorded output of run %q step %d: %w", rec.RunID, rec.StepID, err)
	}
	return out, nil
}

func (j *StepJournal) lookupStep(runID string, stepID int) (stepJournalRecord, bool, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	state, err := j.loadRunLocked(runID)
	if err != nil {
		return stepJournalRecord{}, false, err
	}
	if state.cancelled {
		return stepJournalRecord{}, false, fmt.Errorf("%w: %q", ErrRunCancelled, runID)
	}
	rec, ok := state.steps[stepID]
	return rec, ok, nil
}

// recordStep appends rec, or returns the record another execution already
// committed for the same step.
func (j *StepJournal) recordStep(runID string, rec stepJournalRecord) (*stepJournalRecord, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	state, err := j.loadRunLocked(runID)
	if err != nil {
		return nil, err
	}
	// A cancel committed while the step ran wins; the outcome is discarded.
	if state.cancelled {
		return nil, fmt.Errorf("%w: %q", ErrRunCancelled, runID)
	}
	if prior, ok := state.steps[rec.StepID]; ok {
		if prior.Name != rec.Name {
			return nil, fmt.Errorf("%w: run %q step %d recorded as %q, replayed as %q",
				ErrStepNondeterministic, runID, rec.StepID, prior.Name, rec.Name)
		}
		return &prior, nil
	}
	if err := j.appendLocked(runID, state, &rec); err != nil {
		return nil, err
	}
	state.steps[rec.StepID] = rec
	return nil, nil
}

// CancelRun durably records a cancel for runID, retrying the write with capped
// backoff until it commits or ctx ends. Later steps of the run are refused; a
// run that has not started yet is cancelled pre-emptively.
func (j *StepJournal) CancelRun(ctx context.Context, runID, reason string) error {
	var lastErr error
	for attempt := 1; ; attempt++ {
		done, err := j.tryCancel(runID, reason)
		if done {
			return nil
		}
		if errors.Is(err, ErrStepJournalCorrupt) || errors.Is(err, errStepJournalInvalidArg) {
			return err
		}
		lastErr = err
		if werr := j.wait(ctx, cancelBackoff(attempt)); werr != nil {
			return fmt.Errorf("agentqueue: cancel of run %q not committed after %d attempts: %w", runID, attempt, errors.Join(lastErr, werr))
		}
	}
}

func cancelBackoff(attempt int) time.Duration {
	d := cancelBackoffBase
	for i := 1; i < attempt && d < cancelBackoffMax; i++ {
		d *= 2
	}
	return min(d, cancelBackoffMax)
}

func (j *StepJournal) tryCancel(runID, reason string) (bool, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	state, err := j.loadRunLocked(runID)
	if err != nil {
		return false, err
	}
	if state.cancelled {
		return true, nil
	}
	if err := j.appendLocked(runID, state, &stepJournalRecord{Kind: cancelRecordKind, Reason: reason}); err != nil {
		return false, err
	}
	state.cancelled = true
	return true, nil
}

func (j *StepJournal) appendLocked(runID string, state *stepJournalRun, rec *stepJournalRecord) error {
	rec.Schema = StepJournalSchema
	rec.RunID = runID
	rec.At = j.now().UTC()
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("agentqueue: encode step journal record: %w", err)
	}
	line = append(line, '\n')
	write := j.appendLine
	if write == nil {
		write = appendStepJournalLine
	}
	if err := write(state.path, state.size, line); err != nil {
		return fmt.Errorf("agentqueue: append step journal: %w", err)
	}
	state.size += int64(len(line))
	return nil
}

// appendStepJournalLine writes line at offset and syncs it; on failure it
// truncates back to offset so a retried append never follows a partial line.
func appendStepJournalLine(path string, offset int64, line []byte) (err error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = f.Truncate(offset)
		}
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()
	if _, err = f.WriteAt(line, offset); err != nil {
		return err
	}
	return f.Sync()
}

func (j *StepJournal) runPath(runID string) string {
	sum := sha256.Sum256([]byte(runID))
	return filepath.Join(j.dir, "run-"+hex.EncodeToString(sum[:16])+".steps.jsonl")
}

func (j *StepJournal) loadRunLocked(runID string) (*stepJournalRun, error) {
	if runID == "" {
		return nil, fmt.Errorf("%w: run id is required", errStepJournalInvalidArg)
	}
	if state, ok := j.runs[runID]; ok {
		return state, nil
	}
	state := &stepJournalRun{path: j.runPath(runID), steps: map[int]stepJournalRecord{}}
	body, err := os.ReadFile(state.path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("agentqueue: read step journal: %w", err)
	}
	for offset := 0; offset < len(body); {
		end := bytes.IndexByte(body[offset:], '\n')
		if end < 0 {
			// An unterminated tail is a write torn by a crash; drop it.
			if err := os.Truncate(state.path, int64(offset)); err != nil {
				return nil, fmt.Errorf("agentqueue: truncate torn step journal tail: %w", err)
			}
			break
		}
		var rec stepJournalRecord
		if err := json.Unmarshal(body[offset:offset+end], &rec); err != nil {
			return nil, fmt.Errorf("%w: %s at byte %d: %v", ErrStepJournalCorrupt, state.path, offset, err)
		}
		if rec.Schema != StepJournalSchema || rec.RunID != runID {
			return nil, fmt.Errorf("%w: %s at byte %d: foreign record", ErrStepJournalCorrupt, state.path, offset)
		}
		switch rec.Kind {
		case stepRecordKind:
			if _, dup := state.steps[rec.StepID]; !dup {
				state.steps[rec.StepID] = rec
			}
		case cancelRecordKind:
			state.cancelled = true
		default:
			return nil, fmt.Errorf("%w: %s at byte %d: unknown kind %q", ErrStepJournalCorrupt, state.path, offset, rec.Kind)
		}
		offset += end + 1
		state.size = int64(offset)
	}
	j.runs[runID] = state
	return state, nil
}
