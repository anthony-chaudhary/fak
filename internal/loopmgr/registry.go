package loopmgr

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/parentdir"
	"os"
	"sort"
	"strings"
	"time"
)

// Registry is the on-disk job registry that lives beside the loop ledger so the
// SET of scheduled jobs — their cron cadence, missed-run policy, and
// enabled/disabled state — survives a host or gateway restart and re-arms at
// boot. The fire ledger already persists (the JSONL hash chain); this is the
// schedule DEFINITION the ledger had no home for.
//
// The load-bearing discipline: a STOPPED/disabled job reloads DISABLED, never
// silently re-armed. This is the dos_recall / session-image re-checkable-fact
// rule applied to schedule state — a fact recovered from disk is honored at the
// value it held, not optimistically reset. Persistence covers the definition,
// not a live process; re-arming a reloaded job still goes through the schedule's
// overlap-lock (Schedule.Next).

// SchemaRegistry is the registry-document schema tag.
const SchemaRegistry = "fak.loop-registry.v1"

// JobState is the persisted arm-state of a registered job. The set is closed and
// mirrors the supervisor schedule states the loop ledger already speaks
// (StateArmed / StateStopped / StateDisabled).
type JobState string

const (
	// JobArmed: the job re-arms at boot and the scheduler may fire it.
	JobArmed JobState = JobState(StateArmed)
	// JobStopped: the job is retired. It reloads stopped and is NOT re-armed.
	JobStopped JobState = JobState(StateStopped)
	// JobDisabled: the job is held by the operator. It reloads disabled and is
	// NOT re-armed.
	JobDisabled JobState = JobState(StateDisabled)
)

// ValidJobState reports whether s is a named member of the closed set. The empty
// string is INVALID: a persisted job must carry an explicit state so a reload
// can never guess "armed" for a row whose state was lost.
func ValidJobState(s JobState) bool {
	switch s {
	case JobArmed, JobStopped, JobDisabled:
		return true
	default:
		return false
	}
}

// Armed reports whether a job in this state should re-arm at boot. Only JobArmed
// re-arms; both JobStopped and JobDisabled stay down. This is the single
// predicate the boot path consults so the "never silently re-armed" rule lives
// in one place.
func (s JobState) Armed() bool { return s == JobArmed }

// Job is one persisted registry entry: the schedule definition plus its arm
// state and bookkeeping. Schedule carries the cadence/policy/jitter; State
// carries whether it re-arms.
type Job struct {
	Schedule Schedule `json:"schedule"`
	State    JobState `json:"state"`

	// CreatedUnixNano / UpdatedUnixNano are registry bookkeeping, stamped by the
	// mutating helpers so a reload can show when a job last changed.
	CreatedUnixNano int64          `json:"created_unix_nano,omitempty"`
	UpdatedUnixNano int64          `json:"updated_unix_nano,omitempty"`
	WakeAtUnixNano  int64          `json:"wake_at_unix_nano,omitempty"`
	Duty            *DutyCycle     `json:"duty_cycle,omitempty"`
	Execution       *ExecutionSpec `json:"execution,omitempty"`
}

// JobID is the job's identity, sourced from its schedule.
func (j Job) JobID() string { return j.Schedule.JobID }

// Executable reports whether the job carries a valid execution specification.
func (j Job) Executable() bool {
	return j.Execution != nil && j.Execution.Validate() == nil
}

// Validate checks that the job's schedule, state, and optional duty cycle or
// execution specification are valid.
func (j Job) Validate() error {
	if err := j.Schedule.Validate(); err != nil {
		return err
	}
	if j.Duty != nil {
		if err := j.Duty.Validate(); err != nil {
			return err
		}
	}
	if j.Execution != nil {
		if err := j.Execution.Validate(); err != nil {
			return err
		}
	}
	if !ValidJobState(j.State) {
		return fmt.Errorf("state = %q, want armed|stopped|disabled (never defaulted)", j.State)
	}
	return nil
}

// Registry is the full persisted job set, keyed by job id.
type Registry struct {
	Schema string         `json:"schema"`
	Jobs   map[string]Job `json:"jobs,omitempty"`
}

// Validate checks the registry is well-formed: a known schema and, for every
// job, a valid schedule, a named state, and a key that matches the job's own id.
func (r Registry) Validate() error {
	if r.Schema != "" && r.Schema != SchemaRegistry {
		return fmt.Errorf("loop registry schema = %q, want %q", r.Schema, SchemaRegistry)
	}
	for key, job := range r.Jobs {
		if key != job.JobID() {
			return fmt.Errorf("loop registry key %q != job id %q", key, job.JobID())
		}
		if err := job.Schedule.Validate(); err != nil {
			return fmt.Errorf("loop registry job %q: %w", key, err)
		}
		if job.Duty != nil {
			if err := job.Duty.Validate(); err != nil {
				return fmt.Errorf("loop registry job %q: %w", key, err)
			}
		}
		if job.Execution != nil {
			if err := job.Execution.Validate(); err != nil {
				return fmt.Errorf("loop registry job %q: %w", key, err)
			}
		}
		if !ValidJobState(job.State) {
			return fmt.Errorf("loop registry job %q: state = %q, want armed|stopped|disabled (never defaulted)", key, job.State)
		}
	}
	return nil
}

// LoadRegistry reads the job registry from path. A missing file is NOT an error:
// it returns an empty registry, so a host that has registered no jobs boots
// clean. A present-but-malformed file IS an error — a corrupt registry is loud,
// never silently dropped. The load is STRICT about state: a row that survives is
// honored at the state it held on disk (a disabled job stays disabled).
func LoadRegistry(path string) (Registry, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Registry{Jobs: map[string]Job{}}, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Registry{Jobs: map[string]Job{}}, nil
	}
	if err != nil {
		return Registry{}, fmt.Errorf("read loop registry: %w", err)
	}
	var r Registry
	if err := json.Unmarshal(b, &r); err != nil {
		return Registry{}, fmt.Errorf("decode loop registry %s: %w", path, err)
	}
	if r.Jobs == nil {
		r.Jobs = map[string]Job{}
	}
	if err := r.Validate(); err != nil {
		return Registry{}, err
	}
	return r, nil
}

// SaveRegistry writes the registry to path atomically (temp file + rename) so a
// crashed write never leaves a half-written registry that would fail to reload.
// It stamps the schema and validates before writing — a registry that would not
// reload is never persisted.
func SaveRegistry(path string, r Registry) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return errors.New("loop registry path is required")
	}
	r.Schema = SchemaRegistry
	if r.Jobs == nil {
		r.Jobs = map[string]Job{}
	}
	if err := r.Validate(); err != nil {
		return err
	}

	if err := parentdir.Ensure(path, 0o755); err != nil {
		return fmt.Errorf("create loop registry dir: %w", err)
	}

	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal loop registry: %w", err)
	}
	tmp, err := os.CreateTemp(parentdir.Dir(path), ".loop-registry-*.tmp")
	if err != nil {
		return fmt.Errorf("create loop registry temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("write loop registry temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("close loop registry temp: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("rename loop registry into place: %w", err)
	}
	return nil
}

// Put inserts or replaces a job, stamping bookkeeping with at. A new job gets a
// CreatedUnixNano; an existing job keeps its original CreatedUnixNano and only
// its UpdatedUnixNano advances. The schedule and state are validated before the
// job is admitted, so the registry never holds an unschedulable row.
func (r *Registry) Put(job Job, at time.Time) error {
	if err := job.Schedule.Validate(); err != nil {
		return err
	}
	if job.Duty != nil {
		if err := job.Duty.Validate(); err != nil {
			return err
		}
	}
	if job.Execution != nil {
		if err := job.Execution.Validate(); err != nil {
			return err
		}
	}
	if !ValidJobState(job.State) {
		return fmt.Errorf("job %q: state = %q, want armed|stopped|disabled (never defaulted)", job.JobID(), job.State)
	}
	if r.Jobs == nil {
		r.Jobs = map[string]Job{}
	}
	ts := at.UTC().UnixNano()
	if existing, ok := r.Jobs[job.JobID()]; ok && existing.CreatedUnixNano != 0 {
		job.CreatedUnixNano = existing.CreatedUnixNano
	} else if job.CreatedUnixNano == 0 {
		job.CreatedUnixNano = ts
	}
	job.UpdatedUnixNano = ts
	r.Jobs[job.JobID()] = job
	return nil
}

// SetState transitions a registered job to a new state, stamping UpdatedUnixNano
// with at. This is the operator's disable/stop/re-arm switch — flip a job's
// arm-state without re-registering its schedule. Returns an error for an unknown
// job id or an invalid target state.
func (r *Registry) SetState(jobID string, state JobState, at time.Time) error {
	jobID = strings.TrimSpace(jobID)
	if !ValidJobState(state) {
		return fmt.Errorf("job %q: target state = %q, want armed|stopped|disabled", jobID, state)
	}
	if r.Jobs == nil {
		return fmt.Errorf("loop registry has no job %q", jobID)
	}
	job, ok := r.Jobs[jobID]
	if !ok {
		return fmt.Errorf("loop registry has no job %q", jobID)
	}
	job.State = state
	job.UpdatedUnixNano = at.UTC().UnixNano()
	r.Jobs[jobID] = job
	return nil
}

// Get returns the registered job for an id and whether it exists.
func (r Registry) Get(jobID string) (Job, bool) {
	if r.Jobs == nil {
		return Job{}, false
	}
	job, ok := r.Jobs[strings.TrimSpace(jobID)]
	return job, ok
}

// List returns every registered job in stable job-id order, so a `fak cron`
// re-list after a restart is deterministic.
func (r Registry) List() []Job {
	out := make([]Job, 0, len(r.Jobs))
	for _, job := range r.Jobs {
		out = append(out, job)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].JobID() < out[j].JobID() })
	return out
}

// ArmedJobs returns the jobs that re-arm at boot — JobArmed only — in stable
// order. This is the boot path's single source for "what does the scheduler
// re-arm?": a stopped or disabled row is structurally excluded, so a reload can
// never silently re-arm a job the operator put down.
func (r Registry) ArmedJobs() []Job {
	out := make([]Job, 0, len(r.Jobs))
	for _, job := range r.List() {
		if job.State.Armed() {
			out = append(out, job)
		}
	}
	return out
}

// BGLoopStore adapts the loop registry to bgloop's lower-tier persistence seam.
type BGLoopStore struct{}

func (BGLoopStore) Validate(path string) error {
	_, err := LoadRegistry(path)
	return err
}

func (BGLoopStore) LookupArmed(path, jobID string) (bgloop.Job, bool, error) {
	r, err := LoadRegistry(path)
	if err != nil {
		return nil, false, err
	}
	job, ok := r.Get(jobID)
	if !ok || !job.State.Armed() {
		return nil, false, nil
	}
	return job, true, nil
}

func (BGLoopStore) SetWake(path, jobID string, at time.Time, duty bgloop.DutyCycle, updatedAt time.Time) error {
	r, err := LoadRegistry(path)
	if err != nil {
		return err
	}
	job, ok := r.Get(jobID)
	if !ok {
		return fmt.Errorf("unknown loop job %q", jobID)
	}
	job.WakeAtUnixNano = at.UTC().UnixNano()
	if duty == nil {
		job.Duty = nil
	} else {
		d, ok := duty.(*DutyCycle)
		if !ok {
			return fmt.Errorf("job %q: incompatible duty cycle %T", jobID, duty)
		}
		job.Duty = d
	}
	job.UpdatedUnixNano = updatedAt.UTC().UnixNano()
	r.Jobs[jobID] = job
	return SaveRegistry(path, r)
}

func (BGLoopStore) ArmedWakes(path string) ([]bgloop.StoredWake, error) {
	r, err := LoadRegistry(path)
	if err != nil {
		return nil, err
	}
	out := make([]bgloop.StoredWake, 0)
	for _, job := range r.ArmedJobs() {
		var at time.Time
		if job.WakeAtUnixNano != 0 {
			at = time.Unix(0, job.WakeAtUnixNano)
		}
		var duty bgloop.DutyCycle
		if job.Duty != nil {
			duty = job.Duty
		}
		out = append(out, bgloop.StoredWake{Job: job, WakeAt: at, Duty: duty})
	}
	return out, nil
}

func (BGLoopStore) MoveWake(path, jobID string, at, updatedAt time.Time) error {
	r, err := LoadRegistry(path)
	if err != nil {
		return err
	}
	job, ok := r.Get(jobID)
	if !ok {
		return fmt.Errorf("unknown loop job %q", jobID)
	}
	if at.IsZero() {
		job.WakeAtUnixNano = 0
	} else {
		job.WakeAtUnixNano = at.UTC().UnixNano()
	}
	job.UpdatedUnixNano = updatedAt.UTC().UnixNano()
	r.Jobs[jobID] = job
	return SaveRegistry(path, r)
}

func (BGLoopStore) ClearWake(path, jobID string, updatedAt time.Time) error {
	return BGLoopStore{}.MoveWake(path, jobID, time.Time{}, updatedAt)
}
