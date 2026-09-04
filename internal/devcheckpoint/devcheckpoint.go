// Package devcheckpoint records concise, durable agent progress milestones.
package devcheckpoint

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// Invariant: dev checkpoint progress calculation is fail-closed and monotonic.
// Contract: stage bounds require 1 <= current <= total whenever stages are evaluated.
// Guard: transitions in blocked state require at least one explicit blocker explanation.
// Guard: transitions in progress state require positive non-zero stage bounds.

// State defines the execution lifecycle state of a developer progress checkpoint.
type State string

const (
	// StateStarted indicates the initial kickoff of an agent work session.
	StateStarted State = "started"
	// StateProgress indicates active forward movement with stage tracking.
	StateProgress State = "progress"
	// StateBlocked indicates execution halted due to explicit blocking dependencies.
	StateBlocked State = "blocked"
	// StateHandoff indicates delegation of execution context to a peer or supervisor.
	StateHandoff State = "handoff"
	// StateDone indicates verified terminal completion of the assigned task scope.
	StateDone State = "done"
)

// Stage captures structured phase progress bounds and calculated completion percentage.
type Stage struct {
	Current int    `json:"current"`
	Total   int    `json:"total"`
	Name    string `json:"name,omitempty"`
	Percent int    `json:"percent"`
}

// Record contains an immutable checkpoint entry persisted to the durable progress log.
type Record struct {
	Timestamp time.Time `json:"timestamp"`
	Actor     string    `json:"actor"`
	Scope     string    `json:"scope"`
	State     State     `json:"state"`
	Stage     *Stage    `json:"stage,omitempty"`
	Summary   string    `json:"summary"`
	Evidence  []string  `json:"evidence,omitempty"`
	Next      string    `json:"next,omitempty"`
	Blockers  []string  `json:"blockers,omitempty"`
	GitHub    string    `json:"github,omitempty"`
}

// Input provides parameter values required to construct a valid checkpoint record.
type Input struct {
	Actor        string
	Scope        string
	State        State
	StageCurrent int
	StageTotal   int
	StageName    string
	Summary      string
	Evidence     []string
	Next         string
	Blockers     []string
	GitHub       string
}

// New validates input against fail-closed rules and constructs an immutable Record in UTC.
func New(in Input, now time.Time) (Record, error) {
	in.Actor, in.Scope, in.Summary = strings.TrimSpace(in.Actor), strings.TrimSpace(in.Scope), strings.TrimSpace(in.Summary)
	in.StageName, in.Next, in.GitHub = strings.TrimSpace(in.StageName), strings.TrimSpace(in.Next), strings.TrimSpace(in.GitHub)
	in.Evidence, in.Blockers = clean(in.Evidence), clean(in.Blockers)
	switch {
	case in.Actor == "":
		return Record{}, errors.New("actor is required")
	case in.Scope == "":
		return Record{}, errors.New("scope is required")
	case in.Summary == "":
		return Record{}, errors.New("summary is required")
	}
	switch in.State {
	case StateStarted, StateProgress, StateBlocked, StateHandoff, StateDone:
	default:
		return Record{}, fmt.Errorf("state must be started, progress, blocked, handoff, or done (got %q)", in.State)
	}
	if in.State != StateDone && in.Next == "" {
		return Record{}, fmt.Errorf("next is required for %s", in.State)
	}
	if in.State == StateBlocked && len(in.Blockers) == 0 {
		return Record{}, errors.New("at least one blocker is required for blocked")
	}
	if in.State == StateProgress && (in.StageCurrent == 0 || in.StageTotal == 0) {
		return Record{}, errors.New("stage-current and stage-total are required for progress")
	}
	if in.StageCurrent != 0 || in.StageTotal != 0 || in.StageName != "" {
		if in.StageCurrent < 1 || in.StageTotal < 1 || in.StageCurrent > in.StageTotal {
			return Record{}, errors.New("stage requires 1 <= current <= total")
		}
	}
	r := Record{Timestamp: now.UTC(), Actor: in.Actor, Scope: in.Scope, State: in.State, Summary: in.Summary, Evidence: in.Evidence, Next: in.Next, Blockers: in.Blockers, GitHub: in.GitHub}
	if in.StageCurrent != 0 {
		r.Stage = &Stage{Current: in.StageCurrent, Total: in.StageTotal, Name: in.StageName, Percent: in.StageCurrent * 100 / in.StageTotal}
	}
	return r, nil
}

// Append writes a serialized checkpoint record to disk under an exclusive file lock.
func Append(path string, record Record) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create checkpoint directory: %w", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		err = flock.TryLock(lock)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return err
		}
		if time.Now().After(deadline) {
			return errors.New("checkpoint log lock busy")
		}
		time.Sleep(5 * time.Millisecond)
	}
	defer flock.Unlock(lock)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	b, err := json.Marshal(record)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err = f.Write(b); err != nil {
		return err
	}
	return f.Sync()
}

func clean(values []string) []string {
	var out []string
	for _, value := range values {
		for _, item := range strings.Split(value, ",") {
			if item = strings.TrimSpace(item); item != "" {
				out = append(out, item)
			}
		}
	}
	return out
}
