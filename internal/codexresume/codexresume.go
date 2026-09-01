package codexresume

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Outcome is the typed boundary result of one headless resume process.
type Outcome string

const (
	OutcomeExited                Outcome = "process_exited"
	OutcomeCompleted             Outcome = "completed"
	OutcomeCompletedReclaimed    Outcome = "completed_process_reclaimed"
	OutcomeTurnFailed            Outcome = "turn_failed"
	OutcomeTurnFailedReclaimed   Outcome = "turn_failed_process_reclaimed"
	OutcomeRefused               Outcome = "resume_refused"
	OutcomeCheckOnly             Outcome = "check_only"
	OutcomeExplicitCancelled     Outcome = "explicit_cancelled"
	OutcomeUpstreamInterrupted   Outcome = "upstream_interrupted"
	OutcomeStalledBeforeTerminal Outcome = "process_stalled_before_terminal"
)

// Config binds one owned process to the rollout file that independently witnesses it.
type Config struct {
	Command      []string
	Dir          string
	Env          []string
	RolloutPath  string
	Deadline     time.Duration
	PollInterval time.Duration
	Drain        time.Duration
	Stdout       io.Writer
	Stderr       io.Writer
}

type LaunchState string

const (
	LaunchNotAttempted LaunchState = "not_attempted"
	LaunchStartFailed  LaunchState = "start_failed"
	LaunchStarted      LaunchState = "started"
	LaunchCompleted    LaunchState = "completed"
)

// Result is safe to serialize: it contains no prompt, command, or transcript body.
type Result struct {
	ThreadID          string           `json:"thread_id,omitempty"`
	Preflight         *PreflightResult `json:"preflight,omitempty"`
	Outcome           Outcome          `json:"outcome"`
	ProcessExit       bool             `json:"process_exit"`
	ExitCode          int              `json:"exit_code,omitempty"`
	LaunchPID         int              `json:"launch_pid"`
	LaunchState       LaunchState      `json:"launch_state"`
	TurnID            string           `json:"turn_id,omitempty"`
	TurnStatus        string           `json:"turn_status,omitempty"`
	TurnError         *TurnError       `json:"turn_error,omitempty"`
	TaskStarted       bool             `json:"task_started"`
	UsefulWork        bool             `json:"useful_work_reached"`
	TaskCompleted     bool             `json:"task_completed"`
	Interrupted       bool             `json:"interrupted"`
	ForcedReclaim     bool             `json:"forced_reclaim"`
	WriterLockCleanup string           `json:"writer_lock_cleanup,omitempty"`
	DurationMS        int64            `json:"duration_ms"`
}

type rolloutState struct {
	started, useful, completed, failed, terminal, interrupted bool
	turnID                                                    string
	turnError                                                 *TurnError
}

// TurnError is the provider-authored terminal error attached to task_complete.
// It deliberately carries no prompt, command, or transcript body.
type TurnError struct {
	Type    string `json:"type,omitempty"`
	Message string `json:"message"`
	Status  int    `json:"status,omitempty"`
}

// Run launches exactly one process and trusts the append-only rollout terminal,
// not the upstream process' willingness to exit. A completed task is success;
// if the process does not drain promptly, Run reclaims only that owned tree.
func Run(ctx context.Context, cfg Config) (Result, error) {
	result := Result{LaunchState: LaunchNotAttempted}
	if len(cfg.Command) == 0 {
		return result, errors.New("codexresume: command is required")
	}
	if cfg.RolloutPath == "" {
		return result, errors.New("codexresume: rollout path is required")
	}
	if cfg.Deadline <= 0 {
		cfg.Deadline = 10 * time.Minute
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 100 * time.Millisecond
	}
	if cfg.Drain <= 0 {
		cfg.Drain = time.Second
	}
	startOffset, err := fileSize(cfg.RolloutPath)
	if err != nil {
		return result, fmt.Errorf("codexresume: baseline rollout: %w", err)
	}
	cmd := windowgate.Command(cfg.Command[0], cfg.Command[1:]...)
	cmd.Dir = cfg.Dir
	if cfg.Env != nil {
		cmd.Env = cfg.Env
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return result, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return result, err
	}
	if err := configureOwnedProcess(cmd); err != nil {
		return result, err
	}
	startedAt := time.Now()
	if err := cmd.Start(); err != nil {
		result.LaunchState = LaunchStartFailed
		result.DurationMS = time.Since(startedAt).Milliseconds()
		return result, fmt.Errorf("codexresume: start: %w", err)
	}
	launchPID := cmd.Process.Pid
	result.LaunchPID = launchPID
	result.LaunchState = LaunchStarted
	var copies sync.WaitGroup
	copies.Add(2)
	go func() { defer copies.Done(); _, _ = io.Copy(writerOrDiscard(cfg.Stdout), stdout) }()
	go func() { defer copies.Done(); _, _ = io.Copy(writerOrDiscard(cfg.Stderr), stderr) }()
	exitCh := make(chan error, 1)
	go func() { exitCh <- cmd.Wait() }()

	deadline := time.NewTimer(cfg.Deadline)
	defer deadline.Stop()
	ticker := time.NewTicker(cfg.PollInterval)
	defer ticker.Stop()
	var state rolloutState
	finish := func(r Result) Result {
		r.LaunchPID = launchPID
		r.LaunchState = LaunchCompleted
		r.DurationMS = time.Since(startedAt).Milliseconds()
		return r
	}
	for {
		select {
		case waitErr := <-exitCh:
			copies.Wait()
			state, _ = scanRollout(cfg.RolloutPath, startOffset)
			r := resultFromState(state)
			r.ProcessExit = true
			if cmd.ProcessState != nil {
				r.ExitCode = cmd.ProcessState.ExitCode()
			}
			r.Outcome = outcomeFromState(state, false)
			if waitErr != nil && r.ExitCode == 0 {
				return finish(r), waitErr
			}
			return finish(r), nil
		case <-ticker.C:
			var scanErr error
			state, scanErr = scanRollout(cfg.RolloutPath, startOffset)
			if scanErr != nil {
				_ = killOwnedProcessTree(cmd)
				<-exitCh
				copies.Wait()
				return finish(result), scanErr
			}
			if state.terminal {
				t := time.NewTimer(cfg.Drain)
				select {
				case waitErr := <-exitCh:
					t.Stop()
					copies.Wait()
					r := resultFromState(state)
					r.ProcessExit = true
					r.Outcome = outcomeFromState(state, false)
					if cmd.ProcessState != nil {
						r.ExitCode = cmd.ProcessState.ExitCode()
					}
					if waitErr != nil && r.ExitCode == 0 {
						return finish(r), waitErr
					}
					return finish(r), nil
				case <-t.C:
					_ = killOwnedProcessTree(cmd)
					<-exitCh
					copies.Wait()
					r := resultFromState(state)
					r.Outcome = outcomeFromState(state, true)
					r.ForcedReclaim = true
					return finish(r), nil
				case <-ctx.Done():
					t.Stop()
					_ = killOwnedProcessTree(cmd)
					<-exitCh
					copies.Wait()
					r := resultFromState(state)
					r.Outcome = OutcomeExplicitCancelled
					r.ForcedReclaim = true
					return finish(r), nil
				}
			}

		case <-ctx.Done():
			_ = killOwnedProcessTree(cmd)
			<-exitCh
			copies.Wait()
			r := resultFromState(state)
			r.Outcome = OutcomeExplicitCancelled
			r.ForcedReclaim = true
			return finish(r), nil
		case <-deadline.C:
			_ = killOwnedProcessTree(cmd)
			<-exitCh
			copies.Wait()
			r := resultFromState(state)
			r.Outcome = OutcomeStalledBeforeTerminal
			r.ForcedReclaim = true
			return finish(r), nil
		}
	}
}

func resultFromState(s rolloutState) Result {
	status := ""
	switch {
	case s.completed:
		status = "completed"
	case s.failed:
		status = "failed"
	case s.interrupted:
		status = "interrupted"
	case s.started:
		status = "running"
	}
	return Result{
		TurnID:        s.turnID,
		TurnStatus:    status,
		TurnError:     s.turnError,
		TaskStarted:   s.started,
		UsefulWork:    s.useful,
		TaskCompleted: s.completed,
		Interrupted:   s.interrupted,
	}
}

func outcomeFromState(s rolloutState, reclaimed bool) Outcome {
	switch {
	case s.completed && reclaimed:
		return OutcomeCompletedReclaimed
	case s.completed:
		return OutcomeCompleted
	case s.failed && reclaimed:
		return OutcomeTurnFailedReclaimed
	case s.failed:
		return OutcomeTurnFailed
	case s.interrupted:
		return OutcomeUpstreamInterrupted
	default:
		return OutcomeExited
	}
}
func writerOrDiscard(w io.Writer) io.Writer {
	if w == nil {
		return io.Discard
	}
	return w
}
func fileSize(path string) (int64, error) {
	s, e := os.Stat(path)
	if e != nil {
		return 0, e
	}
	return s.Size(), nil
}

func scanRollout(path string, offset int64) (rolloutState, error) {
	f, err := os.Open(path)
	if err != nil {
		return rolloutState{}, fmt.Errorf("codexresume: scan rollout: %w", err)
	}
	defer f.Close()
	if _, err = f.Seek(offset, io.SeekStart); err != nil {
		return rolloutState{}, err
	}
	var s rolloutState
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 8*1024*1024)
	for sc.Scan() {
		var row struct {
			Type    string `json:"type"`
			Payload struct {
				Type   string          `json:"type"`
				Reason string          `json:"reason"`
				TurnID string          `json:"turn_id"`
				Error  json.RawMessage `json:"error"`
			} `json:"payload"`
		}
		if json.Unmarshal(sc.Bytes(), &row) != nil {
			continue
		}
		if row.Type == "event_msg" {
			switch row.Payload.Type {
			case "task_started":
				s = rolloutState{started: true, turnID: row.Payload.TurnID}
				s.interrupted = false
			case "task_complete":
				if s.started {
					s.terminal = true
					if turnErr := parseTurnError(row.Payload.Error); turnErr != nil {
						s.failed = true
						s.turnError = turnErr
					} else {
						s.completed = true
					}
				}
			case "turn_aborted":
				if s.started && row.Payload.Reason == "interrupted" {
					s.interrupted = true
					s.terminal = true
				}
			}
		}
		if s.started && row.Type == "response_item" && (row.Payload.Type == "function_call_output" || row.Payload.Type == "custom_tool_call_output") {
			s.useful = true
		}
	}
	return s, sc.Err()
}

func parseTurnError(raw json.RawMessage) *TurnError {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var outer struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Status  int    `json:"status"`
	}
	if err := json.Unmarshal(raw, &outer); err != nil {
		return &TurnError{Message: string(raw)}
	}
	out := &TurnError{Type: outer.Type, Message: outer.Message, Status: outer.Status}
	if outer.Message == "" {
		out.Message = string(raw)
		return out
	}
	var nested struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
		Status int `json:"status"`
	}
	if json.Unmarshal([]byte(outer.Message), &nested) == nil && nested.Error.Message != "" {
		out.Type = nested.Error.Type
		out.Message = nested.Error.Message
		out.Status = nested.Status
	}
	return out
}
