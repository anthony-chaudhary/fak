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

// Result is safe to serialize: it contains no prompt, command, or transcript body.
type Result struct {
	Outcome       Outcome `json:"outcome"`
	ProcessExit   bool    `json:"process_exit"`
	ExitCode      int     `json:"exit_code,omitempty"`
	TaskStarted   bool    `json:"task_started"`
	UsefulWork    bool    `json:"useful_work_reached"`
	TaskCompleted bool    `json:"task_completed"`
	Interrupted   bool    `json:"interrupted"`
	ForcedReclaim bool    `json:"forced_reclaim"`
	DurationMS    int64   `json:"duration_ms"`
}

type rolloutState struct {
	started, useful, completed, interrupted bool
}

// Run launches exactly one process and trusts the append-only rollout terminal,
// not the upstream process' willingness to exit. A completed task is success;
// if the process does not drain promptly, Run reclaims only that owned tree.
func Run(ctx context.Context, cfg Config) (Result, error) {
	if len(cfg.Command) == 0 {
		return Result{}, errors.New("codexresume: command is required")
	}
	if cfg.RolloutPath == "" {
		return Result{}, errors.New("codexresume: rollout path is required")
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
		return Result{}, fmt.Errorf("codexresume: baseline rollout: %w", err)
	}
	cmd := windowgate.Command(cfg.Command[0], cfg.Command[1:]...)
	cmd.Dir = cfg.Dir
	if cfg.Env != nil {
		cmd.Env = cfg.Env
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Result{}, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return Result{}, err
	}
	if err := configureOwnedProcess(cmd); err != nil {
		return Result{}, err
	}
	startedAt := time.Now()
	if err := cmd.Start(); err != nil {
		return Result{}, fmt.Errorf("codexresume: start: %w", err)
	}
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
	finish := func(r Result) Result { r.DurationMS = time.Since(startedAt).Milliseconds(); return r }
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
			if state.completed {
				r.Outcome = OutcomeCompleted
			} else {
				r.Outcome = OutcomeExited
			}
			if waitErr != nil && r.ExitCode == 0 {
				return finish(r), waitErr
			}
			return finish(r), nil
		case <-ticker.C:
			var scanErr error
			state, scanErr = scanRollout(cfg.RolloutPath, startOffset)
			if scanErr != nil {
				_ = killOwnedProcessTree(cmd)
				return Result{}, scanErr
			}
			if state.completed {
				t := time.NewTimer(cfg.Drain)
				select {
				case waitErr := <-exitCh:
					t.Stop()
					copies.Wait()
					r := resultFromState(state)
					r.ProcessExit = true
					r.Outcome = OutcomeCompleted
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
					r.Outcome = OutcomeCompletedReclaimed
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
	return Result{TaskStarted: s.started, UsefulWork: s.useful, TaskCompleted: s.completed, Interrupted: s.interrupted}
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
			Type    string                        `json:"type"`
			Payload struct{ Type, Reason string } `json:"payload"`
		}
		if json.Unmarshal(sc.Bytes(), &row) != nil {
			continue
		}
		if row.Type == "event_msg" {
			switch row.Payload.Type {
			case "task_started":
				s.started = true
				s.interrupted = false
			case "task_complete":
				if s.started {
					s.completed = true
				}
			case "turn_aborted":
				if s.started && row.Payload.Reason == "interrupted" {
					s.interrupted = true
				}
			}
		}
		if s.started && row.Type == "response_item" && (row.Payload.Type == "function_call_output" || row.Payload.Type == "custom_tool_call_output") {
			s.useful = true
		}
	}
	return s, sc.Err()
}
