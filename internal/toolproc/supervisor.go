package toolproc

import (
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
	"unicode/utf8"
)

// ProcessState represents the typed lifecycle state of a supervised tool process.
type ProcessState string

const (
	ProcessRunning  ProcessState = "RUNNING"
	ProcessExited   ProcessState = "EXITED"
	ProcessFailed   ProcessState = "FAILED"
	ProcessTimedOut ProcessState = "TIMED_OUT"
)

const (
	// DefaultPollLivelockThreshold is the maximum consecutive poll attempts on a
	// dead handle before livelock suppression trips.
	DefaultPollLivelockThreshold = 4

	// DefaultWriteLivelockThreshold is the maximum consecutive write attempts on
	// a dead handle before the circuit breaker trips.
	DefaultWriteLivelockThreshold = 4

	// DefaultTombstoneCapacity is the maximum number of tombstoned processes retained
	// before oldest entries are evicted FIFO.
	DefaultTombstoneCapacity = 1024

	// DefaultMaxTailBytes is the maximum tail length in bytes retained for stdout/stderr (4KB).
	DefaultMaxTailBytes = 4096
)

// ErrProcessTerminated is returned when an operation is attempted on a terminated process.
type ErrProcessTerminated struct {
	PID        int
	ExitCode   int
	StderrTail string
}

func (e *ErrProcessTerminated) Error() string {
	return fmt.Sprintf("PROCESS_TERMINATED: PID %d exited with code %d; stderr: %s", e.PID, e.ExitCode, e.StderrTail)
}

// ErrLivelockSuppressed is returned when repeated polling on a terminated process exceeds threshold.
type ErrLivelockSuppressed struct {
	PID int
}

func (e *ErrLivelockSuppressed) Error() string {
	return fmt.Sprintf("LIVELOCK_SUPPRESSED: repeated polling on terminated process PID %d", e.PID)
}

// ErrLivelockCircuitBroken is returned when repeated writes on a terminated process trip the circuit breaker.
type ErrLivelockCircuitBroken struct {
	PID int
}

func (e *ErrLivelockCircuitBroken) Error() string {
	return fmt.Sprintf("LIVELOCK_CIRCUIT_BROKEN: PID %d is dead", e.PID)
}

// ErrUnknownProcess is returned when a process PID is neither active nor tombstoned.
type ErrUnknownProcess struct {
	PID int
}

func (e *ErrUnknownProcess) Error() string {
	return fmt.Sprintf("toolproc: unknown process PID %d", e.PID)
}

// Sentinel error for unknown process checks with errors.Is.
var ErrUnknownProcessSentinel = errors.New("toolproc: unknown process")

func (e *ErrUnknownProcess) Is(target error) bool {
	return target == ErrUnknownProcessSentinel
}

// ProcessTombstone retains the terminal execution state and diagnostic tails of
// an exited process.
type ProcessTombstone struct {
	PID               int           `json:"pid"`
	State             ProcessState  `json:"state"`
	ExitCode          int           `json:"exit_code"`
	Elapsed           time.Duration `json:"elapsed"`
	StderrTail        string        `json:"stderr_tail"`
	StdoutTail        string        `json:"stdout_tail"`
	TerminationTime   time.Time     `json:"termination_time"`
	ConsecutivePolls  int           `json:"consecutive_polls"`
	ConsecutiveWrites int           `json:"consecutive_writes"`
}

// ProcessHandle is an active handle for a currently running supervised process.
type ProcessHandle struct {
	PID       int
	Cmd       string
	StartTime time.Time
	State     ProcessState
	Stdin     io.Writer
}

// SetStdin sets the optional stdin writer for the running process.
func (h *ProcessHandle) SetStdin(w io.Writer) {
	h.Stdin = w
}

// ProcessStatus holds the status returned by polling a process.
type ProcessStatus struct {
	PID        int           `json:"pid"`
	State      ProcessState  `json:"state"`
	ExitCode   int           `json:"exit_code"`
	Elapsed    time.Duration `json:"elapsed"`
	StderrTail string        `json:"stderr_tail,omitempty"`
	StdoutTail string        `json:"stdout_tail,omitempty"`
}

// ProcessSupervisor manages background processes with typed polling,
// dead-handle tombstoning, and livelock suppression.
type ProcessSupervisor struct {
	mu                     sync.RWMutex
	active                 map[int]*ProcessHandle
	tombstones             map[int]*ProcessTombstone
	tombstoneOrder         []int
	tombstoneCapacity      int
	pollLivelockThreshold  int
	writeLivelockThreshold int
	maxTailBytes           int
}

// SupervisorOption configures a ProcessSupervisor.
type SupervisorOption func(*ProcessSupervisor)

// WithPollLivelockThreshold sets the consecutive poll threshold before livelock suppression trips.
func WithPollLivelockThreshold(threshold int) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if threshold > 0 {
			s.pollLivelockThreshold = threshold
		}
	}
}

// WithWriteLivelockThreshold sets the consecutive write threshold before circuit breaking trips.
func WithWriteLivelockThreshold(threshold int) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if threshold > 0 {
			s.writeLivelockThreshold = threshold
		}
	}
}

// WithLivelockThreshold sets both poll and write livelock thresholds.
func WithLivelockThreshold(threshold int) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if threshold > 0 {
			s.pollLivelockThreshold = threshold
			s.writeLivelockThreshold = threshold
		}
	}
}

// WithTombstoneCapacity sets the FIFO capacity for retained tombstones.
func WithTombstoneCapacity(capacity int) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if capacity > 0 {
			s.tombstoneCapacity = capacity
		}
	}
}

// WithMaxTailBytes sets the maximum tail length in bytes retained for stdout/stderr.
func WithMaxTailBytes(maxBytes int) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if maxBytes > 0 {
			s.maxTailBytes = maxBytes
		}
	}
}

// NewProcessSupervisor creates a new process supervisor with default or custom options.
func NewProcessSupervisor(opts ...SupervisorOption) *ProcessSupervisor {
	s := &ProcessSupervisor{
		active:                 make(map[int]*ProcessHandle),
		tombstones:             make(map[int]*ProcessTombstone),
		tombstoneCapacity:      DefaultTombstoneCapacity,
		pollLivelockThreshold:  DefaultPollLivelockThreshold,
		writeLivelockThreshold: DefaultWriteLivelockThreshold,
		maxTailBytes:           DefaultMaxTailBytes,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (s *ProcessSupervisor) initLocked() {
	if s.active == nil {
		s.active = make(map[int]*ProcessHandle)
	}
	if s.tombstones == nil {
		s.tombstones = make(map[int]*ProcessTombstone)
	}
	if s.tombstoneCapacity <= 0 {
		s.tombstoneCapacity = DefaultTombstoneCapacity
	}
	if s.pollLivelockThreshold <= 0 {
		s.pollLivelockThreshold = DefaultPollLivelockThreshold
	}
	if s.writeLivelockThreshold <= 0 {
		s.writeLivelockThreshold = DefaultWriteLivelockThreshold
	}
	if s.maxTailBytes <= 0 {
		s.maxTailBytes = DefaultMaxTailBytes
	}
}

// RegisterProcess registers an active background process under supervision.
func (s *ProcessSupervisor) RegisterProcess(pid int, cmd string) *ProcessHandle {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	// Clear any previous tombstone for this PID (e.g. recycled PID)
	if _, exists := s.tombstones[pid]; exists {
		delete(s.tombstones, pid)
		for i, p := range s.tombstoneOrder {
			if p == pid {
				s.tombstoneOrder = append(s.tombstoneOrder[:i], s.tombstoneOrder[i+1:]...)
				break
			}
		}
	}

	handle := &ProcessHandle{
		PID:       pid,
		Cmd:       cmd,
		StartTime: time.Now(),
		State:     ProcessRunning,
	}
	s.active[pid] = handle
	return handle
}

// RecordExit records the termination of a process and moves it to the tombstone table.
func (s *ProcessSupervisor) RecordExit(pid int, exitCode int, stdout, stderr string, elapsed time.Duration) {
	state := ProcessExited
	if exitCode != 0 {
		state = ProcessFailed
	}
	s.RecordExitWithState(pid, state, exitCode, stdout, stderr, elapsed)
}

// RecordExitWithState records process termination with an explicit ProcessState (e.g. TIMED_OUT).
func (s *ProcessSupervisor) RecordExitWithState(pid int, state ProcessState, exitCode int, stdout, stderr string, elapsed time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	if handle, ok := s.active[pid]; ok {
		delete(s.active, pid)
		if elapsed <= 0 && !handle.StartTime.IsZero() {
			elapsed = time.Since(handle.StartTime)
		}
	}

	stderrTail := truncateTail(stderr, s.maxTailBytes)
	stdoutTail := truncateTail(stdout, s.maxTailBytes)

	tb := &ProcessTombstone{
		PID:               pid,
		State:             state,
		ExitCode:          exitCode,
		Elapsed:           elapsed,
		StderrTail:        stderrTail,
		StdoutTail:        stdoutTail,
		TerminationTime:   time.Now(),
		ConsecutivePolls:  0,
		ConsecutiveWrites: 0,
	}

	if _, exists := s.tombstones[pid]; exists {
		s.tombstones[pid] = tb
		return
	}

	// Evict oldest entries if capacity reached (FIFO)
	for len(s.tombstoneOrder) >= s.tombstoneCapacity && len(s.tombstoneOrder) > 0 {
		oldestPID := s.tombstoneOrder[0]
		s.tombstoneOrder = s.tombstoneOrder[1:]
		delete(s.tombstones, oldestPID)
	}

	s.tombstones[pid] = tb
	s.tombstoneOrder = append(s.tombstoneOrder, pid)
}

// RecordTimeout records that a process timed out.
func (s *ProcessSupervisor) RecordTimeout(pid int, stdout, stderr string, elapsed time.Duration) {
	s.RecordExitWithState(pid, ProcessTimedOut, -1, stdout, stderr, elapsed)
}

// PollProcess polls the status of a process by PID.
// If running, returns running status.
// If tombstoned, increments consecutive count. If consecutive count exceeds
// livelock threshold (e.g. 4), returns error/diagnostic:
//
//	LIVELOCK_SUPPRESSED: repeated polling on terminated process PID <id>
//
// Otherwise returns terminal status with exit code and error tail.
// If unknown, returns typed unknown process error.
func (s *ProcessSupervisor) PollProcess(pid int) (ProcessStatus, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	// 1. Running
	if handle, ok := s.active[pid]; ok {
		elapsed := time.Duration(0)
		if !handle.StartTime.IsZero() {
			elapsed = time.Since(handle.StartTime)
		}
		return ProcessStatus{
			PID:     pid,
			State:   ProcessRunning,
			Elapsed: elapsed,
		}, nil
	}

	// 2. Tombstoned
	if tb, ok := s.tombstones[pid]; ok {
		tb.ConsecutivePolls++
		status := ProcessStatus{
			PID:        pid,
			State:      tb.State,
			ExitCode:   tb.ExitCode,
			Elapsed:    tb.Elapsed,
			StderrTail: tb.StderrTail,
			StdoutTail: tb.StdoutTail,
		}
		if tb.ConsecutivePolls > s.pollLivelockThreshold {
			return status, &ErrLivelockSuppressed{PID: pid}
		}
		return status, nil
	}

	// 3. Unknown
	return ProcessStatus{}, &ErrUnknownProcess{PID: pid}
}

// WriteStdin attempts to write data to a supervised process's stdin.
// If tombstoned, returns typed:
//
//	PROCESS_TERMINATED: PID <id> exited with code <exitCode>; stderr: <stderrTail>
//
// Livelock circuit breaker: consecutive write attempts on dead handle trips
// livelock suppression error:
//
//	LIVELOCK_CIRCUIT_BROKEN: PID <id> is dead
func (s *ProcessSupervisor) WriteStdin(pid int, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.initLocked()

	// 1. Tombstoned
	if tb, ok := s.tombstones[pid]; ok {
		tb.ConsecutiveWrites++
		if tb.ConsecutiveWrites > s.writeLivelockThreshold {
			return &ErrLivelockCircuitBroken{PID: pid}
		}
		return &ErrProcessTerminated{
			PID:        pid,
			ExitCode:   tb.ExitCode,
			StderrTail: tb.StderrTail,
		}
	}

	// 2. Active
	if handle, ok := s.active[pid]; ok {
		if handle.Stdin != nil {
			_, err := handle.Stdin.Write(data)
			return err
		}
		return nil
	}

	// 3. Unknown
	return &ErrUnknownProcess{PID: pid}
}

// GetTombstone returns a copy of the tombstone record for a given PID if present.
func (s *ProcessSupervisor) GetTombstone(pid int) (*ProcessTombstone, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	tb, ok := s.tombstones[pid]
	if !ok {
		return nil, false
	}
	cp := *tb
	return &cp, true
}

// ActiveProcess returns a copy of the active process handle for a given PID if running.
func (s *ProcessSupervisor) ActiveProcess(pid int) (*ProcessHandle, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	handle, ok := s.active[pid]
	if !ok {
		return nil, false
	}
	cp := *handle
	return &cp, true
}

// ActiveCount returns the number of currently active processes.
func (s *ProcessSupervisor) ActiveCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.active)
}

// TombstoneCount returns the number of retained tombstones.
func (s *ProcessSupervisor) TombstoneCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tombstones)
}

func truncateTail(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	cut := len(s) - maxBytes
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return s[cut:]
}
