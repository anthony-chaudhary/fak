// Package interactivesession provides interactive agent session lifecycle management,
// state transitions, prompt turn coordination, runaway turn caps, and execution time
// bounding with fail-closed safety floors.
//
// Architecture:
// The package is structured around a central Session model controlled by a finite state
// machine (FSM) spanning Ready, Active, Paused, and Terminated states. Concurrent access
// is synchronized to guarantee sequential turn processing, deterministic token budget
// debits, and fail-closed termination upon boundary violations (e.g. runaway turn loops
// or wall-clock timeouts).
//
// Invariant: Terminal states are absorbing: once StateTerminated is entered, no further
// state transitions, turn executions, or budget debits are permitted.
// Invariant: Turn indices are strictly monotonically increasing positive integers bounded
// by the configured MaxTurns ceiling.
// Guard: fail-closed safety floors reject all turn executions, state mutations, and budget
// debits once limits, timeouts, or terminal conditions are reached.
package interactivesession

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

var (
	// ErrSessionTerminated indicates the session is terminated and cannot accept further operations.
	ErrSessionTerminated = errors.New("session is terminated")

	// ErrTurnLimitExceeded indicates the session has reached or exceeded its configured runaway turn cap.
	ErrTurnLimitExceeded = errors.New("turn limit exceeded")

	// ErrInvalidTransition indicates an illegal lifecycle state transition was attempted.
	ErrInvalidTransition = errors.New("invalid state transition")

	// ErrSessionTimeout indicates the session has exceeded its maximum execution time window.
	ErrSessionTimeout = errors.New("session execution timed out")

	// ErrSessionNotFound indicates the requested session ID was not found in the manager.
	ErrSessionNotFound = errors.New("session not found")

	// ErrTurnAlreadyActive indicates a turn is already in progress and another cannot start concurrently.
	ErrTurnAlreadyActive = errors.New("turn already active")

	// ErrTurnNotActive indicates no turn is active to complete or record.
	ErrTurnNotActive = errors.New("no turn currently active")

	// ErrBudgetExceeded indicates the cumulative token debit exceeded the session budget ceiling.
	ErrBudgetExceeded = errors.New("token budget exceeded")
)

// SessionState represents the operational lifecycle state of an interactive session.
type SessionState string

const (
	// StateReady indicates the session is initialized, idle, and ready to accept the next prompt turn.
	StateReady SessionState = "READY"

	// StateActive indicates a prompt turn is currently actively executing.
	StateActive SessionState = "ACTIVE"

	// StatePaused indicates the session is temporarily suspended and cannot execute turns until resumed.
	StatePaused SessionState = "PAUSED"

	// StateTerminated indicates the session is permanently halted and cannot transition to any other state.
	StateTerminated SessionState = "TERMINATED"
)

// SessionConfig defines configuration bounds, limits, and timeouts for an interactive session.
type SessionConfig struct {
	// MaxTurns is the runaway turn limit ceiling. If <= 0, DefaultMaxTurns is used.
	MaxTurns int

	// MaxExecutionTime is the maximum allowed duration before the session times out. If <= 0, no duration timeout is enforced.
	MaxExecutionTime time.Duration

	// MaxTokenBudget is the total token debit ceiling. If <= 0, budget debiting is unbounded.
	MaxTokenBudget int64

	// FailClosedOnTimeout determines whether timeouts force the session directly into StateTerminated.
	FailClosedOnTimeout bool
}

// DefaultMaxTurns is the default runaway turn cap applied if MaxTurns is not specified.
const DefaultMaxTurns = 100

// NewDefaultSessionConfig constructs a SessionConfig populated with safe default parameters.
func NewDefaultSessionConfig() SessionConfig {
	return SessionConfig{
		MaxTurns:            DefaultMaxTurns,
		MaxExecutionTime:    30 * time.Minute,
		MaxTokenBudget:      0,
		FailClosedOnTimeout: true,
	}
}

// TurnRecord captures the execution metadata and telemetry of a single prompt/response turn.
type TurnRecord struct {
	// TurnIndex is the 1-based sequential sequence number of this turn.
	TurnIndex int

	// Prompt is the input text or instruction driving the turn.
	Prompt string

	// TokensDebited is the count of tokens consumed or debited in this turn.
	TokensDebited int64

	// StartedAt is the wall-clock time when the turn execution began.
	StartedAt time.Time

	// CompletedAt is the wall-clock time when the turn execution completed.
	CompletedAt time.Time

	// Err is the error encountered during turn execution, if any.
	Err error
}

// Session coordinates an interactive agent session lifecycle, turn tracking, and safety bounds.
type Session struct {
	mu                 sync.Mutex
	id                 string
	state              SessionState
	config             SessionConfig
	createdAt          time.Time
	terminatedAt       time.Time
	currentTurnActive  bool
	currentTurnIndex   int
	currentTurnStart   time.Time
	currentTurnPrompt  string
	turns              []TurnRecord
	totalTokensDebited int64
	terminationReason  error
}

// NewSession instantiates a new Session in StateReady with validated configuration.
func NewSession(id string, cfg *SessionConfig) *Session {
	var c SessionConfig
	if cfg != nil {
		c = *cfg
	} else {
		c = NewDefaultSessionConfig()
	}
	if c.MaxTurns <= 0 {
		c.MaxTurns = DefaultMaxTurns
	}

	return &Session{
		id:        id,
		state:     StateReady,
		config:    c,
		createdAt: time.Now().UTC(),
		turns:     make([]TurnRecord, 0),
	}
}

// ID returns the unique session identifier.
func (s *Session) ID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.id
}

// State returns the current lifecycle state of the session.
func (s *Session) State() SessionState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// Config returns a copy of the session configuration.
func (s *Session) Config() SessionConfig {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config
}

// CreatedAt returns the timestamp when the session was created.
func (s *Session) CreatedAt() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.createdAt
}

// TerminatedAt returns the timestamp when the session entered StateTerminated, or zero time if not terminated.
func (s *Session) TerminatedAt() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.terminatedAt
}

// TurnCount returns the number of completed turns recorded in history.
func (s *Session) TurnCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.turns)
}

// TotalTokensDebited returns the cumulative tokens debited against this session.
func (s *Session) TotalTokensDebited() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalTokensDebited
}

// TerminationReason returns the error that triggered termination, or nil if not terminated.
func (s *Session) TerminationReason() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.terminationReason
}

// History returns a copy of all recorded turn history.
func (s *Session) History() []TurnRecord {
	s.mu.Lock()
	defer s.mu.Unlock()
	records := make([]TurnRecord, len(s.turns))
	copy(records, s.turns)
	return records
}

// IsExpired reports whether the session has exceeded its configured MaxExecutionTime.
func (s *Session) IsExpired() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.config.MaxExecutionTime <= 0 {
		return false
	}
	return time.Since(s.createdAt) >= s.config.MaxExecutionTime
}

// CheckTimeout evaluates whether the session has timed out, enforcing fail-closed termination if configured.
func (s *Session) CheckTimeout() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == StateTerminated {
		return ErrSessionTerminated
	}

	if s.config.MaxExecutionTime > 0 && time.Since(s.createdAt) >= s.config.MaxExecutionTime {
		if s.config.FailClosedOnTimeout {
			s.state = StateTerminated
			s.terminatedAt = time.Now().UTC()
			s.terminationReason = ErrSessionTimeout
			s.currentTurnActive = false
		}
		return ErrSessionTimeout
	}
	return nil
}

// TransitionTo validates and executes a lifecycle state transition.
//
// Invariant: StateTerminated is absorbing; transitions out of StateTerminated return ErrSessionTerminated.
// Guard: fail-closed validation blocks illegal transitions with ErrInvalidTransition.
func (s *Session) TransitionTo(target SessionState) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == StateTerminated {
		return ErrSessionTerminated
	}

	if s.state == target {
		return nil
	}

	valid := false
	switch s.state {
	case StateReady:
		valid = (target == StateActive || target == StatePaused || target == StateTerminated)
	case StateActive:
		valid = (target == StateReady || target == StatePaused || target == StateTerminated)
	case StatePaused:
		valid = (target == StateReady || target == StateTerminated)
	}

	if !valid {
		return fmt.Errorf("%w: cannot transition from %s to %s", ErrInvalidTransition, s.state, target)
	}

	s.state = target
	if target == StateTerminated {
		s.terminatedAt = time.Now().UTC()
		s.currentTurnActive = false
	}
	return nil
}

// DebitBudget decrements the session token budget by tokens.
//
// Guard: fail-closed budget check rejects debits exceeding MaxTokenBudget or on terminated sessions.
func (s *Session) DebitBudget(tokens int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == StateTerminated {
		return ErrSessionTerminated
	}
	if tokens < 0 {
		return errors.New("negative token debit prohibited")
	}

	if s.config.MaxTokenBudget > 0 && (s.totalTokensDebited+tokens) > s.config.MaxTokenBudget {
		return ErrBudgetExceeded
	}

	s.totalTokensDebited += tokens
	return nil
}

// BeginTurn initiates a new prompt turn within the session.
//
// Invariant: Turn indices are strictly monotonic and sequential.
// Guard: fail-closed checks enforce turn limits, timeouts, and single-active turn safety floors.
func (s *Session) BeginTurn(prompt string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == StateTerminated {
		return 0, ErrSessionTerminated
	}

	// Guard: check execution timeout.
	if s.config.MaxExecutionTime > 0 && time.Since(s.createdAt) >= s.config.MaxExecutionTime {
		if s.config.FailClosedOnTimeout {
			s.state = StateTerminated
			s.terminatedAt = time.Now().UTC()
			s.terminationReason = ErrSessionTimeout
			s.currentTurnActive = false
		}
		return 0, ErrSessionTimeout
	}

	if s.state == StatePaused {
		return 0, fmt.Errorf("%w: session is paused", ErrInvalidTransition)
	}

	if s.currentTurnActive {
		return 0, ErrTurnAlreadyActive
	}

	// Guard: runaway turn cap check.
	nextIndex := len(s.turns) + 1
	if s.config.MaxTurns > 0 && nextIndex > s.config.MaxTurns {
		s.state = StateTerminated
		s.terminatedAt = time.Now().UTC()
		s.terminationReason = ErrTurnLimitExceeded
		return 0, ErrTurnLimitExceeded
	}

	s.state = StateActive
	s.currentTurnActive = true
	s.currentTurnIndex = nextIndex
	s.currentTurnStart = time.Now().UTC()
	s.currentTurnPrompt = prompt

	return nextIndex, nil
}

// CompleteTurn finalizes an active prompt turn, recording telemetry and resetting state to Ready.
//
// Guard: fail-closed validation guarantees only active turns can be completed and debited.
func (s *Session) CompleteTurn(turnIndex int, tokens int64, turnErr error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.currentTurnActive {
		return ErrTurnNotActive
	}

	if turnIndex != s.currentTurnIndex {
		return fmt.Errorf("%w: turn index mismatch (expected %d, got %d)", ErrInvalidTransition, s.currentTurnIndex, turnIndex)
	}

	// Process token debit.
	if tokens > 0 {
		if s.config.MaxTokenBudget > 0 && (s.totalTokensDebited+tokens) > s.config.MaxTokenBudget {
			s.currentTurnActive = false
			s.state = StateTerminated
			s.terminatedAt = time.Now().UTC()
			s.terminationReason = ErrBudgetExceeded
			return ErrBudgetExceeded
		}
		s.totalTokensDebited += tokens
	}

	now := time.Now().UTC()
	record := TurnRecord{
		TurnIndex:     s.currentTurnIndex,
		Prompt:        s.currentTurnPrompt,
		TokensDebited: tokens,
		StartedAt:     s.currentTurnStart,
		CompletedAt:   now,
		Err:           turnErr,
	}
	s.turns = append(s.turns, record)

	s.currentTurnActive = false
	s.currentTurnPrompt = ""

	// Check if we hit the turn ceiling on completion.
	if s.config.MaxTurns > 0 && len(s.turns) >= s.config.MaxTurns {
		s.state = StateTerminated
		s.terminatedAt = now
		s.terminationReason = ErrTurnLimitExceeded
		return nil
	}

	s.state = StateReady
	return nil
}

// RecordTurn provides a synchronous, atomic turn execution helper invoking BeginTurn and CompleteTurn.
func (s *Session) RecordTurn(ctx context.Context, prompt string, tokens int64, exec func(ctx context.Context) error) (TurnRecord, error) {
	turnIdx, err := s.BeginTurn(prompt)
	if err != nil {
		return TurnRecord{}, err
	}

	var execErr error
	if exec != nil {
		execErr = exec(ctx)
	}

	completeErr := s.CompleteTurn(turnIdx, tokens, execErr)
	lastTurn := s.History()[turnIdx-1]
	if completeErr != nil {
		return lastTurn, completeErr
	}
	return lastTurn, execErr
}

// Terminate shuts down the session with a specified terminal error or reason.
//
// Invariant: StateTerminated is absorbing.
func (s *Session) Terminate(reason error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state == StateTerminated {
		return nil
	}

	s.state = StateTerminated
	s.terminatedAt = time.Now().UTC()
	s.terminationReason = reason
	s.currentTurnActive = false
	return nil
}

// SessionManager coordinates the creation, storage, and lifecycle queries across sessions.
type SessionManager struct {
	mu            sync.RWMutex
	sessions      map[string]*Session
	defaultConfig SessionConfig
}

// NewSessionManager constructs a new SessionManager initialized with default configuration.
func NewSessionManager(defaultConfig SessionConfig) *SessionManager {
	if defaultConfig.MaxTurns <= 0 {
		defaultConfig.MaxTurns = DefaultMaxTurns
	}
	return &SessionManager{
		sessions:      make(map[string]*Session),
		defaultConfig: defaultConfig,
	}
}

// CreateSession registers and returns a new Session with the specified ID and optional config.
func (m *SessionManager) CreateSession(id string, cfg *SessionConfig) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if id == "" {
		return nil, errors.New("session id cannot be empty")
	}

	if _, exists := m.sessions[id]; exists {
		return nil, fmt.Errorf("session already exists: %s", id)
	}

	c := m.defaultConfig
	if cfg != nil {
		c = *cfg
	}
	sess := NewSession(id, &c)
	m.sessions[id] = sess
	return sess, nil
}

// GetSession retrieves an existing Session by ID.
func (m *SessionManager) GetSession(id string) (*Session, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	sess, ok := m.sessions[id]
	return sess, ok
}

// TerminateSession terminates an active session registered with the manager.
func (m *SessionManager) TerminateSession(id string, reason error) error {
	sess, ok := m.GetSession(id)
	if !ok {
		return ErrSessionNotFound
	}
	return sess.Terminate(reason)
}

// ListSessions returns a slice containing all managed sessions.
func (m *SessionManager) ListSessions() []*Session {
	m.mu.RLock()
	defer m.mu.RUnlock()

	list := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		list = append(list, s)
	}
	return list
}

// ActiveCount returns the number of sessions that are currently in StateReady or StateActive.
func (m *SessionManager) ActiveCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	count := 0
	for _, s := range m.sessions {
		st := s.State()
		if st == StateReady || st == StateActive {
			count++
		}
	}
	return count
}

// PruneTerminated removes all sessions that have reached StateTerminated from the manager.
func (m *SessionManager) PruneTerminated() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	pruned := 0
	for id, s := range m.sessions {
		if s.State() == StateTerminated {
			delete(m.sessions, id)
			pruned++
		}
	}
	return pruned
}
