package interactivesession

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestSessionCreation(t *testing.T) {
	// Test with nil config -> default config
	s1 := NewSession("sess-1", nil)
	if s1.ID() != "sess-1" {
		t.Fatalf("expected ID sess-1, got %s", s1.ID())
	}
	if s1.State() != StateReady {
		t.Fatalf("expected initial state %s, got %s", StateReady, s1.State())
	}
	if s1.Config().MaxTurns != DefaultMaxTurns {
		t.Fatalf("expected max turns %d, got %d", DefaultMaxTurns, s1.Config().MaxTurns)
	}
	if s1.CreatedAt().IsZero() {
		t.Fatal("expected non-zero CreatedAt")
	}
	if !s1.TerminatedAt().IsZero() {
		t.Fatal("expected zero TerminatedAt for active session")
	}

	// Test with custom config and MaxTurns <= 0 fallback
	cfg := SessionConfig{
		MaxTurns:         0,
		MaxExecutionTime: 10 * time.Minute,
		MaxTokenBudget:   5000,
	}
	s2 := NewSession("sess-2", &cfg)
	if s2.Config().MaxTurns != DefaultMaxTurns {
		t.Fatalf("expected fallback to default max turns %d, got %d", DefaultMaxTurns, s2.Config().MaxTurns)
	}
	if s2.Config().MaxTokenBudget != 5000 {
		t.Fatalf("expected token budget 5000, got %d", s2.Config().MaxTokenBudget)
	}
}

func TestStateTransitions(t *testing.T) {
	s := NewSession("sess-fsm", nil)

	// Ready -> Ready (no-op)
	if err := s.TransitionTo(StateReady); err != nil {
		t.Fatalf("unexpected error on same-state transition: %v", err)
	}

	// Ready -> Paused
	if err := s.TransitionTo(StatePaused); err != nil {
		t.Fatalf("failed transitioning Ready -> Paused: %v", err)
	}
	if s.State() != StatePaused {
		t.Fatalf("expected state %s, got %s", StatePaused, s.State())
	}

	// Paused -> Active (illegal: must go Paused -> Ready first)
	if err := s.TransitionTo(StateActive); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition for Paused -> Active, got %v", err)
	}

	// Paused -> Ready
	if err := s.TransitionTo(StateReady); err != nil {
		t.Fatalf("failed transitioning Paused -> Ready: %v", err)
	}

	// Ready -> Active
	if err := s.TransitionTo(StateActive); err != nil {
		t.Fatalf("failed transitioning Ready -> Active: %v", err)
	}

	// Active -> Paused
	if err := s.TransitionTo(StatePaused); err != nil {
		t.Fatalf("failed transitioning Active -> Paused: %v", err)
	}

	// Paused -> Terminated
	if err := s.TransitionTo(StateTerminated); err != nil {
		t.Fatalf("failed transitioning Paused -> Terminated: %v", err)
	}
	if s.State() != StateTerminated {
		t.Fatalf("expected state %s, got %s", StateTerminated, s.State())
	}
	if s.TerminatedAt().IsZero() {
		t.Fatal("expected non-zero TerminatedAt upon termination")
	}

	// Terminated -> Ready (illegal: absorbing state)
	if err := s.TransitionTo(StateReady); !errors.Is(err, ErrSessionTerminated) {
		t.Fatalf("expected ErrSessionTerminated transitioning out of Terminated, got %v", err)
	}
}

func TestTurnExecutionAndDebiting(t *testing.T) {
	cfg := SessionConfig{
		MaxTurns:       10,
		MaxTokenBudget: 1000,
	}
	s := NewSession("sess-turns", &cfg)

	// Step 1: Begin Turn 1
	idx, err := s.BeginTurn("hello agent")
	if err != nil {
		t.Fatalf("failed to begin turn 1: %v", err)
	}
	if idx != 1 {
		t.Fatalf("expected turn index 1, got %d", idx)
	}
	if s.State() != StateActive {
		t.Fatalf("expected state %s, got %s", StateActive, s.State())
	}

	// Attempting concurrent turn while one is active
	if _, err := s.BeginTurn("concurrent turn"); !errors.Is(err, ErrTurnAlreadyActive) {
		t.Fatalf("expected ErrTurnAlreadyActive, got %v", err)
	}

	// Complete Turn 1 with 150 tokens
	if err := s.CompleteTurn(idx, 150, nil); err != nil {
		t.Fatalf("failed to complete turn 1: %v", err)
	}
	if s.State() != StateReady {
		t.Fatalf("expected state %s after completion, got %s", StateReady, s.State())
	}
	if s.TurnCount() != 1 {
		t.Fatalf("expected turn count 1, got %d", s.TurnCount())
	}
	if s.TotalTokensDebited() != 150 {
		t.Fatalf("expected 150 total tokens debited, got %d", s.TotalTokensDebited())
	}

	// Verify history record
	history := s.History()
	if len(history) != 1 {
		t.Fatalf("expected 1 history record, got %d", len(history))
	}
	if history[0].Prompt != "hello agent" || history[0].TokensDebited != 150 {
		t.Fatalf("unexpected history record: %+v", history[0])
	}

	// Step 2: Use RecordTurn helper for Turn 2
	customErr := errors.New("command failed in sub-agent")
	rec, err := s.RecordTurn(context.Background(), "run command", 250, func(ctx context.Context) error {
		return customErr
	})
	if !errors.Is(err, customErr) {
		t.Fatalf("expected customErr from RecordTurn, got %v", err)
	}
	if rec.TurnIndex != 2 {
		t.Fatalf("expected turn index 2, got %d", rec.TurnIndex)
	}
	if rec.TokensDebited != 250 {
		t.Fatalf("expected 250 tokens debited, got %d", rec.TokensDebited)
	}
	if s.TotalTokensDebited() != 400 {
		t.Fatalf("expected cumulative tokens 400, got %d", s.TotalTokensDebited())
	}
}

func TestRunawayTurnCutoff(t *testing.T) {
	// Configure session with tight turn cap: 2 turns max
	cfg := SessionConfig{
		MaxTurns: 2,
	}
	s := NewSession("sess-runaway", &cfg)

	// Turn 1
	idx1, err := s.BeginTurn("turn 1")
	if err != nil {
		t.Fatalf("unexpected error on turn 1: %v", err)
	}
	if err := s.CompleteTurn(idx1, 10, nil); err != nil {
		t.Fatalf("failed to complete turn 1: %v", err)
	}
	if s.State() != StateReady {
		t.Fatalf("expected state %s after turn 1, got %s", StateReady, s.State())
	}

	// Turn 2 (reaching max turns)
	idx2, err := s.BeginTurn("turn 2")
	if err != nil {
		t.Fatalf("unexpected error on turn 2: %v", err)
	}
	if err := s.CompleteTurn(idx2, 10, nil); err != nil {
		t.Fatalf("failed to complete turn 2: %v", err)
	}

	// Session should now be terminated due to reaching MaxTurns
	if s.State() != StateTerminated {
		t.Fatalf("expected StateTerminated after reaching MaxTurns, got %s", s.State())
	}
	if !errors.Is(s.TerminationReason(), ErrTurnLimitExceeded) {
		t.Fatalf("expected ErrTurnLimitExceeded reason, got %v", s.TerminationReason())
	}

	// Turn 3: BeginTurn should fail-closed
	if _, err := s.BeginTurn("turn 3"); !errors.Is(err, ErrSessionTerminated) {
		t.Fatalf("expected ErrSessionTerminated on turn 3, got %v", err)
	}
}

func TestBudgetDebitingLimits(t *testing.T) {
	cfg := SessionConfig{
		MaxTurns:       10,
		MaxTokenBudget: 500,
	}
	s := NewSession("sess-budget", &cfg)

	// Negative debit should fail
	if err := s.DebitBudget(-50); err == nil {
		t.Fatal("expected error on negative debit, got nil")
	}

	// Valid debit
	if err := s.DebitBudget(300); err != nil {
		t.Fatalf("unexpected error on debit: %v", err)
	}
	if s.TotalTokensDebited() != 300 {
		t.Fatalf("expected 300 tokens debited, got %d", s.TotalTokensDebited())
	}

	// Debit exceeding remaining budget (300 + 250 = 550 > 500)
	if err := s.DebitBudget(250); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded, got %v", err)
	}
	if s.TotalTokensDebited() != 300 {
		t.Fatalf("budget debit should not have applied, got %d", s.TotalTokensDebited())
	}

	// CompleteTurn exceeding budget should terminate session fail-closed
	idx, err := s.BeginTurn("turn over-budget")
	if err != nil {
		t.Fatalf("unexpected error beginning turn: %v", err)
	}
	err = s.CompleteTurn(idx, 250, nil)
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded from CompleteTurn, got %v", err)
	}
	if s.State() != StateTerminated {
		t.Fatalf("expected StateTerminated on budget breach, got %s", s.State())
	}
}

func TestRecordTurnBudgetExceededDoesNotPanic(t *testing.T) {
	cfg := SessionConfig{
		MaxTurns:       10,
		MaxTokenBudget: 100,
	}
	s := NewSession("sess-record-budget", &cfg)

	// RecordTurn with tokens exceeding budget should cleanly return ErrBudgetExceeded without panicking
	_, err := s.RecordTurn(context.Background(), "overflow turn", 200, nil)
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded from RecordTurn, got %v", err)
	}
	if s.State() != StateTerminated {
		t.Fatalf("expected StateTerminated, got %s", s.State())
	}
}

func TestCompleteTurnWhilePausedMaintainsPausedState(t *testing.T) {
	s := NewSession("sess-pause-turn", nil)
	idx, err := s.BeginTurn("in-flight turn")
	if err != nil {
		t.Fatalf("failed to begin turn: %v", err)
	}

	// Operator pauses session while turn is in-flight
	if err := s.TransitionTo(StatePaused); err != nil {
		t.Fatalf("failed to transition to Paused: %v", err)
	}

	// Complete turn should succeed and maintain Paused state
	if err := s.CompleteTurn(idx, 50, nil); err != nil {
		t.Fatalf("complete turn failed: %v", err)
	}
	if s.State() != StatePaused {
		t.Fatalf("expected StatePaused to be maintained, got %s", s.State())
	}
}

func TestCompleteTurnNegativeTokensRejected(t *testing.T) {
	s := NewSession("sess-neg-tokens", nil)
	idx, err := s.BeginTurn("neg-turn")
	if err != nil {
		t.Fatalf("failed to begin turn: %v", err)
	}

	if err := s.CompleteTurn(idx, -10, nil); err == nil {
		t.Fatal("expected error on negative tokens in CompleteTurn, got nil")
	}
}

func TestCompleteTurnOnTerminatedSessionRejected(t *testing.T) {
	s := NewSession("sess-term-complete", nil)
	idx, err := s.BeginTurn("term-turn")
	if err != nil {
		t.Fatalf("failed to begin turn: %v", err)
	}

	if err := s.Terminate(errors.New("shutdown")); err != nil {
		t.Fatalf("failed to terminate session: %v", err)
	}

	if err := s.CompleteTurn(idx, 10, nil); !errors.Is(err, ErrSessionTerminated) {
		t.Fatalf("expected ErrSessionTerminated from CompleteTurn on terminated session, got %v", err)
	}
}

func TestSessionTimeoutFailClosed(t *testing.T) {
	cfg := SessionConfig{
		MaxTurns:            10,
		MaxExecutionTime:    10 * time.Millisecond,
		FailClosedOnTimeout: true,
	}
	s := NewSession("sess-timeout", &cfg)

	// Before timeout
	if err := s.CheckTimeout(); err != nil {
		t.Fatalf("expected nil before timeout, got %v", err)
	}
	if s.IsExpired() {
		t.Fatal("expected IsExpired false immediately after creation")
	}

	// Wait for expiration
	time.Sleep(25 * time.Millisecond)

	if !s.IsExpired() {
		t.Fatal("expected IsExpired true after sleep")
	}

	// CheckTimeout should trigger fail-closed termination
	if err := s.CheckTimeout(); !errors.Is(err, ErrSessionTimeout) {
		t.Fatalf("expected ErrSessionTimeout, got %v", err)
	}
	if s.State() != StateTerminated {
		t.Fatalf("expected StateTerminated after timeout check, got %s", s.State())
	}
	if !errors.Is(s.TerminationReason(), ErrSessionTimeout) {
		t.Fatalf("expected ErrSessionTimeout termination reason, got %v", s.TerminationReason())
	}

	// Further turns rejected
	if _, err := s.BeginTurn("post timeout"); !errors.Is(err, ErrSessionTerminated) {
		t.Fatalf("expected ErrSessionTerminated on post-timeout turn, got %v", err)
	}
}

func BenchmarkTurnDebiting(b *testing.B) {
	s := NewSession("bench-sess", &SessionConfig{
		MaxTurns:       b.N + 1000,
		MaxTokenBudget: 0, // unbounded for benchmark
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.DebitBudget(1); err != nil {
			b.Fatalf("debit failed at iteration %d: %v", i, err)
		}
	}
}
