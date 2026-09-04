package interactivesession

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSessionManagerLifecycle(t *testing.T) {
	mgr := NewSessionManager(SessionConfig{
		MaxTurns: 50,
	})

	// Create sessions
	s1, err := mgr.CreateSession("sess-1", nil)
	if err != nil {
		t.Fatalf("failed creating sess-1: %v", err)
	}
	if s1.ID() != "sess-1" {
		t.Fatalf("expected ID sess-1, got %s", s1.ID())
	}

	s2, err := mgr.CreateSession("sess-2", &SessionConfig{MaxTurns: 10})
	if err != nil {
		t.Fatalf("failed creating sess-2: %v", err)
	}
	if s2.Config().MaxTurns != 10 {
		t.Fatalf("expected max turns 10 for sess-2, got %d", s2.Config().MaxTurns)
	}

	// Duplicate creation should fail
	if _, err := mgr.CreateSession("sess-1", nil); err == nil {
		t.Fatal("expected error on duplicate session creation, got nil")
	}

	// Empty ID should fail
	if _, err := mgr.CreateSession("", nil); err == nil {
		t.Fatal("expected error on empty session ID, got nil")
	}

	// Get session
	retrieved, ok := mgr.GetSession("sess-1")
	if !ok || retrieved != s1 {
		t.Fatal("failed retrieving sess-1")
	}

	// Non-existent session
	if _, ok := mgr.GetSession("sess-unknown"); ok {
		t.Fatal("expected false for unknown session")
	}

	// ActiveCount
	if mgr.ActiveCount() != 2 {
		t.Fatalf("expected 2 active sessions, got %d", mgr.ActiveCount())
	}

	// Terminate s1
	termErr := errors.New("operator shutdown")
	if err := mgr.TerminateSession("sess-1", termErr); err != nil {
		t.Fatalf("failed to terminate sess-1: %v", err)
	}
	if s1.State() != StateTerminated {
		t.Fatalf("expected StateTerminated for sess-1, got %s", s1.State())
	}

	// Terminate unknown session
	if err := mgr.TerminateSession("sess-unknown", termErr); !errors.Is(err, ErrSessionNotFound) {
		t.Fatalf("expected ErrSessionNotFound for unknown session, got %v", err)
	}

	// ActiveCount after termination
	if mgr.ActiveCount() != 1 {
		t.Fatalf("expected 1 active session, got %d", mgr.ActiveCount())
	}

	// ListSessions
	list := mgr.ListSessions()
	if len(list) != 2 {
		t.Fatalf("expected 2 sessions in list, got %d", len(list))
	}

	// PruneTerminated
	pruned := mgr.PruneTerminated()
	if pruned != 1 {
		t.Fatalf("expected 1 pruned session, got %d", pruned)
	}
	if len(mgr.ListSessions()) != 1 {
		t.Fatalf("expected 1 session remaining after pruning, got %d", len(mgr.ListSessions()))
	}
}

func TestConcurrentTurnsSynchronization(t *testing.T) {
	s := NewSession("sess-concurrency", &SessionConfig{
		MaxTurns: 1000,
	})

	const numWorkers = 8
	const turnsPerWorker = 20
	var wg sync.WaitGroup
	var completedCount int64
	var collisionCount int64

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < turnsPerWorker; j++ {
				prompt := fmt.Sprintf("worker-%d-turn-%d", workerID, j)
				idx, err := s.BeginTurn(prompt)
				if err != nil {
					if errors.Is(err, ErrTurnAlreadyActive) {
						atomic.AddInt64(&collisionCount, 1)
						time.Sleep(time.Millisecond)
						continue
					}
					t.Errorf("unexpected error beginning turn: %v", err)
					return
				}

				// Simulate brief work
				time.Sleep(100 * time.Microsecond)

				if err := s.CompleteTurn(idx, 5, nil); err != nil {
					t.Errorf("unexpected error completing turn: %v", err)
					return
				}
				atomic.AddInt64(&completedCount, 1)
			}
		}(i)
	}

	wg.Wait()

	t.Logf("Completed turns: %d, Collisions prevented by guard: %d", completedCount, collisionCount)
	if completedCount == 0 {
		t.Fatal("expected at least some turns to complete")
	}
	if s.TurnCount() != int(completedCount) {
		t.Fatalf("expected TurnCount %d, got %d", completedCount, s.TurnCount())
	}
	if s.TotalTokensDebited() != completedCount*5 {
		t.Fatalf("expected total tokens %d, got %d", completedCount*5, s.TotalTokensDebited())
	}

	// Verify monotonic turn indices in history
	history := s.History()
	for idx, record := range history {
		if record.TurnIndex != idx+1 {
			t.Fatalf("expected monotonic turn index %d at position %d, got %d", idx+1, idx, record.TurnIndex)
		}
	}
}

func TestTurnWithContextCancellation(t *testing.T) {
	s := NewSession("sess-ctx", &SessionConfig{
		MaxTurns: 10,
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-canceled context

	rec, err := s.RecordTurn(ctx, "canceled turn", 0, func(c context.Context) error {
		return c.Err()
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if rec.Err == nil || !errors.Is(rec.Err, context.Canceled) {
		t.Fatalf("expected recorded error to be context.Canceled, got %v", rec.Err)
	}
	if s.State() != StateReady {
		t.Fatalf("expected session to recover to StateReady after error, got %s", s.State())
	}
}

func TestSessionEdgeCases(t *testing.T) {
	s := NewSession("sess-edge", &SessionConfig{
		MaxTurns: 5,
	})

	// CompleteTurn when not active
	if err := s.CompleteTurn(1, 10, nil); !errors.Is(err, ErrTurnNotActive) {
		t.Fatalf("expected ErrTurnNotActive, got %v", err)
	}

	// BeginTurn
	idx, err := s.BeginTurn("turn-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// CompleteTurn with mismatched index
	if err := s.CompleteTurn(idx+99, 10, nil); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition on index mismatch, got %v", err)
	}

	// Now complete turn properly
	if err := s.CompleteTurn(idx, 10, nil); err != nil {
		t.Fatalf("failed to complete turn: %v", err)
	}

	// Pause session and attempt turn
	if err := s.TransitionTo(StatePaused); err != nil {
		t.Fatalf("failed to pause session: %v", err)
	}
	if _, err := s.BeginTurn("turn-paused"); !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition when paused, got %v", err)
	}

	// Terminate session idempotency
	termErr := errors.New("first reason")
	if err := s.Terminate(termErr); err != nil {
		t.Fatalf("unexpected error on terminate: %v", err)
	}
	if !errors.Is(s.TerminationReason(), termErr) {
		t.Fatalf("expected termErr, got %v", s.TerminationReason())
	}

	// Second terminate should be no-op and preserve original reason
	if err := s.Terminate(errors.New("second reason")); err != nil {
		t.Fatalf("second terminate should succeed, got %v", err)
	}
	if !errors.Is(s.TerminationReason(), termErr) {
		t.Fatalf("expected original termination reason preserved, got %v", s.TerminationReason())
	}
}
