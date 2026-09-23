package agentqueue

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestStoreTerminalLifecyclePersistsAndFences(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	store := FileStore(path)
	startedAt := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	const (
		attemptID = "attempt-a"
		nonce     = "nonce-a"
		pid       = 4242
	)
	seed := Snapshot{
		Schema:     Schema,
		Generation: "generation-registered",
		Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "intent-a", State: IntentQueued},
			{ID: "intent-b", State: IntentQueued},
		},
		Attempts: []Attempt{{
			ID: attemptID, IntentID: "intent-a", State: AttemptLaunching,
			Nonce: nonce, LaunchDeadline: time.Now().Add(time.Minute), PID: pid, StartedAt: startedAt,
		}},
	}
	if err := store.Save(seed); err != nil {
		t.Fatalf("seed registered launch: %v", err)
	}

	running, err := store.MarkRunning(context.Background(), attemptID, nonce, pid, startedAt)
	if err != nil {
		t.Fatalf("MarkRunning: %v", err)
	}
	if running.State != AttemptRunning {
		t.Fatalf("MarkRunning state = %q, want %q", running.State, AttemptRunning)
	}
	loaded, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen running snapshot: %v", err)
	}
	if got := terminalAttempt(t, loaded, attemptID); got.State != AttemptRunning || got.Nonce != nonce || got.PID != pid || !got.StartedAt.Equal(startedAt) {
		t.Fatalf("persisted running attempt = %+v", got)
	}
	if loaded.Intents[0].State != IntentRunning {
		t.Fatalf("running intent state = %q, want %q", loaded.Intents[0].State, IntentRunning)
	}
	if got, err := FileStore(path).MarkRunning(context.Background(), attemptID, nonce, pid, startedAt); err != nil || got.State != AttemptRunning {
		t.Fatalf("idempotent MarkRunning = %+v, err %v", got, err)
	}
	if _, err := store.MarkRunning(context.Background(), attemptID, nonce, pid, startedAt.Add(time.Second)); !errors.Is(err, ErrFenced) {
		t.Fatalf("mismatched MarkRunning error = %v, want ErrFenced", err)
	}

	completed, err := store.CompleteAttempt(context.Background(), attemptID, nonce, pid, startedAt, true)
	if err != nil {
		t.Fatalf("CompleteAttempt: %v", err)
	}
	if completed.State != AttemptSucceeded {
		t.Fatalf("completed state = %q, want %q", completed.State, AttemptSucceeded)
	}
	final, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen completed snapshot: %v", err)
	}
	if got := terminalAttempt(t, final, attemptID); got.State != AttemptSucceeded || got.Nonce != nonce || got.PID != pid || !got.StartedAt.Equal(startedAt) {
		t.Fatalf("persisted completed attempt = %+v", got)
	}
	if final.Intents[0].State != IntentCompleted {
		t.Fatalf("completed intent state = %q, want %q", final.Intents[0].State, IntentCompleted)
	}
	if got, err := FileStore(path).CompleteAttempt(context.Background(), attemptID, nonce, pid, startedAt, true); err != nil || got.State != AttemptSucceeded {
		t.Fatalf("idempotent CompleteAttempt = %+v, err %v", got, err)
	}
	if _, err := store.CompleteAttempt(context.Background(), attemptID, nonce, pid, startedAt, false); !errors.Is(err, ErrFenced) {
		t.Fatalf("conflicting completion result error = %v, want ErrFenced", err)
	}
	if _, err := store.CompleteAttempt(context.Background(), attemptID, nonce, pid+1, startedAt, true); !errors.Is(err, ErrFenced) {
		t.Fatalf("mismatched completion identity error = %v, want ErrFenced", err)
	}

	receipt, err := Reconcile(final)
	if err != nil {
		t.Fatalf("Reconcile completed snapshot: %v", err)
	}
	if receipt.Observed != 0 || len(receipt.Start) != 1 || receipt.Start[0].IntentID != "intent-b" {
		t.Fatalf("reconcile after completion = observed %d, starts %+v; want 0 and intent-b", receipt.Observed, receipt.Start)
	}
}

func TestStoreTerminalRequiresRegisteredWrapper(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	store := FileStore(path)
	startedAt := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	seed := Snapshot{
		Schema:     Schema,
		Generation: "generation-unregistered",
		Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents:    []Intent{{ID: "intent-a", State: IntentQueued}},
		Attempts: []Attempt{{
			ID: "attempt-a", IntentID: "intent-a", State: AttemptLaunching,
			Nonce: "nonce-a", LaunchDeadline: time.Now().Add(time.Minute),
		}},
	}
	if err := store.Save(seed); err != nil {
		t.Fatalf("seed unregistered launch: %v", err)
	}

	if _, err := store.MarkRunning(context.Background(), "attempt-a", "nonce-a", 4242, startedAt); !errors.Is(err, ErrFenced) {
		t.Fatalf("MarkRunning without registration error = %v, want ErrFenced", err)
	}
	if _, err := store.CompleteAttempt(context.Background(), "attempt-a", "nonce-a", 4242, startedAt, true); !errors.Is(err, ErrFenced) {
		t.Fatalf("CompleteAttempt without registration error = %v, want ErrFenced", err)
	}
	loaded, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen unregistered launch: %v", err)
	}
	if got := terminalAttempt(t, loaded, "attempt-a"); got.State != AttemptLaunching || got.PID != 0 || !got.StartedAt.IsZero() {
		t.Fatalf("unregistered attempt mutated = %+v", got)
	}
}

func terminalAttempt(t *testing.T, snapshot Snapshot, id string) Attempt {
	t.Helper()
	for _, attempt := range snapshot.Attempts {
		if attempt.ID == id {
			return attempt
		}
	}
	t.Fatalf("attempt %q not found", id)
	return Attempt{}
}
