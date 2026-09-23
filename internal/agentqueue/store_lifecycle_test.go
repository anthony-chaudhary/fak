package agentqueue

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestStoreLifecycleLaunchingCASPersistsAndFencesWrapper(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	store := FileStore(path)
	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "generation-1",
		Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "intent-a", State: IntentQueued},
			{ID: "intent-b", State: IntentQueued},
		},
		Attempts: []Attempt{{ID: "attempt-a", IntentID: "intent-a", State: AttemptReserved}},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatalf("seed reserved attempt: %v", err)
	}

	deadline := time.Now().UTC().Round(0).Add(time.Minute)
	type beginResult struct {
		nonce   string
		attempt Attempt
		started bool
		err     error
	}
	start := make(chan struct{})
	results := make(chan beginResult, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	for _, nonce := range []string{"nonce-a", "nonce-b"} {
		go func(nonce string) {
			ready.Done()
			<-start
			attempt, started, err := FileStore(path).BeginLaunching(context.Background(), "attempt-a", nonce, deadline)
			results <- beginResult{nonce: nonce, attempt: attempt, started: started, err: err}
		}(nonce)
	}
	ready.Wait()
	close(start)

	var winner beginResult
	winners, fenced := 0, 0
	for range 2 {
		result := <-results
		switch {
		case result.err == nil && result.started:
			winner = result
			winners++
		case errors.Is(result.err, ErrFenced):
			fenced++
		default:
			t.Fatalf("BeginLaunching(%q) = started %v, err %v", result.nonce, result.started, result.err)
		}
	}
	if winners != 1 || fenced != 1 {
		t.Fatalf("launch race winners=%d fenced=%d, want 1/1", winners, fenced)
	}
	if winner.attempt.State != AttemptLaunching || winner.attempt.Nonce != winner.nonce || !winner.attempt.LaunchDeadline.Equal(deadline) {
		t.Fatalf("winning attempt = %+v, want launching nonce=%q deadline=%s", winner.attempt, winner.nonce, deadline)
	}

	reopened := FileStore(path)
	loaded, err := reopened.Load()
	if err != nil {
		t.Fatalf("reopen launching snapshot: %v", err)
	}
	persisted := lifecycleAttempt(t, loaded, "attempt-a")
	if persisted.State != AttemptLaunching || persisted.Nonce != winner.nonce || !persisted.LaunchDeadline.Equal(deadline) {
		t.Fatalf("persisted launching attempt = %+v", persisted)
	}
	receipt, err := Reconcile(loaded)
	if err != nil {
		t.Fatalf("reconcile launching snapshot: %v", err)
	}
	if receipt.Observed != 1 || len(receipt.Start) != 0 {
		t.Fatalf("reconcile launching attempt observed=%d starts=%v, want 1/none", receipt.Observed, receipt.Start)
	}

	same, started, err := reopened.BeginLaunching(context.Background(), "attempt-a", winner.nonce, deadline)
	if err != nil || started {
		t.Fatalf("same-nonce BeginLaunching = started %v, err %v; want idempotent false/nil", started, err)
	}
	if same.State != AttemptLaunching || same.Nonce != winner.nonce {
		t.Fatalf("same-nonce attempt = %+v", same)
	}
	if _, _, err := reopened.BeginLaunching(context.Background(), "attempt-a", "stale-nonce", deadline); !errors.Is(err, ErrFenced) {
		t.Fatalf("other-nonce BeginLaunching error = %v, want ErrFenced", err)
	}

	pid := 4242
	startedAt := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	running, err := reopened.RegisterWrapper(context.Background(), "attempt-a", winner.nonce, pid, startedAt)
	if err != nil {
		t.Fatalf("RegisterWrapper: %v", err)
	}
	if running.State != AttemptLaunching || running.PID != pid || !running.StartedAt.Equal(startedAt) {
		t.Fatalf("registered wrapper attempt = %+v", running)
	}
	idempotent, err := FileStore(path).RegisterWrapper(context.Background(), "attempt-a", winner.nonce, pid, startedAt)
	if err != nil {
		t.Fatalf("idempotent RegisterWrapper: %v", err)
	}
	if idempotent.State != AttemptLaunching || idempotent.PID != pid || !idempotent.StartedAt.Equal(startedAt) {
		t.Fatalf("idempotent wrapper attempt = %+v", idempotent)
	}
	if _, err := reopened.RegisterWrapper(context.Background(), "attempt-a", "stale-nonce", pid, startedAt); !errors.Is(err, ErrFenced) {
		t.Fatalf("wrong-nonce RegisterWrapper error = %v, want ErrFenced", err)
	}
	if _, err := reopened.RegisterWrapper(context.Background(), "attempt-a", winner.nonce, pid, startedAt.Add(time.Second)); !errors.Is(err, ErrFenced) {
		t.Fatalf("reused-PID RegisterWrapper error = %v, want ErrFenced", err)
	}

	final, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen registered snapshot: %v", err)
	}
	registered := lifecycleAttempt(t, final, "attempt-a")
	if registered.State != AttemptLaunching || registered.Nonce != winner.nonce || registered.PID != pid || !registered.StartedAt.Equal(startedAt) {
		t.Fatalf("persisted registered attempt = %+v", registered)
	}
}

func TestStoreLifecycleExpiredBeginLeavesReserved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	store := FileStore(path)
	seed := Snapshot{
		Schema:     Schema,
		Generation: "generation-expired",
		Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents:    []Intent{{ID: "intent-a", State: IntentQueued}},
		Attempts:   []Attempt{{ID: "attempt-a", IntentID: "intent-a", State: AttemptReserved}},
	}
	if err := store.Save(seed); err != nil {
		t.Fatalf("seed reserved attempt: %v", err)
	}

	if _, started, err := store.BeginLaunching(context.Background(), "attempt-a", "nonce-a", time.Now().Add(-time.Second)); err == nil || started {
		t.Fatalf("expired BeginLaunching = started %v, err %v; want false/error", started, err)
	}
	loaded, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen reserved snapshot: %v", err)
	}
	attempt := lifecycleAttempt(t, loaded, "attempt-a")
	if attempt.State != AttemptReserved || attempt.Nonce != "" || !attempt.LaunchDeadline.IsZero() {
		t.Fatalf("attempt mutated after expired begin = %+v", attempt)
	}
}

func TestStoreLifecycleAbortTransitionsFenceIdentityAndState(t *testing.T) {
	store, _, _ := reservedActuatorFixture(t, 2)
	deadline := time.Now().Add(time.Hour)
	if _, started, err := store.BeginLaunching(context.Background(), "attempt-0", "nonce-0", deadline); err != nil || !started {
		t.Fatalf("BeginLaunching = started %v, err %v", started, err)
	}

	if _, err := store.AbortReserved(context.Background(), "attempt-0"); !errors.Is(err, ErrFenced) {
		t.Fatalf("AbortReserved launching error = %v, want ErrFenced", err)
	}
	if _, err := store.AbortLaunching(context.Background(), "attempt-0", "wrong-nonce", 0, time.Time{}); !errors.Is(err, ErrFenced) {
		t.Fatalf("AbortLaunching wrong nonce error = %v, want ErrFenced", err)
	}
	aborted, err := store.AbortLaunching(context.Background(), "attempt-0", "nonce-0", 0, time.Time{})
	if err != nil || aborted.State != AttemptFailed {
		t.Fatalf("AbortLaunching = %+v, err %v", aborted, err)
	}
	if _, err := store.AbortLaunching(context.Background(), "attempt-0", "nonce-0", 0, time.Time{}); err != nil {
		t.Fatalf("idempotent AbortLaunching: %v", err)
	}
	if _, err := store.AbortLaunching(context.Background(), "attempt-1", "nonce-1", 0, time.Time{}); !errors.Is(err, ErrFenced) {
		t.Fatalf("AbortLaunching reserved error = %v, want ErrFenced", err)
	}
	if _, err := store.AbortReserved(context.Background(), "attempt-1"); err != nil {
		t.Fatalf("AbortReserved: %v", err)
	}
}

func TestStoreLifecycleRegisteredWrapperIdempotentAfterDeadline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	pid := 4242
	startedAt := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	store := FileStore(path)
	seed := Snapshot{
		Schema:     Schema,
		Generation: "generation-registered",
		Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "intent-registered", State: IntentQueued},
			{ID: "intent-unregistered", State: IntentQueued},
		},
		Attempts: []Attempt{
			{ID: "attempt-registered", IntentID: "intent-registered", State: AttemptLaunching, Nonce: "nonce-a", LaunchDeadline: time.Now().Add(-time.Second), PID: pid, StartedAt: startedAt},
			{ID: "attempt-unregistered", IntentID: "intent-unregistered", State: AttemptLaunching, Nonce: "nonce-b", LaunchDeadline: time.Now().Add(-time.Second)},
		},
	}
	if err := store.Save(seed); err != nil {
		t.Fatalf("seed expired launching attempts: %v", err)
	}

	got, err := store.RegisterWrapper(context.Background(), "attempt-registered", "nonce-a", pid, startedAt)
	if err != nil {
		t.Fatalf("idempotent registration after deadline: %v", err)
	}
	if got.PID != pid || !got.StartedAt.Equal(startedAt) {
		t.Fatalf("idempotent registered identity = %+v", got)
	}
	if _, err := store.RegisterWrapper(context.Background(), "attempt-registered", "nonce-a", pid, startedAt.Add(time.Second)); !errors.Is(err, ErrFenced) {
		t.Fatalf("mismatched identity after deadline error = %v, want ErrFenced", err)
	}
	if _, err := store.RegisterWrapper(context.Background(), "attempt-unregistered", "nonce-b", pid, startedAt); !errors.Is(err, ErrFenced) {
		t.Fatalf("new identity after deadline error = %v, want ErrFenced", err)
	}
}

func TestStoreLifecycleHoldAttemptPersistsAndFencesIdentityAndReason(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentqueue.json")
	store := FileStore(path)
	pid := 4242
	startedAt := time.Date(2026, 9, 23, 12, 0, 0, 0, time.UTC)
	reason := "agent_guard_denied"
	seed := Snapshot{
		Schema:     Schema,
		Generation: "generation-running",
		Pool:       PoolSpec{ID: "pool", Desired: 1, Max: 1},
		Intents: []Intent{{
			ID: "intent-a", State: IntentRunning, RetryEligible: true, PID: pid,
		}},
		Attempts: []Attempt{{
			ID: "attempt-a", IntentID: "intent-a", State: AttemptRunning,
			Nonce: "nonce-a", LaunchDeadline: time.Now().Add(time.Hour), PID: pid, StartedAt: startedAt,
		}},
	}
	if err := store.Save(seed); err != nil {
		t.Fatalf("seed running attempt: %v", err)
	}

	held, err := store.HoldAttempt(context.Background(), "attempt-a", "nonce-a", pid, startedAt, reason)
	if err != nil {
		t.Fatalf("HoldAttempt: %v", err)
	}
	if held.State != AttemptFailed {
		t.Fatalf("held attempt state = %q, want failed", held.State)
	}
	persisted, err := FileStore(path).Load()
	if err != nil {
		t.Fatalf("reopen held snapshot: %v", err)
	}
	if persisted.Generation == seed.Generation {
		t.Fatal("HoldAttempt did not advance generation")
	}
	gotAttempt := lifecycleAttempt(t, persisted, "attempt-a")
	gotIntent := persisted.Intents[0]
	if gotAttempt.State != AttemptFailed || gotIntent.State != IntentHeld || gotIntent.HoldReason != reason || gotIntent.RetryEligible {
		t.Fatalf("persisted hold = attempt %+v intent %+v", gotAttempt, gotIntent)
	}

	idempotent, err := FileStore(path).HoldAttempt(context.Background(), "attempt-a", "nonce-a", pid, startedAt, reason)
	if err != nil || idempotent.State != AttemptFailed {
		t.Fatalf("idempotent HoldAttempt = %+v, err %v", idempotent, err)
	}
	afterIdempotent, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if afterIdempotent.Generation != persisted.Generation {
		t.Fatalf("idempotent hold advanced generation from %q to %q", persisted.Generation, afterIdempotent.Generation)
	}

	for name, candidate := range map[string]struct {
		nonce     string
		pid       int
		startedAt time.Time
		reason    string
	}{
		"stale nonce":      {nonce: "nonce-stale", pid: pid, startedAt: startedAt, reason: reason},
		"stale pid":        {nonce: "nonce-a", pid: pid + 1, startedAt: startedAt, reason: reason},
		"stale start time": {nonce: "nonce-a", pid: pid, startedAt: startedAt.Add(time.Second), reason: reason},
		"different reason": {nonce: "nonce-a", pid: pid, startedAt: startedAt, reason: "budget_exhausted"},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := store.HoldAttempt(context.Background(), "attempt-a", candidate.nonce, candidate.pid, candidate.startedAt, candidate.reason); !errors.Is(err, ErrFenced) {
				t.Fatalf("HoldAttempt error = %v, want ErrFenced", err)
			}
		})
	}
}

func lifecycleAttempt(t *testing.T, snapshot Snapshot, id string) Attempt {
	t.Helper()
	for _, attempt := range snapshot.Attempts {
		if attempt.ID == id {
			return attempt
		}
	}
	t.Fatalf("attempt %q not found", id)
	return Attempt{}
}
