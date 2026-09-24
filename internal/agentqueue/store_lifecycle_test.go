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

func TestStoreResolveWitnessHeldCompletesExactGreenAttemptAndReplays(t *testing.T) {
	store, proof := newWitnessHeldStore(t)
	before, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}

	resolved, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof)
	if err != nil {
		t.Fatalf("ResolveWitnessHeld: %v", err)
	}
	wantDigest, err := proof.digest()
	if err != nil {
		t.Fatal(err)
	}
	if resolved.State != AttemptSucceeded || resolved.WitnessDigest != wantDigest || resolved.PID != 4242 || resolved.StartedAt.IsZero() {
		t.Fatalf("resolved attempt = %+v", resolved)
	}
	after, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if after.Generation == before.Generation {
		t.Fatal("ResolveWitnessHeld did not advance generation")
	}
	if after.Intents[0].State != IntentCompleted || after.Intents[0].RetryEligible || after.Intents[0].HoldReason != "" {
		t.Fatalf("resolved intent = %+v, want completed without retry/hold", after.Intents[0])
	}
	if got := lifecycleAttempt(t, after, "attempt-witness"); got.State != AttemptSucceeded || got.WitnessDigest != wantDigest {
		t.Fatalf("persisted resolved attempt = %+v", got)
	}

	replayed, err := FileStore(store.Path).ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof)
	if err != nil || replayed.State != AttemptSucceeded || replayed.WitnessDigest != wantDigest {
		t.Fatalf("idempotent ResolveWitnessHeld = %+v, err %v", replayed, err)
	}
	afterReplay, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if afterReplay.Generation != after.Generation {
		t.Fatalf("idempotent replay advanced generation from %q to %q", after.Generation, afterReplay.Generation)
	}
}

func TestStoreResolveWitnessHeldFencesIdentityAndRejectsNonGreenProof(t *testing.T) {
	t.Run("stale nonce", func(t *testing.T) {
		store, proof := newWitnessHeldStore(t)
		if _, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "stale-nonce", proof); !errors.Is(err, ErrFenced) {
			t.Fatalf("ResolveWitnessHeld error = %v, want ErrFenced", err)
		}
		assertWitnessHeldUnchanged(t, store)
	})

	t.Run("wrong attempt", func(t *testing.T) {
		store, proof := newWitnessHeldStore(t)
		if _, err := store.ResolveWitnessHeld(context.Background(), "other-attempt", "nonce-witness", proof); !errors.Is(err, ErrFenced) {
			t.Fatalf("ResolveWitnessHeld error = %v, want ErrFenced", err)
		}
		assertWitnessHeldUnchanged(t, store)
	})

	t.Run("wrong issue", func(t *testing.T) {
		store, proof := newWitnessHeldStore(t)
		proof.Issue++
		if _, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof); !errors.Is(err, ErrFenced) {
			t.Fatalf("ResolveWitnessHeld error = %v, want ErrFenced", err)
		}
		assertWitnessHeldUnchanged(t, store)
	})

	t.Run("wrong pid", func(t *testing.T) {
		store, proof := newWitnessHeldStore(t)
		proof.PID++
		if _, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof); !errors.Is(err, ErrFenced) {
			t.Fatalf("ResolveWitnessHeld error = %v, want ErrFenced", err)
		}
		assertWitnessHeldUnchanged(t, store)
	})

	t.Run("non-green proof", func(t *testing.T) {
		store, proof := newWitnessHeldStore(t)
		proof.TestClaim = "CLAIM_TEST_RED"
		if _, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof); err == nil || errors.Is(err, ErrFenced) {
			t.Fatalf("ResolveWitnessHeld error = %v, want proof validation error", err)
		}
		assertWitnessHeldUnchanged(t, store)
	})

	for _, tc := range []struct {
		name   string
		mutate func(*Attempt)
	}{
		{name: "missing persisted pid", mutate: func(attempt *Attempt) { attempt.PID = 0 }},
		{name: "missing persisted start", mutate: func(attempt *Attempt) { attempt.StartedAt = time.Time{} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store, proof := newWitnessHeldStore(t)
			snapshot, err := store.Load()
			if err != nil {
				t.Fatal(err)
			}
			for index := range snapshot.Attempts {
				if snapshot.Attempts[index].ID == "attempt-witness" {
					tc.mutate(&snapshot.Attempts[index])
				}
			}
			if err := store.Save(snapshot); err != nil {
				t.Fatal(err)
			}
			if _, err := store.ResolveWitnessHeld(context.Background(), "attempt-witness", "nonce-witness", proof); !errors.Is(err, ErrFenced) {
				t.Fatalf("ResolveWitnessHeld error = %v, want ErrFenced", err)
			}
			loaded, err := store.Load()
			if err != nil {
				t.Fatal(err)
			}
			if loaded.Intents[0].State != IntentHeld || loaded.Attempts[0].State != AttemptFailed || loaded.Attempts[0].WitnessDigest != "" {
				t.Fatalf("invalid persisted identity resolved: attempt=%+v intent=%+v", loaded.Attempts[0], loaded.Intents[0])
			}
		})
	}
}

func newWitnessHeldStore(t *testing.T) (Store, WorkWitnessProof) {
	t.Helper()
	store := FileStore(filepath.Join(t.TempDir(), "agentqueue.json"))
	startedAt := time.Date(2026, 9, 23, 14, 0, 0, 0, time.UTC)
	if err := store.Save(Snapshot{
		Schema: Schema, Generation: "generation-witness-running",
		Pool: PoolSpec{ID: "pool-witness", Desired: 1, Max: 1},
		Intents: []Intent{{
			ID: "intent-witness", State: IntentRunning, RetryEligible: true, PID: 4242,
			Launch: LaunchSpec{Issue: 13491, Lane: "cmd"},
		}},
		Attempts: []Attempt{{
			ID: "attempt-witness", IntentID: "intent-witness", State: AttemptRunning,
			Nonce: "nonce-witness", LaunchDeadline: time.Now().Add(time.Hour), PID: 4242, StartedAt: startedAt,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.HoldAttempt(context.Background(), "attempt-witness", "nonce-witness", 4242, startedAt, HoldAwaitingWorkWitness); err != nil {
		t.Fatalf("HoldAttempt: %v", err)
	}
	return store, WorkWitnessProof{
		Issue: 13491, PID: 4242, LogStem: "resolve-13491-20260923-140000", SHA: "0123456789abcdef", Claim: "CLAIM_WITNESSED", Verdict: "OK",
		Witness: "diff-witnessed", TestClaim: "CLAIM_TEST_GREEN",
	}
}

func assertWitnessHeldUnchanged(t *testing.T, store Store) {
	t.Helper()
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Intents[0].State != IntentHeld || loaded.Intents[0].HoldReason != HoldAwaitingWorkWitness ||
		loaded.Attempts[0].State != AttemptFailed || loaded.Attempts[0].WitnessDigest != "" {
		t.Fatalf("held witness changed: attempt=%+v intent=%+v", loaded.Attempts[0], loaded.Intents[0])
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
