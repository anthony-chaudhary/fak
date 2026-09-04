package agentqueue

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestReconcileRestartCrashRecoveryE2E(t *testing.T) {
	livePID := 4242
	deadPID := 9999
	liveness := func(pid int) bool {
		return pid == livePID
	}

	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "gen:pre-crash",
		Pool:       PoolSpec{ID: "pool-recovery", Min: 1, Desired: 2, Max: 2},
		Intents: []Intent{
			{
				ID:     "intent-live",
				State:  IntentRunning,
				Launch: LaunchSpec{Issue: 8887, Lane: "agentqueue"},
			},
			{
				ID:     "intent-dead",
				State:  IntentRunning,
				Launch: LaunchSpec{Issue: 8888, Lane: "agentqueue"},
			},
		},
		Attempts: []Attempt{
			{
				ID:       "att-live",
				IntentID: "intent-live",
				State:    AttemptRunning,
				PID:      livePID,
			},
			{
				ID:       "att-dead",
				IntentID: "intent-dead",
				State:    AttemptRunning,
				PID:      deadPID,
			},
		},
	}

	// 1. Run ReconcileRestart
	rec, updated, err := ReconcileRestart(snapshot, liveness, RestartOptions{})
	if err != nil {
		t.Fatalf("ReconcileRestart failed: %v", err)
	}

	if rec.Schema != RestartReconcileSchema {
		t.Fatalf("rec.Schema = %q, want %q", rec.Schema, RestartReconcileSchema)
	}
	if rec.Generation == snapshot.Generation {
		t.Fatalf("generation was not advanced: %q", rec.Generation)
	}
	if updated.Generation != rec.Generation {
		t.Fatalf("updated snapshot generation mismatch: %q vs %q", updated.Generation, rec.Generation)
	}

	// Live attempt is ADOPTed (count remains 1, not doubled)
	if len(rec.Adopted) != 1 {
		t.Fatalf("len(rec.Adopted) = %d, want 1", len(rec.Adopted))
	}
	if rec.Adopted[0].IntentID != "intent-live" || rec.Adopted[0].Action != AttemptActionAdopt || rec.Adopted[0].PID != livePID {
		t.Fatalf("unexpected adopted disposition: %+v", rec.Adopted[0])
	}

	// Dead attempt is REPLACEd (re-queued exactly once)
	if len(rec.Replaced) != 1 {
		t.Fatalf("len(rec.Replaced) = %d, want 1", len(rec.Replaced))
	}
	if rec.Replaced[0].IntentID != "intent-dead" || rec.Replaced[0].Action != AttemptActionReplace || rec.Replaced[0].PID != deadPID {
		t.Fatalf("unexpected replaced disposition: %+v", rec.Replaced[0])
	}
	if len(rec.Held) != 0 {
		t.Fatalf("len(rec.Held) = %d, want 0", len(rec.Held))
	}

	// Verify updated snapshot states
	var liveIntent, deadIntent *Intent
	for i := range updated.Intents {
		if updated.Intents[i].ID == "intent-live" {
			liveIntent = &updated.Intents[i]
		}
		if updated.Intents[i].ID == "intent-dead" {
			deadIntent = &updated.Intents[i]
		}
	}
	if liveIntent == nil || liveIntent.State != IntentRunning {
		t.Fatalf("liveIntent state = %v, want %v", liveIntent, IntentRunning)
	}
	if deadIntent == nil || deadIntent.State != IntentQueued {
		t.Fatalf("deadIntent state = %v, want %v", deadIntent, IntentQueued)
	}

	// 2. Persist updated snapshot and run controller tick
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	if err := store.Save(updated); err != nil {
		t.Fatalf("save updated snapshot: %v", err)
	}

	runner := &recordingRunner{}
	controller := Controller{Store: store, FakPath: "fak", Runner: runner}

	tickReceipt, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatalf("controller.Tick failed: %v", err)
	}

	// Running controller tick launches only the replaced attempt, keeping adopted attempt running
	if len(tickReceipt.Plan.Start) != 1 {
		t.Fatalf("tick plan start count = %d, want 1", len(tickReceipt.Plan.Start))
	}
	if tickReceipt.Plan.Start[0].IntentID != "intent-dead" {
		t.Fatalf("tick launched intent %q, want 'intent-dead'", tickReceipt.Plan.Start[0].IntentID)
	}
	if len(tickReceipt.Launches) != 1 || tickReceipt.Launches[0].IntentID != "intent-dead" {
		t.Fatalf("tick launches = %+v, want 1 launch for 'intent-dead'", tickReceipt.Launches)
	}
	if len(runner.calls) != 1 {
		t.Fatalf("runner calls = %d, want 1", len(runner.calls))
	}

	// Final check: total active attempts in store is 2 (1 adopted + 1 newly started)
	final, err := store.Load()
	if err != nil {
		t.Fatalf("store.Load: %v", err)
	}
	activeCount := 0
	for _, a := range final.Attempts {
		if a.State == AttemptReserved || a.State == AttemptRunning {
			activeCount++
		}
	}
	if activeCount != 2 {
		t.Fatalf("activeCount = %d, want 2", activeCount)
	}
}

func TestControllerWithReconcileOnStart(t *testing.T) {
	livePID := 5001
	deadPID := 5002
	liveness := func(pid int) bool {
		return pid == livePID
	}

	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	crashed := Snapshot{
		Schema:     Schema,
		Generation: "g-crash",
		Pool:       PoolSpec{ID: "pool-on-start", Min: 1, Desired: 2, Max: 2},
		Intents: []Intent{
			{ID: "live-job", State: IntentRunning, Launch: LaunchSpec{Issue: 101, Lane: "agentqueue"}},
			{ID: "dead-job", State: IntentRunning, Launch: LaunchSpec{Issue: 102, Lane: "agentqueue"}},
		},
		Attempts: []Attempt{
			{ID: "att-live", IntentID: "live-job", State: AttemptRunning, PID: livePID},
			{ID: "att-dead", IntentID: "dead-job", State: AttemptRunning, PID: deadPID},
		},
	}
	if err := store.Save(crashed); err != nil {
		t.Fatalf("store.Save: %v", err)
	}

	runner := &recordingRunner{}
	controller := Controller{
		Store:            store,
		FakPath:          "fak",
		Runner:           runner,
		ReconcileOnStart: true,
		Liveness:         liveness,
	}

	receipt, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick failed: %v", err)
	}

	if receipt.Restart == nil {
		t.Fatal("expected receipt.Restart to be populated")
	}
	if len(receipt.Restart.Adopted) != 1 || receipt.Restart.Adopted[0].IntentID != "live-job" {
		t.Fatalf("unexpected adopted: %+v", receipt.Restart.Adopted)
	}
	if len(receipt.Restart.Replaced) != 1 || receipt.Restart.Replaced[0].IntentID != "dead-job" {
		t.Fatalf("unexpected replaced: %+v", receipt.Restart.Replaced)
	}
	if len(receipt.Launches) != 1 || receipt.Launches[0].IntentID != "dead-job" {
		t.Fatalf("unexpected launches: %+v", receipt.Launches)
	}

	// Subsequent tick shouldn't re-reconcile restart
	secondReceipt, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatalf("second Tick failed: %v", err)
	}
	if secondReceipt.Restart != nil {
		t.Fatalf("expected nil Restart on second tick, got %+v", secondReceipt.Restart)
	}
	if len(secondReceipt.Launches) != 0 {
		t.Fatalf("unexpected launches on second tick: %+v", secondReceipt.Launches)
	}
}

func TestReconcileRestartExpiredLeaseHolds(t *testing.T) {
	now := time.Now()
	expiredTime := now.Add(-5 * time.Minute)

	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "g0",
		Pool:       PoolSpec{ID: "pool-expired", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{
				ID:           "intent-expired",
				State:        IntentRunning,
				LeaseExpires: expiredTime,
				Launch:       LaunchSpec{Issue: 999, Lane: "agentqueue"},
			},
		},
		Attempts: []Attempt{
			{
				ID:       "att-expired",
				IntentID: "intent-expired",
				State:    AttemptRunning,
				PID:      1234,
			},
		},
	}

	liveness := func(pid int) bool { return true }
	rec, updated, err := ReconcileRestart(snapshot, liveness, RestartOptions{Now: now})
	if err != nil {
		t.Fatalf("ReconcileRestart failed: %v", err)
	}

	if len(rec.Held) != 1 {
		t.Fatalf("len(rec.Held) = %d, want 1", len(rec.Held))
	}
	if rec.Held[0].IntentID != "intent-expired" || rec.Held[0].Action != AttemptActionHold {
		t.Fatalf("unexpected held disposition: %+v", rec.Held[0])
	}
	if len(rec.Adopted) != 0 || len(rec.Replaced) != 0 {
		t.Fatalf("expected 0 adopted and 0 replaced, got adopted=%d replaced=%d", len(rec.Adopted), len(rec.Replaced))
	}

	if updated.Intents[0].State != IntentHeld {
		t.Fatalf("intent state = %v, want %v", updated.Intents[0].State, IntentHeld)
	}
	if updated.Attempts[0].State != AttemptFailed {
		t.Fatalf("attempt state = %v, want %v", updated.Attempts[0].State, AttemptFailed)
	}
}

func TestReconcileRestartIndeterminateHolds(t *testing.T) {
	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "g0",
		Pool:       PoolSpec{ID: "pool-indet", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{
				ID:     "intent-indet",
				State:  IntentRunning,
				Launch: LaunchSpec{Issue: 777, Lane: "agentqueue"},
			},
		},
		Attempts: []Attempt{
			{
				ID:       "att-indet",
				IntentID: "intent-indet",
				State:    AttemptRunning,
				PID:      888,
			},
		},
	}

	opts := RestartOptions{
		Indeterminate: func(intent Intent, attempt *Attempt) bool {
			return true
		},
	}
	rec, updated, err := ReconcileRestart(snapshot, nil, opts)
	if err != nil {
		t.Fatalf("ReconcileRestart failed: %v", err)
	}

	if len(rec.Held) != 1 || rec.Held[0].IntentID != "intent-indet" || rec.Held[0].Action != AttemptActionHold {
		t.Fatalf("unexpected held: %+v", rec.Held)
	}
	if updated.Intents[0].State != IntentHeld {
		t.Fatalf("intent state = %v, want %v", updated.Intents[0].State, IntentHeld)
	}
}

func TestReconcileRestartFencesPriorController(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "gen-before-restart",
		Pool:       PoolSpec{ID: "pool-fenced", Min: 0, Desired: 1, Max: 1},
		Intents: []Intent{
			{ID: "job-1", State: IntentRunning, Launch: LaunchSpec{Issue: 50, Lane: "agentqueue"}},
		},
		Attempts: []Attempt{
			{ID: "att-1", IntentID: "job-1", State: AttemptRunning, PID: 1111},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatalf("store.Save: %v", err)
	}

	// Controller 1 observed generation "gen-before-restart"
	priorGen := snapshot.Generation

	// Controller 2 restarts and reconciles
	liveness := func(pid int) bool { return false }
	rec, _, err := store.ReconcileRestart(context.Background(), liveness, RestartOptions{})
	if err != nil {
		t.Fatalf("ReconcileRestart: %v", err)
	}

	if rec.Generation == priorGen {
		t.Fatalf("generation not advanced")
	}

	// Prior controller attempting to reserve with prior generation is fenced
	_, _, err = store.Reserve(context.Background(), priorGen)
	if err == nil {
		t.Fatal("expected ErrGenerationConflict, got nil")
	}
}
