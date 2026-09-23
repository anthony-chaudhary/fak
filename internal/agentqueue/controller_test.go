package agentqueue

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

type refuseOnceRunner struct {
	calls    int
	launches launchingRecordingRunner
}

func (r *refuseOnceRunner) Run(ctx context.Context, name string, args ...string) error {
	r.calls++
	if r.calls == 1 {
		statePath, hasState := commandFlag(args, "--agentqueue-state")
		attemptID, hasAttempt := commandFlag(args, "--agentqueue-attempt")
		nonce, hasNonce := commandFlag(args, "--agentqueue-nonce")
		if !hasState || !hasAttempt || !hasNonce {
			return errors.New("dispatch refusal missing launch identity")
		}
		_, err := FileStore(statePath).AbortLaunching(ctx, attemptID, nonce, 0, time.Time{})
		return err
	}
	return r.launches.Run(ctx, name, args...)
}

func TestControllerSustainsDesiredAndNeverExceedsMax(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	snapshot := Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Min: 1, Desired: 2, Max: 2},
		Intents: []Intent{
			{ID: "a", State: IntentQueued, Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
			{ID: "b", State: IntentQueued, Launch: LaunchSpec{Issue: 102, Lane: "gateway"}},
			{ID: "c", State: IntentQueued, Launch: LaunchSpec{Issue: 103, Lane: "model"}},
		},
	}
	if err := store.Save(snapshot); err != nil {
		t.Fatal(err)
	}
	runner := &launchingRecordingRunner{}
	controller := Controller{Store: store, FakPath: "fak", Runner: runner}
	first, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Launches) != 2 {
		t.Fatalf("first launches=%d, want 2", len(first.Launches))
	}
	second, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Launches) != 0 || second.Plan.Observed != 2 {
		t.Fatalf("capacity tick = %#v", second)
	}

	failed, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	failed.Attempts[0].State = AttemptFailed
	for i := range failed.Intents {
		if failed.Intents[i].ID == failed.Attempts[0].IntentID {
			failed.Intents[i].State = IntentFailed
			failed.Intents[i].RetryEligible = true
		}
	}
	failed.Generation = "after-failure"
	if err := store.Save(failed); err != nil {
		t.Fatal(err)
	}
	replacement, err := controller.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(replacement.Launches) != 1 || replacement.Plan.Observed != 1 {
		t.Fatalf("replacement tick = %#v", replacement)
	}
	final, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	active := 0
	for _, attempt := range final.Attempts {
		if attempt.State == AttemptReserved || attempt.State == AttemptLaunching || attempt.State == AttemptRunning {
			active++
		}
	}
	if active != 2 || active > final.Pool.Max {
		t.Fatalf("active=%d max=%d attempts=%#v", active, final.Pool.Max, final.Attempts)
	}
}

func TestControllerRunStopsOnContext(t *testing.T) {
	store := Store{Path: filepath.Join(t.TempDir(), "queue.json")}
	if err := store.Save(Snapshot{Schema: Schema, Generation: "g", Pool: PoolSpec{ID: "p", Max: 1}}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	controller := Controller{Store: store, FakPath: "fak", Runner: &launchingRecordingRunner{}, Interval: time.Millisecond}
	ticks := 0
	err := controller.Run(ctx, func(TickReceipt) { ticks++; cancel() })
	if err != nil || ticks != 1 {
		t.Fatalf("Run ticks=%d err=%v", ticks, err)
	}
}

func TestControllerRunContinuesAfterProvedPreSpawnRefusal(t *testing.T) {
	store := FileStore(filepath.Join(t.TempDir(), "queue.json"))
	if err := store.Save(Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Desired: 1, Max: 1},
		Intents: []Intent{{
			ID: "intent-a", State: IntentQueued,
			Launch: LaunchSpec{Issue: 101, Lane: "agentqueue"},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	runner := &refuseOnceRunner{}
	controller := Controller{Store: store, FakPath: "fak", Runner: runner, Interval: time.Millisecond}
	var receipts []TickReceipt
	err := controller.Run(ctx, func(receipt TickReceipt) {
		receipts = append(receipts, receipt)
		if len(receipt.Launches) > 0 {
			cancel()
		}
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if runner.calls != 2 || len(runner.launches.calls) != 1 {
		t.Fatalf("runner calls=%d claimed launches=%d, want 2/1", runner.calls, len(runner.launches.calls))
	}
	if len(receipts) != 2 || len(receipts[0].Launches) != 0 || len(receipts[1].Launches) != 1 {
		t.Fatalf("tick receipts = %+v, want deferred tick then launch", receipts)
	}
	if receipts[1].Launches[0].IntentID != "intent-a" {
		t.Fatalf("launched intent = %q, want intent-a", receipts[1].Launches[0].IntentID)
	}

	final, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(final.Attempts) != 2 || final.Attempts[0].State != AttemptFailed || final.Attempts[1].State != AttemptLaunching {
		t.Fatalf("attempts = %+v, want failed deferral then launching retry", final.Attempts)
	}
	if len(final.Intents) != 1 || final.Intents[0].ID != "intent-a" || final.Intents[0].State != IntentQueued {
		t.Fatalf("intent = %+v, want one queued intent", final.Intents)
	}
}

func TestControllerLaunchRestartWhileRunnerBlockedHoldsAttempt(t *testing.T) {
	store := FileStore(filepath.Join(t.TempDir(), "queue.json"))
	if err := store.Save(Snapshot{
		Schema: Schema, Generation: "g0",
		Pool: PoolSpec{ID: "pool", Desired: 1, Max: 1},
		Intents: []Intent{{
			ID: "intent-a", State: IntentQueued,
			Launch: LaunchSpec{Issue: 101, Lane: "agentqueue"},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	runner := &blockingLifecycleRunner{entered: make(chan struct{}), release: make(chan struct{})}
	firstDone := make(chan struct {
		receipt TickReceipt
		err     error
	}, 1)
	go func() {
		receipt, err := (Controller{Store: store, FakPath: "fak", Runner: runner}).Tick(context.Background())
		firstDone <- struct {
			receipt TickReceipt
			err     error
		}{receipt: receipt, err: err}
	}()

	select {
	case <-runner.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("runner did not observe launching attempt")
	}
	loaded, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Attempts) != 1 || loaded.Attempts[0].State != AttemptLaunching || loaded.Attempts[0].Nonce == "" {
		t.Fatalf("blocked runner state = %+v, want one launching attempt", loaded.Attempts)
	}

	restartRunner := &recordingRunner{}
	restart, err := (Controller{Store: store, FakPath: "fak", Runner: restartRunner, ReconcileOnStart: true}).Tick(context.Background())
	if err != nil {
		t.Fatalf("restart Tick: %v", err)
	}
	if len(restart.Restart.Held) != 1 || restart.Restart.Held[0].IntentID != "intent-a" {
		t.Fatalf("restart reconciliation = %+v, want held intent-a", restart.Restart)
	}
	if len(restart.Plan.Start) != 0 || len(restart.Launches) != 0 || len(restartRunner.calls) != 0 {
		t.Fatalf("restart duplicated launch: plan=%+v launches=%+v calls=%v", restart.Plan, restart.Launches, restartRunner.calls)
	}
	afterRestart, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(afterRestart.Attempts) != 1 || afterRestart.Attempts[0].State != AttemptLaunching {
		t.Fatalf("restart changed ambiguous launch: %+v", afterRestart.Attempts)
	}

	close(runner.release)
	select {
	case result := <-firstDone:
		if result.err != nil || len(result.receipt.Launches) != 1 {
			t.Fatalf("original Tick receipt=%+v err=%v", result.receipt, result.err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("original Tick did not finish")
	}
}
