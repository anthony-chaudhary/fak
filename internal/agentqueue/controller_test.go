package agentqueue

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

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
	runner := &recordingRunner{}
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
		if attempt.State == AttemptReserved || attempt.State == AttemptRunning {
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
	controller := Controller{Store: store, FakPath: "fak", Runner: &recordingRunner{}, Interval: time.Millisecond}
	ticks := 0
	err := controller.Run(ctx, func(TickReceipt) { ticks++; cancel() })
	if err != nil || ticks != 1 {
		t.Fatalf("Run ticks=%d err=%v", ticks, err)
	}
}
