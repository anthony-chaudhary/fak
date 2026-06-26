package safecommit

import (
	"context"
	"testing"
)

func TestWindowAIMDGrowsHalvesAndFloors(t *testing.T) {
	w := NewWindow(1)

	w.Observe(Result{Verified: true})
	if got := w.Limit(); got != 2 {
		t.Fatalf("success should grow W to 2, got %d", got)
	}

	w.Observe(Result{Reason: ReasonHookRefused})
	if got := w.Limit(); got != 1 {
		t.Fatalf("failure should halve W to 1, got %d", got)
	}

	w.Observe(Result{Reason: ReasonPushRejected})
	if got := w.Limit(); got != 1 {
		t.Fatalf("failure at W=1 must floor at 1, got %d", got)
	}
}

func TestWindowBlocksWPlusOneConcurrentWriter(t *testing.T) {
	w := NewWindow(1)

	release, ok := w.TryAcquire()
	if !ok {
		t.Fatal("first writer should be admitted")
	}
	if _, ok := w.TryAcquire(); ok {
		t.Fatal("second writer should be blocked when W=1 and one writer is in flight")
	}
	if got := w.InFlight(); got != 1 {
		t.Fatalf("in-flight = %d, want 1", got)
	}

	release(Result{Verified: true})
	if got := w.Limit(); got != 2 {
		t.Fatalf("successful release should grow W to 2, got %d", got)
	}

	r1, ok1 := w.TryAcquire()
	r2, ok2 := w.TryAcquire()
	if !ok1 || !ok2 {
		t.Fatalf("two writers should be admitted at W=2, ok1=%v ok2=%v", ok1, ok2)
	}
	if _, ok := w.TryAcquire(); ok {
		t.Fatal("third writer should be blocked when W=2 and two writers are in flight")
	}
	r1(Result{Verified: true})
	r2(Result{Verified: true})
}

func TestCommitWithWindowFullRefusesBeforeGit(t *testing.T) {
	w := NewWindow(1)
	hold, ok := w.TryAcquire()
	if !ok {
		t.Fatal("setup writer should be admitted")
	}
	defer hold(Result{Verified: true})

	g := &fakeGit{reply: onTrunkBase()}
	opts := baseOpts()
	opts.Window = w

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonWindowFull {
		t.Fatalf("Reason = %q, want %q", res.Reason, ReasonWindowFull)
	}
	if len(g.calls) != 0 {
		t.Fatalf("window-full refusal must happen before git calls, got %v", g.calls)
	}
}

func TestCommitWithUpdatesWindowFromShipResult(t *testing.T) {
	w := NewWindow(1)
	opts := baseOpts()
	opts.Window = w

	g := &fakeGit{reply: onTrunkBase()}
	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if !res.Verified {
		t.Fatalf("happy path should verify, got %+v", res)
	}
	if got := w.Limit(); got != 2 {
		t.Fatalf("successful ship should grow W to 2, got %d", got)
	}

	raced := &fakeGit{reply: onTrunkBase()}
	raced.reply["diff-tree"] = reply{out: "internal/foo/bar.go\ninternal/peer/swept.go\n", code: 0}
	res, err = CommitWith(context.Background(), raced.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonPathspecRace {
		t.Fatalf("raced commit should fail with PATHSPEC_RACE, got %+v", res)
	}
	if got := w.Limit(); got != 1 {
		t.Fatalf("failed ship should halve W back to 1, got %d", got)
	}
}
