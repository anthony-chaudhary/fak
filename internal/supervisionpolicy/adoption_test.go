package supervisionpolicy

import (
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestTakeoverAdoptsLiveChildrenExactlyOnceAndFencesOldCoordinator(t *testing.T) {
	now := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	store := AdoptionStore{Path: filepath.Join(t.TempDir(), "adoption.json")}
	restartA := ChildState{Failures: []time.Time{now.Add(-time.Second)}, LastReceipt: "failure-a"}
	children := []AdoptedChild{
		{ID: "worker-a", Identity: ProcessIdentity{RunID: "run-a", PID: 101, Birth: 1001}, Progress: 4, Restart: restartA},
		{ID: "worker-b", Identity: ProcessIdentity{RunID: "run-b", PID: 102, Birth: 1002}, Progress: 7},
		{ID: "worker-done", Identity: ProcessIdentity{RunID: "run-done", PID: 103, Birth: 1003}, TerminalReceipt: "receipt-done"},
		// Same PID as worker-a with a different birth models PID reuse. It must
		// be quarantined rather than adopted or killed.
		{ID: "stale-pid", Identity: ProcessIdentity{RunID: "run-stale", PID: 101, Birth: 999}},
	}
	if err := store.Initialize("coordinator-1", 1, children); err != nil {
		t.Fatal(err)
	}

	// Worker progress after C1 disappears is durable and visible to C2. The
	// child records are not relaunched or reconstructed by takeover.
	state, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	a := state.Children["worker-a"]
	a.Progress++
	state.Children["worker-a"] = a
	b := state.Children["worker-b"]
	b.Progress++
	state.Children["worker-b"] = b
	if err := store.save(state); err != nil {
		t.Fatal(err)
	}

	var winners atomic.Int32
	var verified atomic.Int32
	results := make(chan AdoptionResult, 2)
	errs := make(chan error, 2)
	var wg sync.WaitGroup
	for _, owner := range []string{"coordinator-2", "coordinator-3"} {
		owner := owner
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, err := store.Takeover(owner, 1, now, 30*time.Second, func(got ProcessIdentity) bool {
				verified.Add(1)
				return got.Birth == 1001 || got.Birth == 1002
			})
			if err == nil {
				winners.Add(1)
				results <- result
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	if winners.Load() != 1 {
		t.Fatalf("takeover winners = %d, want 1", winners.Load())
	}
	var fenced int
	for err := range errs {
		if errors.Is(err, ErrFenced) {
			fenced++
		} else if err != nil {
			t.Fatalf("unexpected takeover error: %v", err)
		}
	}
	if fenced != 1 {
		t.Fatalf("fenced competitors = %d, want 1", fenced)
	}
	result := <-results
	if result.Epoch != 2 || result.Children["worker-a"] != AdoptionAdopted || result.Children["worker-b"] != AdoptionAdopted || result.Children["worker-done"] != AdoptionReconciled || result.Children["stale-pid"] != AdoptionQuarantine {
		t.Fatalf("takeover result = %+v", result)
	}
	if verified.Load() != 3 { // terminal child is reconciled from its receipt.
		t.Fatalf("identity verifications = %d, want 3", verified.Load())
	}

	adopted, err := store.Load()
	if err != nil {
		t.Fatal(err)
	}
	if adopted.Children["worker-a"].Progress != 5 || adopted.Children["worker-b"].Progress != 8 {
		t.Fatalf("post-death progress lost: %+v", adopted.Children)
	}
	gotRestart := adopted.Children["worker-a"].Restart
	if len(gotRestart.Failures) != 1 || gotRestart.LastReceipt != restartA.LastReceipt {
		t.Fatalf("restart budget changed across takeover: %+v", gotRestart)
	}
	winner := adopted.Owner
	if ok, err := store.CanCommand("coordinator-1", 1, "worker-a", now); err != nil || ok {
		t.Fatalf("stale coordinator command: ok=%v err=%v", ok, err)
	}
	if ok, err := store.CanCommand(winner, 2, "worker-a", now.Add(29*time.Second)); err != nil || !ok {
		t.Fatalf("adopted coordinator command: ok=%v err=%v", ok, err)
	}
	if ok, err := store.CanCommand(winner, 2, "worker-a", now.Add(31*time.Second)); err != nil || ok {
		t.Fatalf("expired reconnect command: ok=%v err=%v", ok, err)
	}
	if ok, err := store.CanCommand(winner, 2, "stale-pid", now); err != nil || ok {
		t.Fatalf("quarantined PID command: ok=%v err=%v", ok, err)
	}
}

func TestTakeoverContentionReturnsFenceNotPlatformIOError(t *testing.T) {
	for i := 0; i < 25; i++ {
		store := AdoptionStore{Path: filepath.Join(t.TempDir(), "adoption.json")}
		if err := store.Initialize("coordinator-1", 1, nil); err != nil {
			t.Fatal(err)
		}
		start := make(chan struct{})
		errs := make(chan error, 2)
		for _, owner := range []string{"coordinator-2", "coordinator-3"} {
			owner := owner
			go func() {
				<-start
				_, err := store.Takeover(owner, 1, time.Now(), time.Second, nil)
				errs <- err
			}()
		}
		close(start)
		first, second := <-errs, <-errs
		if (first == nil) == (second == nil) || (!errors.Is(first, ErrFenced) && !errors.Is(second, ErrFenced)) {
			t.Fatalf("iteration %d errors = (%v, %v), want one winner and one fence", i, first, second)
		}
	}
}
