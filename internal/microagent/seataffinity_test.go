package microagent

import (
	"errors"
	"sync"
	"testing"
)

func testSeatPool(t *testing.T, seats, slots int) *SeatPool {
	t.Helper()
	configured := make([]Seat, 0, seats)
	for i := 0; i < seats; i++ {
		scheduler := NewScheduler(slots)
		configured = append(configured, Seat{ID: string(rune('a' + i)), Scheduler: scheduler})
	}
	pool, err := NewSeatPool(configured)
	if err != nil {
		t.Fatal(err)
	}
	return pool
}

func TestSeatPoolKeepsStableAffinityOnOneSeat(t *testing.T) {
	pool := testSeatPool(t, 3, 1)
	first, err := pool.TryAcquire("shared-prefix")
	if err != nil {
		t.Fatal(err)
	}
	want := first.SeatID
	first.Release()
	for i := 0; i < 5; i++ {
		got, err := pool.TryAcquire("shared-prefix")
		if err != nil {
			t.Fatal(err)
		}
		if got.SeatID != want {
			t.Fatalf("seat=%q want stable %q", got.SeatID, want)
		}
		got.Release()
	}
}

func TestSeatPoolFallsBackWhenAffinitySeatBusy(t *testing.T) {
	pool := testSeatPool(t, 2, 1)
	first, err := pool.TryAcquire("shared-prefix")
	if err != nil {
		t.Fatal(err)
	}
	defer first.Release()
	fallback, err := pool.TryAcquire("shared-prefix")
	if err != nil {
		t.Fatal(err)
	}
	defer fallback.Release()
	if fallback.SeatID == first.SeatID {
		t.Fatalf("fallback reused busy seat %q", first.SeatID)
	}
	if _, err := pool.TryAcquire("shared-prefix"); !errors.Is(err, ErrNoSeatAvailable) {
		t.Fatalf("all seats busy err=%v", err)
	}
}

func TestSeatPoolNeverExceedsPerSeatConcurrency(t *testing.T) {
	pool := testSeatPool(t, 3, 1)
	var wg sync.WaitGroup
	start := make(chan struct{})
	results := make(chan *SeatLease, 12)
	errs := make(chan error, 12)
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			lease, err := pool.TryAcquire("")
			if err != nil {
				errs <- err
				return
			}
			results <- lease
		}()
	}
	close(start)
	wg.Wait()
	close(results)
	close(errs)
	seen := map[string]int{}
	for lease := range results {
		seen[lease.SeatID]++
		defer lease.Release()
	}
	for seat, count := range seen {
		if count > 1 {
			t.Fatalf("seat %s admitted %d concurrent calls", seat, count)
		}
	}
	for err := range errs {
		if !errors.Is(err, ErrNoSeatAvailable) {
			t.Fatalf("unexpected error: %v", err)
		}
	}
	if len(seen) != 3 {
		t.Fatalf("admitted seats=%v want all 3 bounded seats", seen)
	}
}

func TestNewSeatPoolRejectsInvalidSeats(t *testing.T) {
	if _, err := NewSeatPool(nil); err == nil {
		t.Fatal("empty pool accepted")
	}
	scheduler := NewScheduler(1)
	if _, err := NewSeatPool([]Seat{{ID: "same", Scheduler: scheduler}, {ID: "same", Scheduler: scheduler}}); err == nil {
		t.Fatal("duplicate ID accepted")
	}
}
