package modelengine

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeSchedulerBlockedDrainDonationPreservesLifecycleSemantics(t *testing.T) {
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}

	fallback := runDrainDonationLifecycle(t, calls, false)
	donated := runDrainDonationLifecycle(t, calls, true)

	if donated.iterations == 0 {
		t.Fatal("blocked drain executed no native scheduler iterations")
	}
	if fallback.iterations != 0 {
		t.Fatalf("fallback recorded %d donated iterations, want 0", fallback.iterations)
	}
	for _, i := range []int{0, 2} {
		if !reflect.DeepEqual(donated.generated[i], fallback.generated[i]) {
			t.Fatalf("donated survivor %d tokens = %v, fallback oracle = %v", i, donated.generated[i], fallback.generated[i])
		}
	}
	if !reflect.DeepEqual(donated.firstReady, []int{1, 2}) {
		t.Fatalf("donated first-ready order = %v, want FIFO [1 2]", donated.firstReady)
	}
	if !reflect.DeepEqual(donated.firstReady, fallback.firstReady) {
		t.Fatalf("donated first-ready order = %v, fallback oracle = %v", donated.firstReady, fallback.firstReady)
	}
	if !reflect.DeepEqual(donated.statuses, fallback.statuses) {
		t.Fatalf("donated statuses = %v, fallback oracle = %v", donated.statuses, fallback.statuses)
	}
	if !reflect.DeepEqual(donated.errs, fallback.errs) {
		t.Fatalf("donated errors = %v, fallback oracle = %v", donated.errs, fallback.errs)
	}
	if !reflect.DeepEqual(donated.reclaimed, []bool{true, true, true}) {
		t.Fatalf("donated reclaim state = %v, want all lanes reclaimed", donated.reclaimed)
	}
}

type drainDonationLifecycle struct {
	generated  [][]int
	firstReady []int
	statuses   []abi.Status
	errs       []error
	reclaimed  []bool
	iterations uint64
}

func runDrainDonationLifecycle(t *testing.T, calls []*abi.ToolCall, donation bool) drainDonationLifecycle {
	t.Helper()

	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetMaxRunning(1)
	s.SetDrainDonation(donation)

	// Register the donor before Admit, as Complete does. That makes the handoff
	// deterministic: the scheduler goroutine remains the disabled-path oracle, while
	// the enabled path must be advanced by this blocked drain.
	if got := s.beginBlockedDrain(); got != donation {
		t.Fatalf("beginBlockedDrain() = %v, want %v", got, donation)
	}

	reqs := make([]abi.EngineRequest, len(calls))
	for i, call := range calls {
		req, err := s.Admit(context.Background(), call)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = req
	}

	firstReady := make(chan int, len(reqs)-1)
	var wg sync.WaitGroup
	for i := 1; i < len(reqs); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			seen := 0
			for range reqs[i].Tokens() {
				seen++
				if seen == 1 {
					firstReady <- i
				}
				if i == 1 && seen == 2 {
					reqs[i].Cancel()
				}
			}
		}(i)
	}

	if donation {
		s.drainWithDonation(reqs[0].Tokens())
		s.endBlockedDrain()
	} else {
		for range reqs[0].Tokens() {
		}
	}
	wg.Wait()
	close(firstReady)

	got := drainDonationLifecycle{
		generated:  make([][]int, len(reqs)),
		statuses:   make([]abi.Status, len(reqs)),
		errs:       make([]error, len(reqs)),
		reclaimed:  make([]bool, len(reqs)),
		iterations: s.DrainDonationReceipt().Iterations,
	}
	for i := range firstReady {
		got.firstReady = append(got.firstReady, i)
	}
	for i, req := range reqs {
		lane, ok := req.(*schedLane)
		if !ok {
			t.Fatalf("request %d type = %T, want *schedLane", i, req)
		}
		result, err := req.Result()
		got.generated[i] = append([]int(nil), lane.gen...)
		got.errs[i] = err
		got.reclaimed[i] = lane.Reclaimed()
		if i == 1 {
			if result != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled request result = %+v, err = %v; want nil, context.Canceled", result, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("request %d Result: %v", i, err)
		}
		if result == nil {
			t.Fatalf("request %d Result = nil", i)
		}
		got.statuses[i] = result.Status
	}

	// Close remains idempotent, and its fail-closed admission boundary is unchanged.
	s.Close()
	s.Close()
	if _, err := s.Admit(context.Background(), calls[0]); !errors.Is(err, errSchedClosed) {
		t.Fatalf("Admit after Close err = %v, want %v", err, errSchedClosed)
	}
	return got
}
