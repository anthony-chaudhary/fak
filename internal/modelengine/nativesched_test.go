package modelengine

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestNativeSchedulerBatchesLanesAndFreesCancelled is the acceptance-#4 witness:
// the SAME abi.LifecycleEngine the per-request in-kernel engine implements also fits
// the continuous-batching shape. Three lanes are admitted and advanced by ONE shared
// StepBatch loop; cancelling one lane mid-run frees it (terminal context.Canceled +
// KV reclaim) WITHOUT disturbing the other two, which decode to completion.
func TestNativeSchedulerBatchesLanesAndFreesCancelled(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	defer s.Close()

	ctx := context.Background()
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}
	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := s.Admit(ctx, c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	const cancelIdx = 1
	const readBeforeCancel = 2

	// Drain the two survivor lanes fully in their own goroutines.
	counts := make([]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		if i == cancelIdx {
			continue
		}
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for range r.Tokens() {
				counts[i]++
			}
		}(i, r)
	}

	// Cancel the middle lane after reading a couple of tokens.
	cr := reqs[cancelIdx]
	got := 0
	for range cr.Tokens() {
		got++
		if got == readBeforeCancel {
			cr.Cancel()
			break
		}
	}
	for range cr.Tokens() { // drain residual so its lane retires
		got++
	}

	wg.Wait()

	// Survivors decode to completion, unaffected by the cancellation.
	for i := range reqs {
		if i == cancelIdx {
			continue
		}
		if counts[i] != genTokens {
			t.Fatalf("survivor lane %d streamed %d tokens, want %d", i, counts[i], genTokens)
		}
		res, err := reqs[i].Result()
		if err != nil {
			t.Fatalf("survivor lane %d Result: %v", i, err)
		}
		if res == nil || res.Status != abi.StatusOK {
			t.Fatalf("survivor lane %d result = %+v, want StatusOK", i, res)
		}
	}

	// The cancelled lane stopped early, ended Canceled, and reclaimed its slot.
	if got >= genTokens {
		t.Fatalf("cancelled lane did not stop early: streamed %d of %d", got, genTokens)
	}
	res, err := cr.Result()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled lane err = %v, want context.Canceled", err)
	}
	if res != nil {
		t.Fatalf("cancelled lane result = %+v, want nil", res)
	}
	ln, ok := cr.(*schedLane)
	if !ok {
		t.Fatalf("Admit returned %T, want *schedLane", cr)
	}
	if !ln.Reclaimed() {
		t.Fatal("cancelled lane did not signal KV reclaim")
	}
}
