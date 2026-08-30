package modelengine

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
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

	receipt := s.SharedWorkReceipt()
	if receipt.Steps == 0 || receipt.Panels == 0 || receipt.MACs == 0 {
		t.Fatalf("scheduler admitted a batch but did not receipt shared model work: %+v", receipt)
	}
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

func TestNativeSchedulerReportsDeterministicCachePhaseLatency(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	defer s.Close()

	var clockMu sync.Mutex
	now := time.Unix(0, 0)
	s.now = func() time.Time {
		clockMu.Lock()
		defer clockMu.Unlock()
		current := now
		now = now.Add(2 * time.Millisecond)
		return current
	}

	req, err := s.Admit(context.Background(), inlineCall("search_flights", `{"from":"SFO"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	for range req.Tokens() {
	}
	if _, err := req.Result(); err != nil {
		t.Fatalf("Result: %v", err)
	}

	receipt := s.CachePhaseLatencyReceipt()
	if got, want := len(receipt.Phases), 3; got != want {
		t.Fatalf("phase cardinality = %d, want %d", got, want)
	}
	if got := receipt.Phases[0]; got.Phase != modelperfobs.CachePipelinePhasePrefill || got.Observations != 1 || got.Total != 2*time.Millisecond {
		t.Fatalf("prefill bucket = %+v, want one 2ms observation", got)
	}
	decode := receipt.Phases[1]
	if decode.Phase != modelperfobs.CachePipelinePhaseDecode || decode.Observations == 0 {
		t.Fatalf("decode bucket = %+v, want bounded non-empty decode observations", decode)
	}
	if decode.Total != time.Duration(decode.Observations)*2*time.Millisecond {
		t.Fatalf("decode total = %s, want %d deterministic 2ms observations", decode.Total, decode.Observations)
	}
	if receipt.Observations != receipt.Phases[0].Observations+decode.Observations || receipt.Total != receipt.Phases[0].Total+decode.Total {
		t.Fatalf("unlabeled receipt does not reconcile with known phases: %+v", receipt)
	}
}
