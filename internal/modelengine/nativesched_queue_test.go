package modelengine

import (
	"context"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestNativeSchedulerWaitingQueueGatesAndPreservesOutput is the issue-#36 acceptance-#2
// witness: the continuous-batching scheduler carries an explicit WAITING/RUNNING queue
// whose running set is recomputed each iteration. Driving the SAME admissions twice —
// uncapped (every lane co-batches) and capped at maxRunning=1 (lanes wait, then run
// strictly serially as slots free) — proves two acceptance bars at once:
//
//   - the waiting queue actually GATES the running set: the loop's own peak running-set
//     high-water mark is the cap (1 when capped, == #admitted when uncapped), so a
//     FINISHED slot is what lets the next waiting lane in — the admit/evict move; and
//   - a surviving sequence's output is BIT-IDENTICAL regardless of how many lanes ran
//     concurrently (acceptance #3): the StepBatch f32 guarantee carried through the
//     queue, so the queue changes WHEN a lane runs, never WHAT it decodes.
func TestNativeSchedulerWaitingQueueGatesAndPreservesOutput(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}

	// Uncapped: every admitted lane is promoted at once and co-batched in one step.
	refOut, refPeak := drainAllLanes(t, m, calls, 0)
	if refPeak != len(calls) {
		t.Fatalf("uncapped peak running = %d, want %d (all lanes co-batched)", refPeak, len(calls))
	}

	// Capped at 1: the waiting queue holds the other lanes; they run only as slots free.
	gatedOut, gatedPeak := drainAllLanes(t, m, calls, 1)
	if gatedPeak != 1 {
		t.Fatalf("maxRunning=1 peak running = %d, want 1 (waiting queue did not gate)", gatedPeak)
	}

	for i := range calls {
		if len(gatedOut[i]) == 0 {
			t.Fatalf("call %d produced no tokens", i)
		}
		if !reflect.DeepEqual(gatedOut[i], refOut[i]) {
			t.Fatalf("call %d: gated output %v != uncapped output %v (queue changed the decode)",
				i, gatedOut[i], refOut[i])
		}
	}
}

// drainAllLanes admits every call into a fresh scheduler capped at maxRunning, drains
// each lane's token stream concurrently, and returns the per-lane token ids plus the
// loop's peak running-set size. With an unbuffered per-lane channel, draining all lanes
// concurrently is what lets a capped run make progress: a running lane finishes only as
// its consumer receives, freeing a slot for the next waiting lane.
func drainAllLanes(t *testing.T, m *model.Model, calls []*abi.ToolCall, maxRunning int) ([][]int, int) {
	t.Helper()
	s := NewNativeScheduler(m)
	s.SetMaxRunning(maxRunning)
	defer s.Close()

	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := s.Admit(context.Background(), c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	out := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for tok := range r.Tokens() {
				out[i] = append(out[i], tok.ID)
			}
		}(i, r)
	}
	wg.Wait()
	return out, s.MaxObservedRunning()
}
