package modelengine

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestAdmitStreamsThenAssembles proves the lifecycle path streams exactly genTokens
// ids one at a time and that Result() carries the SAME finished turn the one-shot
// Complete returns — the streamed ids equal the assembled result's tokens.
func TestAdmitStreamsThenAssembles(t *testing.T) {
	ctx := context.Background()
	e := New()
	req, err := e.Admit(ctx, inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	var streamed []int
	for tok := range req.Tokens() {
		streamed = append(streamed, tok.ID)
	}
	if len(streamed) != genTokens {
		t.Fatalf("streamed %d tokens, want %d", len(streamed), genTokens)
	}
	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res.Status != abi.StatusOK {
		t.Fatalf("status = %v, want OK", res.Status)
	}
	g := decodeGen(t, ctx, res)
	if !equalInts(g.Tokens, streamed) {
		t.Fatalf("streamed ids != assembled result tokens:\n stream=%v\n result=%v", streamed, g.Tokens)
	}
}

// TestCompleteRidesAdmit proves the migration: the one-shot Complete shim yields the
// SAME tokens as the streamed lifecycle path for the same call (Complete now rides
// Admit, byte-identically).
func TestCompleteRidesAdmit(t *testing.T) {
	ctx := context.Background()
	e := New()
	call := inlineCall("get_user_details", `{"id":7}`)

	req, err := e.Admit(ctx, call)
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	var streamed []int
	for tok := range req.Tokens() {
		streamed = append(streamed, tok.ID)
	}

	r2 := mustComplete(t, ctx, e, call)
	g := decodeGen(t, ctx, r2)
	if !equalInts(g.Tokens, streamed) {
		t.Fatalf("Complete tokens != Admit stream:\n complete=%v\n admit=%v", g.Tokens, streamed)
	}
}

// TestAdmitCancelStopsMidDecodeAndReclaims is the acceptance-#3 witness: cancelling
// ctx after a few streamed tokens stops decode BEFORE the fixed genTokens budget and
// signals KV reclaim — the per-step control point the buffered Complete never had.
func TestAdmitCancelStopsMidDecodeAndReclaims(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := New()
	req, err := e.Admit(ctx, inlineCall("calculate", `{"expr":"2+2"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}

	const readBeforeCancel = 3
	got := 0
	for range req.Tokens() {
		got++
		if got == readBeforeCancel {
			cancel()
			break // stop reading; the producer must observe cancel at its next step
		}
	}
	for range req.Tokens() { // drain any tokens already in flight so the goroutine exits
		got++
	}

	if got >= genTokens {
		t.Fatalf("cancel did not stop decode early: streamed %d of %d", got, genTokens)
	}
	res, err := req.Result()
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Result err = %v, want context.Canceled", err)
	}
	if res != nil {
		t.Fatalf("a cancelled request must yield a nil result, got %+v", res)
	}
	ln, ok := req.(*schedLane)
	if !ok {
		t.Fatalf("Admit returned %T, want *schedLane", req)
	}
	if !ln.Reclaimed() {
		t.Fatal("cancelled request did not signal KV reclaim")
	}
}

// TestEngineAdmitUsesNativeContinuousBatching proves the registered Engine's lifecycle path
// now rides NativeScheduler rather than the retired per-request goroutine: requests admitted
// before any consumer drains them are co-promoted into the scheduler's running set and decode
// through the shared StepBatch loop, while each lane's tokens remain serial-bit-identical.
func TestEngineAdmitUsesNativeContinuousBatching(t *testing.T) {
	ctx := context.Background()
	e := New()
	m := e.model()
	calls := []*abi.ToolCall{
		inlineCall("search_flights", `{"from":"SFO"}`),
		inlineCall("get_user_details", `{"id":1}`),
		inlineCall("list_all_airports", `{"region":"EU"}`),
	}

	want := make([][]int, len(calls))
	for i, c := range calls {
		prompt := e.buildPrompt(c.Tool, refBytes(ctx, c.Args), m.Cfg.VocabSize)
		want[i] = serialGreedyTokens(m, prompt)
	}

	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := e.Admit(ctx, c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	got := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for tok := range r.Tokens() {
				got[i] = append(got[i], tok.ID)
			}
		}(i, r)
	}
	wg.Wait()

	if peak := e.nativeScheduler().MaxObservedRunning(); peak < 2 {
		t.Fatalf("engine scheduler peak running = %d, want >=2 (requests did not co-batch)", peak)
	}
	for i := range calls {
		if !equalInts(got[i], want[i]) {
			t.Fatalf("call %d: scheduler tokens %v != serial tokens %v", i, got[i], want[i])
		}
		res, err := reqs[i].Result()
		if err != nil {
			t.Fatalf("Result %d: %v", i, err)
		}
		if res == nil || res.Status != abi.StatusOK {
			t.Fatalf("Result %d = %+v, want StatusOK", i, res)
		}
	}
}

func serialGreedyTokens(m *model.Model, prompt []int) []int {
	sess := m.NewSession()
	logits := sess.Prefill(prompt)
	gen := make([]int, 0, genTokens)
	for i := 0; i < genTokens; i++ {
		next := argmax(logits)
		gen = append(gen, next)
		if sess.M.Cfg.IsEOS(next) {
			break
		}
		logits = sess.Step(next)
	}
	return gen
}
