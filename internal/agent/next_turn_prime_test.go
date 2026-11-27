package agent

// next_turn_prime_test.go - fak-private#3728 [SW-VERIFIED] witnesses for the bounded
// next-turn KV prime: reuse, saturation, cancel, and the loop hook.

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// blockingPrimer parks every prime until release closes or its ctx ends.
type blockingPrimer struct {
	release chan struct{}
	mu      sync.Mutex
	calls   int
}

func (b *blockingPrimer) prime(ctx context.Context, _ []int) (bool, error) {
	b.mu.Lock()
	b.calls++
	b.mu.Unlock()
	select {
	case <-b.release:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func (b *blockingPrimer) callCount() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls
}

func fixedEncode(ids []int) nextTurnEncodeFunc {
	return func(context.Context, []Message, []ToolDef) ([]int, error) { return ids, nil }
}

// With a prime run, the turn-2 demand request reuses at least the primed prefix; without
// one it starts cold.
func TestNextTurnPrimeReusesRenderedPrefix(t *testing.T) {
	cfg := tinyCfg()
	prefix := synthIDs(cfg.VocabSize, 24, 3728)
	turn2 := append(append([]int(nil), prefix...), synthIDs(cfg.VocabSize, 6, 99)...)

	cold := reusePlanner(true, false, cfg)
	if _, matched := decode(cold, turn2, 2); matched != 0 {
		t.Fatalf("cold baseline reused %d tokens, want 0", matched)
	}

	p := reusePlanner(true, false, cfg)
	r := newNextTurnPrimeRunner(func(ctx context.Context, ids []int) (bool, error) {
		rec, err := p.PrimeCacheState(ctx, ids)
		return rec.Usable(), err
	}, p, fixedEncode(prefix), NextTurnPrimeConfig{})
	defer r.Close()
	if !r.Trigger(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil) {
		t.Fatal("Trigger refused an idle runner")
	}
	r.wait()
	if st := r.Stats(); st.Primed != 1 {
		t.Fatalf("stats = %+v, want one primed", st)
	}
	if _, matched := decode(p, turn2, 2); matched < len(prefix) {
		t.Fatalf("turn-2 reused %d tokens, want >= %d", matched, len(prefix))
	}
	// Already covered: a second trigger is skipped without priming.
	r.Trigger(context.Background(), nil, nil)
	r.wait()
	if st := r.Stats(); st.SkippedCovered != 1 || st.Primed != 1 {
		t.Fatalf("stats = %+v, want one covered skip", st)
	}
}

// A saturated semaphore never queues: extra triggers are skipped and counted.
func TestNextTurnPrimeSaturationSkipsWithoutQueue(t *testing.T) {
	bp := &blockingPrimer{release: make(chan struct{})}
	r := newNextTurnPrimeRunner(bp.prime, nil, fixedEncode([]int{1, 2, 3}), NextTurnPrimeConfig{MaxConcurrent: 2})
	defer r.Close()
	started := 0
	for i := 0; i < 5; i++ {
		if r.Trigger(context.Background(), nil, nil) {
			started++
		}
	}
	if started != 2 {
		t.Fatalf("started %d primes, want 2 (semaphore width)", started)
	}
	if st := r.Stats(); st.SkippedSaturated != 3 {
		t.Fatalf("stats = %+v, want 3 saturated skips", st)
	}
	close(bp.release)
	r.wait()
	if got := bp.callCount(); got != 2 {
		t.Fatalf("primer called %d times, want 2 (no queued work)", got)
	}
}

// A cancelled prime returns ctx.Err with Admitted=false; Close cancels in-flight work.
func TestNextTurnPrimeCancelReturnsCtxErr(t *testing.T) {
	cfg := tinyCfg()
	p := reusePlanner(true, false, cfg)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rec, err := p.PrimeCacheState(ctx, synthIDs(cfg.VocabSize, 8, 1))
	if !errors.Is(err, context.Canceled) || rec.Admitted {
		t.Fatalf("receipt=%+v err=%v, want Canceled and Admitted=false", rec, err)
	}

	bp := &blockingPrimer{release: make(chan struct{})}
	got := make(chan error, 1)
	r := newNextTurnPrimeRunner(bp.prime, nil, fixedEncode([]int{1}), NextTurnPrimeConfig{})
	r.onResult = func(usable bool, e error) {
		if usable {
			e = errors.New("admitted despite cancel")
		}
		got <- e
	}
	r.Trigger(context.Background(), nil, nil)
	r.Close()
	select {
	case e := <-got:
		if !errors.Is(e, context.Canceled) {
			t.Fatalf("in-flight prime err = %v, want context.Canceled", e)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not cancel the in-flight prime")
	}
}

// The loop hook fires on a tool-call turn, and is inert when the option is absent.
func TestNextTurnPrimeLoopHookFiresOnToolCalls(t *testing.T) {
	bp := &blockingPrimer{release: make(chan struct{})}
	close(bp.release)
	r := newNextTurnPrimeRunner(bp.prime, nil, fixedEncode([]int{1, 2}), NextTurnPrimeConfig{})
	defer r.Close()
	script := func() *loopTurnScriptedPlanner {
		return &loopTurnScriptedPlanner{replies: []*Completion{{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Type: "function",
				Function: Func{Name: "Read", Arguments: `{}`}}}},
			FinishReason: "tool_calls",
		}}}
	}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Read"}}}
	if _, err := RunArm(t.Context(), script(), "read it", false, 3, nil, WithToolCatalog(tools)); err != nil {
		t.Fatal(err)
	}
	if st := r.Stats(); st.Started != 0 {
		t.Fatalf("prime started without the option: %+v", st)
	}
	if _, err := RunArm(t.Context(), script(), "read it", false, 3, nil, WithToolCatalog(tools), WithNextTurnPrime(r)); err != nil {
		t.Fatal(err)
	}
	r.wait()
	if st := r.Stats(); st.Started != 1 {
		t.Fatalf("stats = %+v, want exactly one prime at tool dispatch", st)
	}
}
