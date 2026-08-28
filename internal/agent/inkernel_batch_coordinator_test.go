package agent

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInKernelPlannerCoalescesConcurrentQwenTurns(t *testing.T) {
	restoreProbe := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		out := make([][]float32, len(ids))
		lanes := 0
		for i, on := range active {
			if !on {
				continue
			}
			out[i] = bs.Seqs[i].Step(ids[i])
			lanes++
		}
		if lanes < 2 {
			return out, 0, 0, true
		}
		return out, 1, int64(lanes), true
	})
	defer restoreProbe()
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	cfg.LayerTypes = []string{"linear_attention"}
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = 8
	cfg.LinearNumKeyHeads = 2
	cfg.LinearValueHeadDim = 8
	cfg.LinearNumValueHeads = 4
	newPlanner := func(enabled bool) *InKernelPlanner {
		m := model.NewSynthetic(cfg)
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-qwen-coalesce", true, nil, false)
		p.maxNew = 6
		p.batchDecode = enabled
		// Synthetic tests install a shared-panel hook; production additionally gates on Metal.
		p.metal = enabled
		return p
	}
	messages := [][]Message{
		{{Role: RoleUser, Content: "alpha"}},
		{{Role: RoleUser, Content: "beta"}},
	}
	serial := newPlanner(false)
	refs := make([]*Completion, len(messages))
	for i := range messages {
		var err error
		refs[i], err = serial.Complete(context.Background(), messages[i], nil)
		if err != nil {
			t.Fatalf("serial[%d]: %v", i, err)
		}
	}

	p := newPlanner(true)
	ready := make(chan struct{})
	var readyOnce sync.Once
	p.coalesceReadyHook = func() { <-ready }
	var mu sync.Mutex
	var batches []int
	p.coalesceBatchHook = func(n int) { mu.Lock(); batches = append(batches, n); mu.Unlock() }
	var sharedPanels int
	var sharedMACs int64
	p.coalesceSharedHook = func(panels int, macs int64) {
		mu.Lock()
		sharedPanels += panels
		sharedMACs += macs
		mu.Unlock()
	}
	type answer struct {
		c   *Completion
		err error
	}
	answers := make([]answer, len(messages))
	var wg sync.WaitGroup
	for i := range messages {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			answers[i].c, answers[i].err = p.Complete(context.Background(), messages[i], nil)
		}(i)
	}
	for {
		p.coalesceMu.Lock()
		n := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if n == 2 {
			break
		}
	}
	readyOnce.Do(func() { close(ready) })
	wg.Wait()
	var cohortID uint64
	for i := range answers {
		if answers[i].c == nil || answers[i].c.InKernelBatch == nil {
			t.Fatalf("coalesced[%d] missing batch receipt", i)
		}
		r := answers[i].c.InKernelBatch
		if r.CohortSize != len(messages) || r.SharedPanels == 0 || r.SharedMACs == 0 || r.SessionCloses != uint32(len(messages)) {
			t.Fatalf("coalesced[%d] receipt=%+v", i, r)
		}
		if cohortID == 0 {
			cohortID = r.CohortID
		} else if r.CohortID != cohortID {
			t.Fatalf("coalesced receipts identify different cohorts: %d != %d", r.CohortID, cohortID)
		}
	}
	for i := range answers {
		if answers[i].err != nil {
			t.Fatalf("coalesced[%d]: %v", i, answers[i].err)
		}
		if answers[i].c.Message.Content != refs[i].Message.Content || answers[i].c.FinishReason != refs[i].FinishReason {
			t.Fatalf("coalesced[%d]=%q/%s, serial=%q/%s", i, answers[i].c.Message.Content, answers[i].c.FinishReason, refs[i].Message.Content, refs[i].FinishReason)
		}
	}
	mu.Lock()
	observed := append([]int(nil), batches...)
	mu.Unlock()
	if len(observed) == 0 || observed[0] < 2 {
		t.Fatalf("observed batches %v, want a forward with B>=2", observed)
	}
	if sharedPanels == 0 || sharedMACs == 0 {
		t.Fatalf("B>=2 was admitted but no shared model work executed: panels=%d macs=%d", sharedPanels, sharedMACs)
	}

	// Cancellation is request-scoped: a canceled ready lane retires while its peer
	// retains the serial output and finish reason.
	p = newPlanner(true)
	ready = make(chan struct{})
	p.coalesceReadyHook = func() { <-ready }
	ctx, cancel := context.WithCancel(context.Background())
	answers = make([]answer, 2)
	wg = sync.WaitGroup{}
	wg.Add(2)
	go func() { defer wg.Done(); answers[0].c, answers[0].err = p.Complete(ctx, messages[0], nil) }()
	go func() {
		defer wg.Done()
		answers[1].c, answers[1].err = p.Complete(context.Background(), messages[1], nil)
	}()
	for {
		p.coalesceMu.Lock()
		n := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if n == 2 {
			break
		}
	}
	cancel()
	close(ready)
	wg.Wait()
	if !errors.Is(answers[0].err, context.Canceled) {
		t.Fatalf("canceled lane err=%v", answers[0].err)
	}
	if answers[1].err != nil || answers[1].c.Message.Content != refs[1].Message.Content || answers[1].c.FinishReason != refs[1].FinishReason {
		t.Fatalf("peer changed: %#v err=%v", answers[1].c, answers[1].err)
	}

	// Singleton uses the same coordinator at B=1; default-off bypasses it.
	p = newPlanner(true)
	p.coalesceReadyHook = func() {}
	batches = nil
	p.coalesceBatchHook = func(n int) { batches = append(batches, n) }
	one, err := p.Complete(context.Background(), messages[0], nil)
	if err != nil || one.Message.Content != refs[0].Message.Content || len(batches) != 1 || batches[0] != 1 {
		t.Fatalf("singleton=%#v err=%v batches=%v", one, err, batches)
	}
	off := newPlanner(false)
	off.coalesceBatchHook = func(int) { t.Fatal("default-off entered coordinator") }
	got, err := off.Complete(context.Background(), messages[0], nil)
	if err != nil || got.Message.Content != refs[0].Message.Content {
		t.Fatalf("default-off=%#v err=%v", got, err)
	}
}
