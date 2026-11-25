package main

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fakeInkling2Clock advances only when told; every armed timer is reported on armed
// so a test knows collect is blocked before it moves time or sends.
type fakeInkling2Clock struct {
	mu     sync.Mutex
	now    time.Time
	timers map[*fakeInkling2Timer]struct{}
	armed  chan time.Duration
}

type fakeInkling2Timer struct {
	at time.Time
	ch chan time.Time
}

func newFakeInkling2Clock() *fakeInkling2Clock {
	return &fakeInkling2Clock{now: time.Unix(1000, 0), timers: map[*fakeInkling2Timer]struct{}{}, armed: make(chan time.Duration, 64)}
}

func (c *fakeInkling2Clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeInkling2Clock) After(d time.Duration) (<-chan time.Time, func()) {
	c.mu.Lock()
	t := &fakeInkling2Timer{at: c.now.Add(d), ch: make(chan time.Time, 1)}
	c.timers[t] = struct{}{}
	c.mu.Unlock()
	c.armed <- d
	return t.ch, func() {
		c.mu.Lock()
		delete(c.timers, t)
		c.mu.Unlock()
	}
}

func (c *fakeInkling2Clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	for t := range c.timers {
		if !t.at.After(c.now) {
			t.ch <- c.now
			delete(c.timers, t)
		}
	}
}

func (c *fakeInkling2Clock) waitArmed(t *testing.T) time.Duration {
	t.Helper()
	select {
	case d := <-c.armed:
		return d
	case <-time.After(10 * time.Second):
		t.Fatal("collect never armed a timer")
		return 0
	}
}

func testInkling2Admission(clk inkling2Clock) inkling2Admission {
	return inkling2Admission{gap: inkling2BatchGapDefault, maxWait: inkling2BatchMaxWaitDefault, max: ggufload.Inkling2LoopMaxBatch, clock: clk}
}

func collectAsync(a inkling2Admission, first *inkling2QueuedRequest, queue chan *inkling2QueuedRequest) <-chan []*inkling2QueuedRequest {
	out := make(chan []*inkling2QueuedRequest, 1)
	go func() { out <- a.collect(first, queue) }()
	return out
}

func TestInkling2BatchAdmissionCoalescesSpreadArrivalsIntoOneCohort(t *testing.T) {
	clk := newFakeInkling2Clock()
	queue := make(chan *inkling2QueuedRequest, 16)
	got := collectAsync(testInkling2Admission(clk), &inkling2QueuedRequest{}, queue)
	// 15 more arrivals 650us apart: ~10ms of spread, every gap under the 4ms window
	for i := 1; i < 16; i++ {
		clk.waitArmed(t)
		clk.advance(650 * time.Microsecond)
		queue <- &inkling2QueuedRequest{}
	}
	if group := <-got; len(group) != 16 {
		t.Fatalf("cohort width %d, want 16", len(group))
	}
}

func TestInkling2BatchAdmissionGathersAlreadyQueuedWithoutWaiting(t *testing.T) {
	clk := newFakeInkling2Clock()
	queue := make(chan *inkling2QueuedRequest, 16)
	for i := 0; i < 15; i++ {
		queue <- &inkling2QueuedRequest{}
	}
	if group := testInkling2Admission(clk).collect(&inkling2QueuedRequest{}, queue); len(group) != 16 {
		t.Fatalf("cohort width %d, want 16", len(group))
	}
	if len(clk.armed) != 0 {
		t.Fatal("a full queue must not arm a timer")
	}
}

func TestInkling2BatchAdmissionMaxWaitBoundsTheCohort(t *testing.T) {
	clk := newFakeInkling2Clock()
	queue := make(chan *inkling2QueuedRequest, 16)
	start := clk.Now()
	got := collectAsync(testInkling2Admission(clk), &inkling2QueuedRequest{}, queue)
	var group []*inkling2QueuedRequest
	for group == nil {
		select {
		case d := <-clk.armed:
			if d >= 3*time.Millisecond {
				clk.advance(3 * time.Millisecond)
				queue <- &inkling2QueuedRequest{}
			} else {
				clk.advance(d)
			}
		case group = <-got:
		case <-time.After(10 * time.Second):
			t.Fatal("collect hung")
		}
	}
	// arrivals every 3ms never let the 4ms gap fire; the 25ms wait does
	if elapsed := clk.Now().Sub(start); elapsed != inkling2BatchMaxWaitDefault {
		t.Fatalf("cohort closed after %s, want %s", elapsed, inkling2BatchMaxWaitDefault)
	}
	if len(group) != 9 {
		t.Fatalf("cohort width %d, want 9 (first + 8 arrivals in 25ms)", len(group))
	}
}

func TestInkling2BatchAdmissionLoneRequestRunsScalarAfterGap(t *testing.T) {
	clk := newFakeInkling2Clock()
	var scalarCalls int
	e := newInkling2LoopBatchEngineWith(
		func(_ context.Context, _ []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
			scalarCalls++
			emit(7)
			return agent.Inkling2LoopGeneration{Generated: 1}, nil
		},
		func([]ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			t.Fatal("a lone request must not run the batch")
			return nil, nil
		},
		testInkling2Admission(clk), 4096)
	go e.run()
	defer close(e.queue)
	done := make(chan agent.Inkling2LoopGeneration, 1)
	go func() {
		gen, err := e.Generate(context.Background(), []int{1}, 4, nil, func(int) bool { return true })
		if err != nil {
			t.Error(err)
		}
		done <- gen
	}()
	if d := clk.waitArmed(t); d != inkling2BatchGapDefault {
		t.Fatalf("lone request armed %s, want the %s gap", d, inkling2BatchGapDefault)
	}
	clk.advance(inkling2BatchGapDefault)
	if gen := <-done; gen.Generated != 1 || scalarCalls != 1 || e.scalarGroups.Load() != 1 || e.batchGroups.Load() != 0 {
		t.Fatalf("gen=%+v scalar=%d groups scalar=%d batch=%d", gen, scalarCalls, e.scalarGroups.Load(), e.batchGroups.Load())
	}
}

// Real time end to end: concurrent Generate callers land in one cohort and each
// receives only its own row's tokens.
func TestInkling2BatchEngineConcurrentCallersShareOneCohort(t *testing.T) {
	var mu sync.Mutex
	var widths []int
	adm := inkling2Admission{gap: inkling2BatchDurationMax, maxWait: inkling2BatchDurationMax, max: ggufload.Inkling2LoopMaxBatch, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(
		func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
			t.Error("concurrent callers fell back to scalar")
			return agent.Inkling2LoopGeneration{}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			mu.Lock()
			widths = append(widths, len(reqs))
			mu.Unlock()
			out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
			for i, r := range reqs {
				if !r.Emit(r.Prompt[0] * 10) {
					out[i].Stopped = true
					continue
				}
				out[i].Generated = 1
			}
			return out, nil
		},
		adm, 4096)
	go e.run()
	defer close(e.queue)
	var wg sync.WaitGroup
	for i := 1; i <= 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			time.Sleep(time.Duration(i) * 600 * time.Microsecond)
			var got []int
			gen, err := e.Generate(context.Background(), []int{i}, 1, nil, func(id int) bool { got = append(got, id); return true })
			if err != nil || gen.Generated != 1 || len(got) != 1 || got[0] != i*10 {
				t.Errorf("caller %d: gen=%+v err=%v got=%v", i, gen, err, got)
			}
		}(i)
	}
	wg.Wait()
	if len(widths) != 1 || widths[0] != 16 || e.peak.Load() != 16 {
		t.Fatalf("cohort widths %v peak %d, want one cohort of 16", widths, e.peak.Load())
	}
}

// A panicking cohort fails every row and leaves the worker alive for the next one.
func TestInkling2BatchWorkerPanicFailsCohortAndSurvives(t *testing.T) {
	adm := inkling2Admission{gap: inkling2BatchDurationMax, maxWait: inkling2BatchDurationMax, max: 2, clock: inkling2RealClock{}}
	calls := 0
	e := newInkling2LoopBatchEngineWith(
		func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
			t.Error("pairs must not run scalar")
			return agent.Inkling2LoopGeneration{}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			calls++
			if calls == 1 {
				panic("device fault")
			}
			return make([]ggufload.Inkling2LoopBatchResult, len(reqs)), nil
		},
		adm, 4096)
	go e.run()
	defer close(e.queue)
	for round, wantErr := range []bool{true, false} {
		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := e.Generate(context.Background(), []int{1}, 1, nil, func(int) bool { return true })
				if (err != nil) != wantErr {
					t.Errorf("round %d: err=%v, want error=%v", round, err, wantErr)
				}
			}()
		}
		wg.Wait()
	}
}

func TestInkling2BatchRejectsRequestPastContextBeforeAdmission(t *testing.T) {
	e := newInkling2LoopBatchEngineWith(nil, nil, testInkling2Admission(inkling2RealClock{}), 64)
	if _, err := e.Generate(context.Background(), make([]int, 64), 2, nil, func(int) bool { return true }); err == nil || len(e.queue) != 0 {
		t.Fatalf("past-context request err=%v queued=%d", err, len(e.queue))
	}
}

func TestInkling2BatchAdmissionEnv(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	a, err := inkling2AdmissionFromEnv(env(nil))
	if err != nil || a.gap != 4*time.Millisecond || a.maxWait != 25*time.Millisecond || a.max != 16 {
		t.Fatalf("defaults = %+v err=%v", a, err)
	}
	a, err = inkling2AdmissionFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_GAP": "8ms", "FAK_INKLING2_BATCH_WAIT": "100ms"}))
	if err != nil || a.gap != 8*time.Millisecond || a.maxWait != 100*time.Millisecond {
		t.Fatalf("override = %+v err=%v", a, err)
	}
	for _, bad := range []string{"0", "-1ms", "101ms", "soon"} {
		if _, err := inkling2AdmissionFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_WAIT": bad})); err == nil {
			t.Fatalf("FAK_INKLING2_BATCH_WAIT=%q accepted", bad)
		}
		if _, err := inkling2AdmissionFromEnv(env(map[string]string{"FAK_INKLING2_BATCH_GAP": bad})); err == nil {
			t.Fatalf("FAK_INKLING2_BATCH_GAP=%q accepted", bad)
		}
	}
}

func TestInkling2ServeBatchGateDefaultsOff(t *testing.T) {
	// the process env must not leak in: every knob arrives through lookup
	t.Setenv("FAK_INKLING2_SERVE_BATCH", "1")
	gen, _, err := newServeInkling2Generator(nil, nil, func(string) (string, bool) { return "", false })
	if _, ok := gen.(*inkling2LoopEngine); err != nil || !ok {
		t.Fatalf("default gen=%T err=%v, want the serialized engine", gen, err)
	}
	on := func(k string) (string, bool) { return "1", k == "FAK_INKLING2_SERVE_BATCH" }
	gen, msg, err := newServeInkling2Generator(nil, nil, on)
	if _, ok := gen.(*inkling2LoopEngine); err != nil || !ok {
		t.Fatalf("ineligible load gen=%T err=%v msg=%q, want the serialized engine", gen, err, msg)
	}
	bad := func(k string) (string, bool) {
		switch k {
		case "FAK_INKLING2_SERVE_BATCH":
			return "1", true
		case "FAK_INKLING2_BATCH_GAP":
			return "bad", true
		}
		return "", false
	}
	if _, _, err := newServeInkling2Generator(nil, nil, bad); err == nil {
		t.Fatal("an invalid admission env must refuse the load")
	}
}

// pairEngine runs every pair of callers as one cohort through batch.
func pairEngine(t *testing.T, batch inkling2BatchGenerate) *inkling2LoopBatchEngine {
	adm := inkling2Admission{gap: inkling2BatchDurationMax, maxWait: inkling2BatchDurationMax, max: 2, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
		t.Error("pairs must not run scalar")
		return agent.Inkling2LoopGeneration{}, nil
	}, batch, adm, 4096)
	go e.run()
	t.Cleanup(func() { close(e.queue) })
	return e
}

// A cancelled caller returns while its cohort is still running, and its emit is
// never called after it returned even though the cohort tries to.
func TestInkling2BatchCancelledCallerReturnsBeforeCohortDrains(t *testing.T) {
	release := make(chan struct{})
	cohortDone := make(chan struct{})
	e := pairEngine(t, func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		defer close(cohortDone)
		<-release
		out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
		for i, r := range reqs {
			if r.Emit(1) {
				out[i].Generated = 1
			}
		}
		return out, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	var mu sync.Mutex
	returned, lateEmit := false, false
	cancelled := make(chan error, 1)
	go func() {
		_, err := e.Generate(ctx, []int{1}, 1, nil, func(int) bool {
			mu.Lock()
			defer mu.Unlock()
			lateEmit = returned
			return true
		})
		mu.Lock()
		returned = true
		mu.Unlock()
		cancelled <- err
	}()
	other := make(chan agent.Inkling2LoopGeneration, 1)
	go func() {
		gen, _ := e.Generate(context.Background(), []int{2}, 1, nil, func(int) bool { return true })
		other <- gen
	}()
	for e.batchGroups.Load() == 0 {
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case err := <-cancelled:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled caller err=%v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("cancelled caller waited for the cohort to drain")
	}
	close(release)
	<-cohortDone
	if gen := <-other; gen.Generated != 1 {
		t.Fatalf("surviving row gen=%+v", gen)
	}
	mu.Lock()
	defer mu.Unlock()
	if lateEmit {
		t.Fatal("emit ran after its caller returned")
	}
}

// A row the cohort reports done answers its caller at once, not when the cohort ends.
func TestInkling2BatchRowDoneAnswersBeforeCohortEnds(t *testing.T) {
	release := make(chan struct{})
	e := pairEngine(t, func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		early := ggufload.Inkling2LoopBatchResult{Generated: 3, Stopped: true}
		reqs[0].Done(early)
		<-release
		return []ggufload.Inkling2LoopBatchResult{early, {Generated: 5}}, nil
	})
	gens := make(chan agent.Inkling2LoopGeneration, 2)
	for i := 0; i < 2; i++ {
		go func() {
			gen, err := e.Generate(context.Background(), []int{1}, 8, nil, func(int) bool { return true })
			if err != nil {
				t.Error(err)
			}
			gens <- gen
		}()
	}
	select {
	case gen := <-gens:
		if gen.Generated != 3 || !gen.Stopped {
			t.Fatalf("early row gen=%+v", gen)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("a finished row waited for the cohort")
	}
	close(release)
	if gen := <-gens; gen.Generated != 5 {
		t.Fatalf("late row gen=%+v", gen)
	}
}

// The response timings name the width each turn actually decoded in: a pair that
// shared a cohort reports 2 and batch, a lone turn 1 and scalar.
func TestInkling2BatchTimingsCarryCohortWidth(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	canned, err := tok.Encode(`<|message_model|><|content_text|>ok<|end_message|><|content_model_end_sampling|>`)
	if err != nil {
		t.Fatal(err)
	}
	scalar, _ := inkling2FakeEngine(canned)
	adm := inkling2Admission{gap: inkling2BatchDurationMax, maxWait: inkling2BatchDurationMax, max: 2, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(scalar.Generate, func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
		for i, r := range reqs {
			turn := ggufload.Inkling2LoopTurn{MaxTokens: r.MaxTokens, Stop: r.Stop, Emit: r.Emit}
			for _, id := range canned {
				if !turn.Accept(id) {
					break
				}
			}
			out[i].Generated, out[i].Stopped = turn.Generated, turn.Stopped
		}
		return out, nil
	}, adm, 4096)
	go e.run()
	defer close(e.queue)
	p, err := newServeInkling2LoopPlanner(e, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	timings := func() map[string]any {
		comp, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "hi"}}, nil)
		if err != nil || comp.Timings == nil {
			t.Errorf("complete: comp=%+v err=%v", comp, err)
			return nil
		}
		raw, _ := json.Marshal(comp.Timings)
		var m map[string]any
		_ = json.Unmarshal(raw, &m)
		return m
	}
	var wg sync.WaitGroup
	pair := make([]map[string]any, 2)
	for i := range pair {
		wg.Add(1)
		go func() { defer wg.Done(); pair[i] = timings() }()
	}
	wg.Wait()
	for i, m := range pair {
		if m["inkling2_width"] != float64(2) || m["inkling2_mode"] != "batch" {
			t.Fatalf("cohort turn %d timings %v, want width 2 batch", i, m)
		}
	}
	if m := timings(); m["inkling2_width"] != float64(1) || m["inkling2_mode"] != "scalar" {
		t.Fatalf("lone turn timings %v, want width 1 scalar", m)
	}
}

// A lone request after a cohort frees the cohort arenas before its scalar session
// allocates, while back-to-back cohorts keep reusing them.
func TestInkling2BatchLoneRequestAfterCohortReleasesArenasFirst(t *testing.T) {
	var events []string
	e := newInkling2LoopBatchEngineWith(
		func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
			events = append(events, "scalar")
			return agent.Inkling2LoopGeneration{}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			events = append(events, "batch")
			return make([]ggufload.Inkling2LoopBatchResult, len(reqs)), nil
		},
		testInkling2Admission(inkling2RealClock{}), 4096)
	e.releaseArenas = func() { events = append(events, "release") }
	group := func(width int) []*inkling2QueuedRequest {
		g := make([]*inkling2QueuedRequest, width)
		for i := range g {
			g[i] = &inkling2QueuedRequest{ctx: context.Background(), prompt: []int{1}, maxNew: 1,
				sink: &inkling2RowSink{emit: func(int) bool { return true }}, done: make(chan inkling2QueuedResult, 1)}
		}
		return g
	}
	for _, width := range []int{1, 2, 2, 1, 1, 2, 1} {
		e.execute(group(width))
	}
	want := []string{"scalar", "batch", "batch", "release", "scalar", "scalar", "batch", "release", "scalar"}
	if len(events) != len(want) {
		t.Fatalf("events %v, want %v", events, want)
	}
	for i := range want {
		if events[i] != want[i] {
			t.Fatalf("events %v, want %v", events, want)
		}
	}
}
