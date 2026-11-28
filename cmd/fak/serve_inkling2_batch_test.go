package main

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type testInkling2CUDA struct {
	compute.Backend
	compute.Inkling2BatchDevice
	compute.Inkling2PrefillHalfKV
	w4a4    bool
	halfErr error
}

func (c testInkling2CUDA) Ink2PHalfKVReady() error       { return c.halfErr }
func (c testInkling2CUDA) Ink2PNVFP4W4A4Available() bool { return c.w4a4 }

// fak-test:runtime fast est=1ms
func TestInkling2CUDAPrefillDefaults(t *testing.T) {
	none := inkling2EnvLookup(nil)
	deviceW4 := testInkling2CUDA{w4a4: true}
	deviceNoW4 := testInkling2CUDA{w4a4: false}
	if !inkling2CUDAPrefillReady(deviceW4, none) ||
		!inkling2CUDAPrefillReady(deviceNoW4, none) ||
		inkling2CUDAPrefillReady(nil, none) ||
		inkling2CUDAPrefillReady(testInkling2CUDA{w4a4: true, halfErr: errors.New("half unavailable")}, none) ||
		inkling2CUDAPrefillReady(testInkling2CUDA{w4a4: false, halfErr: errors.New("half unavailable")}, none) {
		t.Fatal("CUDA cohort defaults must require working half KV and activate regardless of W4A4 availability")
	}
	if !inkling2CUDAPrefillW4A4Active(deviceW4, none) {
		t.Fatal("W4A4 label must report active when W4A4 is available and not opted out")
	}
	if inkling2CUDAPrefillW4A4Active(deviceNoW4, none) {
		t.Fatal("W4A4 label must report inactive when W4A4 is unavailable")
	}
	// Disabling W4A4 prefill keeps cohort concurrency enabled and only clears the W4A4 label.
	noW4Env := inkling2EnvLookup(map[string]string{"FAK_INKLING2_PREFILL_NVFP4_W4A4": "0"})
	if !inkling2CUDAPrefillReady(deviceW4, noW4Env) {
		t.Fatal("FAK_INKLING2_PREFILL_NVFP4_W4A4=0 must not disable cohort concurrency")
	}
	if inkling2CUDAPrefillW4A4Active(deviceW4, noW4Env) {
		t.Fatal("FAK_INKLING2_PREFILL_NVFP4_W4A4=0 must report W4A4 inactive")
	}
	for _, tc := range []struct{ key, value string }{
		{"FAK_INKLING2_SERVE_BATCH", "0"},
		{"FAK_INKLING2_SERVE_BATCH", "false"},
		{"FAK_INKLING2_SERVE_BATCH", "off"},
		{"FAK_INKLING2_BATCHED_PREFILL", "0"},
		{"FAK_INKLING2_COHORT_PREFILL", "0"},
		{"FAK_INKLING2_BATCH_SEQOPS", "0"},
		{"FAK_INKLING2_COHORT_PREFILL_KV", "off"},
	} {
		if inkling2CUDAPrefillReady(deviceW4, inkling2EnvLookup(map[string]string{tc.key: tc.value})) {
			t.Fatalf("overrode %s=%s", tc.key, tc.value)
		}
		if inkling2CUDAPrefillReady(deviceNoW4, inkling2EnvLookup(map[string]string{tc.key: tc.value})) {
			t.Fatalf("overrode %s=%s on non-W4A4 device", tc.key, tc.value)
		}
	}
	lookup := inkling2CUDAPrefillDefaults(none)
	admission, err := inkling2AdmissionFromEnv(lookup)
	if err != nil || !inkling2ServeBatchEnabled(lookup) || admission.gap != 25*time.Millisecond ||
		admission.maxWait != 75*time.Millisecond {
		t.Fatalf("unexpected CUDA defaults: %+v, %v", admission, err)
	}
	for _, key := range []string{"FAK_INKLING2_BATCH_KV", "FAK_INKLING2_COHORT_PREFILL_KV"} {
		if value, _ := lookup(key); value != "f16" {
			t.Fatalf("%s=%s, want f16", key, value)
		}
	}
	for _, want := range []struct{ key, val string }{
		{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB", "4096"},
		{"FAK_INKLING2_PREFIX_SLOTS", "4"},
		{"FAK_INKLING2_HOL_BACKFILL_MAX", "4"},
	} {
		if value, _ := lookup(want.key); value != want.val {
			t.Fatalf("%s=%s, want %s", want.key, value, want.val)
		}
	}
	optOut := inkling2CUDAPrefillDefaults(inkling2EnvLookup(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "0"}))
	if value, _ := optOut("FAK_INKLING2_BATCH_PREFIX_CACHE_MIB"); value != "0" {
		t.Fatalf("explicit opt-out FAK_INKLING2_BATCH_PREFIX_CACHE_MIB=%s, want 0", value)
	}
	if _, set := lookup("FAK_INKLING2_CTX"); set {
		t.Fatal("CUDA defaults set a context cap")
	}
	lookup = inkling2CUDAPrefillDefaults(inkling2EnvLookup(map[string]string{
		"FAK_INKLING2_BATCH_GAP": "8ms", "FAK_INKLING2_BATCH_WAIT": "40ms",
		"FAK_INKLING2_BATCH_KV": "f32", "FAK_INKLING2_CTX": "16384",
	}))
	admission, err = inkling2AdmissionFromEnv(lookup)
	kv, _ := lookup("FAK_INKLING2_BATCH_KV")
	ctx, _ := lookup("FAK_INKLING2_CTX")
	if err != nil || admission.gap != 8*time.Millisecond || admission.maxWait != 40*time.Millisecond ||
		kv != "f32" || ctx != "16384" {
		t.Fatalf("explicit CUDA settings changed: %+v, KV %s, context %s, %v", admission, kv, ctx, err)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2ConfigSanityWarnings(t *testing.T) {
	if w := inkling2ConfigSanityWarnings(inkling2EnvLookup(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "4096"}), true, inkling2PrefixReuseOn); len(w) != 0 {
		t.Fatalf("expected no warnings with defaults on, got %v", w)
	}
	wOff := inkling2ConfigSanityWarnings(inkling2EnvLookup(map[string]string{"FAK_INKLING2_BATCH_PREFIX_CACHE_MIB": "0"}), true, inkling2PrefixReuseOff)
	if len(wOff) != 2 {
		t.Fatalf("expected 2 warnings when both batch prefix cache and prefix reuse are off, got %v", wOff)
	}
}

// fak-test:runtime fast est=1ms
func TestInkling2PrefixHitWatch(t *testing.T) {
	var watch inkling2PrefixHitWatch
	if msg := watch.observe(1000, 0, true); msg != "" {
		t.Fatalf("small prompt should not trigger: %q", msg)
	}
	if msg := watch.observe(5000, 0, true); msg != "" {
		t.Fatalf("miss 1 should not alert: %q", msg)
	}
	if msg := watch.observe(5000, 0, true); !strings.Contains(msg, "RED FLAG: kv_prefix_zero_hit_streak") {
		t.Fatalf("miss 2 should emit RED FLAG alert, got %q", msg)
	}
	if msg := watch.observe(5000, 0, true); msg != "" {
		t.Fatalf("miss 3 should be rate-limited: %q", msg)
	}
	if msg := watch.observe(5000, 128, true); msg != "" {
		t.Fatalf("cache hit should reset without alerting: %q", msg)
	}
	if msg := watch.observe(5000, 0, true); msg != "" {
		t.Fatalf("miss 1 after reset should not alert: %q", msg)
	}
}

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
	first := &inkling2QueuedRequest{}
	sent := []*inkling2QueuedRequest{first}
	got := collectAsync(testInkling2Admission(clk), first, queue)
	// 15 more arrivals 650us apart: ~10ms of spread, every gap under the 4ms window
	for i := 1; i < 16; i++ {
		clk.waitArmed(t)
		clk.advance(650 * time.Microsecond)
		r := &inkling2QueuedRequest{}
		sent = append(sent, r)
		queue <- r
	}
	clk.waitArmed(t)
	clk.advance(inkling2BatchGapDefault)
	requireInkling2Cohort(t, <-got, sent)
}

// requireInkling2Cohort asserts the collected cohort is exactly the requests that were sent, in any order.
func requireInkling2Cohort(t *testing.T, got, sent []*inkling2QueuedRequest) {
	t.Helper()
	if len(got) != len(sent) {
		t.Fatalf("cohort width %d, want %d (every sent request, none extra)", len(got), len(sent))
	}
	want := make(map[*inkling2QueuedRequest]bool, len(sent))
	for _, r := range sent {
		want[r] = true
	}
	for _, r := range got {
		if !want[r] {
			t.Fatal("cohort holds a request that was not sent or was already admitted")
		}
		delete(want, r)
	}
}

func TestInkling2BatchAdmissionGathersAlreadyQueuedWithoutWaiting(t *testing.T) {
	clk := newFakeInkling2Clock()
	queue := make(chan *inkling2QueuedRequest, ggufload.Inkling2LoopMaxBatch)
	var sent []*inkling2QueuedRequest
	for i := 0; i < ggufload.Inkling2LoopMaxBatch-1; i++ {
		r := &inkling2QueuedRequest{}
		sent = append(sent, r)
		queue <- r
	}
	first := &inkling2QueuedRequest{}
	sent = append(sent, first)
	requireInkling2Cohort(t, testInkling2Admission(clk).collect(first, queue), sent)
	if len(clk.armed) != 0 {
		t.Fatal("a full queue must not arm a timer")
	}
}

func TestInkling2BatchAdmissionMaxWaitBoundsTheCohort(t *testing.T) {
	clk := newFakeInkling2Clock()
	queue := make(chan *inkling2QueuedRequest, 16)
	start := clk.Now()
	first := &inkling2QueuedRequest{}
	sent := []*inkling2QueuedRequest{first}
	got := collectAsync(testInkling2Admission(clk), first, queue)
	var group []*inkling2QueuedRequest
	for group == nil {
		select {
		case d := <-clk.armed:
			if d >= 3*time.Millisecond {
				clk.advance(3 * time.Millisecond)
				r := &inkling2QueuedRequest{}
				sent = append(sent, r)
				queue <- r
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
	requireInkling2Cohort(t, group, sent)
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
	if err != nil || a.gap != 4*time.Millisecond || a.maxWait != 25*time.Millisecond || a.max != ggufload.Inkling2LoopMaxBatch {
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

// An ineligible load serves through the serialized engine, which keeps prefix
// reuse; the lone-session guard only matters beside a cohort engine.
func TestInkling2ServeBatchIneligibleKeepsPrefixReuse(t *testing.T) {
	on := func(k string) (string, bool) { return "1", k == "FAK_INKLING2_SERVE_BATCH" }
	gen, _, err := newServeInkling2Generator(nil, nil, on)
	eng, ok := gen.(*inkling2LoopEngine)
	if err != nil || !ok {
		t.Fatalf("gen=%T err=%v, want the serialized engine", gen, err)
	}
	if eng.prefixReuse.disabled() {
		t.Fatal("ineligible load disabled prefix reuse on the engine it returns")
	}
}

// A row whose caller already gave up neither takes a cohort slot nor is prefilled,
// and it is answered with its context error so nothing waits on it.
func TestInkling2BatchCollectDropsCancelled(t *testing.T) {
	adm := inkling2Admission{gap: inkling2BatchGapDefault, maxWait: inkling2BatchMaxWaitDefault, max: 2, clock: newFakeInkling2Clock()}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	dead := &inkling2QueuedRequest{ctx: ctx, done: make(chan inkling2QueuedResult, 1)}
	live := &inkling2QueuedRequest{ctx: context.Background(), done: make(chan inkling2QueuedResult, 1)}
	tail := &inkling2QueuedRequest{ctx: context.Background(), done: make(chan inkling2QueuedResult, 1)}
	queue := make(chan *inkling2QueuedRequest, 4)
	queue <- dead
	queue <- tail
	group := adm.collect(live, queue)
	if len(group) != 2 || group[0] != live || group[1] != tail {
		t.Fatalf("cohort has %d rows, want the two live rows (cancelled row must not take a slot)", len(group))
	}
	select {
	case res := <-dead.done:
		if !errors.Is(res.err, context.Canceled) {
			t.Fatalf("dropped row err=%v, want context.Canceled", res.err)
		}
	default:
		t.Fatal("dropped row was never answered")
	}
	if g := adm.collect(dead, make(chan *inkling2QueuedRequest)); g != nil {
		t.Fatalf("cancelled first row formed cohort %d rows, want none", len(g))
	}
}

// admittingEngine is a serve engine whose cohorts run through run, at cohort width two.
// The scalar path and the plain batch are forbidden: every row must come through admission.
func admittingEngine(t *testing.T, run inkling2BatchAdmitting) *inkling2LoopBatchEngine {
	t.Helper()
	adm := inkling2Admission{gap: inkling2BatchDurationMax, maxWait: inkling2BatchDurationMax, max: 2, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
		t.Error("admitting cohort rows must not run scalar")
		return agent.Inkling2LoopGeneration{}, nil
	}, func([]ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		t.Error("admitting engine ran the plain batch")
		return nil, nil
	}, adm, 4096)
	e.batchAdmitting = run
	go e.run()
	t.Cleanup(func() { close(e.queue) })
	return e
}

// waitQueued blocks until n requests wait in the queue, so a test admits only after
// the caller it means to admit has arrived.
func waitQueued(t *testing.T, e *inkling2LoopBatchEngine, n int) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for len(e.queue) < n {
		if time.Now().After(deadline) {
			t.Fatalf("queue holds %d requests, want %d", len(e.queue), n)
		}
		time.Sleep(time.Millisecond)
	}
}

// generateAsync runs one Generate in the background and reports its outcome.
func generateAsync(e *inkling2LoopBatchEngine, prompt []int, maxNew int, emitted *[]int) <-chan error {
	out := make(chan error, 1)
	go func() {
		_, err := e.Generate(context.Background(), prompt, maxNew, nil, func(id int) bool {
			*emitted = append(*emitted, id)
			return true
		})
		out <- err
	}()
	return out
}

// A request queued while a cohort runs joins the slot of a row that ended, and its
// caller is answered with the cohort's width.
func TestInkling2BatchAdmittedRequestJoinsFinishedSlot(t *testing.T) {
	const maxPositions = 64
	var e *inkling2LoopBatchEngine
	var emitA, emitB, emitC []int
	var errC <-chan error
	e = admittingEngine(t, func(reqs []ggufload.Inkling2LoopBatchRequest, admit func(int) (ggufload.Inkling2LoopBatchRequest, bool)) ([]ggufload.Inkling2LoopBatchResult, error) {
		if len(reqs) != 2 {
			t.Errorf("cohort width %d, want 2", len(reqs))
		}
		reqs[0].Emit(reqs[0].Prompt[0] * 10)
		reqs[0].Done(ggufload.Inkling2LoopBatchResult{Generated: 1})
		errC = generateAsync(e, []int{3}, 1, &emitC)
		waitQueued(t, e, 1)
		c, ok := admit(maxPositions)
		if !ok {
			t.Error("queued request not admitted into the finished slot")
			return nil, nil
		}
		c.Emit(c.Prompt[0] * 10)
		c.Done(ggufload.Inkling2LoopBatchResult{Generated: 1})
		reqs[1].Emit(reqs[1].Prompt[0] * 10)
		reqs[1].Done(ggufload.Inkling2LoopBatchResult{Generated: 1})
		return nil, nil
	})
	errA := generateAsync(e, []int{1}, 1, &emitA)
	errB := generateAsync(e, []int{2}, 1, &emitB)
	// errC is set by the cohort before it answers B, so it is safe to read after B
	for name, ch := range map[string]<-chan error{"A": errA, "B": errB} {
		if err := <-ch; err != nil {
			t.Errorf("row %s: %v", name, err)
		}
	}
	if err := <-errC; err != nil {
		t.Errorf("row C: %v", err)
	}
	if !reflect.DeepEqual(emitA, []int{10}) || !reflect.DeepEqual(emitB, []int{20}) || !reflect.DeepEqual(emitC, []int{30}) {
		t.Fatalf("emitted A=%v B=%v C=%v", emitA, emitB, emitC)
	}
	if e.admittedRows.Load() != 1 {
		t.Fatalf("admitted rows %d, want 1", e.admittedRows.Load())
	}
}

// A request that does not fit the cohort's capacity is held as the next cohort's first
// row: admission refuses everything behind it, and the next cohort starts with it.
func TestInkling2BatchNonFittingRequestHeldAsNextCohortFirst(t *testing.T) {
	const maxPositions = 4
	var mu sync.Mutex
	var cohorts [][]int
	var e *inkling2LoopBatchEngine
	var errX, errY <-chan error
	cohort := 0
	e = admittingEngine(t, func(reqs []ggufload.Inkling2LoopBatchRequest, admit func(int) (ggufload.Inkling2LoopBatchRequest, bool)) ([]ggufload.Inkling2LoopBatchResult, error) {
		firsts := make([]int, len(reqs))
		for i, r := range reqs {
			firsts[i] = r.Prompt[0]
		}
		mu.Lock()
		cohorts = append(cohorts, firsts)
		mu.Unlock()
		cohort++
		if cohort > 1 {
			return make([]ggufload.Inkling2LoopBatchResult, len(reqs)), nil
		}
		// X needs 6 positions against 4; Y would fit but queues behind X
		// X is queued before Y, so it is the request admission sees first
		errX = generateAsync(e, []int{1, 2, 3}, 3, new([]int))
		waitQueued(t, e, 1)
		errY = generateAsync(e, []int{4}, 1, new([]int))
		waitQueued(t, e, 2)
		if _, ok := admit(maxPositions); ok {
			t.Error("admitted a request that does not fit")
		}
		if _, ok := admit(maxPositions); ok {
			t.Error("admitted a request behind a held one")
		}
		if n := len(e.queue); n != 1 {
			t.Errorf("queue holds %d requests after the refusal, want Y alone", n)
		}
		return make([]ggufload.Inkling2LoopBatchResult, len(reqs)), nil
	})
	e.holBackfillMax = 0
	errA := generateAsync(e, []int{5}, 1, new([]int))
	errB := generateAsync(e, []int{6}, 1, new([]int))
	// the cohort sets errX and errY before it answers A and B, so they are read after
	for name, ch := range map[string]<-chan error{"A": errA, "B": errB} {
		if err := <-ch; err != nil {
			t.Errorf("row %s: %v", name, err)
		}
	}
	for name, ch := range map[string]<-chan error{"X": errX, "Y": errY} {
		if err := <-ch; err != nil {
			t.Errorf("row %s: %v", name, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	// the cohort's row order follows arrival, so the first cohort is checked as a set
	if len(cohorts) != 2 || len(cohorts[0]) != 2 || cohorts[0][0]+cohorts[0][1] != 11 || !reflect.DeepEqual(cohorts[1], []int{1, 4}) {
		t.Fatalf("cohorts %v, want {5 6} then [1 4]: X first, Y after it", cohorts)
	}
}

// A panic while a cohort runs answers the rows it admitted as well as the rows it
// started with, so no caller waits on a worker that is gone.
func TestInkling2BatchPanicAnswersAdmittedRows(t *testing.T) {
	var e *inkling2LoopBatchEngine
	var errC <-chan error
	e = admittingEngine(t, func(reqs []ggufload.Inkling2LoopBatchRequest, admit func(int) (ggufload.Inkling2LoopBatchRequest, bool)) ([]ggufload.Inkling2LoopBatchResult, error) {
		errC = generateAsync(e, []int{3}, 1, new([]int))
		waitQueued(t, e, 1)
		if _, ok := admit(64); !ok {
			t.Error("queued request not admitted")
		}
		panic("device fault")
	})
	errA := generateAsync(e, []int{1}, 1, new([]int))
	errB := generateAsync(e, []int{2}, 1, new([]int))
	// the panic is recovered after errC is set, so it is safe to read once A and B answer
	for name, ch := range map[string]<-chan error{"A": errA, "B": errB} {
		if err := <-ch; err == nil {
			t.Errorf("row %s answered without an error after the cohort panicked", name)
		}
	}
	if err := <-errC; err == nil {
		t.Error("row C answered without an error after the cohort panicked")
	}
}

// admitQueued admits a request whose len(prompt)+maxNew-1 equals maxPositions
// (matching Generate's context bound) and holds one that exceeds it by 1.
func TestInkling2BatchAdmitQueuedBoundary(t *testing.T) {
	const maxPositions = 8
	e := newInkling2LoopBatchEngineWith(nil, nil, testInkling2Admission(inkling2RealClock{}), maxPositions)
	var owned []*inkling2QueuedRequest
	c := &inkling2Cohort{width: 2, owned: &owned}

	// exact boundary: 7 prompt + 2 maxNew - 1 == 8 -> fits
	exact := &inkling2QueuedRequest{ctx: context.Background(), prompt: make([]int, 7), maxNew: 2,
		sink: &inkling2RowSink{emit: func(int) bool { return true }}, done: make(chan inkling2QueuedResult, 1)}
	e.queue <- exact
	if _, ok := e.admitQueued(c, maxPositions); !ok {
		t.Fatal("request with len(prompt)+maxNew-1 == maxPositions was refused")
	}
	if e.pending != nil {
		t.Fatal("fitting boundary request was held as pending")
	}

	// one over boundary: 7 prompt + 3 maxNew - 1 == 9 > 8 -> held as pending
	over := &inkling2QueuedRequest{ctx: context.Background(), prompt: make([]int, 7), maxNew: 3,
		sink: &inkling2RowSink{emit: func(int) bool { return true }}, done: make(chan inkling2QueuedResult, 1)}
	e.queue <- over
	if _, ok := e.admitQueued(c, maxPositions); ok {
		t.Fatal("request with len(prompt)+maxNew-1 > maxPositions was admitted")
	}
	if e.pending != over {
		t.Fatal("oversized request was not held as pending")
	}
}

// High-context batch engines accept requests up to len(prompt)+maxNew-1 == contextTokens
// and reject requests one token past it.
func TestInkling2BatchHighContextEngine(t *testing.T) {
	for _, ctxTokens := range []int{16384, 32768} {
		adm := inkling2Admission{gap: time.Millisecond, maxWait: time.Millisecond, max: 2, clock: inkling2RealClock{}}
		e := newInkling2LoopBatchEngineWith(
			func(_ context.Context, _ []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
				emit(42)
				return agent.Inkling2LoopGeneration{Generated: 1}, nil
			},
			func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
				out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
				for i, r := range reqs {
					r.Emit(42)
					out[i].Generated = 1
				}
				return out, nil
			},
			adm, ctxTokens)
		go e.run()

		prompt := make([]int, ctxTokens-4)
		// exact boundary: (ctxTokens-4) + 5 - 1 == ctxTokens -> accepted
		gen, err := e.Generate(context.Background(), prompt, 5, nil, func(int) bool { return true })
		if err != nil || gen.Generated != 1 {
			close(e.queue)
			t.Fatalf("ctx=%d exact boundary: gen=%+v err=%v", ctxTokens, gen, err)
		}
		// one past boundary: (ctxTokens-4) + 6 - 1 == ctxTokens+1 -> rejected
		if _, err := e.Generate(context.Background(), prompt, 6, nil, func(int) bool { return true }); err == nil {
			close(e.queue)
			t.Fatalf("ctx=%d past boundary accepted", ctxTokens)
		}
		close(e.queue)
	}
}

// TestInkling2LongCtxScalarFallbackAndCohortResume verifies that a request whose
// required context exceeds the multi-row (C>=2) cohort budget routes directly to
// runScalar (releasing cached cohort arenas even when loneCohort is true), and that
// subsequent short bursts automatically resume C>=2 cohorts.
func TestInkling2LongCtxScalarFallbackAndCohortResume(t *testing.T) {
	adm := inkling2Admission{gap: 5 * time.Millisecond, maxWait: 10 * time.Millisecond, max: 4, clock: inkling2RealClock{}}
	var scalarCalls, batchCalls, arenaReleases int
	var batchWidths []int
	e := newInkling2LoopBatchEngineWith(
		func(_ context.Context, prompt []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
			scalarCalls++
			emit(7)
			return agent.Inkling2LoopGeneration{Generated: 1}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			batchCalls++
			batchWidths = append(batchWidths, len(reqs))
			out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
			for i, r := range reqs {
				r.Emit(9)
				out[i].Generated = 1
			}
			return out, nil
		},
		adm, 131072)
	e.loneCohort = true
	e.arenasCached = true
	e.releaseArenas = func() { arenaReleases++ }
	e.fitsMultiRow = func(positions int) bool { return positions <= 16384 }
	go e.run()
	defer close(e.queue)

	// 1) Long-context (65536) request exceeds multi-row budget -> routed directly to runScalar, releasing arenas.
	gen, err := e.Generate(context.Background(), make([]int, 65000), 536, nil, func(int) bool { return true })
	if err != nil || gen.Width != 1 || scalarCalls != 1 || batchCalls != 0 || arenaReleases != 1 {
		t.Fatalf("long-ctx request: gen=%+v err=%v scalar=%d batch=%d releases=%d", gen, err, scalarCalls, batchCalls, arenaReleases)
	}

	// 2) Subsequent 2-request short burst automatically resumes C>=2 cohort batching.
	var wg sync.WaitGroup
	errs := make([]error, 2)
	widths := make([]int, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			g, gerr := e.Generate(context.Background(), make([]int, 512), 64, nil, func(int) bool { return true })
			errs[idx] = gerr
			widths[idx] = g.Width
		}(i)
	}
	wg.Wait()
	for i := 0; i < 2; i++ {
		if errs[i] != nil || widths[i] != 2 {
			t.Fatalf("short burst row %d: width=%d err=%v, want width=2", i, widths[i], errs[i])
		}
	}
	if batchCalls != 1 || len(batchWidths) != 1 || batchWidths[0] != 2 {
		t.Fatalf("short burst did not resume C=2 cohort: batchCalls=%d widths=%v", batchCalls, batchWidths)
	}
}

// TestInkling2AdmitHoLBoundedBackfill verifies that admitQueued allows up to K
// (default 2, configurable via FAK_INKLING2_HOL_BACKFILL_MAX, 0 disables) short
// requests to backfill finished slots while an oversized e.pending waits, then
// drains for e.pending.
func TestInkling2AdmitHoLBoundedBackfill(t *testing.T) {
	mkReq := func(promptLen, maxNew int) *inkling2QueuedRequest {
		return &inkling2QueuedRequest{
			ctx: context.Background(), prompt: make([]int, promptLen), maxNew: maxNew,
			sink: &inkling2RowSink{emit: func(int) bool { return true }}, done: make(chan inkling2QueuedResult, 1),
		}
	}

	// Default K=2: oversized request sets e.pending, then 2 short requests backfill, 3rd is held for drain.
	adm := inkling2Admission{gap: time.Millisecond, maxWait: time.Millisecond, max: 8, clock: inkling2RealClock{}}
	e := newInkling2LoopBatchEngineWith(nil, nil, adm, 65536)
	e.initialDecodeHeadroom = 0
	if got := inkling2HoLBackfillMax(func(string) (string, bool) { return "", false }); got != 2 {
		t.Fatalf("default HoL backfill max = %d, want 2", got)
	}
	if got := inkling2HoLBackfillMax(func(k string) (string, bool) { return "0", true }); got != 0 {
		t.Fatalf("explicit 0 HoL backfill max = %d, want 0", got)
	}
	if got := inkling2HoLBackfillMax(func(k string) (string, bool) { return "4", true }); got != 4 {
		t.Fatalf("explicit 4 HoL backfill max = %d, want 4", got)
	}

	var owned []*inkling2QueuedRequest
	c := &inkling2Cohort{width: 2, owned: &owned}
	big := mkReq(1000, 10)
	s1, s2, s3 := mkReq(10, 4), mkReq(10, 4), mkReq(10, 4)
	e.queue <- big
	e.queue <- s1
	e.queue <- s2
	e.queue <- s3

	if _, ok := e.admitQueued(c, 100); !ok || e.pending != big || c.holBackfills != 1 {
		t.Fatalf("1st backfill failed: ok=%v pending=%v backfills=%d", ok, e.pending == big, c.holBackfills)
	}
	if _, ok := e.admitQueued(c, 100); !ok || c.holBackfills != 2 {
		t.Fatalf("2nd backfill failed: ok=%v backfills=%d", ok, c.holBackfills)
	}
	// 3rd short request must NOT backfill once K=2 is reached; cohort drains for big.
	if _, ok := e.admitQueued(c, 100); ok {
		t.Fatal("3rd short request backfilled past K=2 limit")
	}

	// K=0 disables backfill immediately when big sets e.pending.
	e0 := newInkling2LoopBatchEngineWith(nil, nil, adm, 65536)
	e0.initialDecodeHeadroom = 0
	e0.holBackfillMax = 0
	c0 := &inkling2Cohort{width: 2, owned: &owned}
	e0.queue <- big
	e0.queue <- s1
	if _, ok := e0.admitQueued(c0, 100); ok || e0.pending != big || c0.holBackfills != 0 {
		t.Fatalf("K=0 admitted backfill: ok=%v pending=%v backfills=%d", ok, e0.pending == big, c0.holBackfills)
	}
}

// fak-test:runtime fast est=5ms
func TestInkling2AdmitCohortUsesLastArenaEstimateToShedProactively(t *testing.T) {
	ledger := newInkling2ByteLedger(1000)
	ledger.Set("prefix_store", inkling2LedgerPrefixStore, 700)
	arena := int64(600)
	var shedNeed int64
	e := &inkling2LoopBatchEngine{
		ledger:           ledger,
		cohortArenaBytes: func() int64 { return arena },
		shedIdleSlots:    func(n int64) int64 { shedNeed = n; return n },
	}
	e.syncCohortLedger()
	if e.lastArenaBytes != 600 {
		t.Fatalf("lastArenaBytes = %d, want 600", e.lastArenaBytes)
	}
	ledger.Delete("cohort_arena")
	e.admitCohortMemory(e.lastArenaBytes)
	if shedNeed != 300 {
		t.Fatalf("shed need = %d, want 300 (700+600-1000)", shedNeed)
	}
}
