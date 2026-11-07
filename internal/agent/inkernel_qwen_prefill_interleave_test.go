package agent

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// This fixture exercises real host hybrid append, recurrent state, sampling and
// the production request coalescer. Its Q4_K selector admits the existing chunk
// boundary, but synthetic weights execute host arithmetic, not GPU qualification.
type qwenInterleaveSession struct {
	*model.Session
	after func(bool) error
}

func (s qwenInterleaveSession) PrefillNoLogitsContext(ctx context.Context, ids []int) error {
	if err := s.Session.PrefillNoLogitsContext(ctx, ids); err != nil {
		return err
	}
	return s.after(false)
}

func (s qwenInterleaveSession) PrefillContext(ctx context.Context, ids []int) ([]float32, error) {
	logits, err := s.Session.PrefillContext(ctx, ids)
	if err == nil {
		err = s.after(true)
	}
	return logits, err
}

type qwenInterleaveAnswer struct {
	result inKernelGenerateResult
	tokens []int
	cache  *model.KVCache
	err    error
	panic  any
}

type qwenInterleaveFixture struct {
	t       *testing.T
	p       *InKernelPlanner
	prompts [][]int
	budgets []int
	started []atomic.Int32
	closed  []atomic.Int32
	reqs    []*inKernelCoalesceRequest
	done    []chan qwenInterleaveAnswer
	ctx     []context.Context
	cancel  []context.CancelFunc
	mu      sync.Mutex
	events  []string
	chunk   func(int, int, bool) error
}

func newQwenInterleaveFixture(t *testing.T, budgets ...int) *qwenInterleaveFixture {
	t.Helper()
	// These fixtures pin the #14226 interleaving choreography through probe-hook
	// synchronization that assumes every cohort step takes the batched driver.
	// Disable the adaptive batched->serial governor so driver selection cannot
	// divert steps out of the probe; the governor itself is pinned by
	// TestBatchStepHealth.
	t.Setenv(batchStepHealthEnv, "0")
	m := model.NewSyntheticQwen38MTP()
	f := &qwenInterleaveFixture{
		t: t, p: &InKernelPlanner{m: m, q4k: true, qwenQ4KPrefillChunkTokens: 2, coalesceReadyHook: func() {}},
		budgets: budgets, prompts: make([][]int, len(budgets)), started: make([]atomic.Int32, len(budgets)),
		closed: make([]atomic.Int32, len(budgets)), reqs: make([]*inKernelCoalesceRequest, len(budgets)),
		done: make([]chan qwenInterleaveAnswer, len(budgets)), ctx: make([]context.Context, len(budgets)), cancel: make([]context.CancelFunc, len(budgets)),
	}
	for i := range budgets {
		f.prompts[i] = []int{1, 2, 3 + i, 4, 5}
		f.done[i] = make(chan qwenInterleaveAnswer, 1)
		f.ctx[i], f.cancel[i] = context.WithCancel(context.Background())
	}
	t.Cleanup(func() {
		for _, cancel := range f.cancel {
			cancel()
		}
	})
	return f
}

func (f *qwenInterleaveFixture) note(event string) {
	f.mu.Lock()
	f.events = append(f.events, event)
	f.mu.Unlock()
}

func (f *qwenInterleaveFixture) run(i int) {
	go func() {
		var out qwenInterleaveAnswer
		defer func() { out.panic = recover(); f.done[i] <- out }()
		out.result, out.err = f.p.runCoalescedGenerate(f.ctx[i], func(ctx context.Context) (inKernelGenerateResult, error) {
			f.started[i].Add(1)
			req := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest)
			f.reqs[i] = req
			s := f.p.m.NewSession()
			s.Quant = false
			defer func() { s.Close(); f.closed[i].Add(1) }()
			chunks := 0
			observed := qwenInterleaveSession{Session: s, after: func(final bool) error {
				chunks++
				f.note(fmt.Sprintf("p%d:%d:%t", i, s.SequencePosition(), final))
				if f.chunk != nil {
					return f.chunk(i, chunks, final)
				}
				return nil
			}}
			logits, err := f.p.prefillDivergentSuffix(ctx, observed, f.prompts[i])
			if err != nil {
				return inKernelGenerateResult{}, err
			}
			f.note(fmt.Sprintf("ready%d", i))
			lane := &decodeLane{s: s, logits: logits, rng: rand.New(rand.NewSource(1)), stops: map[int]bool{}, maxNew: f.budgets[i]}
			lane.emit = func(id int) bool { out.tokens = append(out.tokens, id); return false }
			_, err = coalescedDecode(ctx, lane)
			out.cache = s.Cache.Clone()
			return inKernelGenerateResult{gen: lane.gen}, err
		})
	}()
}

// The production retry wrapper runs over a capability fixture with one-token
// append chunks. GDN math is deliberately synthetic here: this test qualifies
// abort/retry ownership, while the preceding fixture checks real host math.
type qwenInterleaveRetryBackend struct {
	*prefixReuseQwenBackend
	trims int
}

func (*qwenInterleaveRetryBackend) Qwen35SequencePrefillPath() string {
	return compute.Qwen35SequencePrefillPath
}
func (*qwenInterleaveRetryBackend) Qwen35SequenceEmbeddingRowsPath() string {
	return compute.Qwen35SequenceEmbeddingRowsPath
}
func (*qwenInterleaveRetryBackend) Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error) {
	panic("one-token lifecycle witness must use the scalar capability fixture")
}
func (b *qwenInterleaveRetryBackend) Recycle()      {}
func (b *qwenInterleaveRetryBackend) TrimLarge(int) { b.trims++ }

// fak-test:runtime fast est=1s lane=default
func TestInKernelQwenPrefillInterleaveDoesNotRetryPreparingOOM(t *testing.T) {
	t.Setenv(inKernelDecodeShareEnv, "0.5")
	t.Setenv(inKernelPromptRowBudgetEnv, "0")
	f := newQwenInterleaveFixture(t, 12)
	be := &qwenInterleaveRetryBackend{prefixReuseQwenBackend: &prefixReuseQwenBackend{Backend: compute.Default()}}
	prefiller := &InKernelPlanner{m: f.p.m, backend: be, q4k: true, qwenQ4KPrefillChunkTokens: 1}
	// Ensure the fixture could retry a causal OOM before any coordinated decode
	// pass. A nil backend would make a broken typed abort look safely non-retryable.
	causal := &InKernelOOMError{Bytes: 4096, Class: compute.MemoryScratchpad}
	if retry, err := prefiller.admitDeviceOOMRetry(context.Background(), causal, 5, 4); !retry || err != nil {
		t.Fatalf("retry positive control was not admitted: retry=%t err=%v", retry, err)
	}
	be.trims = 0
	first, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	calls := 0
	restore := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		calls++
		if calls == 2 {
			close(first)
			<-release
		}
		if calls == 3 {
			panic(&compute.DeviceAllocError{Bytes: causal.Bytes, Class: causal.Class})
		}
		out := bs.StepBatchActive(ids, active)
		return out, bs.LastStepSharedPanels(), bs.LastStepMACs(), true
	})
	t.Cleanup(restore)
	f.run(0)
	awaitQwenInterleave(t, first, "retry witness live decoder")
	done := make(chan qwenInterleaveAnswer, 1)
	var request *inKernelCoalesceRequest
	retried, emitted := 0, 0
	go func() {
		var out qwenInterleaveAnswer
		defer func() { out.panic = recover(); done <- out }()
		out.result, out.err = f.p.runCoalescedGenerate(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			request = ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest)
			return prefiller.generateReusedWithOOMRetry(ctx, []int{1, 2, 3, 4, 5}, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { emitted++; return false }, func() { retried++ })
		})
	}()
	awaitCoalescedQueue(t, f.p, 1)
	once.Do(func() { close(release) })
	leader := f.answer(0)
	var recovered *InKernelOOMError
	if !errors.As(leader.err, &recovered) || recovered.Bytes != causal.Bytes {
		t.Errorf("decoder did not retain causal OOM: %+v", leader)
	}
	select {
	case out := <-done:
		if !errors.Is(out.err, errInKernelQwenCohortAborted) || out.panic != nil || request == nil || request.decodePass.Load() != 0 || retried != 0 || emitted != 0 || be.trims != 0 {
			t.Errorf("preparation replayed or lost terminal abort: out=%+v request=%v retries=%d emitted=%d trims=%d", out, request, retried, emitted, be.trims)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("OOM retry wrapper stranded preparing request")
	}
	f.assertIdle()
}

func (f *qwenInterleaveFixture) answer(i int) qwenInterleaveAnswer {
	f.t.Helper()
	select {
	case out := <-f.done[i]:
		return out
	case <-time.After(3 * time.Second):
		f.t.Fatalf("request %d stranded", i)
		return qwenInterleaveAnswer{}
	}
}

func awaitQwenInterleave(t *testing.T, ready <-chan struct{}, where string) {
	t.Helper()
	select {
	case <-ready:
	case <-time.After(3 * time.Second):
		t.Fatalf("did not reach %s", where)
	}
}

func (f *qwenInterleaveFixture) assertIdle() {
	f.t.Helper()
	f.p.coalesceMu.Lock()
	pending, running := len(f.p.coalesceReady), f.p.coalesceRunning
	f.p.coalesceMu.Unlock()
	if pending != 0 || running {
		f.t.Errorf("coordinator retained pending=%d running=%t", pending, running)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := f.p.devMu.LockContext(ctx); err != nil {
		f.t.Errorf("device ownership retained: %v", err)
	} else {
		f.p.devMu.Unlock()
	}
}

// fak-test:runtime fast est=1s lane=default
func TestInKernelQwenPrefillInterleaveNativeRequests(t *testing.T) {
	for _, tc := range []struct {
		name, share string
		budget      int
		wantBetween bool
	}{
		{"selected", "0.5", 12, true},
		{"old_lane_retires", "0.5", 3, false},
		// Unset now defaults to share 0.5, so the default row interleaves exactly
		// like the explicit 0.5 row; only an explicit out-of-(0,1) value keeps
		// whole-prompt arrival ordering.
		{"default", "", 12, true},
		{"invalid", "junk", 12, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(inKernelDecodeShareEnv, tc.share)
			t.Setenv(inKernelPromptRowBudgetEnv, "0")
			f := newQwenInterleaveFixture(t, tc.budget, 4)
			oldRetired, releaseRemainder := make(chan struct{}), make(chan struct{})
			var remainderOnce sync.Once
			t.Cleanup(func() { remainderOnce.Do(func() { close(releaseRemainder) }) })
			if tc.name == "old_lane_retires" {
				f.chunk = func(i, chunks int, _ bool) error {
					if i == 1 && chunks == 2 {
						close(oldRetired)
						<-releaseRemainder
					}
					return nil
				}
			}
			first, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			t.Cleanup(func() { once.Do(func() { close(release) }) })
			calls := 0
			restore := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
				calls++
				if calls == 2 {
					close(first)
					<-release
				}
				out := bs.StepBatchActive(ids, active)
				f.note("decode")
				return out, bs.LastStepSharedPanels(), bs.LastStepMACs(), true
			})
			t.Cleanup(restore)
			f.run(0)
			awaitQwenInterleave(t, first, "live decoder")
			f.run(1)
			awaitCoalescedQueue(t, f.p, 1)
			once.Do(func() { close(release) })
			if tc.name == "old_lane_retires" {
				awaitQwenInterleave(t, oldRetired, "prefill after old decoder retirement")
			}
			leader := f.answer(0)
			remainderOnce.Do(func() { close(releaseRemainder) })
			answers := []qwenInterleaveAnswer{leader, f.answer(1)}
			for i, out := range answers {
				lane, sink := newDecodeLaneOver(f.p.m, f.prompts[i], f.budgets[i], 1)
				inKernelDecodeSerial(context.Background(), lane)
				wantCache := lane.s.Cache.Clone()
				lane.s.Close()
				if out.err != nil || out.panic != nil || !reflect.DeepEqual(out.tokens, sink.toks) || f.closed[i].Load() != 1 {
					t.Errorf("request %d: tokens=%v want=%v err=%v panic=%v closes=%d", i, out.tokens, sink.toks, out.err, out.panic, f.closed[i].Load())
				}
				if !reflect.DeepEqual(out.cache, wantCache) {
					t.Errorf("request %d retained attention KV/recurrent state differs from serial", i)
				}
			}
			between, afterFinal, during, finished, ready := 0, 0, false, false, false
			for _, event := range f.events {
				if event == "p1:2:false" {
					during = true
				} else if event == "p1:5:true" {
					during, finished = false, true
				} else if event == "ready1" {
					ready = true
				} else if during && event == "decode" {
					between++
				} else if finished && !ready && event == "decode" {
					afterFinal++
				}
			}
			if !finished || !ready || afterFinal != 0 || (between > 0) != tc.wantBetween || (tc.wantBetween && between != 2) {
				t.Fatalf("executed inter-chunk forwards=%d want selected=%t; events=%v", between, tc.wantBetween, f.events)
			}
			f.assertIdle()
		})
	}
}

// Two late requests are selected together. The second is no longer queued when
// the first yields; every abnormal owner exit must resolve that sibling too.
// fak-test:runtime fast est=1s lane=default
func TestInKernelQwenPrefillInterleaveUnwind(t *testing.T) {
	for _, mode := range []string{"panic", "device_oom", "owner_goexit", "prepare_goexit", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv(inKernelDecodeShareEnv, "0.5")
			t.Setenv(inKernelPromptRowBudgetEnv, "0")
			f := newQwenInterleaveFixture(t, 12, 4, 4, 2)
			first, releaseArrival := make(chan struct{}), make(chan struct{})
			paused, releaseBoundary := make(chan struct{}), make(chan struct{})
			var arrivalOnce, boundaryOnce sync.Once
			t.Cleanup(func() {
				arrivalOnce.Do(func() { close(releaseArrival) })
				boundaryOnce.Do(func() { close(releaseBoundary) })
			})
			primary := errors.New("interleaved decoder failure")
			oom := &compute.DeviceAllocError{Bytes: 4096, Class: compute.MemoryScratchpad, Site: "interleaved-decode"}
			if mode == "prepare_goexit" {
				f.chunk = func(i, chunks int, _ bool) error {
					if i == 1 && chunks == 1 {
						runtime.Goexit()
					}
					return nil
				}
			}
			calls := 0
			forwarded := make(map[*model.Session]int)
			var leaderSession *model.Session
			restore := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
				calls++
				if leaderSession == nil {
					leaderSession = bs.Seqs[0]
				}
				if calls == 2 {
					close(first)
					<-releaseArrival
				}
				if calls == 3 && mode != "prepare_goexit" {
					close(paused)
					<-releaseBoundary
				}
				out := bs.StepBatchActive(ids, active)
				for i, on := range active {
					if on {
						forwarded[bs.Seqs[i]]++
					}
				}
				if calls == 3 && mode != "prepare_goexit" {
					// Fail after real Session state advanced, not before the
					// forward. Restarting this lane would apply its token twice.
					switch mode {
					case "panic":
						panic(primary)
					case "device_oom":
						panic(oom)
					case "owner_goexit":
						runtime.Goexit()
					}
				}
				return out, bs.LastStepSharedPanels(), bs.LastStepMACs(), true
			})
			t.Cleanup(restore)
			f.run(0)
			awaitQwenInterleave(t, first, "initial decode")
			f.run(1)
			awaitCoalescedQueue(t, f.p, 1)
			f.run(2)
			awaitCoalescedQueue(t, f.p, 2)
			arrivalOnce.Do(func() { close(releaseArrival) })
			if mode != "prepare_goexit" {
				awaitQwenInterleave(t, paused, "paused prefill")
				if f.started[1].Load() != 1 || f.started[2].Load() != 0 {
					t.Fatal("witness did not reach first of two selected preparations")
				}
				if mode == "cancel" {
					f.cancel[1]()
					if f.closed[1].Load() != 0 {
						t.Fatal("cancellation closed prefill before decoder acknowledged")
					}
				}
				f.run(3)
				awaitCoalescedQueue(t, f.p, 1)
				boundaryOnce.Do(func() { close(releaseBoundary) })
			}
			out := []qwenInterleaveAnswer{f.answer(0), f.answer(1), f.answer(2)}
			if mode == "prepare_goexit" {
				f.run(3)
			}
			successor := f.answer(3)
			if successor.err != nil || successor.panic != nil || f.closed[3].Load() != 1 {
				t.Fatalf("next request lost handoff: %+v", successor)
			}
			if mode == "cancel" || mode == "prepare_goexit" {
				want := error(errInKernelQwenCohortAborted)
				if mode == "cancel" {
					want = context.Canceled
				}
				if !errors.Is(out[1].err, want) || out[0].err != nil || out[2].err != nil || f.closed[2].Load() != 1 {
					t.Errorf("lane-local exit spread: %+v", out)
				}
			} else {
				if forwarded[leaderSession] != 3 || len(out[1].tokens) != 0 || len(out[2].tokens) != 0 {
					t.Errorf("fatal forward replayed or unprepared tokens escaped: forwards=%d tokens=%v/%v", forwarded[leaderSession], out[1].tokens, out[2].tokens)
				}
				if !errors.Is(out[1].err, errInKernelQwenCohortAborted) || !errors.Is(out[2].err, errInKernelQwenCohortAborted) || f.started[2].Load() != 0 {
					t.Errorf("unstarted sibling or preparation escaped fatal abort: %+v", out)
				}
				if mode == "panic" && out[0].panic != primary {
					t.Errorf("primary panic=%v", out[0].panic)
				}
				var recoveredOOM *InKernelOOMError
				if mode == "device_oom" && (!errors.As(out[0].err, &recoveredOOM) || recoveredOOM.Bytes != oom.Bytes || recoveredOOM.Site != oom.Site) {
					t.Errorf("causal OOM=%v", out[0].err)
				}
				if mode == "owner_goexit" && !errors.Is(out[0].err, errInKernelQwenCohortAborted) {
					t.Errorf("Goexit result=%+v", out[0])
				}
			}
			if f.closed[0].Load() != 1 || f.closed[1].Load() != 1 {
				t.Errorf("joined closes=%d/%d", f.closed[0].Load(), f.closed[1].Load())
			}
			f.assertIdle()
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestInKernelQwenPrefillInterleaveDebtSegments(t *testing.T) {
	share := newInKernelDecodeShare(0.5, 2)
	// Keep the serving fixture buildable on its parent revision for the
	// negative witness; only this added lifecycle unit case needs the method.
	retirement, ok := any(&share).(interface{ retired(bool) })
	if !ok {
		t.Fatal("decode-share retirement lifecycle is absent")
	}
	share.stepped(100, true)
	share.rounded(40)
	retirement.retired(true)
	if share.debtNs != 60 {
		t.Fatalf("first segment debt=%d", share.debtNs)
	}
	share.stepped(30, true)
	share.rounded(20)
	if share.debtNs != 70 {
		t.Fatalf("second segment debt=%d", share.debtNs)
	}
	retirement.retired(false)
	share.stepped(1000, false)
	if share.debtNs != 0 || !share.due(true) || share.promptShare != 0.5 || share.rowBudget != 2 {
		t.Fatalf("new decoder inherited obsolete debt or lost configuration: %+v", share)
	}
}

// Exercise the ordinary Complete gate and generateReusedWithOOMRetry closure,
// not only a custom coalescer request. The established synthetic Metal gate is
// used solely for host reachability; no hardware/shared-panel credit is minted.
// fak-test:runtime fast est=2s lane=default
func TestInKernelQwenPrefillInterleaveCompleteReachability(t *testing.T) {
	t.Setenv(inKernelDecodeShareEnv, "0.5")
	t.Setenv(inKernelPromptRowBudgetEnv, "0")
	cfg := tinyConcurrencyConfig()
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	cfg.LinearConvKernelDim, cfg.LinearKeyHeadDim, cfg.LinearNumKeyHeads = 3, 8, 2
	cfg.LinearValueHeadDim, cfg.LinearNumValueHeads = 8, 4
	newPlanner := func(coalesced bool) *InKernelPlanner {
		m := model.NewSynthetic(cfg)
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "prefill-interleave-reachability", true, nil, false)
		p.maxNew, p.qwenQ4KPrefillChunkTokens = 12, 2
		p.batchDecode, p.metal = coalesced, coalesced
		p.coalesceReadyHook = func() {}
		return p
	}
	messages := [][]Message{
		{{Role: RoleUser, Content: "alpha"}},
		{{Role: RoleUser, Content: "beta has a longer divergent prompt"}},
	}
	serial := newPlanner(false)
	want := make([]*Completion, len(messages))
	for i := range messages {
		var err error
		want[i], err = serial.Complete(context.Background(), messages[i], nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	previous := enginestep.Default
	recorder := enginestep.New(enginestep.MaxRecent)
	enginestep.Default = recorder
	t.Cleanup(func() { enginestep.Default = previous })
	p := newPlanner(true)
	if !p.coalescesQwenDecode() || !p.qwenQ4KPrefillChunkTarget() {
		t.Fatal("fixture does not reach both native gates")
	}
	first, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	var late *inKernelCoalesceRequest
	calls, duringPrefill := 0, 0
	restore := installQwenSharedReceiptProbeForTest(func(bs *model.BatchSession, ids []int, active []bool) ([][]float32, int, int64, bool) {
		calls++
		if calls == 2 {
			close(first)
			<-release
		}
		out := bs.StepBatchActive(ids, active)
		if late != nil && late.decodePass.Load() == 0 {
			for _, row := range recorder.Snapshot(enginestep.MaxRecent, enginestep.KindPrefillChunk).Recent {
				if row.RequestID == "complete-late" {
					duringPrefill++
					break
				}
			}
		}
		return out, bs.LastStepSharedPanels(), bs.LastStepMACs(), true
	})
	t.Cleanup(restore)
	type answer struct {
		completion *Completion
		err        error
	}
	done := []chan answer{make(chan answer, 1), make(chan answer, 1)}
	run := func(i int) {
		go func() {
			ctx := enginestep.WithRequestObservation(context.Background(), "complete-first", "interleave-test")
			if i == 1 {
				ctx = enginestep.WithRequestObservation(context.Background(), "complete-late", "interleave-test")
			}
			comp, err := p.Complete(ctx, messages[i], nil)
			done[i] <- answer{comp, err}
		}()
	}
	run(0)
	awaitQwenInterleave(t, first, "Complete live decoder")
	run(1)
	awaitCoalescedQueue(t, p, 1)
	p.coalesceMu.Lock()
	late = p.coalesceReady[0]
	p.coalesceMu.Unlock()
	once.Do(func() { close(release) })
	for i := range done {
		select {
		case out := <-done[i]:
			if out.err != nil || out.completion == nil || out.completion.InKernelBatch == nil {
				t.Fatalf("Complete %d=%+v", i, out)
			}
			if out.completion.Message.Content != want[i].Message.Content || out.completion.FinishReason != want[i].FinishReason {
				t.Errorf("Complete %d differs from serial", i)
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("Complete %d stranded", i)
		}
	}
	if duringPrefill == 0 {
		t.Fatal("Complete path completed no actual decode forward between late prompt chunks")
	}
	(&qwenInterleaveFixture{t: t, p: p}).assertIdle()
}
