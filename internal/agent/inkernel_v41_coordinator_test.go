package agent

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func v41CoordinatorRequest(ctx context.Context, run func(context.Context) (inKernelGenerateResult, error)) *inKernelCoalesceRequest {
	return &inKernelCoalesceRequest{
		kind: inKernelCoalesceV41, ctx: ctx, run: run,
		prepared: make(chan *decodeLane, 1), proceed: make(chan error, 1),
		result: make(chan inKernelCoalesceResult, 1), done: make(chan struct{}),
		drain: make(chan struct{}, 1), receiptReady: make(chan struct{}),
	}
}

type v41CoordinatorTrimProbe struct {
	compute.Backend
	trims atomic.Int32
}

func (b *v41CoordinatorTrimProbe) Trim() { b.trims.Add(1) }

// Scripted positive results below qualify request/driver state transitions only.
// No Flash head arithmetic or actual Complete admission is fabricated.
// fak-test:runtime fast est=30ms lane=default
func TestInKernelV41CoordinatorLaneOutcomes(t *testing.T) {
	lanes, sinks := v41DriverLanes(4, 3)
	m := model.NewSynthetic(tinyCfg())
	backend := &v41CoordinatorTrimProbe{Backend: compute.Default()}
	p := &InKernelPlanner{m: m, backend: backend}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	fault := &InKernelOOMError{Bytes: 17, Site: "scripted-cohort-fault"}
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		laneCtx := context.Background()
		if i == 1 {
			laneCtx = ctx
		}
		requests[i] = v41CoordinatorRequest(laneCtx, func(runCtx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			coordinated, err := coalescedDecode(runCtx, lane)
			if !coordinated {
				t.Error("request lost coalesced context")
			}
			if lane.err != nil {
				err = lane.err
			}
			if err != nil {
				if retry, _ := p.admitDeviceOOMRetry(runCtx, fault, 1, 3); retry {
					t.Error("accepted/emitted request authorized whole-request retry")
				}
			}
			return inKernelGenerateResult{gen: lane.gen}, err
		})
	}
	calls, waits, scalar := 0, 0, 0
	var first []model.V41BatchLane
	ops := inKernelV41DecodeOps{
		batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
			calls++
			switch calls {
			case 1:
				first = append([]model.V41BatchLane(nil), input...)
				return v41DriverRefusal(input), model.ErrV41BatchBusy
			case 2:
				if !reflect.DeepEqual(first, input) {
					t.Fatal("owner-busy retry changed chosen tokens or stable IDs")
				}
				cancel()
				return model.V41BatchResult{Selected: true, SharedHeadPanels: 1, SharedHeadRows: 4, Lanes: []model.V41BatchLaneResult{
					{ID: input[3].ID, Committed: true, Err: fault},
					{ID: input[1].ID, Err: context.Canceled},
					{ID: input[0].ID, Committed: true, Logits: []float32{0, 0, 10}},
					{ID: input[2].ID, Err: fault},
				}}, fault
			case 3:
				if len(input) != 1 || input[0].ID != 1 || input[0].Token != 2 {
					t.Fatal("failed or committed-error lane was replayed")
				}
				return v41DriverRefusal(input), model.ErrV41BatchUnavailable
			default:
				t.Fatal("unexpected batch replay")
				return model.V41BatchResult{}, nil
			}
		},
		serial: func(s *model.Session, token int) ([]float32, error) {
			scalar++
			if s != lanes[0].s || token != 2 {
				t.Fatal("scalar fallback changed surviving request")
			}
			return []float32{0, 0, 10}, nil
		},
		wait: func(context.Context) error {
			waits++
			for i, lane := range lanes {
				if lane.gen != 1 || lane.counts[1] != 1 || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
					t.Fatal("busy wait resampled or emitted twice")
				}
			}
			return nil
		},
	}
	p.runV41DecodeCohort(requests, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
		return inKernelDecodeV41LanesWithOps(ctx, got, ops)
	})
	if calls != 3 || waits != 1 || scalar != 1 || closed.Load() != 4 || backend.trims.Load() != 0 {
		t.Fatalf("calls/waits/scalar/closed/trims=%d/%d/%d/%d/%d", calls, waits, scalar, closed.Load(), backend.trims.Load())
	}
	for i, req := range requests {
		out := <-req.result
		<-req.receiptReady
		want := error(fault)
		if i == 0 {
			want = nil
		} else if i == 1 {
			want = context.Canceled
		}
		if out.err != want || req.decodePass.Load() != 1 || req.receipt.SharedPanels != 1 || req.receipt.SharedRows != 4 || req.receipt.SharedMACs != 0 || req.receipt.SessionCloses != 4 {
			t.Fatalf("lane %d outcome=%v receipt=%+v passes=%d", i, out.err, req.receipt, req.decodePass.Load())
		}
	}
	if !reflect.DeepEqual(sinks[0].toks, []int{1, 2, 2}) || !reflect.DeepEqual(lanes[0].forwarded, []int{1, 2}) || len(lanes[1].forwarded) != 0 || len(lanes[2].forwarded) != 0 || !reflect.DeepEqual(lanes[3].forwarded, []int{1}) {
		t.Fatal("per-lane scatter lost healthy output or actual commit accounting")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInKernelV41CoordinatorPanicJoinsTeardown(t *testing.T) {
	lanes, sinks := v41DriverLanes(2, 3)
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	primary := &struct{ name string }{"decode panic"}
	secondary := &struct{ name string }{"cleanup panic"}
	cleanupStarted, releaseCleanup := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseCleanup) }) }
	defer release()
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		requests[i] = v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() {
				if i == 0 {
					close(cleanupStarted)
					<-releaseCleanup
				}
				lane.s.Close()
				closed.Add(1)
				if i == 1 {
					panic(secondary)
				}
			}()
			_, err := coalescedDecode(ctx, lane)
			return inKernelGenerateResult{}, err
		})
	}
	finished := make(chan any, 1)
	go func() {
		defer func() { finished <- recover() }()
		p.runV41DecodeCohort(requests, func(ctx context.Context, got []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, got, inKernelV41DecodeOps{
				batch:  func([]model.V41BatchLane) (model.V41BatchResult, error) { panic(primary) },
				serial: func(*model.Session, int) ([]float32, error) { panic("unexpected replay") },
				wait:   func(context.Context) error { panic("unexpected retry") },
			})
		})
	}()
	select {
	case <-cleanupStarted:
	case <-time.After(time.Second):
		t.Fatal("request cleanup never began")
	}
	select {
	case got := <-finished:
		t.Fatalf("panic escaped while cleanup blocked: %v", got)
	case <-time.After(10 * time.Millisecond):
	}
	for _, req := range requests {
		select {
		case <-req.receiptReady:
			t.Fatal("receipt published before all teardown joined")
		default:
		}
	}
	release()
	select {
	case got := <-finished:
		if got != primary {
			t.Fatalf("panic identity changed: %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("fatal cohort failed to join")
	}
	for i, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; !errors.Is(out.err, errInKernelV41CohortAborted) || req.decodePass.Load() != 1 || !reflect.DeepEqual(sinks[i].toks, []int{1}) {
			t.Fatal("fatal request replayed or lost terminal cause")
		}
	}
	if closed.Load() != 2 {
		t.Fatal("panic escaped before every Session.Close")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := p.devMu.LockContext(ctx); err != nil {
		t.Fatal("fatal cohort retained planner gate")
	}
	p.devMu.Unlock()
}

// This executes real queue/handoff and dense CPU sessions, not Flash admission.
// fak-test:runtime fast est=100ms lane=default
func TestInKernelV41CoordinatorFIFOAndCap(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	release := make(chan struct{})
	var releaseOnce sync.Once
	open := func() { releaseOnce.Do(func() { close(release) }) }
	p := &InKernelPlanner{m: m, coalesceReadyHook: func() { <-release }}
	kinds := []inKernelCoalesceKind{inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceV41, inKernelCoalesceQwen, inKernelCoalesceQwen, inKernelCoalesceV41}
	var mu sync.Mutex
	var order, sizes []int
	p.coalesceBatchHook = func(n int) { mu.Lock(); sizes = append(sizes, n); mu.Unlock() }
	answers := make([]chan inKernelCoalesceResult, len(kinds))
	defer open()
	for i, kind := range kinds {
		answers[i] = make(chan inKernelCoalesceResult, 1)
		go func(i int, kind inKernelCoalesceKind) {
			res, err := p.runCoalescedGenerateKind(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
				mu.Lock()
				order = append(order, i)
				mu.Unlock()
				lane, _ := newDecodeLaneOver(m, []int{1, 2}, 1, 0)
				defer lane.s.Close()
				_, err := coalescedDecode(ctx, lane)
				return inKernelGenerateResult{gen: lane.gen}, err
			}, kind)
			answers[i] <- inKernelCoalesceResult{result: res, err: err}
		}(i, kind)
		awaitCoalescedQueue(t, p, i+1)
	}
	open()
	wantID := []uint64{1, 1, 1, 1, 2, 3, 3, 4}
	for i, done := range answers {
		select {
		case out := <-done:
			if out.err != nil || out.result.gen != 1 || out.result.batchReceipt.CohortID != wantID[i] || out.result.batchReceipt.SharedPanels != 0 || out.result.batchReceipt.SharedRows != 0 {
				t.Fatalf("request %d: %+v %v", i, out.result, out.err)
			}
		case <-time.After(time.Second):
			t.Fatal("FIFO request stranded")
		}
	}
	if !reflect.DeepEqual(order, []int{0, 1, 2, 3, 4, 5, 6, 7}) || !reflect.DeepEqual(sizes, []int{4, 1, 2, 1}) || p.coalesceRunning || len(p.coalesceReady) != 0 {
		t.Fatalf("FIFO order=%v sizes=%v idle=%t", order, sizes, !p.coalesceRunning)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestInKernelV41CoordinatorFatalHandsOffQueue(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	release := make(chan struct{})
	var once sync.Once
	open := func() { once.Do(func() { close(release) }) }
	defer open()
	p := &InKernelPlanner{m: m, coalesceReadyHook: func() { <-release }}
	primary := &struct{ value int }{17}
	calls := 0
	p.coalesceBatchHook = func(int) {
		calls++
		if calls == 1 {
			panic(primary)
		}
	}
	var closed atomic.Int32
	run := func(ctx context.Context) (inKernelGenerateResult, error) {
		lane, _ := newDecodeLaneOver(m, []int{1, 2}, 1, 0)
		defer func() { lane.s.Close(); closed.Add(1) }()
		_, err := coalescedDecode(ctx, lane)
		return inKernelGenerateResult{gen: lane.gen}, err
	}
	leader := make(chan any, 1)
	go func() {
		defer func() { leader <- recover() }()
		_, _ = p.runCoalescedGenerateKind(context.Background(), run, inKernelCoalesceV41)
	}()
	awaitCoalescedQueue(t, p, 1)
	follower := make(chan error, 1)
	go func() { _, err := p.runCoalescedGenerate(context.Background(), run); follower <- err }()
	awaitCoalescedQueue(t, p, 2)
	open()
	select {
	case got := <-leader:
		if got != primary || closed.Load() < 1 {
			t.Fatal("leader escaped before cleanup or changed fatal identity")
		}
	case <-time.After(time.Second):
		t.Fatal("fatal leader stranded")
	}
	if err := awaitCoalescedError(t, follower); err != nil {
		t.Fatal(err)
	}
	if closed.Load() != 2 || calls != 2 || p.coalesceRunning || len(p.coalesceReady) != 0 {
		t.Fatal("fatal Flash cohort failed to hand off the queued Qwen cohort")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInKernelV41CoordinatorPreparationPanic(t *testing.T) {
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	lane, _ := newDecodeLaneOver(m, []int{1, 2}, 2, 0)
	primary := &struct{ stage string }{"prepare"}
	var closed, unstarted atomic.Int32
	requests := []*inKernelCoalesceRequest{
		v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			_, err := coalescedDecode(ctx, lane)
			return inKernelGenerateResult{}, err
		}),
		v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			s := m.NewSession()
			defer func() { s.Close(); closed.Add(1) }()
			panic(primary)
		}),
		v41CoordinatorRequest(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			unstarted.Add(1)
			return inKernelGenerateResult{}, nil
		}),
	}
	var got any
	func() {
		defer func() { got = recover() }()
		p.runV41DecodeCohort(requests, func(context.Context, []*decodeLane, *model.Model) (inKernelV41DecodeReceipt, error) {
			t.Fatal("fatal preparation reached decode")
			return inKernelV41DecodeReceipt{}, nil
		})
	}()
	if got != primary || closed.Load() != 2 || unstarted.Load() != 0 || lane.gen != 0 {
		t.Fatal("preparation panic changed identity, skipped cleanup or started later work")
	}
	for _, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; !errors.Is(out.err, errInKernelV41CohortAborted) {
			t.Fatalf("fatal preparation did not terminate every request: %v", out.err)
		}
	}
}

type v41CoordinatorOwnerProbe struct {
	compute.Backend
	owner *sync.Mutex
}

func (b *v41CoordinatorOwnerProbe) ForwardExecutionOwner() *sync.Mutex { return b.owner }

// No positive capability is advertised or execution initialized by these cases.
// fak-test:runtime fast est=5ms lane=default
func TestInKernelV41CoordinatorAdvisoryRefusal(t *testing.T) {
	m := &model.Model{Cfg: model.Config{ModelType: "deepseek_v41"}}
	for _, p := range []*InKernelPlanner{
		nil,
		{m: m, backend: compute.Default()},
		{m: m, batchDecode: true, backend: compute.Default()},
		{m: m, batchDecode: true, backend: &v41CoordinatorOwnerProbe{Backend: compute.Default()}},
		{m: m, batchDecode: true, backend: &v41CoordinatorOwnerProbe{Backend: compute.Default(), owner: &sync.Mutex{}}},
	} {
		_, err := p.runV41CoalescedGenerate(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			t.Fatal("missing prerequisites started a request")
			return inKernelGenerateResult{}, nil
		})
		if !errors.Is(err, model.ErrV41BatchUnavailable) {
			t.Fatalf("advisory refusal = %v", err)
		}
	}
}

// A selected lane error is not proof that a later panic came from cleanup.
// fak-test:runtime fast est=20ms lane=default
func TestInKernelV41CoordinatorPostDecodePanic(t *testing.T) {
	lanes, _ := v41DriverLanes(2, 2)
	m := model.NewSynthetic(tinyCfg())
	p := &InKernelPlanner{m: m}
	fault := errors.New("selected lane fault")
	primary := &struct{ stage string }{"unrelated post-decode work"}
	var closed atomic.Int32
	requests := make([]*inKernelCoalesceRequest, len(lanes))
	for i, lane := range lanes {
		lane.s = m.NewSession()
		requests[i] = v41CoordinatorRequest(context.Background(), func(ctx context.Context) (inKernelGenerateResult, error) {
			defer func() { lane.s.Close(); closed.Add(1) }()
			_, err := coalescedDecode(ctx, lane)
			if i == 0 {
				panic(primary)
			}
			return inKernelGenerateResult{}, err
		})
	}
	var got any
	func() {
		defer func() { got = recover() }()
		p.runV41DecodeCohort(requests, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
			return inKernelDecodeV41LanesWithOps(ctx, lanes, inKernelV41DecodeOps{
				batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
					out := v41DriverRefusal(input)
					out.Selected = true
					for i := range out.Lanes {
						out.Lanes[i].Err = fault
					}
					return out, fault
				},
				serial: func(*model.Session, int) ([]float32, error) { panic("selected replay") },
				wait:   func(context.Context) error { panic("selected retry") },
			})
		})
	}()
	if got != primary || closed.Load() != 2 {
		t.Fatalf("lane error suppressed unknown panic or escaped cleanup: %v closed=%d", got, closed.Load())
	}
	for _, req := range requests {
		<-req.done
		<-req.receiptReady
		if out := <-req.result; out.err != fault {
			t.Fatalf("unknown panic erased per-lane result: %v", out.err)
		}
	}
}
