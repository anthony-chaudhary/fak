package agent

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func queuedCancelRequest() *inKernelCoalesceRequest {
	return &inKernelCoalesceRequest{drain: make(chan struct{}, 1)}
}

// fak-test:runtime fast est=10ms lane=default
func TestCoalescedWithdrawalPreservesQueueAndActiveOwner(t *testing.T) {
	for _, index := range []int{0, 1, 2} {
		a, b, c := queuedCancelRequest(), queuedCancelRequest(), queuedCancelRequest()
		backing := []*inKernelCoalesceRequest{a, b, c}
		target := backing[index]
		want := append([]*inKernelCoalesceRequest(nil), backing[:index]...)
		want = append(want, backing[index+1:]...)
		p := &InKernelPlanner{coalesceReady: backing, coalesceRunning: true}
		if !p.withdrawCoalescedRequest(target, false) {
			t.Fatal("queued follower was not withdrawn")
		}
		if len(p.coalesceReady) != 2 || p.coalesceReady[0] != want[0] || p.coalesceReady[1] != want[1] {
			t.Fatalf("removal at %d changed survivor FIFO", index)
		}
		if backing[2] != nil {
			t.Fatal("removed tail retains a request")
		}
		if !p.coalesceRunning {
			t.Fatal("withdrawal stole the active drainer's ownership")
		}
		for _, req := range []*inKernelCoalesceRequest{a, b, c} {
			if len(req.drain) != 0 {
				t.Fatal("withdrawal manufactured a baton")
			}
		}
	}
	req := queuedCancelRequest()
	p := &InKernelPlanner{coalesceReady: []*inKernelCoalesceRequest{req}, coalesceRunning: true}
	if !p.withdrawCoalescedRequest(req, false) || len(p.coalesceReady) != 0 || !p.coalesceRunning {
		t.Fatal("empty queue without baton must retain active ownership")
	}
	if p.withdrawCoalescedRequest(req, false) {
		t.Fatal("selected/absent request must not be withdrawn")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCoalescedWithdrawalTransfersOnlyOwnedBaton(t *testing.T) {
	for _, received := range []bool{false, true} {
		for _, survivor := range []bool{false, true} {
			req, next := queuedCancelRequest(), queuedCancelRequest()
			p := &InKernelPlanner{coalesceReady: []*inKernelCoalesceRequest{req}, coalesceRunning: true}
			if survivor {
				p.coalesceReady = append(p.coalesceReady, next)
			}
			req.drain <- struct{}{}
			if received {
				<-req.drain
			}
			if !p.withdrawCoalescedRequest(req, received) {
				t.Fatal("baton owner remained queued")
			}
			if len(req.drain) != 0 {
				t.Fatal("withdrawn owner retained baton")
			}
			if survivor {
				if !p.coalesceRunning || len(p.coalesceReady) != 1 || p.coalesceReady[0] != next || len(next.drain) != 1 {
					t.Fatal("baton did not transfer to next queue head")
				}
			} else {
				if p.coalesceRunning || len(p.coalesceReady) != 0 {
					t.Fatal("empty baton-owned queue did not release leadership")
				}
				var calls int
				p.coalesceReadyHook = func() {}
				_, err := p.runCoalescedGenerate(context.Background(), func(context.Context) (inKernelGenerateResult, error) { calls++; return inKernelGenerateResult{}, nil })
				if err != nil || calls != 1 || p.coalesceRunning {
					t.Fatalf("later arrival could not elect and finish: calls=%d running=%t err=%v", calls, p.coalesceRunning, err)
				}
			}
		}
	}
}

func awaitCoalescedQueue(t *testing.T, p *InKernelPlanner, n int) {
	t.Helper()
	timeout := time.NewTimer(time.Second)
	defer timeout.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		p.coalesceMu.Lock()
		got := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if got == n {
			return
		}
		select {
		case <-timeout.C:
			t.Fatalf("queue length=%d, want %d", got, n)
		case <-tick.C:
		}
	}
}

func awaitCoalescedError(t *testing.T, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(time.Second):
		t.Fatal("coalesced request did not return")
		return nil
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestCoalescedCanceledFollowerReturnsBeforeActiveCohort(t *testing.T) {
	p := &InKernelPlanner{coalesceReadyHook: func() {}}
	activeStarted := make(chan struct{})
	releaseActive := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(releaseActive) }) }
	defer release()
	activeDone := make(chan error, 1)
	go func() {
		_, err := p.runCoalescedGenerate(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			close(activeStarted)
			<-releaseActive
			return inKernelGenerateResult{}, nil
		})
		activeDone <- err
	}()
	select {
	case <-activeStarted:
	case <-time.After(time.Second):
		t.Fatal("active cohort did not start")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var canceledRuns, survivorRuns atomic.Int32
	canceledDone := make(chan error, 1)
	go func() {
		_, err := p.runCoalescedGenerate(ctx, func(ctx context.Context) (inKernelGenerateResult, error) {
			canceledRuns.Add(1)
			return inKernelGenerateResult{}, ctx.Err()
		})
		canceledDone <- err
	}()
	awaitCoalescedQueue(t, p, 1)
	cancel()
	if err := awaitCoalescedError(t, canceledDone); !errors.Is(err, context.Canceled) {
		t.Fatalf("follower = %v", err)
	}
	if canceledRuns.Load() != 0 {
		t.Fatal("withdrawn follower's run closure started")
	}
	p.coalesceMu.Lock()
	pending, running := len(p.coalesceReady), p.coalesceRunning
	p.coalesceMu.Unlock()
	if pending != 0 || !running {
		t.Fatal("queued cancellation altered active cohort ownership")
	}
	select {
	case <-activeDone:
		t.Fatal("queued cancellation preempted active owner")
	default:
	}
	survivorDone := make(chan error, 1)
	go func() {
		_, err := p.runCoalescedGenerate(context.Background(), func(context.Context) (inKernelGenerateResult, error) {
			survivorRuns.Add(1)
			return inKernelGenerateResult{}, nil
		})
		survivorDone <- err
	}()
	awaitCoalescedQueue(t, p, 1)
	release()
	if err := awaitCoalescedError(t, activeDone); err != nil {
		t.Fatal(err)
	}
	if err := awaitCoalescedError(t, survivorDone); err != nil {
		t.Fatal(err)
	}
	if survivorRuns.Load() != 1 || canceledRuns.Load() != 0 {
		t.Fatalf("survivor runs=%d canceled runs=%d", survivorRuns.Load(), canceledRuns.Load())
	}
	p.coalesceMu.Lock()
	pending, running = len(p.coalesceReady), p.coalesceRunning
	p.coalesceMu.Unlock()
	if pending != 0 || running {
		t.Fatal("coalescer did not return to idle")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestCoalescedSelectedCancellationWaitsForResultAndCleanupReceipt(t *testing.T) {
	// Stand in for a drainer to control selection and the two publication barriers
	// independently. No session or shared device operation is preempted here.
	p := &InKernelPlanner{coalesceRunning: true}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan inKernelCoalesceResult, 1)
	finished := make(chan struct{})
	go func() {
		result, err := p.runCoalescedGenerate(ctx, func(context.Context) (inKernelGenerateResult, error) {
			t.Error("test drainer must own execution")
			return inKernelGenerateResult{}, nil
		})
		done <- inKernelCoalesceResult{result: result, err: err}
		close(finished)
	}()
	awaitCoalescedQueue(t, p, 1)
	p.coalesceMu.Lock()
	req := p.coalesceReady[0]
	p.coalesceReady[0] = nil
	p.coalesceReady = p.coalesceReady[:0]
	p.coalesceMu.Unlock()
	var resultOnce, receiptOnce sync.Once
	publishResult := func() {
		resultOnce.Do(func() { req.result <- inKernelCoalesceResult{err: context.Canceled} })
	}
	publishReceipt := func() {
		receiptOnce.Do(func() {
			req.receipt = InKernelBatchReceipt{CohortID: 1}
			close(req.receiptReady)
		})
	}
	defer func() {
		publishResult()
		publishReceipt()
		select {
		case <-finished:
		case <-time.After(time.Second):
			t.Error("selected waiter did not finish after test cleanup")
		}
	}()
	// The production selector removes requests under this same lock. Once absent,
	// cancellation cannot release Complete's resources before selected work ends.
	if p.withdrawCoalescedRequest(req, false) {
		t.Fatal("selected request was withdrawn")
	}
	cancel()
	select {
	case err := <-done:
		t.Fatalf("selected request returned before result: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	publishResult()
	select {
	case err := <-done:
		t.Fatalf("selected request returned before cleanup receipt: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	publishReceipt()
	select {
	case out := <-done:
		if !errors.Is(out.err, context.Canceled) || out.result.batchReceipt.CohortID != 1 {
			t.Fatalf("selected result = %v, receipt = %+v", out.err, out.result.batchReceipt)
		}
	case <-time.After(time.Second):
		t.Fatal("selected request did not return after both publication barriers")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCoalescedDeviceGateCancellation(t *testing.T) {
	for _, kind := range []inKernelCoalesceKind{inKernelCoalesceQwen, inKernelCoalesceV41} {
		for _, allCanceled := range []bool{true, false} {
			name := "qwen"
			if kind == inKernelCoalesceV41 {
				name = "v41"
			}
			if allCanceled {
				name += "/all-canceled"
			} else {
				name += "/live-sibling"
			}
			t.Run(name, func(t *testing.T) {
				p := &InKernelPlanner{coalesceRunning: true}
				p.devMu.Lock()
				var releaseOnce sync.Once
				release := func() { releaseOnce.Do(p.devMu.Unlock) }
				defer release()
				var runs, liveRuns atomic.Int32
				contexts := make([]context.Context, 2)
				cancels := make([]context.CancelFunc, 2)
				requests := make([]*inKernelCoalesceRequest, 2)
				for i := range requests {
					contexts[i], cancels[i] = context.WithCancel(context.Background())
					defer cancels[i]()
					requests[i] = &inKernelCoalesceRequest{
						kind: kind, ctx: contexts[i],
						prepared: make(chan *decodeLane, 1), proceed: make(chan error, 1),
						result: make(chan inKernelCoalesceResult, 1), done: make(chan struct{}),
						drain: make(chan struct{}, 1), receiptReady: make(chan struct{}),
						run: func(ctx context.Context) (inKernelGenerateResult, error) {
							runs.Add(1)
							if err := ctx.Err(); err != nil {
								return inKernelGenerateResult{}, err
							}
							liveRuns.Add(1)
							return inKernelGenerateResult{}, nil
						},
					}
				}
				p.coalesceReady = append([]*inKernelCoalesceRequest(nil), requests...)
				drained := make(chan struct{})
				go func() { p.drainCoalescedGenerates(); close(drained) }()
				defer func() {
					release()
					select {
					case <-drained:
					case <-time.After(time.Second):
						t.Error("drainer did not join after cleanup")
					}
				}()
				awaitCoalescedQueue(t, p, 0) // Selected, but the device owner still holds its slot.
				cancels[0]()
				if allCanceled {
					cancels[1]()
					select {
					case <-drained:
					case <-time.After(time.Second):
						t.Fatal("canceled selected cohort waited for unrelated device owner")
					}
					if runs.Load() != 0 {
						t.Fatal("all-canceled cohort invoked a run closure")
					}
					// A canceled cohort must never unlock the unrelated owner's slot.
					if len(p.devMu.slot) != 1 {
						t.Fatal("canceled cohort released unrelated device owner")
					}
				} else {
					select {
					case <-drained:
						t.Fatal("one canceled participant aborted its live sibling")
					case <-time.After(10 * time.Millisecond):
					}
					release()
					select {
					case <-drained:
					case <-time.After(time.Second):
						t.Fatal("live sibling did not finish after device release")
					}
					if liveRuns.Load() != 1 {
						t.Fatalf("live runs = %d, want 1", liveRuns.Load())
					}
				}
				for i, req := range requests {
					select {
					case <-req.done:
					default:
						t.Fatal("drainer returned before participant cleanup")
					}
					select {
					case <-req.receiptReady:
					default:
						t.Fatal("drainer returned before cleanup receipt publication")
					}
					out := <-req.result
					_, err := finishCoalescedRequest(req, out)
					if want := contexts[i].Err(); !errors.Is(err, want) {
						t.Fatalf("participant %d error = %v, want %v", i, err, want)
					}
					if req.closes.Load() != 0 {
						t.Fatal("sessionless waiter reported a close")
					}
				}
				p.coalesceMu.Lock()
				running, pending := p.coalesceRunning, len(p.coalesceReady)
				p.coalesceMu.Unlock()
				if running || pending != 0 {
					t.Fatal("completed cohort retained drainer ownership")
				}
				release()
				if err := p.devMu.LockContext(context.Background()); err != nil {
					t.Fatal(err)
				}
				p.devMu.Unlock()
			})
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCoalescedMixedCancellationReturnsBeforeOwner(t *testing.T) {
	for _, kind := range []inKernelCoalesceKind{inKernelCoalesceQwen, inKernelCoalesceV41} {
		name := "qwen"
		if kind == inKernelCoalesceV41 {
			name = "v41"
		}
		t.Run(name, func(t *testing.T) {
			p := &InKernelPlanner{coalesceRunning: true}
			p.devMu.Lock()
			var once sync.Once
			release := func() { once.Do(p.devMu.Unlock) }
			defer release()
			canceled, cancel := context.WithCancel(context.Background())
			defer cancel()
			order := make(chan int, 3)
			requests := make([]*inKernelCoalesceRequest, 3)
			for i := range requests {
				ctx := context.Background()
				if i == 1 {
					ctx = canceled
				}
				id := i
				requests[i] = &inKernelCoalesceRequest{kind: kind, ctx: ctx, prepared: make(chan *decodeLane, 1), proceed: make(chan error, 1), result: make(chan inKernelCoalesceResult, 1), done: make(chan struct{}), drain: make(chan struct{}, 1), receiptReady: make(chan struct{}), enqueued: time.Now(), run: func(ctx context.Context) (inKernelGenerateResult, error) {
					order <- id
					return inKernelGenerateResult{}, ctx.Err()
				}}
			}
			p.coalesceReady = append([]*inKernelCoalesceRequest(nil), requests...)
			drained := make(chan struct{})
			go func() { p.drainCoalescedGenerates(); close(drained) }()
			defer func() {
				release()
				select {
				case <-drained:
				case <-time.After(time.Second):
					t.Error("drainer did not join")
				}
			}()
			awaitCoalescedQueue(t, p, 0)
			cancel()
			select {
			case <-requests[1].receiptReady:
			case <-time.After(time.Second):
				t.Fatal("canceled selected member waited behind unrelated owner despite live siblings")
			}
			select {
			case <-requests[1].done:
			default:
				t.Fatal("receipt preceded canceled completion")
			}
			out := <-requests[1].result
			if !errors.Is(out.err, context.Canceled) {
				t.Fatalf("canceled member error: %v", out.err)
			}
			if requests[1].receipt != (InKernelBatchReceipt{}) || requests[1].closes.Load() != 0 {
				t.Fatal("pre-start cancellation claimed execution or cleanup")
			}
			if len(p.devMu.slot) != 1 {
				t.Fatal("cancellation released foreign owner")
			}
			select {
			case id := <-order:
				t.Fatalf("request %d ran under foreign owner", id)
			default:
			}
			select {
			case <-drained:
				t.Fatal("cancellation abandoned live siblings")
			default:
			}
			release()
			select {
			case <-drained:
			case <-time.After(time.Second):
				t.Fatal("live siblings did not finish")
			}
			for _, want := range []int{0, 2} {
				select {
				case got := <-order:
					if got != want {
						t.Fatalf("survivor order %d want %d", got, want)
					}
				default:
					t.Fatal("missing live sibling")
				}
			}
			select {
			case id := <-order:
				t.Fatalf("canceled or duplicate run %d", id)
			default:
			}
			for _, i := range []int{0, 2} {
				select {
				case <-requests[i].receiptReady:
				default:
					t.Fatal("survivor receipt missing")
				}
			}
			p.coalesceMu.Lock()
			running, pending := p.coalesceRunning, len(p.coalesceReady)
			p.coalesceMu.Unlock()
			if running || pending != 0 {
				t.Fatal("cohort retained leadership")
			}
			if err := p.devMu.LockContext(context.Background()); err != nil {
				t.Fatal(err)
			}
			p.devMu.Unlock()
		})
	}
}
