package gateway

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// fak-test:runtime fast est=2ms lane=default
func TestAdmissionRetiredQueueSlotsReleaseReferences(t *testing.T) {
	type acquireResult struct {
		lease *AdmissionLease
		err   error
	}
	startAcquire := func(ctl *AdmissionController, ctx context.Context, req SeqRequest) <-chan acquireResult {
		result := make(chan acquireResult, 1)
		go func() {
			lease, err := ctl.Acquire(ctx, req)
			result <- acquireResult{lease: lease, err: err}
		}()
		return result
	}
	awaitQueued := func(t *testing.T, queued <-chan struct{}) {
		t.Helper()
		select {
		case <-queued:
		case <-time.After(2 * time.Second):
			t.Fatal("Acquire never recorded its queued decision")
		}
	}
	awaitResult := func(t *testing.T, result <-chan acquireResult) acquireResult {
		t.Helper()
		select {
		case got := <-result:
			return got
		case <-time.After(2 * time.Second):
			t.Fatal("Acquire did not finish after its ownership transition")
			return acquireResult{}
		}
	}
	newHeldController := func(t *testing.T) (*AdmissionController, *AdmissionLease, <-chan struct{}) {
		t.Helper()
		ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 2, TokenBudget: 10})
		running, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "running", Tokens: 1})
		if err != nil || running == nil {
			t.Fatalf("running Acquire = (%v, %v), want lease", running, err)
		}
		t.Cleanup(running.Release)
		queued := make(chan struct{}, 4)
		ctl.SetClock(func() time.Time {
			select {
			case queued <- struct{}{}:
			default:
			}
			return time.Unix(1, 0)
		})
		return ctl, running, queued
	}
	assertEmpty := func(t *testing.T, ctl *AdmissionController) {
		t.Helper()
		if got := ctl.Stats(); got.Running != 0 || got.Waiting != 0 || got.TokensInUse != 0 || got.QueuedTokens != 0 {
			t.Fatalf("final live gauges = %+v, want zero", got)
		}
	}

	t.Run("canceled_waiter", func(t *testing.T) {
		ctl, running, queued := newHeldController(t)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		result := startAcquire(ctl, ctx, SeqRequest{TraceID: "canceled", Tokens: 3})
		awaitQueued(t, queued)
		// Acquiring the mutex after the queued receipt joins Acquire's initial
		// scheduling round. Holding it across cancel makes cancelAdmission, not
		// the initial canceled-waiter scan, retire this actual blocked request.
		ctl.mu.Lock()
		if len(ctl.waiting) != 1 || ctl.waiting[0].ctx != ctx {
			ctl.mu.Unlock()
			t.Fatal("queued request/context not owned by the real waiting array")
		}
		backing := ctl.waiting[:cap(ctl.waiting)]
		cancel()
		ctl.mu.Unlock()
		got := awaitResult(t, result)
		defer got.lease.Release()
		if got.lease != nil || !errors.Is(got.err, context.Canceled) {
			t.Fatalf("canceled Acquire = (%v, %v), want (nil, context.Canceled)", got.lease, got.err)
		}
		ctl.mu.Lock()
		retired := backing[0]
		ctl.mu.Unlock()
		if retired.ctx != nil || retired.ready != nil || retired.req.TraceID != "" {
			t.Fatalf("retired canceled slot retains request/context/ready: %+v", retired)
		}
		if st := ctl.Stats(); st.Running != 1 || st.Waiting != 0 || st.TokensInUse != 1 || st.QueuedTokens != 0 {
			t.Fatalf("cancellation changed live owner/accounting: %+v", st)
		}
		running.Release()
		assertEmpty(t, ctl)
	})

	t.Run("weighted_winner", func(t *testing.T) {
		ctl, running, queued := newHeldController(t)
		ctl.SetOrder(session.WeightedFair)
		followerCtx, cancelFollower := context.WithCancel(context.Background())
		defer cancelFollower()
		follower := startAcquire(ctl, followerCtx, SeqRequest{TraceID: "follower", SessionID: "follower", Tokens: 2, Priority: 10})
		awaitQueued(t, queued)
		winnerCtx, cancelWinner := context.WithCancel(context.Background())
		defer cancelWinner()
		winner := startAcquire(ctl, winnerCtx, SeqRequest{TraceID: "winner", SessionID: "winner", Tokens: 3, Priority: 0})
		awaitQueued(t, queued)
		ctl.mu.Lock()
		if len(ctl.waiting) != 2 || ctl.waiting[0].ctx != followerCtx || ctl.waiting[1].ctx != winnerCtx {
			ctl.mu.Unlock()
			t.Fatal("weighted waiters not present in original queue order")
		}
		backing := ctl.waiting[:cap(ctl.waiting)]
		ctl.mu.Unlock()
		running.Release()
		gotWinner := awaitResult(t, winner)
		defer gotWinner.lease.Release()
		if gotWinner.err != nil || gotWinner.lease == nil {
			t.Fatalf("weighted winner = (%v, %v), want lease", gotWinner.lease, gotWinner.err)
		}
		ctl.mu.Lock()
		retired := backing[1]
		liveFollower := len(ctl.waiting) == 1 && ctl.waiting[0].ctx == followerCtx && ctl.waiting[0].ready != nil
		ctl.mu.Unlock()
		if retired.ctx != nil || retired.ready != nil || retired.req.TraceID != "" {
			t.Fatalf("retired weighted slot retains request/context/ready: %+v", retired)
		}
		if !liveFollower {
			t.Fatal("retiring weighted winner changed the live follower")
		}
		if st := ctl.Stats(); st.Running != 1 || st.Waiting != 1 || st.TokensInUse != 3 || st.QueuedTokens != 2 {
			t.Fatalf("weighted promotion changed accounting: %+v", st)
		}
		gotWinner.lease.Release()
		gotFollower := awaitResult(t, follower)
		defer gotFollower.lease.Release()
		if gotFollower.err != nil || gotFollower.lease == nil {
			t.Fatalf("live follower = (%v, %v), want lease", gotFollower.lease, gotFollower.err)
		}
		gotFollower.lease.Release()
		assertEmpty(t, ctl)
	})
}
