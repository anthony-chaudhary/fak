package agent

// inkernel_warm_state_test.go — CW-27 (fak#13345): the planner-owned startup KV-cache
// warm lifecycle. These tests witness ONLY the lifecycle contract — the bounded
// single-profile coalescing slot, explicit cancellation, at-most-once release callback,
// and idempotent Release/Reset — not any cache warming. No model, no tokenizer, no cache
// is constructed: the state's zero value is a usable idle slot (the same idiom as
// inKernelTurnTaxState and moeResidencyState), which is itself part of the contract.

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestInKernelWarmStateLifecycle(t *testing.T) {
	t.Run("idle zero value begins a profile", func(t *testing.T) {
		var s InKernelWarmState
		if s.Active() {
			t.Fatal("zero-value warm state reports Active before Begin")
		}
		ctx, coalesced, err := s.Begin(context.Background(), "startup-prefix")
		if err != nil {
			t.Fatalf("Begin on idle zero value: %v", err)
		}
		if coalesced {
			t.Fatal("first Begin must start work, not coalesce")
		}
		if ctx == nil {
			t.Fatal("Begin returned nil context")
		}
		if !s.Active() {
			t.Fatal("state not Active after Begin")
		}
		if got := s.ProfileID(); got != "startup-prefix" {
			t.Fatalf("ProfileID = %q, want startup-prefix", got)
		}
		if err := ctx.Err(); err != nil {
			t.Fatalf("fresh warm context already errored: %v", err)
		}
		s.Release()
	})

	t.Run("same profile coalesces onto the active slot", func(t *testing.T) {
		var s InKernelWarmState
		first, coalesced, err := s.Begin(context.Background(), "p")
		if err != nil || coalesced {
			t.Fatalf("first Begin = (coalesced=%v, err=%v), want (false, nil)", coalesced, err)
		}
		second, coalesced, err := s.Begin(context.Background(), "p")
		if err != nil {
			t.Fatalf("same-profile Begin: %v", err)
		}
		if !coalesced {
			t.Fatal("same-profile Begin must coalesce, not start new work")
		}
		if second != first {
			t.Fatal("coalesced Begin returned a different context than the active slot")
		}
		if got := s.CoalescedCount(); got != 1 {
			t.Fatalf("CoalescedCount = %d, want 1", got)
		}
		if got := s.RefusedCount(); got != 0 {
			t.Fatalf("RefusedCount = %d, want 0 (same profile never refuses)", got)
		}
		s.Release()
	})

	t.Run("different profile is refused by the bounded slot", func(t *testing.T) {
		var s InKernelWarmState
		if _, _, err := s.Begin(context.Background(), "a"); err != nil {
			t.Fatalf("Begin a: %v", err)
		}
		ctx, coalesced, err := s.Begin(context.Background(), "b")
		if !errors.Is(err, ErrInKernelWarmProfileBusy) {
			t.Fatalf("Begin b err = %v, want ErrInKernelWarmProfileBusy", err)
		}
		if coalesced || ctx != nil {
			t.Fatalf("refused Begin returned (coalesced=%v, ctx=%v), want (false, nil)", coalesced, ctx)
		}
		if got := s.ProfileID(); got != "a" {
			t.Fatalf("ProfileID after refusal = %q, want the incumbent a", got)
		}
		if got := s.RefusedCount(); got != 1 {
			t.Fatalf("RefusedCount = %d, want 1", got)
		}
		s.Release()
	})

	t.Run("release callback fires at most once on Complete", func(t *testing.T) {
		var s InKernelWarmState
		var calls atomic.Int64
		s.BindRelease(func() { calls.Add(1) })
		if _, _, err := s.Begin(context.Background(), "p"); err != nil {
			t.Fatalf("Begin: %v", err)
		}
		s.Complete()
		s.Complete() // idempotent
		if got := calls.Load(); got != 1 {
			t.Fatalf("release callback fired %d times on Complete, want exactly 1", got)
		}
		if s.Active() {
			t.Fatal("state still Active after Complete")
		}
		s.Release() // must not re-fire
		if got := calls.Load(); got != 1 {
			t.Fatalf("release callback fired %d times after Release, want exactly 1", got)
		}
	})

	t.Run("release cancels the in-flight warm context", func(t *testing.T) {
		var s InKernelWarmState
		ctx, _, err := s.Begin(context.Background(), "p")
		if err != nil {
			t.Fatalf("Begin: %v", err)
		}
		select {
		case <-ctx.Done():
			t.Fatal("warm context done before Release")
		default:
		}
		s.Release()
		select {
		case <-ctx.Done():
		case <-time.After(2 * time.Second):
			t.Fatal("release did not cancel the in-flight warm context")
		}
		if !errors.Is(ctx.Err(), context.Canceled) {
			t.Fatalf("warm ctx err = %v, want context.Canceled", ctx.Err())
		}
	})

	t.Run("release fires the callback once and is idempotent", func(t *testing.T) {
		var s InKernelWarmState
		var calls atomic.Int64
		s.BindRelease(func() { calls.Add(1) })
		if _, _, err := s.Begin(context.Background(), "p"); err != nil {
			t.Fatalf("Begin: %v", err)
		}
		s.Release()
		s.Release()
		if got := calls.Load(); got != 1 {
			t.Fatalf("release callback fired %d times, want exactly 1", got)
		}
		if s.Active() {
			t.Fatal("state Active after Release")
		}
		if got := s.ProfileID(); got != "" {
			t.Fatalf("ProfileID after Release = %q, want empty", got)
		}
	})

	t.Run("Begin after Release is refused", func(t *testing.T) {
		var s InKernelWarmState
		s.Release()
		if _, _, err := s.Begin(context.Background(), "p"); !errors.Is(err, ErrInKernelWarmReleased) {
			t.Fatalf("Begin after Release err = %v, want ErrInKernelWarmReleased", err)
		}
	})

	t.Run("Reset returns a released slot to a usable idle state", func(t *testing.T) {
		var s InKernelWarmState
		s.BindRelease(func() {})
		if _, _, err := s.Begin(context.Background(), "a"); err != nil {
			t.Fatalf("Begin: %v", err)
		}
		s.Release()
		if _, _, err := s.Begin(context.Background(), "b"); !errors.Is(err, ErrInKernelWarmReleased) {
			t.Fatalf("Begin before Reset err = %v, want ErrInKernelWarmReleased", err)
		}
		s.Reset()
		ctx, coalesced, err := s.Begin(context.Background(), "b")
		if err != nil || coalesced {
			t.Fatalf("Begin after Reset = (coalesced=%v, err=%v), want (false, nil)", coalesced, err)
		}
		if ctx == nil {
			t.Fatal("Begin after Reset returned nil context")
		}
		if got := s.ProfileID(); got != "b" {
			t.Fatalf("ProfileID after Reset+Begin = %q, want b", got)
		}
		s.Release()
	})

	t.Run("completion refuses a same-profile restart until Reset", func(t *testing.T) {
		var s InKernelWarmState
		if _, _, err := s.Begin(context.Background(), "p"); err != nil {
			t.Fatalf("Begin: %v", err)
		}
		s.Complete()
		// The completed slot is kept (not idle): a stale second startup must not reopen it.
		if _, _, err := s.Begin(context.Background(), "p"); !errors.Is(err, ErrInKernelWarmProfileBusy) {
			t.Fatalf("Begin on completed slot err = %v, want ErrInKernelWarmProfileBusy", err)
		}
		s.Reset()
		if _, coalesced, err := s.Begin(context.Background(), "p"); err != nil || coalesced {
			t.Fatalf("Begin after Reset = (coalesced=%v, err=%v), want (false, nil)", coalesced, err)
		}
		s.Release()
	})

	t.Run("concurrent same-profile Begins coalesce onto one slot", func(t *testing.T) {
		var s InKernelWarmState
		const n = 32
		var wg sync.WaitGroup
		var started atomic.Int64
		results := make([]bool, n)
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				if _, coalesced, err := s.Begin(context.Background(), "p"); err == nil {
					results[i] = coalesced
					if !coalesced {
						started.Add(1)
					}
				}
			}(i)
		}
		wg.Wait()
		if got := started.Load(); got != 1 {
			t.Fatalf("concurrent same-profile Begins started %d workers, want exactly 1", got)
		}
		if got := s.CoalescedCount(); got != n-1 {
			t.Fatalf("CoalescedCount = %d, want %d", got, n-1)
		}
		s.Release()
	})

	t.Run("planner accessor aliases the owned state and shutdown delegates", func(t *testing.T) {
		p := &InKernelPlanner{}
		state := p.StartupWarmState()
		if state == nil {
			t.Fatal("StartupWarmState returned nil for a non-nil planner")
		}
		var calls atomic.Int64
		state.BindRelease(func() { calls.Add(1) })
		if _, _, err := state.Begin(context.Background(), "p"); err != nil {
			t.Fatalf("Begin via planner state: %v", err)
		}
		p.ReleaseStartupWarm()
		if got := calls.Load(); got != 1 {
			t.Fatalf("planner ReleaseStartupWarm fired callback %d times, want 1", got)
		}
		if state.Active() {
			t.Fatal("planner state still Active after ReleaseStartupWarm")
		}
		// A nil planner must be a safe no-op, matching the package's nil-receiver idiom.
		var nilPlanner *InKernelPlanner
		nilPlanner.ReleaseStartupWarm()
		if nilPlanner.StartupWarmState() != nil {
			t.Fatal("nil planner returned a non-nil state")
		}
	})
}
