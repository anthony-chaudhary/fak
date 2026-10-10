package agent

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms lane=default
func TestInKernelDeviceGateCanceledWaiterPreservesOwner(t *testing.T) {
	var gate inKernelDeviceGate
	gate.Lock()
	held := true
	defer func() {
		if held {
			gate.Unlock()
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- gate.LockContext(ctx) }()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("waiter error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled waiter did not return while holder still owned the gate")
	}
	if len(gate.slot) != 1 {
		t.Fatal("canceled waiter released the holder's slot")
	}
	gate.Unlock()
	held = false
	next, cancelNext := context.WithTimeout(context.Background(), time.Second)
	defer cancelNext()
	if err := gate.LockContext(next); err != nil {
		t.Fatalf("next waiter cannot acquire after release: %v", err)
	}
	gate.Unlock()
	// A canceled or expired request must not seize even an initially free slot.
	if err := gate.LockContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("already canceled = %v", err)
	}
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()
	if err := gate.LockContext(expired); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired = %v", err)
	}
	if len(gate.slot) != 0 {
		t.Fatal("canceled free-slot acquisition leaked ownership")
	}
	gate.Lock()
	gate.Unlock()
}

// fak-test:runtime fast est=20ms lane=default
func TestInKernelDeviceGateSerializesRequestAndCohortOwners(t *testing.T) {
	var gate inKernelDeviceGate
	var active, overlaps, completed atomic.Int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(cohort bool) {
			defer wg.Done()
			<-start
			for j := 0; j < 32; j++ {
				if cohort {
					gate.Lock()
				} else if err := gate.LockContext(context.Background()); err != nil {
					overlaps.Add(1)
					return
				}
				if active.Add(1) != 1 {
					overlaps.Add(1)
				}
				runtime.Gosched()
				active.Add(-1)
				completed.Add(1)
				gate.Unlock()
			}
		}(i%2 == 0)
	}
	close(start)
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("mixed owners failed to make progress")
	}
	if overlaps.Load() != 0 || active.Load() != 0 || completed.Load() != 256 {
		t.Fatalf("overlaps=%d active=%d completed=%d", overlaps.Load(), active.Load(), completed.Load())
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelCompleteCancellationReturnsBeforeDeviceOwnerReleases(t *testing.T) {
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref unavailable")
	}
	m := model.NewSynthetic(tinyConcurrencyConfig())
	p := NewInKernelPlanner(m, loadProbeTok(t), "queued-device-cancel", false, base, false)
	p.concurrencyProfile = newConcurrencyProfiler()
	if !p.requiresDeviceSerialization() || p.coalescesQwenDecode() {
		t.Fatal("fixture must use serialized device route")
	}
	p.devMu.Lock()
	held := true
	defer func() {
		if held {
			p.devMu.Unlock()
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var verdicts atomic.Int32
	ctx = p.WithExecutionDeadlineAdmission(ctx, func(context.Context, int, int, int) error { verdicts.Add(1); return nil })
	done := make(chan error, 1)
	go func() {
		_, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "wait for the device"}}, nil, WithMaxTokens(1))
		done <- err
	}()
	// The existing profiler marks the exact pre-lock boundary, avoiding a sleep
	// that could cancel during prompt preparation instead of the device wait.
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		p.concurrencyProfile.mu.Lock()
		arrived := len(p.concurrencyProfile.phases) > 0
		p.concurrencyProfile.mu.Unlock()
		if arrived {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("Complete returned before reaching the held gate: %v", err)
		case <-timer.C:
			cancel()
			p.devMu.Unlock()
			held = false
			select {
			case <-done:
			case <-time.After(time.Second):
			}
			t.Fatal("Complete did not reach device wait")
		case <-tick.C:
		}
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Complete error = %v", err)
		}
	case <-time.After(time.Second):
		p.devMu.Unlock()
		held = false
		select {
		case <-done:
		case <-time.After(time.Second):
		}
		t.Fatal("Complete ignored cancellation while waiting for the device")
	}
	if verdicts.Load() != 0 {
		t.Fatal("canceled queued request reached owned-session admission")
	}
	if len(p.devMu.slot) != 1 {
		t.Fatal("Complete released somebody else's device slot")
	}
	p.concurrencyProfile.mu.Lock()
	entered := p.concurrencyProfile.open
	p.concurrencyProfile.mu.Unlock()
	if entered != 0 {
		t.Fatal("canceled queued request entered forward execution")
	}
	p.devMu.Unlock()
	held = false
	next, cancelNext := context.WithTimeout(context.Background(), time.Second)
	defer cancelNext()
	if err := p.devMu.LockContext(next); err != nil {
		t.Fatalf("device ownership leaked after cancellation: %v", err)
	}
	p.devMu.Unlock()
}
