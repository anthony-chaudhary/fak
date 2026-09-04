package breathgate

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestBasicAcquireRelease(t *testing.T) {
	gate := New(DefaultConfig())

	m0 := gate.Metrics()
	if m0.TotalAcquires != 0 || m0.ActiveHolders != 0 {
		t.Fatalf("expected initial zero metrics, got %+v", m0)
	}

	ctx := context.Background()
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	m1 := gate.Metrics()
	if m1.TotalAcquires != 1 || m1.ActiveHolders != 1 {
		t.Errorf("expected 1 acquire and 1 active holder, got %+v", m1)
	}

	gate.Release()

	m2 := gate.Metrics()
	if m2.TotalReleases != 1 || m2.ActiveHolders != 0 {
		t.Errorf("expected 1 release and 0 active holders, got %+v", m2)
	}
}

func TestNilContext(t *testing.T) {
	gate := New(DefaultConfig())

	err := gate.Acquire(nil)
	if !errors.Is(err, ErrNilContext) {
		t.Fatalf("expected ErrNilContext, got %v", err)
	}

	m := gate.Metrics()
	if m.ActiveHolders != 0 || m.TotalAcquires != 0 {
		t.Errorf("metrics mutated on nil context admission rejection: %+v", m)
	}
}

func TestPreCanceledContext(t *testing.T) {
	gate := New(DefaultConfig())

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := gate.Acquire(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	m := gate.Metrics()
	if m.ActiveHolders != 0 || m.TotalAcquires != 0 {
		t.Errorf("metrics mutated on canceled context admission rejection: %+v", m)
	}
}

func TestContextCancellationWhileWaiting(t *testing.T) {
	gate := New(Config{
		MaxConcurrent: 1,
		MinInterval:   0,
	})

	if err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("first acquire failed: %v", err)
	}

	ctxTimeout, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := gate.Acquire(ctxTimeout)
	elapsed := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
	if elapsed < 20*time.Millisecond {
		t.Errorf("Acquire returned prematurely after %v, expected at least ~20ms", elapsed)
	}

	// Active holders must still reflect only the first holder.
	if h := gate.Metrics().ActiveHolders; h != 1 {
		t.Errorf("expected active holders 1, got %d", h)
	}

	gate.Release()

	// Subsequent acquire should now succeed cleanly.
	if err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("subsequent acquire failed: %v", err)
	}
	gate.Release()
}

func TestCooldown(t *testing.T) {
	gate := New(Config{
		MaxConcurrent: 1,
		MinInterval:   0,
	})

	gate.Cooldown(40 * time.Millisecond)
	if !gate.IsCoolingDown() {
		t.Fatal("expected gate to be in cooldown")
	}

	m := gate.Metrics()
	if m.Cooldowns != 1 {
		t.Errorf("expected 1 cooldown recorded, got %d", m.Cooldowns)
	}

	start := time.Now()
	if err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("acquire after cooldown failed: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed < 30*time.Millisecond {
		t.Errorf("expected cooldown wait of at least 30ms, took %v", elapsed)
	}
	if gate.IsCoolingDown() {
		t.Error("gate should no longer be cooling down after wait expired")
	}

	gate.Release()
}

func TestThrottleDefaultBackoff(t *testing.T) {
	gate := New(Config{
		MaxConcurrent:     1,
		MinInterval:       0,
		BackoffOnThrottle: 50 * time.Millisecond,
	})

	gate.Throttle()
	if !gate.IsCoolingDown() {
		t.Fatal("expected gate to be in cooldown following throttle")
	}

	start := time.Now()
	if err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed < 35*time.Millisecond {
		t.Errorf("expected throttle backoff wait of at least 35ms, took %v", elapsed)
	}

	gate.Release()
}

func TestCooldownExtension(t *testing.T) {
	gate := New(Config{MaxConcurrent: 1})

	gate.Cooldown(30 * time.Millisecond)
	m1 := gate.Metrics()

	// Extending with longer cooldown should advance CooldownUntil.
	gate.Cooldown(100 * time.Millisecond)
	m2 := gate.Metrics()
	if !m2.CooldownUntil.After(m1.CooldownUntil) {
		t.Errorf("expected cooldown deadline extension: prev %v, curr %v", m1.CooldownUntil, m2.CooldownUntil)
	}

	// Applying shorter duration must NOT truncate existing cooldown.
	gate.Cooldown(10 * time.Millisecond)
	m3 := gate.Metrics()
	if m3.CooldownUntil.Before(m2.CooldownUntil) {
		t.Errorf("shorter cooldown truncated deadline: prev %v, curr %v", m2.CooldownUntil, m3.CooldownUntil)
	}
}

func TestCloseFailClosed(t *testing.T) {
	gate := New(Config{MaxConcurrent: 1})

	if err := gate.Acquire(context.Background()); err != nil {
		t.Fatalf("initial acquire failed: %v", err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	var waiterErr error
	go func() {
		defer wg.Done()
		waiterErr = gate.Acquire(context.Background())
	}()

	// Allow goroutine to enter wait state.
	time.Sleep(20 * time.Millisecond)

	if err := gate.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	wg.Wait()
	if !errors.Is(waiterErr, ErrClosed) {
		t.Errorf("expected waiter to receive ErrClosed, got %v", waiterErr)
	}

	// Subsequent acquire must immediately reject.
	err := gate.Acquire(context.Background())
	if !errors.Is(err, ErrClosed) {
		t.Errorf("expected ErrClosed on closed gate, got %v", err)
	}

	// Idempotent close.
	if err := gate.Close(); err != nil {
		t.Errorf("idempotent Close returned error: %v", err)
	}
}

func TestReleaseUnderflowProtection(t *testing.T) {
	gate := New(DefaultConfig())

	// Call Release multiple times without Acquire.
	gate.Release()
	gate.Release()

	m := gate.Metrics()
	if m.ActiveHolders != 0 {
		t.Errorf("expected 0 active holders, got %d", m.ActiveHolders)
	}
	if m.TotalReleases != 0 {
		t.Errorf("expected 0 total releases recorded for underflow calls, got %d", m.TotalReleases)
	}
}

func TestReset(t *testing.T) {
	gate := New(Config{MaxConcurrent: 1})

	gate.Cooldown(5 * time.Second)
	if !gate.IsCoolingDown() {
		t.Fatal("expected gate in cooldown")
	}

	gate.Reset()
	if gate.IsCoolingDown() {
		t.Fatal("expected gate cooldown cleared after Reset")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("acquire after reset failed: %v", err)
	}
	gate.Release()
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MaxConcurrent != 1 {
		t.Errorf("expected MaxConcurrent 1, got %d", cfg.MaxConcurrent)
	}
	if cfg.MinInterval <= 0 {
		t.Errorf("expected positive MinInterval, got %v", cfg.MinInterval)
	}
	if cfg.BackoffOnThrottle <= 0 {
		t.Errorf("expected positive BackoffOnThrottle, got %v", cfg.BackoffOnThrottle)
	}

	// Zero-value config must be sanitized to safe defaults by New.
	g := New(Config{})
	sanitized := g.Config()
	if sanitized.MaxConcurrent != 1 {
		t.Errorf("expected sanitized MaxConcurrent 1, got %d", sanitized.MaxConcurrent)
	}
	if sanitized.BackoffOnThrottle <= 0 {
		t.Errorf("expected positive default BackoffOnThrottle, got %v", sanitized.BackoffOnThrottle)
	}
}
