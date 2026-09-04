package breathgate

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// simClock provides a thread-safe deterministic simulated clock for testing.
type simClock struct {
	mu  sync.Mutex
	now time.Time
}

func newSimClock(start time.Time) *simClock {
	return &simClock{now: start}
}

func (c *simClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *simClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func (c *simClock) Sleep(ctx context.Context, d time.Duration) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.Advance(d)
	return nil
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MinInterval != 100*time.Millisecond {
		t.Fatalf("expected MinInterval 100ms, got %v", cfg.MinInterval)
	}
	if cfg.MaxInterval != 500*time.Millisecond {
		t.Fatalf("expected MaxInterval 500ms, got %v", cfg.MaxInterval)
	}
	if cfg.Cooldown != 5*time.Second {
		t.Fatalf("expected Cooldown 5s, got %v", cfg.Cooldown)
	}
	if cfg.Jitter != 0.10 {
		t.Fatalf("expected Jitter 0.10, got %v", cfg.Jitter)
	}
	if cfg.BurstLimit != 20 {
		t.Fatalf("expected BurstLimit 20, got %d", cfg.BurstLimit)
	}
}

func TestNewGateConfigSanitization(t *testing.T) {
	cfg := Config{
		MinInterval: -10 * time.Millisecond,
		MaxInterval: -20 * time.Millisecond,
		Cooldown:    -5 * time.Second,
		Jitter:      -0.5,
		BurstLimit:  -3,
	}
	g := New(cfg)
	c := g.Config()

	if c.MinInterval != 0 {
		t.Errorf("expected MinInterval 0, got %v", c.MinInterval)
	}
	if c.MaxInterval != 0 {
		t.Errorf("expected MaxInterval 0, got %v", c.MaxInterval)
	}
	if c.Cooldown != 0 {
		t.Errorf("expected Cooldown 0, got %v", c.Cooldown)
	}
	if c.Jitter != 0.0 {
		t.Errorf("expected Jitter 0.0, got %v", c.Jitter)
	}
	if c.BurstLimit != 0 {
		t.Errorf("expected BurstLimit 0, got %d", c.BurstLimit)
	}
	if c.BurstWindow <= 0 {
		t.Errorf("expected positive default BurstWindow, got %v", c.BurstWindow)
	}

	// MaxInterval less than MinInterval should clamp to MinInterval.
	g2 := New(Config{
		MinInterval: 200 * time.Millisecond,
		MaxInterval: 50 * time.Millisecond,
		Jitter:      2.5, // should clamp to 1.0
	})
	c2 := g2.Config()
	if c2.MaxInterval != 200*time.Millisecond {
		t.Errorf("expected MaxInterval clamped to MinInterval 200ms, got %v", c2.MaxInterval)
	}
	if c2.Jitter != 1.0 {
		t.Errorf("expected Jitter clamped to 1.0, got %v", c2.Jitter)
	}
}

func TestGate_Check_InitialState(t *testing.T) {
	g := New(DefaultConfig())
	if !g.Check() {
		t.Fatal("expected Check() == true on freshly created Gate")
	}
	if rem := g.Remaining(); rem != 0 {
		t.Fatalf("expected Remaining() == 0, got %v", rem)
	}
	stats := g.Stats()
	if stats.TotalTurns != 0 || stats.InCooldown || stats.CooldownCount != 0 {
		t.Fatalf("unexpected initial stats: %+v", stats)
	}
}

func TestGate_Check_And_RecordTurn_Pacing(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 100 * time.Millisecond,
		MaxInterval: 100 * time.Millisecond,
		Jitter:      0,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	// Initial check: should pass.
	if !g.Check() {
		t.Fatal("expected initial Check to pass")
	}

	// Turn 1 executes.
	g.RecordTurn()

	// Immediately after turn 1: Check should fail.
	if g.Check() {
		t.Fatal("expected Check to fail immediately after RecordTurn")
	}
	if rem := g.Remaining(); rem != 100*time.Millisecond {
		t.Fatalf("expected Remaining 100ms, got %v", rem)
	}

	// Advance clock by 50ms: Check should still fail.
	clk.Advance(50 * time.Millisecond)
	if g.Check() {
		t.Fatal("expected Check to fail at 50ms")
	}
	if rem := g.Remaining(); rem != 50*time.Millisecond {
		t.Fatalf("expected Remaining 50ms, got %v", rem)
	}

	// Advance clock by remaining 50ms (total 100ms): Check should succeed.
	clk.Advance(50 * time.Millisecond)
	if !g.Check() {
		t.Fatal("expected Check to pass after 100ms")
	}
	if rem := g.Remaining(); rem != 0 {
		t.Fatalf("expected Remaining 0, got %v", rem)
	}

	// Turn 2 executes.
	g.RecordTurn()
	if g.Check() {
		t.Fatal("expected Check to fail immediately after Turn 2")
	}

	stats := g.Stats()
	if stats.TotalTurns != 2 {
		t.Fatalf("expected TotalTurns 2, got %d", stats.TotalTurns)
	}
}

func TestGate_Wait_Pacing(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 100 * time.Millisecond,
		MaxInterval: 100 * time.Millisecond,
		Jitter:      0,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	ctx := context.Background()

	// First Wait should return immediately with 0 delay.
	if err := g.Wait(ctx); err != nil {
		t.Fatalf("unexpected error on first Wait: %v", err)
	}
	g.RecordTurn()

	// Second Wait immediately after should sleep for 100ms.
	tBefore := clk.Now()
	if err := g.Wait(ctx); err != nil {
		t.Fatalf("unexpected error on second Wait: %v", err)
	}
	g.RecordTurn()
	tAfter := clk.Now()

	if elapsed := tAfter.Sub(tBefore); elapsed != 100*time.Millisecond {
		t.Fatalf("expected 100ms wait, got %v", elapsed)
	}

	stats := g.Stats()
	if stats.TotalTurns != 2 {
		t.Fatalf("expected 2 turns, got %d", stats.TotalTurns)
	}
	if stats.ThrottledTurns != 1 {
		t.Fatalf("expected 1 throttled turn, got %d", stats.ThrottledTurns)
	}
	if stats.TotalWaitTime != 100*time.Millisecond {
		t.Fatalf("expected 100ms total wait time, got %v", stats.TotalWaitTime)
	}
}

func TestGate_Wait_ContextCanceled(t *testing.T) {
	g := New(Config{
		MinInterval: 10 * time.Second,
		Cooldown:    10 * time.Second,
	})

	// 1. Context already canceled before Wait.
	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()

	err := g.Wait(ctxCanceled)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// 2. Context deadline exceeded during Wait.
	g.RecordTurn() // Arms 10s wait.
	ctxTimeout, cancelTimeout := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancelTimeout()

	err = g.Wait(ctxTimeout)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
}

func TestGate_Jitter(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	var rngVal float64
	mockRNG := func() float64 { return rngVal }

	g := New(Config{
		MinInterval: 100 * time.Millisecond,
		MaxInterval: 200 * time.Millisecond,
		Jitter:      0.50, // up to 50% additional delay (100ms + 50ms = 150ms)
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
		RNG:         mockRNG,
	})

	// RNG returns 0.0 -> delta is 0 -> interval is 100ms.
	rngVal = 0.0
	g.RecordTurn()
	rem0 := g.Remaining()
	if rem0 != 100*time.Millisecond {
		t.Fatalf("expected 100ms with RNG=0.0, got %v", rem0)
	}

	// Advance clock to clear.
	clk.Advance(100 * time.Millisecond)

	// RNG returns 1.0 -> delta is 50ms -> interval is 150ms.
	rngVal = 1.0
	g.RecordTurn()
	rem1 := g.Remaining()
	if rem1 != 150*time.Millisecond {
		t.Fatalf("expected 150ms with RNG=1.0, got %v", rem1)
	}

	// Verify MaxInterval clamping.
	clk.Advance(150 * time.Millisecond)
	gClamped := New(Config{
		MinInterval: 100 * time.Millisecond,
		MaxInterval: 120 * time.Millisecond, // ceiling lower than 100ms + 50ms
		Jitter:      0.50,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
		RNG:         mockRNG,
	})
	rngVal = 1.0
	gClamped.RecordTurn()
	remClamped := gClamped.Remaining()
	if remClamped != 120*time.Millisecond {
		t.Fatalf("expected clamped interval 120ms, got %v", remClamped)
	}
}

func TestGate_Cooldown_BurstLimit(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 10 * time.Millisecond,
		MaxInterval: 10 * time.Millisecond,
		Cooldown:    500 * time.Millisecond,
		BurstLimit:  3,
		BurstWindow: 100 * time.Millisecond,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	// Turn 1
	g.RecordTurn()
	if g.Stats().InCooldown {
		t.Fatal("turn 1 should not trigger cooldown")
	}

	// Turn 2 within burst window
	clk.Advance(15 * time.Millisecond)
	g.RecordTurn()
	if g.Stats().InCooldown {
		t.Fatal("turn 2 should not trigger cooldown")
	}

	// Turn 3: hits burst limit of 3 -> triggers cooldown!
	clk.Advance(15 * time.Millisecond)
	g.RecordTurn()

	stats := g.Stats()
	if !stats.InCooldown {
		t.Fatal("turn 3 should trigger cooldown")
	}
	if stats.CooldownCount != 1 {
		t.Fatalf("expected CooldownCount 1, got %d", stats.CooldownCount)
	}
	if g.Check() {
		t.Fatal("Check should return false during cooldown")
	}
	if rem := g.Remaining(); rem != 500*time.Millisecond {
		t.Fatalf("expected Remaining 500ms, got %v", rem)
	}

	// Advance clock by 300ms: still in cooldown.
	clk.Advance(300 * time.Millisecond)
	if !g.Stats().InCooldown {
		t.Fatal("should still be in cooldown at 300ms")
	}

	// Advance clock by remaining 200ms (total 500ms): cooldown cleared.
	clk.Advance(200 * time.Millisecond)
	if g.Stats().InCooldown {
		t.Fatal("cooldown should be cleared after 500ms")
	}
	if !g.Check() {
		t.Fatal("Check should return true after cooldown expires")
	}
}

func TestGate_Cooldown_BurstWindowReset(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 10 * time.Millisecond,
		Cooldown:    1 * time.Second,
		BurstLimit:  3,
		BurstWindow: 50 * time.Millisecond,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	// Turn 1
	g.RecordTurn()
	// Pause longer than BurstWindow (100ms > 50ms)
	clk.Advance(100 * time.Millisecond)
	// Turn 2: should reset consecutive turns counter to 1
	g.RecordTurn()
	if stats := g.Stats(); stats.ConsecutiveTurns != 1 {
		t.Fatalf("expected ConsecutiveTurns reset to 1, got %d", stats.ConsecutiveTurns)
	}

	// Turn 3: consecutive turns becomes 2
	clk.Advance(15 * time.Millisecond)
	g.RecordTurn()
	if stats := g.Stats(); stats.ConsecutiveTurns != 2 {
		t.Fatalf("expected ConsecutiveTurns 2, got %d", stats.ConsecutiveTurns)
	}
	if g.Stats().InCooldown {
		t.Fatal("should not be in cooldown since window reset")
	}
}

func TestGate_Cooldown_Manual(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 10 * time.Millisecond,
		Cooldown:    250 * time.Millisecond,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	// Trigger default cooldown.
	g.TriggerCooldown()
	if !g.Stats().InCooldown {
		t.Fatal("expected gate in cooldown after TriggerCooldown")
	}
	if rem := g.Remaining(); rem != 250*time.Millisecond {
		t.Fatalf("expected Remaining 250ms, got %v", rem)
	}

	clk.Advance(250 * time.Millisecond)
	if g.Stats().InCooldown {
		t.Fatal("expected cooldown cleared")
	}

	// Trigger custom cooldown duration (e.g. HTTP 429 Retry-After).
	g.TriggerCooldownFor(600 * time.Millisecond)
	if !g.Stats().InCooldown {
		t.Fatal("expected gate in cooldown after TriggerCooldownFor")
	}
	if rem := g.Remaining(); rem != 600*time.Millisecond {
		t.Fatalf("expected Remaining 600ms, got %v", rem)
	}

	clk.Advance(600 * time.Millisecond)
	if !g.Check() {
		t.Fatal("expected Check true after custom cooldown")
	}
}

func TestGate_Reset(t *testing.T) {
	startTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	clk := newSimClock(startTime)

	g := New(Config{
		MinInterval: 1 * time.Second,
		Cooldown:    10 * time.Second,
		BurstLimit:  2,
		Clock:       clk.Now,
		Sleep:       clk.Sleep,
	})

	// Enter cooldown.
	g.TriggerCooldown()
	if g.Check() {
		t.Fatal("gate should be closed in cooldown")
	}

	// Reset must clear cooldown immediately.
	g.Reset()
	if !g.Check() {
		t.Fatal("gate should be open immediately after Reset")
	}
	if rem := g.Remaining(); rem != 0 {
		t.Fatalf("expected Remaining 0 after Reset, got %v", rem)
	}
	if g.Stats().InCooldown {
		t.Fatal("InCooldown should be false after Reset")
	}
}

func TestGate_Concurrency(t *testing.T) {
	g := New(Config{
		MinInterval: 1 * time.Millisecond,
		MaxInterval: 2 * time.Millisecond,
		Cooldown:    10 * time.Millisecond,
		Jitter:      0.1,
		BurstLimit:  50,
	})

	var wg sync.WaitGroup
	workers := 16
	turnsPerWorker := 20
	var successCount atomic.Int64

	ctx := context.Background()

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < turnsPerWorker; j++ {
				if err := g.Wait(ctx); err != nil {
					return
				}
				g.RecordTurn()
				successCount.Add(1)
			}
		}()
	}

	// Concurrent observers calling Check, Stats, Remaining, Reset
	stopObservers := make(chan struct{})
	go func() {
		for {
			select {
			case <-stopObservers:
				return
			default:
				_ = g.Check()
				_ = g.Stats()
				_ = g.Remaining()
				time.Sleep(200 * time.Microsecond)
			}
		}
	}()

	wg.Wait()
	close(stopObservers)

	totalExpected := int64(workers * turnsPerWorker)
	if successCount.Load() != totalExpected {
		t.Fatalf("expected %d successful turns, got %d", totalExpected, successCount.Load())
	}
	stats := g.Stats()
	if stats.TotalTurns != totalExpected {
		t.Fatalf("expected TotalTurns %d, got %d", totalExpected, stats.TotalTurns)
	}
}

func BenchmarkGate(b *testing.B) {
	b.Run("Check", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = g.Check()
		}
	})

	b.Run("RecordTurn", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.RecordTurn()
		}
	})

	b.Run("WaitZeroInterval", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		ctx := context.Background()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = g.Wait(ctx)
		}
	})

	b.Run("Stats", func(b *testing.B) {
		g := New(DefaultConfig())
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = g.Stats()
		}
	})

	b.Run("ParallelCheckAndRecord", func(b *testing.B) {
		g := New(Config{MinInterval: 0})
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				if g.Check() {
					g.RecordTurn()
				}
			}
		})
	})
}
