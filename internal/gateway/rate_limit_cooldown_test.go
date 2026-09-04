package gateway

import (
	"sync"
	"testing"
	"time"
)

// TestRateLimitCooldownCoordinatesConcurrentSessions pins #10659:
// Five concurrent sessions hitting 429 produce one shared coordinated cooldown,
// aggregate attempts are bounded, and held requests do not stampede upstream.
func TestRateLimitCooldownCoordinatesConcurrentSessions(t *testing.T) {
	c := NewRateLimitCooldownController()
	t0 := time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC)
	currTime := t0
	var timeMu sync.Mutex
	c.SetClock(func() time.Time {
		timeMu.Lock()
		defer timeMu.Unlock()
		return currTime
	})
	c.SetMaxAttempts(5)

	target := "provider-anthropic/claude-sonnet"

	// Initial check: admission granted
	rc := c.CheckAdmission(target)
	if rc.Outcome != "admitted" {
		t.Fatalf("expected initial request admitted, got %s", rc.Outcome)
	}

	// First 429 arrives with Retry-After: 30
	receipt := c.RecordRateLimit(target, "30")
	if receipt.Outcome != "cooldown_armed" {
		t.Fatalf("expected cooldown_armed, got %s", receipt.Outcome)
	}
	if receipt.RemainingCooldown != 30*time.Second {
		t.Fatalf("expected 30s remaining cooldown, got %v", receipt.RemainingCooldown)
	}

	// 5 concurrent requests check admission during cooldown
	var wg sync.WaitGroup
	outcomes := make([]string, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			r := c.CheckAdmission(target)
			outcomes[idx] = r.Outcome
		}(i)
	}
	wg.Wait()

	for i, outcome := range outcomes {
		if outcome != "held" && outcome != "retry_exhausted" {
			t.Errorf("request %d: expected held or retry_exhausted, got %s", i, outcome)
		}
	}

	// Check admission reports remaining cooldown accurately
	check := c.CheckAdmission(target)
	if check.Outcome != "held" {
		t.Fatalf("expected held, got %s", check.Outcome)
	}
	if check.RemainingCooldown <= 0 || check.RemainingCooldown > 30*time.Second {
		t.Fatalf("expected remaining cooldown in (0, 30s], got %v", check.RemainingCooldown)
	}

	// Advance time past cooldown (+35s)
	timeMu.Lock()
	currTime = t0.Add(35 * time.Second)
	timeMu.Unlock()

	// Target self-heals: requests admitted after cooldown expires
	afterCooldown := c.CheckAdmission(target)
	if afterCooldown.Outcome != "admitted" {
		t.Fatalf("expected admitted after cooldown expired, got %s", afterCooldown.Outcome)
	}

	// Reset confirms clean recovered receipt
	recovered := c.Reset(target)
	if recovered.Outcome != "recovered" || recovered.AggregateAttempts != 0 {
		t.Fatalf("expected recovered receipt with 0 attempts, got %+v", recovered)
	}
}

func TestRateLimitCooldownRetryAfterHTTPDateAndFallback(t *testing.T) {
	c := NewRateLimitCooldownController()
	t0 := time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC)
	c.SetClock(func() time.Time { return t0 })

	target := "provider-openai/gpt-4"

	// 1. HTTP-Date Retry-After header
	futureDate := "Thu, 03 Sep 2026 14:00:20 GMT"
	rc := c.RecordRateLimit(target, futureDate)
	if rc.RemainingCooldown != 20*time.Second {
		t.Fatalf("expected 20s cooldown from HTTP-date, got %v", rc.RemainingCooldown)
	}

	// 2. Clamped max backoff
	hugeSecs := "3600"
	rcHuge := c.RecordRateLimit(target, hugeSecs)
	if rcHuge.RemainingCooldown != 60*time.Second {
		t.Fatalf("expected 60s clamped cooldown, got %v", rcHuge.RemainingCooldown)
	}

	// 3. Fallback jittered backoff when header is empty
	c2 := NewRateLimitCooldownController()
	c2.SetClock(func() time.Time { return t0 })
	rcFallback := c2.RecordRateLimit("target2", "")
	if rcFallback.RemainingCooldown < 2*time.Second || rcFallback.RemainingCooldown > 3*time.Second {
		t.Fatalf("expected fallback cooldown in [2s, 3s], got %v", rcFallback.RemainingCooldown)
	}
}
