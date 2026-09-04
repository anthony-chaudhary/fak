package gateway

import (
	"math/rand"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// RateLimitCooldownReceipt is the privacy-safe structured record of a cooldown or retry event (#10659).
type RateLimitCooldownReceipt struct {
	Target             string        `json:"target"`
	Outcome            string        `json:"outcome"` // "cooldown_armed" | "admitted" | "held" | "retry_exhausted" | "recovered"
	RemainingCooldown  time.Duration `json:"remaining_cooldown_ms"`
	AggregateAttempts  int           `json:"aggregate_attempts"`
	ObservedRetryAfter string        `json:"observed_retry_after,omitempty"`
}

// RateLimitCooldownController coordinates 429 backpressure and cooldown across concurrent sessions (#10659).
type RateLimitCooldownController struct {
	mu          sync.RWMutex
	cooldowns   map[string]time.Time
	attempts    map[string]int
	maxAttempts int
	defaultWait time.Duration
	maxWait     time.Duration
	clock       func() time.Time
}

// NewRateLimitCooldownController constructs a new shared rate limit cooldown controller.
func NewRateLimitCooldownController() *RateLimitCooldownController {
	return &RateLimitCooldownController{
		cooldowns:   make(map[string]time.Time),
		attempts:    make(map[string]int),
		maxAttempts: 5,
		defaultWait: 2 * time.Second,
		maxWait:     60 * time.Second,
		clock:       time.Now,
	}
}

// SetClock overrides the time source for testing.
func (c *RateLimitCooldownController) SetClock(clk func() time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.clock = clk
}

// SetMaxAttempts overrides the aggregate attempts bound.
func (c *RateLimitCooldownController) SetMaxAttempts(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.maxAttempts = n
}

// RecordRateLimit records a 429 from an upstream target, parses Retry-After, and arms the shared cooldown.
func (c *RateLimitCooldownController) RecordRateLimit(target string, retryAfterHeader string) RateLimitCooldownReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := c.clock()
	dur := c.parseRetryAfter(retryAfterHeader, now)

	expiry := now.Add(dur)
	if cur, ok := c.cooldowns[target]; !ok || expiry.After(cur) {
		c.cooldowns[target] = expiry
	}
	c.attempts[target]++

	return RateLimitCooldownReceipt{
		Target:             target,
		Outcome:            "cooldown_armed",
		RemainingCooldown:  dur,
		AggregateAttempts:  c.attempts[target],
		ObservedRetryAfter: retryAfterHeader,
	}
}

// CheckAdmission evaluates whether a request to target is admitted or held under cooldown.
func (c *RateLimitCooldownController) CheckAdmission(target string) RateLimitCooldownReceipt {
	c.mu.RLock()
	defer c.mu.RUnlock()

	now := c.clock()
	expiry, exists := c.cooldowns[target]
	if !exists || now.After(expiry) {
		return RateLimitCooldownReceipt{
			Target:            target,
			Outcome:           "admitted",
			RemainingCooldown: 0,
			AggregateAttempts: c.attempts[target],
		}
	}

	remaining := expiry.Sub(now)
	attempts := c.attempts[target]
	outcome := "held"
	if c.maxAttempts > 0 && attempts >= c.maxAttempts {
		outcome = "retry_exhausted"
	}

	return RateLimitCooldownReceipt{
		Target:            target,
		Outcome:           outcome,
		RemainingCooldown: remaining,
		AggregateAttempts: attempts,
	}
}

// Reset clears cooldown and attempt counters when upstream capacity recovers.
func (c *RateLimitCooldownController) Reset(target string) RateLimitCooldownReceipt {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.cooldowns, target)
	delete(c.attempts, target)

	return RateLimitCooldownReceipt{
		Target:            target,
		Outcome:           "recovered",
		RemainingCooldown: 0,
		AggregateAttempts: 0,
	}
}

func (c *RateLimitCooldownController) parseRetryAfter(header string, now time.Time) time.Duration {
	if header == "" {
		jitter := time.Duration(rand.Int63n(int64(c.defaultWait / 2)))
		return c.defaultWait + jitter
	}
	if secs, err := strconv.Atoi(header); err == nil && secs > 0 {
		dur := time.Duration(secs) * time.Second
		if dur > c.maxWait {
			dur = c.maxWait
		}
		return dur
	}
	if t, err := http.ParseTime(header); err == nil {
		diff := t.Sub(now)
		if diff <= 0 {
			return c.defaultWait
		}
		if diff > c.maxWait {
			diff = c.maxWait
		}
		return diff
	}
	return c.defaultWait
}
