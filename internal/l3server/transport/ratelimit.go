package transport

import (
	"sync/atomic"
	"time"
)

// ConnRateLimiter is a lock-free atomic token bucket for per-connection rate
// limiting. Tokens refill continuously based on elapsed time. Safe for
// concurrent use from multiple goroutines.
type ConnRateLimiter struct {
	maxTokens  int64 // bucket capacity (ops/sec = tokens refilled per second)
	tokens     atomic.Int64
	lastRefill atomic.Int64 // UnixNano of last refill
}

// NewConnRateLimiter creates a rate limiter allowing opsPerSec operations
// per second. Pass 0 to disable (Allow always returns true).
func NewConnRateLimiter(opsPerSec int64) *ConnRateLimiter {
	if opsPerSec <= 0 {
		return nil
	}
	rl := &ConnRateLimiter{maxTokens: opsPerSec}
	rl.tokens.Store(opsPerSec)
	rl.lastRefill.Store(time.Now().UnixNano())
	return rl
}

// Allow consumes one token and returns true, or returns false if the bucket
// is empty. Refills tokens based on elapsed time since last refill.
func (rl *ConnRateLimiter) Allow() bool {
	return rl.allowAt(time.Now().UnixNano())
}

// allowAt keeps the clock input explicit for deterministic refill checks.
func (rl *ConnRateLimiter) allowAt(now int64) bool {
	if rl == nil {
		return true
	}
	// Try to refill based on elapsed time
	last := rl.lastRefill.Load()
	elapsed := now - last
	if elapsed >= int64(time.Millisecond) {
		// One second fills the bucket. Split shorter products before multiplying
		// so even a large configured capacity cannot overflow int64.
		second := int64(time.Second)
		add := rl.maxTokens
		if elapsed < second {
			add = (rl.maxTokens/second)*elapsed + (rl.maxTokens%second)*elapsed/second
		}
		if add > 0 && rl.lastRefill.CompareAndSwap(last, now) {
			for {
				cur := rl.tokens.Load()
				next := rl.maxTokens
				if cur < rl.maxTokens-add {
					next = cur + add
				}
				// Saturate atomically without restoring a concurrently spent token.
				if rl.tokens.CompareAndSwap(cur, next) {
					break
				}
			}
		}
	}
	// Try to consume one token
	for {
		cur := rl.tokens.Load()
		if cur <= 0 {
			return false
		}
		if rl.tokens.CompareAndSwap(cur, cur-1) {
			return true
		}
	}
}
