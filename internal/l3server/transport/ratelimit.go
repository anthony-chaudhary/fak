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
	if rl == nil {
		return true
	}
	// Try to refill based on elapsed time
	now := time.Now().UnixNano()
	last := rl.lastRefill.Load()
	elapsed := now - last
	if elapsed >= int64(time.Millisecond) {
		// Calculate tokens to add (proportional to elapsed time)
		add := rl.maxTokens * elapsed / int64(time.Second)
		if add > 0 && rl.lastRefill.CompareAndSwap(last, now) {
			cur := rl.tokens.Add(add)
			// Cap at maxTokens
			if cur > rl.maxTokens {
				rl.tokens.Store(rl.maxTokens)
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
