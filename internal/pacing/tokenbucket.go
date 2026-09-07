package pacing

import (
	"context"
	"math"
	"sync"
	"time"
)

// TokenBucketConfig configures rate limits for model tokens and requests.
type TokenBucketConfig struct {
	// TPMLimit is the maximum tokens permitted per minute (0 = unlimited).
	TPMLimit int64 `json:"tpm_limit"`
	// RPMLimit is the maximum requests permitted per minute (0 = unlimited).
	RPMLimit int64 `json:"rpm_limit"`
}

// TokenBucket implements a dual token-per-minute (TPM) and request-per-minute (RPM)
// rate limiter with fractional fill and adaptive backoff.
type TokenBucket struct {
	mu sync.Mutex

	tpmLimit int64
	rpmLimit int64

	tokensCapacity   float64
	tokensAvailable  float64
	requestsCapacity float64
	requestsAvail    float64

	lastRefill time.Time

	backoffUntil time.Time
	backoffMult  float64
}

// NewTokenBucket creates a token bucket with the specified limits.
func NewTokenBucket(cfg TokenBucketConfig) *TokenBucket {
	tb := &TokenBucket{
		tpmLimit:    cfg.TPMLimit,
		rpmLimit:    cfg.RPMLimit,
		lastRefill:  time.Now(),
		backoffMult: 1.0,
	}
	if cfg.TPMLimit > 0 {
		tb.tokensCapacity = float64(cfg.TPMLimit)
		tb.tokensAvailable = tb.tokensCapacity
	}
	if cfg.RPMLimit > 0 {
		tb.requestsCapacity = float64(cfg.RPMLimit)
		tb.requestsAvail = tb.requestsCapacity
	}
	return tb
}

// refill replenishes tokens and requests based on elapsed time. Must be called with tb.mu locked.
func (tb *TokenBucket) refill(now time.Time) {
	if tb.lastRefill.IsZero() {
		tb.lastRefill = now
		return
	}
	elapsed := now.Sub(tb.lastRefill).Seconds()
	if elapsed <= 0 {
		return
	}
	tb.lastRefill = now

	if tb.tpmLimit > 0 {
		// Fill rate = TPMLimit tokens per 60 seconds
		fillRate := float64(tb.tpmLimit) / 60.0
		tb.tokensAvailable = math.Min(tb.tokensCapacity, tb.tokensAvailable+elapsed*fillRate)
	}

	if tb.rpmLimit > 0 {
		// Fill rate = RPMLimit requests per 60 seconds
		fillRate := float64(tb.rpmLimit) / 60.0
		tb.requestsAvail = math.Min(tb.requestsCapacity, tb.requestsAvail+elapsed*fillRate)
	}
}

// Allow checks if the requested tokens and 1 request can be granted immediately.
func (tb *TokenBucket) Allow(tokens int64) bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	if now.Before(tb.backoffUntil) {
		return false
	}
	tb.refill(now)

	reqTokens := float64(tokens)
	if tb.tpmLimit > 0 && tb.tokensAvailable < reqTokens {
		return false
	}
	if tb.rpmLimit > 0 && tb.requestsAvail < 1.0 {
		return false
	}

	if tb.tpmLimit > 0 {
		tb.tokensAvailable -= reqTokens
	}
	if tb.rpmLimit > 0 {
		tb.requestsAvail -= 1.0
	}
	return true
}

// Reserve waits until sufficient tokens and request capacity are available or ctx is cancelled.
func (tb *TokenBucket) Reserve(ctx context.Context, estimatedTokens int64) error {
	reqTokens := float64(estimatedTokens)

	for {
		tb.mu.Lock()
		now := time.Now()
		tb.refill(now)

		// Check backoff
		var backoffWait time.Duration
		if now.Before(tb.backoffUntil) {
			backoffWait = tb.backoffUntil.Sub(now)
		}

		var tokenWait time.Duration
		if tb.tpmLimit > 0 && tb.tokensAvailable < reqTokens {
			fillRate := float64(tb.tpmLimit) / 60.0
			deficit := reqTokens - tb.tokensAvailable
			tokenWait = time.Duration(deficit/fillRate*float64(time.Second)) + time.Millisecond
		}

		var reqWait time.Duration
		if tb.rpmLimit > 0 && tb.requestsAvail < 1.0 {
			fillRate := float64(tb.rpmLimit) / 60.0
			deficit := 1.0 - tb.requestsAvail
			reqWait = time.Duration(deficit/fillRate*float64(time.Second)) + time.Millisecond
		}

		waitDur := backoffWait
		if tokenWait > waitDur {
			waitDur = tokenWait
		}
		if reqWait > waitDur {
			waitDur = reqWait
		}

		if waitDur <= 0 {
			// Sufficient capacity
			if tb.tpmLimit > 0 {
				tb.tokensAvailable -= reqTokens
			}
			if tb.rpmLimit > 0 {
				tb.requestsAvail -= 1.0
			}
			tb.mu.Unlock()
			return nil
		}

		tb.mu.Unlock()

		// Sleep until waitDur expires or context is cancelled
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(waitDur):
		}
	}
}

// ApplyBackoff sets an adaptive backoff window during which acquisitions are delayed.
func (tb *TokenBucket) ApplyBackoff(dur time.Duration) {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	until := now.Add(dur)
	if until.After(tb.backoffUntil) {
		tb.backoffUntil = until
	}
	tb.backoffMult = math.Min(tb.backoffMult*1.5, 8.0)
}

// ResetBackoff reduces backoff multipliers after successful operations.
func (tb *TokenBucket) ResetBackoff() {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	tb.backoffMult = math.Max(1.0, tb.backoffMult*0.8)
	if time.Now().After(tb.backoffUntil) {
		tb.backoffUntil = time.Time{}
	}
}

// TokenBucketStats contains snapshot statistics for the token bucket.
type TokenBucketStats struct {
	TPMLimit        int64     `json:"tpm_limit"`
	RPMLimit        int64     `json:"rpm_limit"`
	TokensAvailable float64   `json:"tokens_available"`
	RequestsAvail   float64   `json:"requests_available"`
	BackoffActive   bool      `json:"backoff_active"`
	BackoffUntil    time.Time `json:"backoff_until,omitempty"`
}

// Stats returns a snapshot of token bucket statistics.
func (tb *TokenBucket) Stats() TokenBucketStats {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	tb.refill(now)

	return TokenBucketStats{
		TPMLimit:        tb.tpmLimit,
		RPMLimit:        tb.rpmLimit,
		TokensAvailable: tb.tokensAvailable,
		RequestsAvail:   tb.requestsAvail,
		BackoffActive:   now.Before(tb.backoffUntil),
		BackoffUntil:    tb.backoffUntil,
	}
}
