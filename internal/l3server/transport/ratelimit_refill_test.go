package transport

import (
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestConnRateLimiterRefillBounds(t *testing.T) {
	t.Parallel()
	const maxTokens = int64(^uint64(0) >> 1)
	const start = int64(24 * time.Hour)
	for _, tc := range []struct {
		name       string
		capacity   int64
		tokens     int64
		elapsed    time.Duration
		wantTokens int64
		wantAllow  bool
		refilled   bool
	}{
		{name: "default rate long idle", capacity: 500_000, elapsed: 6 * time.Hour, wantTokens: 499_999, wantAllow: true, refilled: true},
		{name: "configured rate within TCP idle timeout", capacity: 50_000_000, elapsed: 4 * time.Minute, wantTokens: 49_999_999, wantAllow: true, refilled: true},
		{name: "full maximum capacity", capacity: maxTokens, tokens: maxTokens, elapsed: time.Second, wantTokens: maxTokens - 1, wantAllow: true, refilled: true},
		{name: "existing balance addition overflow", capacity: maxTokens - 5_000_000_000, tokens: maxTokens - 5_000_000_000, elapsed: 2 * time.Second, wantTokens: maxTokens - 5_000_000_001, wantAllow: true, refilled: true},
		{name: "maximum capacity fractional refill", capacity: maxTokens, elapsed: time.Millisecond, wantTokens: maxTokens/1000 - 1, wantAllow: true, refilled: true},
		{name: "near full saturates", capacity: maxTokens, tokens: maxTokens - 2, elapsed: time.Millisecond, wantTokens: maxTokens - 1, wantAllow: true, refilled: true},
		{name: "default rate millisecond", capacity: 500_000, elapsed: time.Millisecond, wantTokens: 499, wantAllow: true, refilled: true},
		{name: "fraction rounds down", capacity: 3, elapsed: time.Second / 2, wantAllow: true, refilled: true},
		{name: "fraction has no whole token", capacity: 1, elapsed: time.Second - 1},
		{name: "exact second", capacity: 1, elapsed: time.Second, wantAllow: true, refilled: true},
		{name: "submillisecond gate", capacity: 500_000, elapsed: time.Millisecond - 1},
		{name: "backward clock preserves bucket", capacity: 10, tokens: 2, elapsed: -time.Second, wantTokens: 1, wantAllow: true},
		{name: "unchanged clock spends token", capacity: 10, tokens: 2, wantTokens: 1, wantAllow: true},
		{name: "empty bucket refuses", capacity: 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rl := &ConnRateLimiter{maxTokens: tc.capacity}
			rl.tokens.Store(tc.tokens)
			rl.lastRefill.Store(start)
			now := start + int64(tc.elapsed)
			if got := rl.allowAt(now); got != tc.wantAllow {
				t.Fatalf("allow = %v, want %v", got, tc.wantAllow)
			}
			if got := rl.tokens.Load(); got != tc.wantTokens {
				t.Fatalf("tokens = %d, want %d", got, tc.wantTokens)
			}
			wantRefill := start
			if tc.refilled {
				wantRefill = now
			}
			if got := rl.lastRefill.Load(); got != wantRefill {
				t.Fatalf("last refill = %d, want %d", got, wantRefill)
			}
		})
	}
	for _, disabled := range []int64{0, -1} {
		if rl := NewConnRateLimiter(disabled); rl != nil || !rl.Allow() {
			t.Fatalf("disabled rate %d did not preserve nil/allow behavior", disabled)
		}
	}
}
