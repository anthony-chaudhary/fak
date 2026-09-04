package util

import (
	"sync/atomic"
	"time"
)

// LogGate is a lock-free time gate for rate-limiting hot-path log output.
// Only the first call within each interval returns true; subsequent calls
// return false until the interval elapses. Safe for concurrent use.
type LogGate struct {
	intervalNs int64
	lastNs     atomic.Int64
}

// NewLogGate creates a gate that allows one log per interval.
func NewLogGate(interval time.Duration) *LogGate {
	return &LogGate{intervalNs: int64(interval)}
}

// Allow returns true if the interval has elapsed since the last Allow()==true call.
// Uses atomic CAS — no locks, no allocations.
func (g *LogGate) Allow() bool {
	now := time.Now().UnixNano()
	last := g.lastNs.Load()
	if now-last < g.intervalNs {
		return false
	}
	return g.lastNs.CompareAndSwap(last, now)
}
