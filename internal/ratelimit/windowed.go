// DIRECT-PORT of sethvargo/go-limiter memorystore/store.go and store.go (Store
// interface) at cd11927274b607c56cff42adb6cb7b972a85a11a (Apache-2.0).
// Ported: the per-key interval-reset token bucket (Take/Get/Set/Burst/Close,
// full refill at each interval boundary, clock-backward re-anchor, TTL purge).
// fak adds: an injected clock, a typed reset instant, refusal of every method
// after Close, and an explicit Sweep in place of the upstream purge goroutine.
//
// This is the interval-reset variant, distinct from window.go's
// continuous-refill BucketCap: here a bucket jumps back to full at each
// boundary counted from the key's first use, and never accrues partial tokens.
package ratelimit

import (
	"errors"
	"math"
	"sync"
	"time"
)

// ErrWindowedStopped is returned by every WindowedStore method after Close.
var ErrWindowedStopped = errors.New("ratelimit: windowed store stopped")

// Upstream defaults for a zero WindowedConfig field.
const (
	WindowedDefaultTokens      uint64        = 1
	WindowedDefaultInterval    time.Duration = time.Second
	WindowedDefaultSweepMinTTL time.Duration = 12 * time.Hour
)

// WindowedConfig sizes a WindowedStore. Zero fields take the Windowed* defaults.
type WindowedConfig struct {
	Tokens   uint64
	Interval time.Duration
	// SweepMinTTL is how long a key must be idle before Sweep removes it. It
	// should be at least Interval, or a key can be purged before its limit
	// applies.
	SweepMinTTL time.Duration
}

// WindowedTake is the result of one Take: the key's per-interval limit, the
// tokens left after this take, when the current interval ends, and whether a
// token was granted.
type WindowedTake struct {
	Tokens    uint64
	Remaining uint64
	Reset     time.Time
	OK        bool
}

// WindowedStore is an in-process, per-key interval-reset token bucket.
type WindowedStore struct {
	mu          sync.Mutex
	now         func() time.Time
	tokens      uint64
	interval    time.Duration
	sweepMinTTL time.Duration
	stopped     bool
	buckets     map[string]*windowedBucket
}

type windowedBucket struct {
	start     time.Time
	maxTokens uint64
	interval  time.Duration
	available uint64
	lastTick  int64
}

// NewWindowedStore builds a store on the wall clock.
func NewWindowedStore(cfg WindowedConfig) *WindowedStore {
	return NewWindowedStoreWithClock(cfg, time.Now)
}

// NewWindowedStoreWithClock builds a store on the given clock; a nil clock is
// the wall clock.
func NewWindowedStoreWithClock(cfg WindowedConfig, now func() time.Time) *WindowedStore {
	if now == nil {
		now = time.Now
	}
	s := &WindowedStore{
		now:         now,
		tokens:      WindowedDefaultTokens,
		interval:    WindowedDefaultInterval,
		sweepMinTTL: WindowedDefaultSweepMinTTL,
		buckets:     make(map[string]*windowedBucket),
	}
	if cfg.Tokens > 0 {
		s.tokens = cfg.Tokens
	}
	if cfg.Interval > 0 {
		s.interval = cfg.Interval
	}
	if cfg.SweepMinTTL > 0 {
		s.sweepMinTTL = cfg.SweepMinTTL
	}
	return s
}

// Take removes one token from key, creating the bucket on first use.
func (s *WindowedStore) Take(key string) (WindowedTake, error) {
	return s.TakeAt(key, s.now())
}

// TakeAt is Take evaluated at an explicit instant.
func (s *WindowedStore) TakeAt(key string, at time.Time) (WindowedTake, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return WindowedTake{}, ErrWindowedStopped
	}
	b, ok := s.buckets[key]
	if !ok {
		b = newWindowedBucket(s.tokens, s.interval, at)
		s.buckets[key] = b
	}
	return b.take(at), nil
}

// Get reports key's limit and currently available tokens, or (0, 0) for an
// unknown key. It does not apply a pending interval refill.
func (s *WindowedStore) Get(key string) (tokens, remaining uint64, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return 0, 0, ErrWindowedStopped
	}
	if b, ok := s.buckets[key]; ok {
		return b.maxTokens, b.available, nil
	}
	return 0, 0, nil
}

// Set replaces key's bucket with a fresh, full one of the given size. A
// non-positive interval falls back to the store interval so tick math never
// divides by zero.
func (s *WindowedStore) Set(key string, tokens uint64, interval time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return ErrWindowedStopped
	}
	if interval <= 0 {
		interval = s.interval
	}
	s.buckets[key] = newWindowedBucket(tokens, interval, s.now())
	return nil
}

// Burst adds n tokens to key's available count, saturating at MaxUint64. The
// surplus may exceed the limit until the next interval refill resets it.
func (s *WindowedStore) Burst(key string, n uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return ErrWindowedStopped
	}
	b, ok := s.buckets[key]
	if !ok {
		b = newWindowedBucket(s.tokens, s.interval, s.now())
		s.buckets[key] = b
	}
	if b.available > math.MaxUint64-n {
		b.available = math.MaxUint64
	} else {
		b.available += n
	}
	return nil
}

// Sweep removes keys idle longer than SweepMinTTL as of at and returns how
// many it removed. Callers own the cadence; the store runs no goroutine.
func (s *WindowedStore) Sweep(at time.Time) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	removed := 0
	for k, b := range s.buckets {
		last := b.start.Add(time.Duration(b.lastTick) * b.interval)
		if last.After(at) {
			last = at
		}
		if at.Sub(last) > s.sweepMinTTL {
			delete(s.buckets, k)
			removed++
		}
	}
	return removed
}

// Close stops the store and drops every bucket. It is idempotent.
func (s *WindowedStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return nil
	}
	s.stopped = true
	s.buckets = make(map[string]*windowedBucket)
	return nil
}

func newWindowedBucket(tokens uint64, interval time.Duration, at time.Time) *windowedBucket {
	return &windowedBucket{start: at, maxTokens: tokens, interval: interval, available: tokens}
}

func (b *windowedBucket) take(at time.Time) WindowedTake {
	// A clock that stepped backward would make elapsed negative; re-anchor
	// so the bucket keeps ticking from the new now.
	if at.Before(b.start) {
		b.start = at
		b.lastTick = 0
	}
	currTick := int64(at.Sub(b.start) / b.interval)
	res := WindowedTake{
		Tokens: b.maxTokens,
		Reset:  b.start.Add(time.Duration(currTick+1) * b.interval),
	}
	if b.lastTick < currTick {
		b.available = b.maxTokens
		b.lastTick = currTick
	}
	if b.available > 0 {
		b.available--
		res.OK = true
		res.Remaining = b.available
	}
	return res
}
