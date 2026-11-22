package ratelimit

import (
	"errors"
	"math"
	"testing"
	"time"
)

type windowedClock struct{ t time.Time }

func (c *windowedClock) now() time.Time          { return c.t }
func (c *windowedClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func TestWindowedTokenBucket(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)

	t.Run("upstream defaults", func(t *testing.T) {
		s := NewWindowedStoreWithClock(WindowedConfig{}, nil)
		if s.tokens != 1 || s.interval != time.Second || s.sweepMinTTL != 12*time.Hour {
			t.Fatalf("defaults = %d %v %v, want 1 1s 12h", s.tokens, s.interval, s.sweepMinTTL)
		}
	})

	t.Run("take drains then refuses and reset is start+interval", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 3, Interval: time.Minute}, c.now)
		for want := uint64(2); ; want-- {
			r, err := s.Take("k")
			if err != nil || !r.OK || r.Remaining != want || r.Tokens != 3 {
				t.Fatalf("take: %+v %v, want ok remaining=%d", r, err, want)
			}
			if !r.Reset.Equal(t0.Add(time.Minute)) {
				t.Fatalf("reset = %v, want %v", r.Reset, t0.Add(time.Minute))
			}
			if want == 0 {
				break
			}
		}
		c.advance(59 * time.Second)
		if r, _ := s.Take("k"); r.OK || r.Remaining != 0 {
			t.Fatalf("over-limit take admitted: %+v", r)
		}
		if tok, rem, _ := s.Get("k"); tok != 3 || rem != 0 {
			t.Fatalf("get = %d,%d want 3,0", tok, rem)
		}
	})

	t.Run("boundary refills to full once, not cumulatively", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 2, Interval: time.Minute}, c.now)
		s.Take("k")
		s.Take("k")
		c.advance(time.Minute)
		r, _ := s.Take("k")
		if !r.OK || r.Remaining != 1 || !r.Reset.Equal(t0.Add(2*time.Minute)) {
			t.Fatalf("after one interval: %+v", r)
		}
		c.advance(10 * time.Minute)
		r, _ = s.Take("k")
		if !r.OK || r.Remaining != 1 || !r.Reset.Equal(t0.Add(12*time.Minute)) {
			t.Fatalf("after many intervals: %+v, want remaining=1 (full=2, no accrual)", r)
		}
	})

	t.Run("backward clock re-anchors start", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 1, Interval: time.Minute}, c.now)
		s.Take("k")
		c.advance(-time.Hour)
		r, _ := s.Take("k")
		if r.OK || !r.Reset.Equal(t0.Add(-time.Hour+time.Minute)) {
			t.Fatalf("re-anchor: %+v", r)
		}
		c.advance(time.Minute)
		if r, _ := s.Take("k"); !r.OK {
			t.Fatalf("no refill after re-anchored interval: %+v", r)
		}
	})

	t.Run("set overrides per key and isolates keys", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 1, Interval: time.Minute}, c.now)
		if err := s.Set("big", 5, 0); err != nil {
			t.Fatal(err)
		}
		r, _ := s.Take("big")
		if r.Tokens != 5 || r.Remaining != 4 || !r.Reset.Equal(t0.Add(time.Minute)) {
			t.Fatalf("set bucket: %+v (zero interval must fall back to store interval)", r)
		}
		if r, _ := s.Take("small"); r.Tokens != 1 || r.Remaining != 0 || !r.OK {
			t.Fatalf("other key affected: %+v", r)
		}
		if tok, rem, err := s.Get("absent"); tok != 0 || rem != 0 || err != nil {
			t.Fatalf("absent get = %d,%d,%v", tok, rem, err)
		}
	})

	t.Run("burst adds and saturates", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 2, Interval: time.Minute}, c.now)
		s.Burst("k", 3)
		if tok, rem, _ := s.Get("k"); tok != 2 || rem != 5 {
			t.Fatalf("burst on absent key = %d,%d want 2,5", tok, rem)
		}
		s.Burst("k", math.MaxUint64)
		if _, rem, _ := s.Get("k"); rem != math.MaxUint64 {
			t.Fatalf("burst did not saturate: %d", rem)
		}
	})

	t.Run("sweep drops idle keys past TTL", func(t *testing.T) {
		c := &windowedClock{t: t0}
		s := NewWindowedStoreWithClock(WindowedConfig{Tokens: 1, Interval: time.Minute, SweepMinTTL: time.Hour}, c.now)
		s.Take("old")
		c.advance(2 * time.Hour)
		s.Take("fresh")
		if n := s.Sweep(c.now()); n != 1 {
			t.Fatalf("swept %d, want 1", n)
		}
		if tok, _, _ := s.Get("old"); tok != 0 {
			t.Fatal("old key survived sweep")
		}
		if tok, _, _ := s.Get("fresh"); tok != 1 {
			t.Fatal("fresh key was swept")
		}
	})

	t.Run("close refuses every call", func(t *testing.T) {
		s := NewWindowedStoreWithClock(WindowedConfig{}, (&windowedClock{t: t0}).now)
		s.Take("k")
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatalf("second close: %v", err)
		}
		if _, err := s.Take("k"); !errors.Is(err, ErrWindowedStopped) {
			t.Fatalf("take after close: %v", err)
		}
		if _, _, err := s.Get("k"); !errors.Is(err, ErrWindowedStopped) {
			t.Fatalf("get after close: %v", err)
		}
		if err := s.Set("k", 1, 0); !errors.Is(err, ErrWindowedStopped) {
			t.Fatalf("set after close: %v", err)
		}
		if err := s.Burst("k", 1); !errors.Is(err, ErrWindowedStopped) {
			t.Fatalf("burst after close: %v", err)
		}
	})
}
