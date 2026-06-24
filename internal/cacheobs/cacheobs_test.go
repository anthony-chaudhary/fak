package cacheobs

import (
	"sync"
	"testing"
)

func TestObserveAccumulatesAndBuckets(t *testing.T) {
	o := New()
	// turn 1: cold first prefill (nothing reused)
	o.Observe(100, 0)
	// turn 2: partial reuse
	o.Observe(200, 100)
	// turn 3: frozen-regime reuse (>= 90%)
	o.Observe(1000, 990)

	s := o.Snapshot()
	if s.Turns != 3 {
		t.Fatalf("turns = %d, want 3", s.Turns)
	}
	if s.PromptTokens != 1300 || s.ReusedTokens != 1090 {
		t.Fatalf("tokens = prompt %d reused %d, want 1300/1090", s.PromptTokens, s.ReusedTokens)
	}
	if s.ColdTurns != 1 || s.PartialTurns != 1 || s.FrozenTurns != 1 {
		t.Fatalf("buckets cold=%d partial=%d frozen=%d, want 1/1/1", s.ColdTurns, s.PartialTurns, s.FrozenTurns)
	}
	want := 1090.0 / 1300.0
	if d := s.ReuseRatio - want; d > 1e-9 || d < -1e-9 {
		t.Fatalf("reuse ratio = %v, want %v", s.ReuseRatio, want)
	}
}

func TestBucketBoundaries(t *testing.T) {
	// exactly FrozenFloor -> frozen; just under -> partial; exactly ColdCeil -> partial;
	// just under ColdCeil -> cold.
	cases := []struct {
		prompt, reused        int
		frozen, partial, cold uint64
	}{
		{100, 90, 1, 0, 0}, // 0.90 == FrozenFloor -> frozen
		{100, 89, 0, 1, 0}, // 0.89 -> partial
		{100, 10, 0, 1, 0}, // 0.10 == ColdCeil -> NOT cold (cold is strictly <)
		{100, 9, 0, 0, 1},  // 0.09 -> cold
	}
	for _, c := range cases {
		o := New()
		o.Observe(c.prompt, c.reused)
		s := o.Snapshot()
		if s.FrozenTurns != c.frozen || s.PartialTurns != c.partial || s.ColdTurns != c.cold {
			t.Errorf("Observe(%d,%d): buckets frozen=%d partial=%d cold=%d, want %d/%d/%d",
				c.prompt, c.reused, s.FrozenTurns, s.PartialTurns, s.ColdTurns, c.frozen, c.partial, c.cold)
		}
	}
}

func TestObserveClampsAndIgnores(t *testing.T) {
	o := New()
	o.Observe(0, 50)    // non-positive prompt: ignored
	o.Observe(-5, 1)    // negative prompt: ignored
	o.Observe(100, 250) // reused > prompt: clamped to prompt
	o.Observe(100, -10) // negative reused: clamped to 0
	s := o.Snapshot()
	if s.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (the two zero/negative-prompt calls are ignored)", s.Turns)
	}
	if s.ReusedTokens != 100 { // 100 (clamped from 250) + 0 (clamped from -10)
		t.Fatalf("reused = %d, want 100 (clamped)", s.ReusedTokens)
	}
	if s.PromptTokens != 200 {
		t.Fatalf("prompt = %d, want 200", s.PromptTokens)
	}
}

func TestNilObserverSafe(t *testing.T) {
	var o *Observer
	o.Observe(100, 50) // must not panic
	if s := o.Snapshot(); s.Turns != 0 {
		t.Fatalf("nil observer snapshot turns = %d, want 0", s.Turns)
	}
}

func TestIdleRatioIsZero(t *testing.T) {
	if s := New().Snapshot(); s.ReuseRatio != 0 {
		t.Fatalf("idle reuse ratio = %v, want 0 (no phantom ratio)", s.ReuseRatio)
	}
}

func TestConcurrentObserveIsRace_free(t *testing.T) {
	o := New()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				o.Observe(100, 90)
			}
		}()
	}
	wg.Wait()
	if s := o.Snapshot(); s.Turns != 5000 {
		t.Fatalf("turns = %d, want 5000", s.Turns)
	}
}
