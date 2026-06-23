package model

import (
	"os"
	"strconv"
	"testing"
)

// These are the bounded-memory WINDOWED-decode witnesses the sliding-window doc
// (SLIDING-WINDOW-RESULTS.md) and v0.12.0 release notes cite for
// Session.TrimToWindow. They run on the weight-free synthetic model (swaTestCfg),
// so they are always available; the cross-path numerics vs HuggingFace are proven
// separately by the rung-3 oracle (TestKVQuarantineEqualsNeverSaw) that
// TrimToWindow composes via KVCache.Evict.

// TestBoundedWindowMatchesFullWindow is the argmax-identity witness: a windowed
// decode whose aged-out K/V is trimmed every step (so the cache peaks at 2·window)
// is ARGMAX-IDENTICAL, step for step (teacher-forced), to the same windowed decode
// over the full cache. This holds because TrimToWindow composes the proven
// KVCache.Evict, which renumbers survivors into a contiguous [0, w) window; RoPE is
// relative, so every within-window attention score is preserved exactly — the same
// argument TestSlidingWindowSurvivesEvict makes for a one-shot evict, exercised here
// as a rolling trim over a long stream.
func TestBoundedWindowMatchesFullWindow(t *testing.T) {
	const W = 4
	cfg := swaTestCfg()
	cfg.Window = []int{W, W}
	m := NewSynthetic(cfg)
	const slack = W // peak cache = W + slack = 2·W (the doc's "peaks at 2·window")

	// A deterministic teacher-forced stream well past the window so trimming fires
	// repeatedly. Vocab is 97 (swaTestCfg); ids are in range and non-degenerate.
	stream := []int{3, 17, 5, 23, 41, 2, 19, 8, 31, 14, 7, 11, 29, 13, 37, 6, 25, 9, 33, 21, 15, 27, 1, 10}

	full := m.NewSession()
	bounded := m.NewSession()
	fullLogits := full.Prefill(stream[:W])
	boundedLogits := bounded.Prefill(stream[:W])
	bounded.TrimToWindow(slack)

	var maxDelta float64
	steps := len(stream) - W
	for i := 0; i < steps; i++ {
		id := stream[W+i]
		fullLogits = full.Step(id)
		boundedLogits = bounded.Step(id)
		bounded.TrimToWindow(slack)

		if fa, ba := argmax(fullLogits), argmax(boundedLogits); fa != ba {
			t.Fatalf("step %d: argmax diverged — full=%d bounded=%d (cache lens %d vs %d)",
				i, fa, ba, full.Cache.Len(), bounded.Cache.Len())
		}
		if d, _ := maxAbsDiff(fullLogits, boundedLogits); d > maxDelta {
			maxDelta = d
		}
		if bounded.Cache.Len() > W+slack {
			t.Fatalf("step %d: bounded cache %d exceeded W+slack=%d", i, bounded.Cache.Len(), W+slack)
		}
	}

	// Non-vacuous: the full cache grew with the stream while the bounded one did not.
	if full.Cache.Len() != len(stream) {
		t.Fatalf("full cache len %d != stream len %d (the full-cache arm must grow)", full.Cache.Len(), len(stream))
	}
	if bounded.Cache.Len() > W+slack {
		t.Fatalf("final bounded cache %d > %d", bounded.Cache.Len(), W+slack)
	}
	t.Logf("OK: %d-step windowed decode argmax-identical under a rolling trim; full cache=%d, bounded peak≤%d, max|Δlogits|=%.3e",
		steps, full.Cache.Len(), W+slack, maxDelta)
}

// TestBoundedWindowMemoryIsBounded is the memory witness: the cache stays
// O(window) — never exceeding W+slack — over a long stream that would otherwise
// grow without bound. Default 4096 tokens (fast); FAK_LONGCTX_N raises it (the
// doc's million-token knob) to prove the bound holds at extreme length.
func TestBoundedWindowMemoryIsBounded(t *testing.T) {
	const W = 4
	n := 4096
	if v := os.Getenv("FAK_LONGCTX_N"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			n = parsed
		}
	}
	cfg := swaTestCfg()
	cfg.Window = []int{W, W}
	m := NewSynthetic(cfg)
	const slack = W

	s := m.NewSession()
	s.Prefill([]int{3, 17, 5, 23})
	peak := s.Cache.Len()
	id := 5
	for i := 0; i < n; i++ {
		s.Step(id)
		s.TrimToWindow(slack)
		if l := s.Cache.Len(); l > peak {
			peak = l
		}
		if s.Cache.Len() > W+slack {
			t.Fatalf("step %d: cache %d exceeded W+slack=%d — the bound broke", i, s.Cache.Len(), W+slack)
		}
	}
	if peak > W+slack {
		t.Fatalf("peak cache %d > W+slack=%d over a %d-token stream", peak, W+slack, n)
	}
	t.Logf("OK: %d-token stream, cache peaked at %d positions (W=%d, slack=%d) — O(window), not O(N)",
		n, peak, W, slack)
}

// TestTrimToWindowNoOpWithoutWindow guards the no-op story: with no window
// configured (the Llama/Qwen default), MaxWindow()==0 and TrimToWindow is a no-op,
// so a non-SWA model is byte-identical to a run that never called it.
func TestTrimToWindowNoOpWithoutWindow(t *testing.T) {
	m := NewSynthetic(swaTestCfg()) // Window nil -> MaxWindow()==0
	s := m.NewSession()
	s.Prefill([]int{3, 17, 5, 23, 41})
	before := s.Cache.Len()
	if ev := s.TrimToWindow(0); ev != 0 {
		t.Fatalf("TrimToWindow with no window evicted %d positions, want 0 (no-op)", ev)
	}
	if s.Cache.Len() != before {
		t.Fatalf("cache changed %d -> %d with no window configured", before, s.Cache.Len())
	}
	if w := s.M.Cfg.MaxWindow(); w != 0 {
		t.Fatalf("MaxWindow()=%d with no window, want 0", w)
	}
}
