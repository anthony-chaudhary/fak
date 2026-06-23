package model

// This file holds the bounded-memory WINDOWED-decode API — the follow-on to the
// sliding-window read-mask (windowLo / windowLoContig in weights.go). The read-mask
// is the correctness precondition for DROPPING aged-out K/V: a position older than
// every layer's window is provably never attended to again, so removing it leaves
// every future token's output unchanged. MaxWindow + TrimToWindow turn that fact
// into a bounded cache, O(window) for a stream of any length, through the proven
// KVCache.Evict (which renumbers and re-RoPEs the survivors). Because RoPE is
// relative, renumbering the surviving contiguous window preserves every
// within-window attention score, so a trimmed windowed decode is argmax-identical
// to the same decode over the full cache (TestBoundedWindowMatchesFullWindow).
//
// Both are opt-in: no production decode path calls TrimToWindow, and it is a no-op
// when no window is configured, so non-SWA (Llama/Qwen) models are byte-identical
// to before. Nothing here touches the frozen ABI.

// MaxWindow returns the widest per-layer sliding window when EVERY layer configures
// one, else 0. It is the keep-count a bounded-window decode trims down to: a
// position older than every layer's window is provably un-attended, so its K/V may
// be dropped without changing any future token. A single full-causal layer (no
// window, or an unset entry) attends to every position, so no trim is ever safe —
// that yields 0, which TrimToWindow reads as "unbounded / do not trim". 0 is
// therefore the sentinel for both "no window" and "not safely trimmable".
func (c Config) MaxWindow() int {
	max := 0
	for l := 0; l < c.NumLayers; l++ {
		w := c.windowForLayer(l) // -1 (or <=0) means full causal / unset for this layer
		if w <= 0 {
			return 0
		}
		if w > max {
			max = w
		}
	}
	return max
}

// TrimToWindow bounds the session's KV cache to O(window) for a stream of any
// length. When the cache grows past MaxWindow()+slack, it evicts the oldest
// positions back down to MaxWindow() through the proven KVCache.Evict. The slack
// is hysteresis on WHEN to trim (peak cache is MaxWindow()+slack); the target
// after a trim is MaxWindow(). Returns positions evicted (0 when nothing changed).
//
// It is a no-op when no window is configured (MaxWindow()==0), so a non-SWA model
// is byte-identical to a run that never called it. A negative slack is treated as
// 0. It never trims below MaxWindow() itself.
func (s *Session) TrimToWindow(slack int) int {
	w := s.M.Cfg.MaxWindow()
	if w <= 0 {
		return 0 // no window configured -> nothing is safely droppable
	}
	if slack < 0 {
		slack = 0
	}
	if s.Cache.Len() <= w+slack {
		return 0
	}
	over := s.Cache.Len() - w
	return s.Cache.Evict(0, over)
}
