package engine

// recurrent_prefix_test.go — independent TEST-AUTHOR witnesses for the
// oss-port-p0-recurrent-prefix-reuse ticket. These probe the FROZEN exported
// contract (RecurrentPrefixCache + the ContinuousBatcher wiring), NOT the
// implementation's internals. Each case names the failure it would catch:
//
//   - strict-extend TAKE semantics (an entry consumed by a hit is gone; a
//     re-issue of the same prompt is a MISS, not a hit),
//   - extend-only refusals on an irreversible GDN fold (backward rewind,
//     prefix divergence) and empty-input refusals,
//   - LRU bounding at capacity plus the documented default,
//   - engine wiring: reuse fires ONLY for a hybrid model with reuse enabled,
//     and is fully disabled otherwise,
//   - the decisive behavioral-equivalence witness: a reused turn-2 decode
//     produces EXACTLY the token stream of a cold full-prompt prefill.

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// rpHybridConfig is the smallest synthetic Qwen3.5-family hybrid config: three
// Gated-DeltaNet linear-attention layers and one gated full-attention layer, so
// model.Config.IsHybrid() is true and the recurrent fold is genuinely exercised.
func rpHybridConfig() model.Config {
	return model.Config{
		Name:                  "rp-hybrid",
		HiddenSize:            32,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      8,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    8,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

// rpHybridModel builds a fresh deterministic synthetic hybrid model.
func rpHybridModel() *model.Model { return model.NewSynthetic(rpHybridConfig()) }

// rpSnapshot builds a real PrefixSnapshot by prefilling a throwaway session,
// using exactly the session construction the batcher itself uses. The stored
// token list is supplied to the cache independently, so the snapshot only needs
// to be a non-nil, restorable boundary.
func rpSnapshot(t *testing.T, m *model.Model) *model.PrefixSnapshot {
	t.Helper()
	s := &model.Session{M: m, Cache: model.NewKVCache(m.Cfg)}
	s.Prefill([]int{1, 2, 3})
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	if snap == nil {
		t.Fatal("PrefixSnapshot returned a nil snapshot")
	}
	return snap
}

// rpDrive runs StepPhase until sessionID's slot retires with at least target
// generated tokens and returns a copy of the full generated token stream. It is
// bounded so a wedged scheduler fails the test instead of hanging.
func rpDrive(t *testing.T, cb *ContinuousBatcher, sessionID string, target int) []int {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < 1000; i++ {
		slot, ok := cb.GetSlot(sessionID)
		if !ok {
			t.Fatalf("slot %q not found while driving", sessionID)
		}
		if slot.State == SlotStateFinished && slot.TokensGenerated() >= target {
			return append([]int(nil), slot.GeneratedTokens...)
		}
		if _, err := cb.StepPhase(ctx); err != nil {
			t.Fatalf("StepPhase for %q failed at iteration %d: %v", sessionID, i, err)
		}
	}
	t.Fatalf("session %q never retired within the step bound", sessionID)
	return nil
}

// TestRecurrentPrefixCacheStrictExtendTakeSemantics pins the TAKE contract: a
// strict-extend lookup returns the held length and the snapshot, REMOVES the
// entry, and a second identical lookup is therefore a miss.
func TestRecurrentPrefixCacheStrictExtendTakeSemantics(t *testing.T) {
	t.Parallel()
	m := rpHybridModel()
	c := NewRecurrentPrefixCache(0)

	if err := c.Store("c", []int{1, 2, 3}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store(c) failed: %v", err)
	}

	matched, snap, hit := c.Lookup("c", []int{1, 2, 3, 4})
	if !hit {
		t.Fatal("strict-extend lookup = miss, want hit")
	}
	if matched != 3 {
		t.Fatalf("matched = %d, want 3", matched)
	}
	if snap == nil {
		t.Fatal("hit returned a nil snapshot")
	}
	snap.Close() // caller owns the taken snapshot

	// The entry was TAKEN: an identical re-issue must miss.
	if matched2, snap2, hit2 := c.Lookup("c", []int{1, 2, 3, 4}); hit2 {
		t.Fatalf("second lookup = hit (matched=%d, snap nil=%v), want miss: take semantics violated", matched2, snap2 == nil)
	}

	stats := c.Stats()
	if stats.Hits != 1 || stats.Misses != 1 {
		t.Fatalf("Stats hits=%d misses=%d, want 1 hit / 1 miss", stats.Hits, stats.Misses)
	}
}

// TestRecurrentPrefixCacheIdenticalPromptMiss pins the strict-extend boundary:
// re-issuing the EXACT stored prompt (or a shorter one) is a miss, because there
// is no remaining suffix to prefill.
func TestRecurrentPrefixCacheIdenticalPromptMiss(t *testing.T) {
	t.Parallel()
	m := rpHybridModel()
	c := NewRecurrentPrefixCache(0)
	if err := c.Store("c", []int{1, 2, 3}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store failed: %v", err)
	}
	defer c.Evict("c")

	if _, snap, hit := c.Lookup("c", []int{1, 2, 3}); hit {
		snap.Close()
		t.Fatal("identical-prompt lookup = hit, want miss (no suffix remains)")
	}
	if _, snap, hit := c.Lookup("c", []int{1, 2}); hit {
		snap.Close()
		t.Fatal("shorter-prompt lookup = hit, want miss")
	}
	if got := c.Stats().Misses; got != 2 {
		t.Fatalf("Misses = %d, want 2", got)
	}
	if got := c.Stats().Hits; got != 0 {
		t.Fatalf("Hits = %d, want 0", got)
	}

	// Adversarial: an empty key or empty prompt can never match, even though a
	// populated entry exists under a different key.
	if _, snap, hit := c.Lookup("", []int{1, 2, 3, 4}); hit {
		snap.Close()
		t.Fatal("Lookup(empty key) = hit, want miss")
	}
	if _, snap, hit := c.Lookup("c", nil); hit {
		snap.Close()
		t.Fatal("Lookup(empty prompt) = hit, want miss")
	}
	// The real entry must have SURVIVED those two misses (a miss is not a take).
	if _, snap, hit := c.Lookup("c", []int{1, 2, 3, 4}); !hit {
		t.Fatal("entry lost after empty-key/prompt misses; a miss must not consume the entry")
	} else {
		snap.Close()
	}
}

// TestRecurrentPrefixCacheRefusesRewindAndDivergence pins the extend-only
// invariant on an irreversible linear-recurrence fold: a shorter store is a
// backward-rewind refusal, a divergent store is a divergence refusal, and each
// refusal increments Stats.Refusals.
func TestRecurrentPrefixCacheRefusesRewindAndDivergence(t *testing.T) {
	t.Parallel()
	m := rpHybridModel()
	c := NewRecurrentPrefixCache(0)
	if err := c.Store("c", []int{1, 2, 3}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store failed: %v", err)
	}
	defer c.Evict("c")

	err := c.Store("c", []int{1, 2}, rpSnapshot(t, m))
	if !errors.Is(err, ErrRecurrentPrefixBackwardRewind) {
		t.Fatalf("shorter Store err = %v, want ErrRecurrentPrefixBackwardRewind", err)
	}

	err = c.Store("c", []int{1, 9, 3, 4}, rpSnapshot(t, m))
	if !errors.Is(err, ErrRecurrentPrefixDivergence) {
		t.Fatalf("divergent Store err = %v, want ErrRecurrentPrefixDivergence", err)
	}

	if got := c.Stats().Refusals; got != 2 {
		t.Fatalf("Refusals = %d, want 2", got)
	}
	if got := c.Stats().Puts; got != 1 {
		t.Fatalf("Puts = %d, want 1 (only the original accepted store)", got)
	}
}

// TestRecurrentPrefixCacheRefusesEmptyInputs pins the closed empty-input
// refusals: an empty key, and a nil/empty token list (or nil snapshot).
func TestRecurrentPrefixCacheRefusesEmptyInputs(t *testing.T) {
	t.Parallel()
	m := rpHybridModel()
	c := NewRecurrentPrefixCache(0)

	if err := c.Store("", []int{1}, rpSnapshot(t, m)); !errors.Is(err, ErrRecurrentPrefixEmptyKey) {
		t.Fatalf("Store(empty key) err = %v, want ErrRecurrentPrefixEmptyKey", err)
	}
	if err := c.Store("c", nil, rpSnapshot(t, m)); !errors.Is(err, ErrRecurrentPrefixEmptyTokens) {
		t.Fatalf("Store(nil tokens) err = %v, want ErrRecurrentPrefixEmptyTokens", err)
	}
	if err := c.Store("c", []int{1}, nil); !errors.Is(err, ErrRecurrentPrefixEmptyTokens) {
		t.Fatalf("Store(nil snapshot) err = %v, want ErrRecurrentPrefixEmptyTokens", err)
	}
	if got := c.Stats().Refusals; got != 3 {
		t.Fatalf("Refusals = %d, want 3", got)
	}
	if got := c.Len(); got != 0 {
		t.Fatalf("Len = %d after only refusals, want 0", got)
	}
}

// TestRecurrentPrefixCacheLRUCapacityAndDefault pins the documented default when
// capacity <= 0 and LRU eviction at capacity: a capacity-2 cache storing three
// distinct keys keeps Len==2 and evicts at least the oldest.
func TestRecurrentPrefixCacheLRUCapacityAndDefault(t *testing.T) {
	t.Parallel()
	m := rpHybridModel()

	if got := NewRecurrentPrefixCache(0).Capacity(); got != DefaultRecurrentPrefixCapacity {
		t.Fatalf("NewRecurrentPrefixCache(0).Capacity() = %d, want %d", got, DefaultRecurrentPrefixCapacity)
	}
	if got := NewRecurrentPrefixCache(-1).Capacity(); got != DefaultRecurrentPrefixCapacity {
		t.Fatalf("NewRecurrentPrefixCache(-1).Capacity() = %d, want %d", got, DefaultRecurrentPrefixCapacity)
	}

	c := NewRecurrentPrefixCache(2)
	if err := c.Store("a", []int{1, 2, 3}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store(a): %v", err)
	}
	if err := c.Store("b", []int{4, 5, 6}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store(b): %v", err)
	}
	if err := c.Store("c", []int{7, 8, 9}, rpSnapshot(t, m)); err != nil {
		t.Fatalf("Store(c): %v", err)
	}

	if got := c.Len(); got != 2 {
		t.Fatalf("Len = %d, want 2 (capacity-bounded)", got)
	}
	if got := c.Stats().Evictions; got < 1 {
		t.Fatalf("Evictions = %d, want >= 1", got)
	}
	// The oldest key ("a") must have been the one evicted.
	if _, _, hit := c.Lookup("a", []int{1, 2, 3, 4}); hit {
		t.Fatal("key \"a\" survived in a capacity-2 LRU cache; the oldest entry was not evicted")
	}
}

// TestRecurrentPrefixBatcherWiringHitAndDisable pins the engine wiring: reuse is
// enabled only for a hybrid model with reuse NOT disabled. Turn 1 must retain a
// boundary (Puts>=1, Len>=1); a strictly-extending turn 2 must report a hit and
// reuse the matched prefix token count. With DisableRecurrentPrefixReuse the
// cache is absent and no reuse is ever reported.
func TestRecurrentPrefixBatcherWiringHitAndDisable(t *testing.T) {
	turn1 := []int{1, 2, 3}
	turn2 := []int{1, 2, 3, 4, 5, 6}

	newBatcher := func(t *testing.T, disable bool) *ContinuousBatcher {
		t.Helper()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.PrefillBudget = 2 // chunked prefill
		cfg.Model = rpHybridModel()
		cfg.DisableRecurrentPrefixReuse = disable
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher: %v", err)
		}
		t.Cleanup(func() { _ = cb.Close() })
		return cb
	}

	t.Run("hybrid_enabled_reuses", func(t *testing.T) {
		cb := newBatcher(t, false)
		if _, enabled := cb.PrefixCacheStats(); !enabled {
			t.Fatal("PrefixCacheStats enabled = false for a hybrid model with reuse on, want true")
		}

		if _, err := cb.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: turn1, TargetTokens: 2, ChunkedPrefill: true}); err != nil {
			t.Fatalf("Submit(turn1): %v", err)
		}
		rpDrive(t, cb, "conv", 2)

		stats1, _ := cb.PrefixCacheStats()
		if stats1.Puts < 1 {
			t.Fatalf("after turn 1 Puts = %d, want >= 1", stats1.Puts)
		}
		if got := cb.PrefixCacheLen(); got < 1 {
			t.Fatalf("after turn 1 PrefixCacheLen = %d, want >= 1", got)
		}

		if _, err := cb.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: turn2, TargetTokens: 2, ChunkedPrefill: true}); err != nil {
			t.Fatalf("Submit(turn2): %v", err)
		}
		res, err := cb.StepPhase(context.Background())
		if err != nil {
			t.Fatalf("StepPhase(turn2): %v", err)
		}
		if res.PrefixHits < 1 {
			t.Fatalf("turn-2 admission PrefixHits = %d, want >= 1 (reuse not wired)", res.PrefixHits)
		}
		if res.PrefixReuseTokens <= 0 {
			t.Fatalf("turn-2 PrefixReuseTokens = %d, want > 0", res.PrefixReuseTokens)
		}
		stats2, _ := cb.PrefixCacheStats()
		if stats2.Hits < 1 {
			t.Fatalf("Stats.Hits = %d after an extending turn 2, want >= 1", stats2.Hits)
		}
	})

	t.Run("disabled_no_cache_no_reuse", func(t *testing.T) {
		cb := newBatcher(t, true)
		if _, enabled := cb.PrefixCacheStats(); enabled {
			t.Fatal("PrefixCacheStats enabled = true with DisableRecurrentPrefixReuse, want false")
		}
		if got := cb.PrefixCacheLen(); got != 0 {
			t.Fatalf("PrefixCacheLen = %d with reuse disabled, want 0", got)
		}

		if _, err := cb.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: turn1, TargetTokens: 2, ChunkedPrefill: true}); err != nil {
			t.Fatalf("Submit(turn1): %v", err)
		}
		rpDrive(t, cb, "conv", 2)
		if _, err := cb.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: turn2, TargetTokens: 2, ChunkedPrefill: true}); err != nil {
			t.Fatalf("Submit(turn2): %v", err)
		}
		res, err := cb.StepPhase(context.Background())
		if err != nil {
			t.Fatalf("StepPhase(turn2): %v", err)
		}
		if res.PrefixHits != 0 || res.PrefixReuseTokens != 0 {
			t.Fatalf("disabled reuse reported hits=%d reuseTokens=%d, want 0/0", res.PrefixHits, res.PrefixReuseTokens)
		}
	})
}

// TestRecurrentPrefixBehavioralEquivalence is the DECISIVE witness: a turn 2
// that reuses a stored recurrent boundary and prefills only the suffix must
// generate the SAME token stream as a cold prefill of the full turn-2 prompt.
//
// Two independent batchers over identical deterministic synthetic hybrid models:
//
//	(A) reuse enabled: turn 1 (prefix) then turn 2 (strict extension);
//	(B) reuse disabled baseline: one fresh request over the full turn-2 prompt.
//
// If the streams differ, reuse is NOT bit-exact and this test FAILS loudly.
func TestRecurrentPrefixBehavioralEquivalence(t *testing.T) {
	prefix := []int{1, 2, 3}
	suffix := []int{4, 5, 6}
	full := append(append([]int(nil), prefix...), suffix...)
	const target = 3

	// (A) reuse-enabled: turn 1 then an extending turn 2.
	cfgA := DefaultContinuousBatcherConfig()
	cfgA.MaxSlots = 2
	cfgA.PrefillBudget = 2
	cfgA.Model = rpHybridModel()
	cbA, err := NewContinuousBatcher(cfgA)
	if err != nil {
		t.Fatalf("NewContinuousBatcher(A): %v", err)
	}
	defer func() { _ = cbA.Close() }()

	if _, err := cbA.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: prefix, TargetTokens: 2, ChunkedPrefill: true}); err != nil {
		t.Fatalf("Submit(A turn1): %v", err)
	}
	rpDrive(t, cbA, "conv", 2)
	if got := cbA.PrefixCacheLen(); got < 1 {
		t.Fatalf("A retained no boundary after turn 1 (Len=%d); equivalence check would be vacuous", got)
	}

	if _, err := cbA.Submit(&SubagentRequest{SessionID: "conv", PromptTokens: full, TargetTokens: target, ChunkedPrefill: true}); err != nil {
		t.Fatalf("Submit(A turn2): %v", err)
	}
	reuseTokens := rpDrive(t, cbA, "conv", target)

	// Non-vacuity guard: turn 2 must ACTUALLY have taken a stored boundary,
	// otherwise this would compare two cold prefills and prove nothing.
	if statsA, _ := cbA.PrefixCacheStats(); statsA.Hits < 1 {
		t.Fatalf("turn 2 took no cached boundary (Hits=%d); equivalence check would be vacuous", statsA.Hits)
	}

	// (B) reuse-disabled baseline: one fresh full-prompt request.
	cfgB := DefaultContinuousBatcherConfig()
	cfgB.MaxSlots = 2
	cfgB.PrefillBudget = 2
	cfgB.Model = rpHybridModel()
	cfgB.DisableRecurrentPrefixReuse = true
	cbB, err := NewContinuousBatcher(cfgB)
	if err != nil {
		t.Fatalf("NewContinuousBatcher(B): %v", err)
	}
	defer func() { _ = cbB.Close() }()

	if _, err := cbB.Submit(&SubagentRequest{SessionID: "cold", PromptTokens: full, TargetTokens: target, ChunkedPrefill: true}); err != nil {
		t.Fatalf("Submit(B): %v", err)
	}
	coldTokens := rpDrive(t, cbB, "cold", target)

	if len(reuseTokens) != len(coldTokens) {
		t.Fatalf("token stream length differs: reuse=%v (len %d) vs cold=%v (len %d)",
			reuseTokens, len(reuseTokens), coldTokens, len(coldTokens))
	}
	t.Logf("reuse stream=%v cold stream=%v (bit-identical)", reuseTokens, coldTokens)
	for i := range coldTokens {
		if reuseTokens[i] != coldTokens[i] {
			t.Fatalf("reuse is NOT bit-exact: token[%d] reuse=%d cold=%d\n reuse=%v\n cold =%v",
				i, reuseTokens[i], coldTokens[i], reuseTokens, coldTokens)
		}
	}
}

// TestRecurrentPrefixBatcherNonHybridNoCache pins the model gate: a non-hybrid
// synthetic model — and the nil-model case — never build the cache, so
// PrefixCacheStats reports disabled and PrefixCacheLen is 0.
func TestRecurrentPrefixBatcherNonHybridNoCache(t *testing.T) {
	t.Parallel()

	t.Run("dense_model", func(t *testing.T) {
		t.Parallel()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cfg.Model = model.NewSynthetic(wireModelConfig()) // no linear_attention layers
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher: %v", err)
		}
		defer func() { _ = cb.Close() }()

		if _, enabled := cb.PrefixCacheStats(); enabled {
			t.Fatal("PrefixCacheStats enabled = true for a non-hybrid model, want false")
		}
		if got := cb.PrefixCacheLen(); got != 0 {
			t.Fatalf("PrefixCacheLen = %d for a non-hybrid model, want 0", got)
		}
	})

	t.Run("nil_model", func(t *testing.T) {
		t.Parallel()
		cfg := DefaultContinuousBatcherConfig()
		cfg.MaxSlots = 2
		cb, err := NewContinuousBatcher(cfg)
		if err != nil {
			t.Fatalf("NewContinuousBatcher: %v", err)
		}
		defer func() { _ = cb.Close() }()

		if _, enabled := cb.PrefixCacheStats(); enabled {
			t.Fatal("PrefixCacheStats enabled = true with a nil model, want false")
		}
		if got := cb.PrefixCacheLen(); got != 0 {
			t.Fatalf("PrefixCacheLen = %d with a nil model, want 0", got)
		}
	})
}
