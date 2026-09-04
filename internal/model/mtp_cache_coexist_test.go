package model

import (
	"testing"
)

func TestMTPCacheCoexist_EmptyDataSpecTolerance(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	t.Cleanup(promptCache.Clear)

	cfg := DefaultMTPCacheCoexistConfig()
	cfg.EmptyDataSpecTolerance = true

	session, err := NewMTPCacheCoexistSession(target, promptCache, cfg)
	if err != nil {
		t.Fatalf("failed to create MTPCacheCoexistSession: %v", err)
	}
	t.Cleanup(session.Close)

	prompt := []int{0, 1, 2, 1}
	res, err := session.PrefillWithCache(prompt)
	if err != nil {
		t.Fatalf("prefill failed: %v", err)
	}
	if !res.DraftReady || !res.MTPRetained {
		t.Fatalf("expected MTP ready and retained, got %+v", res)
	}

	initialCacheLen := target.Cache.Len()
	if initialCacheLen != len(prompt) {
		t.Fatalf("target cache len = %d, want %d", initialCacheLen, len(prompt))
	}
	if promptCache.Len() != 1 {
		t.Fatalf("prompt cache len = %d, want 1", promptCache.Len())
	}

	// Capture a prefill-time checkpoint with EMPTY data_spec
	cp, err := session.Checkpoint(true, nil)
	if err != nil {
		t.Fatalf("checkpoint failed: %v", err)
	}
	if len(cp.DataSpec) != 0 {
		t.Fatalf("expected empty DataSpec in prefill checkpoint, got %v", cp.DataSpec)
	}

	// Rollback with empty data_spec tolerance enabled
	if err := session.Rollback(cp); err != nil {
		t.Fatalf("rollback with empty data_spec failed: %v", err)
	}

	// Verify KV cache and prompt cache are retained!
	if target.Cache.Len() != initialCacheLen {
		t.Fatalf("target cache wiped or corrupted! len = %d, want %d", target.Cache.Len(), initialCacheLen)
	}
	if promptCache.Len() != 1 {
		t.Fatalf("prompt cache wiped! len = %d, want 1", promptCache.Len())
	}
	if session.EmptySpecRollbackCount != 1 {
		t.Fatalf("EmptySpecRollbackCount = %d, want 1", session.EmptySpecRollbackCount)
	}

	// Now test with tolerance disabled: should wipe cache
	session.Config.EmptyDataSpecTolerance = false
	cp2, err := session.Checkpoint(true, []int{})
	if err != nil {
		t.Fatalf("checkpoint 2 failed: %v", err)
	}
	err = session.Rollback(cp2)
	if err == nil {
		t.Fatal("expected error on empty data_spec rollback when tolerance is disabled")
	}
	if target.Cache.Len() != 0 {
		t.Fatalf("expected target cache wiped (len 0), got %d", target.Cache.Len())
	}
	if promptCache.Len() != 0 {
		t.Fatalf("expected prompt cache wiped (len 0), got %d", promptCache.Len())
	}
}

func TestMTPCacheCoexist_RecurrentRollbackWithinDepth(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	t.Cleanup(promptCache.Clear)

	cfg := DefaultMTPCacheCoexistConfig()
	cfg.MaxRecurrentRollbackDepth = 4

	session, err := NewMTPCacheCoexistSession(target, promptCache, cfg)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	t.Cleanup(session.Close)

	// Step 1: Initial prefill
	prompt1 := []int{0, 1, 2, 0, 1, 2}
	res1, err := session.PrefillWithCache(prompt1)
	if err != nil {
		t.Fatalf("initial prefill: %v", err)
	}
	if res1.CacheHit {
		t.Fatal("expected cold prefill for prompt1")
	}

	// Step 2: New prompt shares first 4 tokens {0, 1, 2, 0}, diverges by 2 tokens {2, 1}
	prompt2 := []int{0, 1, 2, 0, 2, 1}
	res2, err := session.PrefillWithCache(prompt2)
	if err != nil {
		t.Fatalf("divergent prefill: %v", err)
	}

	if !res2.CacheHit {
		t.Fatal("expected cache hit for shared prefix")
	}
	if res2.CachedTokens != 4 {
		t.Fatalf("cached tokens = %d, want 4", res2.CachedTokens)
	}
	if res2.Divergence != 2 {
		t.Fatalf("divergence = %d, want 2", res2.Divergence)
	}
	if !res2.RollbackApplied {
		t.Fatal("expected RollbackApplied to be true")
	}
	if res2.ColdPrefillFallback {
		t.Fatal("expected recurrent rollback, not cold prefill fallback")
	}
	if !res2.DraftReady || !res2.MTPRetained {
		t.Fatalf("expected MTP retained and draft ready, got %+v", res2)
	}

	// Verify MTP draft proposals work after recurrent rollback
	draft, err := session.Draft(prompt2)
	if err != nil {
		t.Fatalf("draft proposal after recurrent rollback failed: %v", err)
	}
	if len(draft) != cfg.DraftDepth {
		t.Fatalf("draft length = %d, want %d", len(draft), cfg.DraftDepth)
	}
}

// TestMTPCacheCoexistPromptDivergenceRollbackAndFallback binds the witness cited in docs/claims.
func TestMTPCacheCoexistPromptDivergenceRollbackAndFallback(t *testing.T) {
	t.Run("RecurrentRollbackWithinDepth", TestMTPCacheCoexist_RecurrentRollbackWithinDepth)
	t.Run("PromptDivergenceExceedsRollbackDepth_GracefulFallback", TestMTPCacheCoexist_PromptDivergenceExceedsRollbackDepth_GracefulFallback)
}

func TestMTPCacheCoexist_PromptDivergenceExceedsRollbackDepth_GracefulFallback(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	t.Cleanup(promptCache.Clear)

	cfg := DefaultMTPCacheCoexistConfig()
	cfg.MaxRecurrentRollbackDepth = 4

	session, err := NewMTPCacheCoexistSession(target, promptCache, cfg)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	t.Cleanup(session.Close)

	// Cached prompt has 10 tokens
	prompt1 := []int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0}
	_, err = session.PrefillWithCache(prompt1)
	if err != nil {
		t.Fatalf("initial prefill: %v", err)
	}

	// New prompt diverges at index 2 (shares {0, 1}, divergence = 10 - 2 = 8 tokens > 4)
	prompt2 := []int{0, 1, 1, 2, 0}
	res2, err := session.PrefillWithCache(prompt2)
	if err != nil {
		t.Fatalf("divergent prefill should not error: %v", err)
	}

	// Must trigger graceful fallback
	if !res2.ColdPrefillFallback {
		t.Fatal("expected ColdPrefillFallback to be true when divergence exceeds rollback depth")
	}
	if res2.RollbackApplied {
		t.Fatal("expected RollbackApplied to be false for fallback")
	}
	if !res2.MTPRetained || !res2.DraftReady {
		t.Fatalf("expected MTP retained and draft ready after fallback, got %+v", res2)
	}
	if session.ColdPrefillFallbackCount != 1 {
		t.Fatalf("ColdPrefillFallbackCount = %d, want 1", session.ColdPrefillFallbackCount)
	}

	// Prompt cache store entries must survive!
	if promptCache.Len() < 1 {
		t.Fatalf("prompt cache should retain prior entries, got len %d", promptCache.Len())
	}

	// Draft recovery: Proposing drafts on the new prompt succeeds!
	draft, err := session.Draft(prompt2)
	if err != nil {
		t.Fatalf("draft proposal failed after cold fallback: %v", err)
	}
	if len(draft) != cfg.DraftDepth {
		t.Fatalf("draft length = %d, want %d", len(draft), cfg.DraftDepth)
	}
}

func TestMTPCacheCoexist_MultiTurnCacheRetentionAndDraftRecovery(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	t.Cleanup(promptCache.Clear)

	cfg := DefaultMTPCacheCoexistConfig()
	session, err := NewMTPCacheCoexistSession(target, promptCache, cfg)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}
	t.Cleanup(session.Close)

	// Turn 1: Cold prefill prompt A
	turn1Prompt := []int{0, 1, 2}
	res1, err := session.PrefillWithCache(turn1Prompt)
	if err != nil || res1.CacheHit {
		t.Fatalf("turn 1: err=%v, hit=%v", err, res1.CacheHit)
	}
	draft1, err := session.Draft(turn1Prompt)
	if err != nil || len(draft1) != cfg.DraftDepth {
		t.Fatalf("turn 1 draft: err=%v, len=%d", err, len(draft1))
	}

	// Turn 2: Extension of prompt A (cache hit) + prefill checkpoint with empty data_spec
	turn2Prompt := append(append([]int(nil), turn1Prompt...), 0, 1)
	res2, err := session.PrefillWithCache(turn2Prompt)
	if err != nil || !res2.CacheHit {
		t.Fatalf("turn 2: err=%v, hit=%v", err, res2.CacheHit)
	}
	if res2.CachedTokens != len(turn1Prompt) {
		t.Fatalf("turn 2 cached tokens = %d, want %d", res2.CachedTokens, len(turn1Prompt))
	}

	cp2, err := session.Checkpoint(true, nil)
	if err != nil {
		t.Fatalf("turn 2 checkpoint: %v", err)
	}
	if err := session.Rollback(cp2); err != nil {
		t.Fatalf("turn 2 empty data_spec rollback: %v", err)
	}
	if target.Cache.Len() != len(turn2Prompt) {
		t.Fatalf("turn 2 cache wiped: len=%d, want %d", target.Cache.Len(), len(turn2Prompt))
	}

	// Turn 3: Severely divergent prompt -> graceful fallback
	turn3Prompt := []int{1, 2, 0, 1}
	res3, err := session.PrefillWithCache(turn3Prompt)
	if err != nil {
		t.Fatalf("turn 3 prefill: %v", err)
	}
	if !res3.ColdPrefillFallback {
		t.Fatal("turn 3 expected cold prefill fallback")
	}

	// Turn 4: Subsequent query proves draft recovery!
	turn4Prompt := append(append([]int(nil), turn3Prompt...), 2)
	res4, err := session.PrefillWithCache(turn4Prompt)
	if err != nil {
		t.Fatalf("turn 4 prefill: %v", err)
	}
	if !res4.CacheHit {
		t.Fatal("turn 4 expected cache hit on prompt 3 prefix")
	}
	draft4, err := session.Draft(turn4Prompt)
	if err != nil {
		t.Fatalf("turn 4 draft recovery failed: %v", err)
	}
	if len(draft4) != cfg.DraftDepth {
		t.Fatalf("turn 4 draft length = %d, want %d", len(draft4), cfg.DraftDepth)
	}
}

func TestMTPCacheCoexist_SessionErrorsAndEdgeCases(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	target := m.NewSession()
	t.Cleanup(target.Close)

	promptCache := NewMTPPromptCache()
	t.Cleanup(promptCache.Clear)

	cfg := DefaultMTPCacheCoexistConfig()
	session, err := NewMTPCacheCoexistSession(target, promptCache, cfg)
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	// Empty prompt error
	if _, err := session.PrefillWithCache(nil); err != ErrMTPCacheEmptyPrompt {
		t.Fatalf("expected ErrMTPCacheEmptyPrompt, got %v", err)
	}
	if _, err := session.PrefillWithCache([]int{}); err != ErrMTPCacheEmptyPrompt {
		t.Fatalf("expected ErrMTPCacheEmptyPrompt for empty slice, got %v", err)
	}

	// Invalid checkpoint error
	if err := session.Rollback(nil); err != ErrMTPCacheInvalidCheckpoint {
		t.Fatalf("expected ErrMTPCacheInvalidCheckpoint for nil cp, got %v", err)
	}
	otherCP := &MTPCacheCheckpoint{}
	if err := session.Rollback(otherCP); err != ErrMTPCacheInvalidCheckpoint {
		t.Fatalf("expected ErrMTPCacheInvalidCheckpoint for unowned cp, got %v", err)
	}

	// Close session and verify ErrMTPCacheSessionClosed
	session.Close()
	if _, err := session.PrefillWithCache([]int{1, 2}); err != ErrMTPCacheSessionClosed {
		t.Fatalf("expected ErrMTPCacheSessionClosed on prefill, got %v", err)
	}
	if _, err := session.Checkpoint(true, nil); err != ErrMTPCacheSessionClosed {
		t.Fatalf("expected ErrMTPCacheSessionClosed on checkpoint, got %v", err)
	}
	if err := session.Rollback(otherCP); err != ErrMTPCacheSessionClosed {
		t.Fatalf("expected ErrMTPCacheSessionClosed on rollback, got %v", err)
	}
	if _, err := session.Draft([]int{1, 2}); err != ErrMTPCacheSessionClosed {
		t.Fatalf("expected ErrMTPCacheSessionClosed on draft, got %v", err)
	}

	// Multiple close should be safe
	session.Close()
}

func TestMTPCache_PromptCacheConcurrentThreadSafety(t *testing.T) {
	cache := NewMTPPromptCache()
	t.Cleanup(cache.Clear)

	const goroutines = 8
	const iterations = 50

	done := make(chan struct{}, goroutines)
	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer func() { done <- struct{}{} }()
			for i := 0; i < iterations; i++ {
				p := []int{id, i % 5, i % 3}
				_ = cache.Put(p, nil)
				_ = cache.Get(p)
				_, _ = cache.MatchPrefix([]int{id, i % 5})
				_ = cache.Len()
			}
		}(g)
	}

	for g := 0; g < goroutines; g++ {
		<-done
	}

	if cache.Len() <= 0 {
		t.Fatal("expected positive prompt cache length after concurrent runs")
	}

	// Nil cache safety
	var nilCache *MTPPromptCache
	if nilCache.Len() != 0 {
		t.Fatal("nilCache.Len should return 0")
	}
	if nilCache.Get([]int{1}) != nil {
		t.Fatal("nilCache.Get should return nil")
	}
	if e, l := nilCache.MatchPrefix([]int{1}); e != nil || l != 0 {
		t.Fatal("nilCache.MatchPrefix should return nil, 0")
	}
	if err := nilCache.Put([]int{1}, nil); err != nil {
		t.Fatal("nilCache.Put should return nil")
	}
	nilCache.Clear() // should not panic
}
