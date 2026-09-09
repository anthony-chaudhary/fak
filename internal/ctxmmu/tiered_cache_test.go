package ctxmmu

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
)

// TestTieredPromptCache_EvictionPriorityAndTrimming is the primary verifiable witness for Issue #12593.
// It verifies:
//  1. Tiered role-based eviction policy: assistant (evict first) -> user (evict next) -> system (retain longest).
//     Under memory pressure, assistant responses are purged while system prompts remain preserved in cache.
//  2. Trimmable prefix slices: cached sequence with 1,000 tokens trimmed to 800 tokens in-place on buffer.
//  3. Zero-allocation logical trimming and prefix trie indexing.
func TestTieredPromptCache_EvictionPriorityAndTrimming(t *testing.T) {
	// =========================================================================
	// Part 1: Tiered Role-Based Eviction (Assistant -> User -> System)
	// =========================================================================
	t.Run("TieredEvictionPriority_SystemPreserved", func(t *testing.T) {
		// Capacity set to 400 tokens total
		cache := NewTieredPromptCacheWithMaxTokens(400)

		// Cache entries totaling 400 tokens
		// Fill with sized tokens
		makeTokens := func(start, count int) []int {
			toks := make([]int, count)
			for i := 0; i < count; i++ {
				toks[i] = start + i
			}
			return toks
		}

		sTokens := makeTokens(1000, 100)  // 100 tokens, System
		uTokens1 := makeTokens(2000, 100) // 100 tokens, User
		aTokens1 := makeTokens(3000, 100) // 100 tokens, Assistant 1
		aTokens2 := makeTokens(4000, 100) // 100 tokens, Assistant 2

		if _, err := cache.PutWithID("sys-prompt", RoleSystem, sTokens, nil); err != nil {
			t.Fatalf("failed to insert system prompt: %v", err)
		}
		if _, err := cache.PutWithID("user-turn-1", RoleUser, uTokens1, nil); err != nil {
			t.Fatalf("failed to insert user prompt 1: %v", err)
		}
		if _, err := cache.PutWithID("asst-turn-1", RoleAssistant, aTokens1, nil); err != nil {
			t.Fatalf("failed to insert assistant prompt 1: %v", err)
		}
		if _, err := cache.PutWithID("asst-turn-2", RoleAssistant, aTokens2, nil); err != nil {
			t.Fatalf("failed to insert assistant prompt 2: %v", err)
		}

		if cache.TotalTokens() != 400 {
			t.Fatalf("expected total tokens = 400, got %d", cache.TotalTokens())
		}
		if cache.Len() != 4 {
			t.Fatalf("expected 4 entries, got %d", cache.Len())
		}

		// Inject memory pressure: Insert new user turn of 100 tokens
		// Capacity is 400, adding 100 requires evicting 100 tokens.
		// Assistant 1 (oldest assistant turn) must be evicted first!
		uTokens2 := makeTokens(5000, 100)
		if _, err := cache.PutWithID("user-turn-2", RoleUser, uTokens2, nil); err != nil {
			t.Fatalf("failed to insert user prompt 2: %v", err)
		}

		if cache.TotalTokens() != 400 {
			t.Fatalf("expected total tokens to remain capped at 400, got %d", cache.TotalTokens())
		}

		// Verify Assistant 1 was evicted
		if cache.ContainsID("asst-turn-1") {
			t.Errorf("expected asst-turn-1 to be evicted first, but still found in cache")
		}
		// Verify System prompt is still in cache!
		if !cache.ContainsID("sys-prompt") {
			t.Errorf("system prompt must be retained under memory pressure")
		}
		// Verify User turn 1 is still in cache
		if !cache.ContainsID("user-turn-1") {
			t.Errorf("user-turn-1 should still be in cache")
		}
		// Verify Assistant 2 is still in cache
		if !cache.ContainsID("asst-turn-2") {
			t.Errorf("asst-turn-2 should still be in cache")
		}

		// Inject further memory pressure: Insert another user turn of 100 tokens
		// Assistant 2 (remaining assistant turn) must be evicted next!
		uTokens3 := makeTokens(6000, 100)
		if _, err := cache.PutWithID("user-turn-3", RoleUser, uTokens3, nil); err != nil {
			t.Fatalf("failed to insert user prompt 3: %v", err)
		}

		// Verify Assistant 2 was evicted
		if cache.ContainsID("asst-turn-2") {
			t.Errorf("expected asst-turn-2 to be evicted, but still found")
		}
		// Verify System prompt STILL retained!
		if !cache.ContainsID("sys-prompt") {
			t.Errorf("system prompt must remain retained after all assistant turns are evicted")
		}

		// Inject further memory pressure: Insert another 100 tokens
		// No assistant turns left! Now the oldest User turn (user-turn-1) must be evicted.
		uTokens4 := makeTokens(7000, 100)
		if _, err := cache.PutWithID("user-turn-4", RoleUser, uTokens4, nil); err != nil {
			t.Fatalf("failed to insert user prompt 4: %v", err)
		}

		// Verify User turn 1 evicted
		if cache.ContainsID("user-turn-1") {
			t.Errorf("expected user-turn-1 to be evicted when no assistant turns remain")
		}
		// Verify System prompt STILL retained!
		if !cache.ContainsID("sys-prompt") {
			t.Errorf("system prompt must remain retained above user turns")
		}
	})

	// =========================================================================
	// Part 2: Trimmable Prefix Slices (1,000 tokens trimmed to 800 tokens)
	// =========================================================================
	t.Run("TrimmablePrefixSlices_InPlaceReuse", func(t *testing.T) {
		cache := NewTieredPromptCacheWithMaxTokens(10000)

		// Create a cached sequence with 1,000 tokens
		seq1000 := make([]int, 1000)
		for i := 0; i < 1000; i++ {
			seq1000[i] = 100 + i
		}

		buffer := NewCacheBuffer(1000, 1000, 2)
		entry, err := cache.PutWithID("long-sequence", RoleSystem, seq1000, buffer)
		if err != nil {
			t.Fatalf("failed to cache 1000-token sequence: %v", err)
		}

		if entry.Len() != 1000 {
			t.Fatalf("expected entry length 1000, got %d", entry.Len())
		}
		if buffer.TrimCount != 0 {
			t.Fatalf("expected initial trim count 0, got %d", buffer.TrimCount)
		}

		// Incoming prompt matches first 800 tokens, then diverges
		incomingPrompt := make([]int, 850)
		copy(incomingPrompt, seq1000[:800])
		for i := 800; i < 850; i++ {
			incomingPrompt[i] = 99999 + i // divergent tokens
		}

		// Perform trimmable lookup
		matchedEntry, matchedLen, trimLen, ok := cache.LookupTrimmable(incomingPrompt)
		if !ok {
			t.Fatalf("expected trimmable match for prefix, got false")
		}
		if matchedEntry.ID != "long-sequence" {
			t.Fatalf("expected matched entry long-sequence, got %s", matchedEntry.ID)
		}
		if matchedLen != 800 {
			t.Fatalf("expected matched length 800, got %d", matchedLen)
		}
		if trimLen != 200 {
			t.Fatalf("expected trim length 200 (1000 - 800), got %d", trimLen)
		}

		// Execute in-place cache trimming directly on the buffer
		trimmedEntry, err := cache.TrimCache(seq1000, 800)
		if err != nil {
			t.Fatalf("TrimCache failed: %v", err)
		}

		if trimmedEntry.Len() != 800 {
			t.Fatalf("expected trimmed length 800, got %d", trimmedEntry.Len())
		}
		if buffer.SequenceLen != 800 {
			t.Fatalf("expected buffer sequence len 800, got %d", buffer.SequenceLen)
		}
		if buffer.TrimCount != 1 {
			t.Fatalf("expected buffer TrimCount == 1, got %d", buffer.TrimCount)
		}
		if buffer.Capacity != 1000 {
			t.Fatalf("buffer capacity should remain 1000 (no new allocation), got %d", buffer.Capacity)
		}

		// Verify cache total tokens dropped from 1000 to 800
		if cache.TotalTokens() != 800 {
			t.Fatalf("expected cache total tokens 800, got %d", cache.TotalTokens())
		}

		// An exact lookup for the 800-token prefix now hits directly
		exactEntry, found := cache.Get(seq1000[:800])
		if !found {
			t.Fatalf("expected exact match for trimmed 800-token prefix")
		}
		if exactEntry.ID != "long-sequence" {
			t.Fatalf("expected exactEntry ID long-sequence, got %s", exactEntry.ID)
		}
	})
}

func TestTieredPromptCache_PrefixTrie(t *testing.T) {
	trie := NewPromptTrie()

	entry1 := &PromptEntry{ID: "e1", Role: RoleSystem, Tokens: []int{1, 2, 3}}
	entry2 := &PromptEntry{ID: "e2", Role: RoleUser, Tokens: []int{1, 2, 3, 4, 5}}
	entry3 := &PromptEntry{ID: "e3", Role: RoleAssistant, Tokens: []int{1, 2, 3, 4, 5, 6, 7}}

	trie.Insert(entry1.Tokens, entry1)
	trie.Insert(entry2.Tokens, entry2)
	trie.Insert(entry3.Tokens, entry3)

	if trie.Size() != 3 {
		t.Fatalf("expected size 3, got %d", trie.Size())
	}

	// Exact match
	if got := trie.ExactMatch([]int{1, 2, 3}); got != entry1 {
		t.Errorf("exact match e1 failed: got %v", got)
	}
	if got := trie.ExactMatch([]int{1, 2, 3, 4, 5}); got != entry2 {
		t.Errorf("exact match e2 failed: got %v", got)
	}
	if got := trie.ExactMatch([]int{1, 2}); got != nil {
		t.Errorf("expected nil for non-entry prefix, got %v", got)
	}

	// Longest prefix match
	e, matchLen := trie.LongestPrefix([]int{1, 2, 3, 4, 5, 99})
	if e != entry2 || matchLen != 5 {
		t.Errorf("expected e2 with matchLen 5, got %v with matchLen %d", e, matchLen)
	}

	// Trimmable match: query with [1, 2, 3, 4]
	// Node at [1, 2, 3, 4] has no direct entry, but has descendant e2 (len 5) and e3 (len 7).
	// Should return descendant e2 (or e3), matchedLen = 4, trimLen = 1 (or 3).
	eTrim, matchedLen, trimLen, ok := trie.FindTrimmablePrefix([]int{1, 2, 3, 4})
	if !ok {
		t.Fatalf("expected trimmable match for [1, 2, 3, 4]")
	}
	if matchedLen != 4 {
		t.Errorf("expected matchedLen = 4, got %d", matchedLen)
	}
	if trimLen <= 0 {
		t.Errorf("expected positive trimLen, got %d", trimLen)
	}
	if eTrim == nil {
		t.Fatalf("expected non-nil entry")
	}

	// Deletion
	if !trie.Delete([]int{1, 2, 3}) {
		t.Errorf("expected delete true for e1")
	}
	if trie.ExactMatch([]int{1, 2, 3}) != nil {
		t.Errorf("e1 should be deleted from trie")
	}
	if trie.Size() != 2 {
		t.Errorf("expected size 2 after deletion, got %d", trie.Size())
	}
}

func TestTieredPromptCache_InPlaceTrimming(t *testing.T) {
	cache := NewTieredPromptCacheWithMaxTokens(1000)

	tokens := []int{10, 20, 30, 40, 50}
	buf := NewCacheBuffer(5, 10, 2)
	entry, err := cache.PutWithID("seq1", RoleSystem, tokens, buf)
	if err != nil {
		t.Fatalf("failed to put: %v", err)
	}

	// Standalone TrimCache helper
	trimmedTokens, err := TrimCache(tokens, 3)
	if err != nil {
		t.Fatalf("TrimCache standalone failed: %v", err)
	}
	if !reflect.DeepEqual(trimmedTokens, []int{10, 20, 30}) {
		t.Fatalf("unexpected trimmed tokens: %v", trimmedTokens)
	}

	// Cache TrimCache
	trimmedEntry, err := cache.TrimCache(tokens, 3)
	if err != nil {
		t.Fatalf("cache.TrimCache failed: %v", err)
	}
	if !reflect.DeepEqual(trimmedEntry.Tokens, []int{10, 20, 30}) {
		t.Fatalf("unexpected entry tokens: %v", trimmedEntry.Tokens)
	}
	if buf.SequenceLen != 3 {
		t.Fatalf("expected buffer sequence len 3, got %d", buf.SequenceLen)
	}
	if buf.TrimCount != 1 {
		t.Fatalf("expected TrimCount 1, got %d", buf.TrimCount)
	}

	// Out of bounds trim
	if _, err := cache.TrimCache(trimmedEntry.Tokens, 10); err != ErrTokenOutOfBounds {
		t.Errorf("expected ErrTokenOutOfBounds, got %v", err)
	}
	if _, err := cache.TrimCache(trimmedEntry.Tokens, -1); err != ErrTokenOutOfBounds {
		t.Errorf("expected ErrTokenOutOfBounds for negative, got %v", err)
	}

	_ = entry
}

func TestTieredPromptCache_Concurrent(t *testing.T) {
	cache := NewTieredPromptCacheWithMaxTokens(5000)

	var wg sync.WaitGroup
	workers := 8
	iterations := 100

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				role := RoleUser
				if i%3 == 0 {
					role = RoleSystem
				} else if i%3 == 1 {
					role = RoleAssistant
				}

				toks := []int{workerID*1000 + i, workerID*1000 + i + 1, workerID*1000 + i + 2}
				buf := NewCacheBuffer(3, 10, 2)
				id := fmt.Sprintf("w%d-i%d", workerID, i)

				_, _ = cache.PutWithID(id, role, toks, buf)
				_, _ = cache.Get(toks)
				_, _, _, _ = cache.LookupTrimmable(toks)
				if i%10 == 0 {
					_, _ = cache.TrimCache(toks, 2)
				}
				_ = cache.Stats()
			}
		}(w)
	}

	wg.Wait()

	stats := cache.Stats()
	if stats.TotalEntries <= 0 {
		t.Errorf("expected entries in cache after concurrent runs, got %d", stats.TotalEntries)
	}
}
