package model

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
)

func makeTestGDNSnapshot(sessionID string, tokenIDs []int, numLayers int, stateDim int) StateSnapshot {
	layers := make([]GDNLayerState, numLayers)
	for l := 0; l < numLayers; l++ {
		conv := make([]float32, 2)
		conv[0] = float32(l + 1)
		conv[1] = float32(len(tokenIDs))

		rec := make([]float32, stateDim)
		for i := 0; i < stateDim; i++ {
			rec[i] = float32((l+1)*100 + i + len(tokenIDs))
		}
		layers[l] = GDNLayerState{
			Layer:     l,
			Conv:      conv,
			Recurrent: rec,
		}
	}
	return StateSnapshot{
		SessionID: sessionID,
		TokenIDs:  append([]int(nil), tokenIDs...),
		Layers:    layers,
	}
}

func TestGDNPrefixCache_ExactPrefixExtensionHits(t *testing.T) {
	cache := NewGDNPrefixCache(4)
	if cache.Capacity() != 4 {
		t.Fatalf("cache.Capacity() = %d, want 4", cache.Capacity())
	}

	sessionID := "session-agent-001"
	prefix1 := []int{101, 102, 103, 104}
	snap1 := makeTestGDNSnapshot(sessionID, prefix1, 48, 16)

	if err := cache.Put(sessionID, prefix1, snap1); err != nil {
		t.Fatalf("initial Put failed: %v", err)
	}

	// Exact match hit
	got, hit := cache.Get(sessionID, prefix1)
	if !hit {
		t.Fatalf("expected hit on exact prefix match")
	}
	if len(got.TokenIDs) != len(prefix1) {
		t.Fatalf("got token count = %d, want %d", len(got.TokenIDs), len(prefix1))
	}
	if got.LayerCount() != 48 {
		t.Fatalf("got layers = %d, want 48", got.LayerCount())
	}

	// Extension hit 1: prompt has 3 additional tokens
	prompt2 := []int{101, 102, 103, 104, 105, 106, 107}
	got2, hit2 := cache.Get(sessionID, prompt2)
	if !hit2 {
		t.Fatalf("expected hit on prefix extension prompt2")
	}
	if len(got2.TokenIDs) != 4 {
		t.Fatalf("got2 held tokens = %d, want 4", len(got2.TokenIDs))
	}
	for i, tok := range prefix1 {
		if got2.TokenIDs[i] != tok {
			t.Fatalf("got2 token mismatch at %d: got %d, want %d", i, got2.TokenIDs[i], tok)
		}
	}

	// Update cache with extended snapshot
	snap2 := makeTestGDNSnapshot(sessionID, prompt2, 48, 16)
	if err := cache.Put(sessionID, prompt2, snap2); err != nil {
		t.Fatalf("Put prompt2 failed: %v", err)
	}

	// Extension hit 2: prompt extends further
	prompt3 := []int{101, 102, 103, 104, 105, 106, 107, 108, 109, 110}
	got3, hit3 := cache.Get(sessionID, prompt3)
	if !hit3 {
		t.Fatalf("expected hit on prefix extension prompt3")
	}
	if len(got3.TokenIDs) != 7 {
		t.Fatalf("got3 held tokens = %d, want 7", len(got3.TokenIDs))
	}

	stats := cache.Stats()
	if stats.Hits < 3 {
		t.Fatalf("stats.Hits = %d, want >= 3", stats.Hits)
	}
	if stats.Puts < 2 {
		t.Fatalf("stats.Puts = %d, want >= 2", stats.Puts)
	}
	if stats.Refusals != 0 {
		t.Fatalf("stats.Refusals = %d, want 0", stats.Refusals)
	}
}

func TestGDNPrefixCache_DivergentPrefixRejectionAndMiss(t *testing.T) {
	cache := NewGDNPrefixCache(4)
	sessionID := "session-diverge-test"
	heldPrefix := []int{10, 20, 30, 40, 50}
	snap := makeTestGDNSnapshot(sessionID, heldPrefix, 48, 16)

	if err := cache.Put(sessionID, heldPrefix, snap); err != nil {
		t.Fatalf("initial Put failed: %v", err)
	}

	t.Run("BackwardRewind_GetMiss", func(t *testing.T) {
		// Prompt is shorter than held recurrence state: irreversible fold cannot rewind
		shorter := []int{10, 20, 30}
		_, hit := cache.Get(sessionID, shorter)
		if hit {
			t.Fatalf("expected miss when prompt is shorter than held prefix")
		}
	})

	t.Run("BackwardRewind_PutRefused", func(t *testing.T) {
		// Put attempts backward rewind: must be rejected with ErrBackwardRewind
		shorter := []int{10, 20, 30}
		err := cache.Put(sessionID, shorter, snap)
		if err == nil {
			t.Fatalf("expected error on backward rewind Put")
		}
		if !errors.Is(err, ErrBackwardRewind) {
			t.Fatalf("expected ErrBackwardRewind, got: %v", err)
		}
	})

	t.Run("DivergentToken_GetMiss", func(t *testing.T) {
		// Divergence at index 2 (token 999 instead of 30)
		divergent := []int{10, 20, 999, 40, 50, 60}
		_, hit := cache.Get(sessionID, divergent)
		if hit {
			t.Fatalf("expected miss on divergent prefix prompt")
		}
	})

	t.Run("DivergentToken_PutRefused", func(t *testing.T) {
		// Put attempts divergent extension: must be rejected with ErrPrefixDivergence
		divergent := []int{10, 20, 999, 40, 50, 60}
		err := cache.Put(sessionID, divergent, snap)
		if err == nil {
			t.Fatalf("expected error on divergent prefix Put")
		}
		if !errors.Is(err, ErrPrefixDivergence) {
			t.Fatalf("expected ErrPrefixDivergence, got: %v", err)
		}
	})

	t.Run("DivergentFirstToken", func(t *testing.T) {
		divergentFirst := []int{999, 20, 30, 40, 50, 60}
		_, hit := cache.Get(sessionID, divergentFirst)
		if hit {
			t.Fatalf("expected miss when first token diverges")
		}
		err := cache.Put(sessionID, divergentFirst, snap)
		if !errors.Is(err, ErrPrefixDivergence) {
			t.Fatalf("expected ErrPrefixDivergence on first token divergence, got: %v", err)
		}
	})

	t.Run("EmptyValidation", func(t *testing.T) {
		if _, hit := cache.Get("", []int{10, 20}); hit {
			t.Fatalf("expected miss on empty sessionID")
		}
		if _, hit := cache.Get(sessionID, nil); hit {
			t.Fatalf("expected miss on nil tokenIDs")
		}
		if err := cache.Put("", []int{10}, snap); !errors.Is(err, ErrEmptySessionID) {
			t.Fatalf("expected ErrEmptySessionID, got: %v", err)
		}
		if err := cache.Put(sessionID, nil, snap); !errors.Is(err, ErrEmptyTokens) {
			t.Fatalf("expected ErrEmptyTokens, got: %v", err)
		}
	})

	t.Run("ResetAfterExplicitEvict", func(t *testing.T) {
		// After Evict, a new conversation with divergent tokens is accepted cleanly
		cache.Evict(sessionID)
		freshPrefix := []int{999, 1000, 1001}
		freshSnap := makeTestGDNSnapshot(sessionID, freshPrefix, 48, 16)
		if err := cache.Put(sessionID, freshPrefix, freshSnap); err != nil {
			t.Fatalf("Put after Evict failed: %v", err)
		}
		got, hit := cache.Get(sessionID, []int{999, 1000, 1001, 1002})
		if !hit {
			t.Fatalf("expected hit on fresh session after evict")
		}
		if len(got.TokenIDs) != 3 {
			t.Fatalf("got held tokens = %d, want 3", len(got.TokenIDs))
		}
	})
}

func TestGDNPrefixCache_BoundedSessionRetention(t *testing.T) {
	// Retain up to 3 conversation sessions (LRU capacity = 3)
	cache := NewGDNPrefixCache(3)
	if cache.Capacity() != 3 {
		t.Fatalf("capacity = %d, want 3", cache.Capacity())
	}

	snapA := makeTestGDNSnapshot("S1", []int{1, 2}, 4, 8)
	snapB := makeTestGDNSnapshot("S2", []int{3, 4}, 4, 8)
	snapC := makeTestGDNSnapshot("S3", []int{5, 6}, 4, 8)

	if err := cache.Put("S1", []int{1, 2}, snapA); err != nil {
		t.Fatal(err)
	}
	if err := cache.Put("S2", []int{3, 4}, snapB); err != nil {
		t.Fatal(err)
	}
	if err := cache.Put("S3", []int{5, 6}, snapC); err != nil {
		t.Fatal(err)
	}

	if cache.Len() != 3 {
		t.Fatalf("cache.Len() = %d, want 3", cache.Len())
	}

	// Access S1 to promote it to MRU. Order is now S2 (oldest), S3, S1.
	if _, hit := cache.Get("S1", []int{1, 2, 7}); !hit {
		t.Fatalf("expected hit for S1")
	}

	// Insert S4. Capacity exceeded -> S2 (oldest) must be evicted.
	snapD := makeTestGDNSnapshot("S4", []int{7, 8}, 4, 8)
	if err := cache.Put("S4", []int{7, 8}, snapD); err != nil {
		t.Fatal(err)
	}

	if cache.Len() != 3 {
		t.Fatalf("cache.Len() after eviction = %d, want 3", cache.Len())
	}

	// S2 must be evicted
	if _, hit := cache.Get("S2", []int{3, 4, 9}); hit {
		t.Fatalf("S2 should have been evicted by bounded capacity")
	}

	// S1, S3, S4 must still be present
	for _, id := range []string{"S1", "S3", "S4"} {
		tokens, ok := cache.HeldTokens(id)
		if !ok || len(tokens) == 0 {
			t.Fatalf("expected %s to remain in cache", id)
		}
	}

	// After Get(S3): S3 is moved to front.
	// List order from front to back: S3, S4, S1. S1 is now at the back (oldest).
	if _, hit := cache.Get("S3", []int{5, 6, 10}); !hit {
		t.Fatalf("expected hit for S3")
	}

	// Insert S5 -> S1 (oldest at back) must be evicted.
	snapE := makeTestGDNSnapshot("S5", []int{9, 10}, 4, 8)
	if err := cache.Put("S5", []int{9, 10}, snapE); err != nil {
		t.Fatal(err)
	}

	if _, hit := cache.Get("S1", []int{1, 2, 7}); hit {
		t.Fatalf("S1 should have been evicted")
	}

	// S3, S4, S5 must still be present
	for _, id := range []string{"S3", "S4", "S5"} {
		tokens, ok := cache.HeldTokens(id)
		if !ok || len(tokens) == 0 {
			t.Fatalf("expected %s to remain in cache", id)
		}
	}

	// Explicit Evict of S4
	cache.Evict("S4")
	if cache.Len() != 2 {
		t.Fatalf("cache.Len() after explicit Evict = %d, want 2", cache.Len())
	}
	if _, hit := cache.Get("S4", []int{7, 8, 11}); hit {
		t.Fatalf("S4 should be gone after Evict")
	}

	stats := cache.Stats()
	if stats.Evictions < 3 {
		t.Fatalf("stats.Evictions = %d, want >= 3 (2 capacity + 1 explicit)", stats.Evictions)
	}
}

// simulateGDNStep computes one step of Gated DeltaNet linear recurrence across 48 layers:
// S_t = decay * S_{t-1} + key^T * val - (S_{t-1} * key) * key^T
// with a 1D convolution tail buffer.
func simulateGDNStep(layer int, token int, conv []float32, recurrent []float32, dim int) ([]float32, []float32) {
	decay := float32(0.95 - 0.001*float64(layer%10))

	// Update convolution tail: shift and insert new token
	newConv := make([]float32, len(conv))
	copy(newConv[1:], conv[:len(conv)-1])
	newConv[0] = float32(token * (layer + 1))

	// Linear recurrence update
	newRec := make([]float32, len(recurrent))
	tF := float32(token)
	lF := float32(layer + 1)
	for i := 0; i < dim; i++ {
		key := float32(math.Sin(float64(tF*3.1415 + lF*0.1 + float32(i))))
		val := float32(math.Cos(float64(tF*1.4142 + lF*0.2 + float32(i))))
		// Simplified rank-1 DeltaNet recurrent fold
		oldVal := recurrent[i]
		newRec[i] = decay*oldVal + key*val - (oldVal*key)*key*0.01
	}
	return newConv, newRec
}

// computeReferenceGDN folds tokens from scratch across all 48 layers.
func computeReferenceGDN(sessionID string, tokens []int, numLayers, dim int) StateSnapshot {
	convSize := 3
	layers := make([]GDNLayerState, numLayers)
	for l := 0; l < numLayers; l++ {
		conv := make([]float32, convSize)
		rec := make([]float32, dim)
		for _, tok := range tokens {
			conv, rec = simulateGDNStep(l, tok, conv, rec, dim)
		}
		layers[l] = GDNLayerState{
			Layer:     l,
			Conv:      conv,
			Recurrent: rec,
		}
	}
	return StateSnapshot{
		SessionID: sessionID,
		TokenIDs:  append([]int(nil), tokens...),
		Layers:    layers,
	}
}

// extendGDN resumes linear recurrence from an existing StateSnapshot for only the suffix tokens.
func extendGDN(base StateSnapshot, prompt []int, dim int) StateSnapshot {
	startPos := len(base.TokenIDs)
	suffix := prompt[startPos:]

	layers := make([]GDNLayerState, len(base.Layers))
	for i, l := range base.Layers {
		conv := append([]float32(nil), l.Conv...)
		rec := append([]float32(nil), l.Recurrent...)
		for _, tok := range suffix {
			conv, rec = simulateGDNStep(l.Layer, tok, conv, rec, dim)
		}
		layers[i] = GDNLayerState{
			Layer:     l.Layer,
			Conv:      conv,
			Recurrent: rec,
		}
	}
	return StateSnapshot{
		SessionID: base.SessionID,
		TokenIDs:  append([]int(nil), prompt...),
		Layers:    layers,
	}
}

func TestGDNPrefixCache_MultiTurnSimulatedAgentConversationSteps(t *testing.T) {
	const (
		numLayers = 48
		stateDim  = 16
	)

	cache := NewGDNPrefixCache(4)
	sessionID := "agent-multiturn-session"

	// Define 4 conversation turns in an agent coding loop:
	// Turn 1: System prompt + initial user prompt (32 tokens)
	tokensTurn1 := make([]int, 32)
	for i := range tokensTurn1 {
		tokensTurn1[i] = 100 + i
	}

	// Turn 2: Agent tool call appended (extended to 64 tokens)
	tokensTurn2 := make([]int, 64)
	copy(tokensTurn2, tokensTurn1)
	for i := 32; i < 64; i++ {
		tokensTurn2[i] = 200 + i
	}

	// Turn 3: Tool response + intermediate reasoning (extended to 112 tokens)
	tokensTurn3 := make([]int, 112)
	copy(tokensTurn3, tokensTurn2)
	for i := 64; i < 112; i++ {
		tokensTurn3[i] = 300 + i
	}

	// Turn 4: User follow-up + final assistant answer (extended to 160 tokens)
	tokensTurn4 := make([]int, 160)
	copy(tokensTurn4, tokensTurn3)
	for i := 112; i < 160; i++ {
		tokensTurn4[i] = 400 + i
	}

	turns := [][]int{tokensTurn1, tokensTurn2, tokensTurn3, tokensTurn4}
	totalTokensSaved := 0

	for turnIdx, prompt := range turns {
		turnNum := turnIdx + 1

		// 1. Look up cached prefix state
		cachedSnap, hit := cache.Get(sessionID, prompt)

		var currentSnap StateSnapshot
		if !hit {
			// First turn: cold prefill
			if turnNum != 1 {
				t.Fatalf("turn %d: unexpected cache miss on valid prefix extension", turnNum)
			}
			currentSnap = computeReferenceGDN(sessionID, prompt, numLayers, stateDim)
		} else {
			// Subsequent turns: extend only from cached prefix
			heldCount := len(cachedSnap.TokenIDs)
			if heldCount >= len(prompt) {
				t.Fatalf("turn %d: held count %d >= prompt count %d", turnNum, heldCount, len(prompt))
			}
			tokensSaved := heldCount
			totalTokensSaved += tokensSaved

			currentSnap = extendGDN(cachedSnap, prompt, stateDim)
		}

		// 2. Validate mathematical bit-exact equivalence against full recomputation from scratch
		refSnap := computeReferenceGDN(sessionID, prompt, numLayers, stateDim)
		for l := 0; l < numLayers; l++ {
			gotL, _ := currentSnap.Layer(l)
			refL, _ := refSnap.Layer(l)

			for i := range gotL.Conv {
				if gotL.Conv[i] != refL.Conv[i] {
					t.Fatalf("turn %d layer %d conv[%d] mismatch: got %g, want %g",
						turnNum, l, i, gotL.Conv[i], refL.Conv[i])
				}
			}
			for i := range gotL.Recurrent {
				diff := math.Abs(float64(gotL.Recurrent[i] - refL.Recurrent[i]))
				if diff > 1e-7 {
					t.Fatalf("turn %d layer %d recurrent[%d] mismatch: got %g, want %g (diff %g)",
						turnNum, l, i, gotL.Recurrent[i], refL.Recurrent[i], diff)
				}
			}
		}

		// 3. Store extended snapshot in cache for next turn
		if err := cache.Put(sessionID, prompt, currentSnap); err != nil {
			t.Fatalf("turn %d Put failed: %v", turnNum, err)
		}
	}

	// Verify total tokens saved across 4 turns:
	// Turn 2: 32 tokens saved
	// Turn 3: 64 tokens saved
	// Turn 4: 112 tokens saved
	// Total = 208 tokens saved
	if totalTokensSaved != (32 + 64 + 112) {
		t.Fatalf("totalTokensSaved = %d, want %d", totalTokensSaved, 32+64+112)
	}

	stats := cache.Stats()
	if stats.Hits != 3 {
		t.Fatalf("stats.Hits = %d, want 3", stats.Hits)
	}
	if stats.Misses != 1 {
		t.Fatalf("stats.Misses = %d, want 1", stats.Misses)
	}
	if stats.Puts != 4 {
		t.Fatalf("stats.Puts = %d, want 4", stats.Puts)
	}
}

func TestGDNPrefixCache_DeepCopyIsolation(t *testing.T) {
	cache := NewGDNPrefixCache(2)
	sessionID := "isolation-test"
	prefix := []int{1, 2, 3}
	snap := makeTestGDNSnapshot(sessionID, prefix, 2, 4)

	if err := cache.Put(sessionID, prefix, snap); err != nil {
		t.Fatal(err)
	}

	// Mutate input snap after Put: cache must not be affected
	snap.Layers[0].Recurrent[0] = 99999.0
	snap.TokenIDs[0] = 88888

	got, hit := cache.Get(sessionID, prefix)
	if !hit {
		t.Fatalf("expected hit")
	}
	if got.Layers[0].Recurrent[0] == 99999.0 {
		t.Fatalf("cache was corrupted by mutation of input snapshot")
	}
	if got.TokenIDs[0] == 88888 {
		t.Fatalf("cache TokenIDs corrupted by mutation")
	}

	// Mutate retrieved snapshot: cache must still not be affected
	got.Layers[0].Recurrent[0] = 77777.0
	got2, hit2 := cache.Get(sessionID, prefix)
	if !hit2 {
		t.Fatalf("expected hit on second get")
	}
	if got2.Layers[0].Recurrent[0] == 77777.0 {
		t.Fatalf("cache was corrupted by mutation of retrieved snapshot")
	}
}

func TestGDNPrefixCache_NativeMetalLayerSnapshotsRoundTrip(t *testing.T) {
	sessionID := "metal-roundtrip"
	tokens := []int{5, 6, 7}
	metalSnaps := []qwen35GDNLayerSnapshot{
		{layer: 0, conv: []float32{1.0, 2.0}, recurrent: []float32{10.0, 20.0}},
		{layer: 1, conv: []float32{3.0, 4.0}, recurrent: []float32{30.0, 40.0}},
	}

	snap := FromQwen35GDNLayerSnapshots(sessionID, tokens, metalSnaps)
	if snap.SessionID != sessionID || len(snap.TokenIDs) != 3 || snap.LayerCount() != 2 {
		t.Fatalf("FromQwen35GDNLayerSnapshots mismatch: %+v", snap)
	}

	converted := snap.ToQwen35GDNLayerSnapshots()
	if len(converted) != len(metalSnaps) {
		t.Fatalf("converted len = %d, want %d", len(converted), len(metalSnaps))
	}
	for i := range metalSnaps {
		if converted[i].layer != metalSnaps[i].layer {
			t.Fatalf("layer %d != %d", converted[i].layer, metalSnaps[i].layer)
		}
		for j := range metalSnaps[i].conv {
			if converted[i].conv[j] != metalSnaps[i].conv[j] {
				t.Fatalf("conv mismatch at %d,%d", i, j)
			}
		}
		for j := range metalSnaps[i].recurrent {
			if converted[i].recurrent[j] != metalSnaps[i].recurrent[j] {
				t.Fatalf("recurrent mismatch at %d,%d", i, j)
			}
		}
	}
}

func TestGDNPrefixCache_ConcurrentAccess(t *testing.T) {
	cache := NewGDNPrefixCache(8)
	const numGoroutines = 16
	const opsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			sessID := fmt.Sprintf("concurrent-session-%d", gid%4)
			prefix := []int{gid + 1, gid + 2}

			for op := 0; op < opsPerGoroutine; op++ {
				snap := makeTestGDNSnapshot(sessID, prefix, 2, 4)
				_ = cache.Put(sessID, prefix, snap)

				extended := append(prefix, op+100)
				cache.Get(sessID, extended)

				if op%10 == 0 {
					cache.Evict(sessID)
				}
			}
		}(g)
	}

	wg.Wait()
}
