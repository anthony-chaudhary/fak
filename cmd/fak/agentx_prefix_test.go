package main

import (
	"reflect"
	"testing"
)

// wantRange builds the expected contiguous ID range [start, start+n).
func wantRange(start, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = start + i
	}
	return out
}

// TestPrefixAllocatorLayers verifies the layered prefix allocator ported from
// ai-dynamo/aiperf@ab610add (src/aiperf/dataset/agentic_code_gen/
// prefix_model.py, Apache-2.0): layer assignment, offset arithmetic, and
// round-trip allocation behavior against the source's semantics.
func TestPrefixAllocatorLayers(t *testing.T) {
	// block_size 128; L1 = 32000 tokens -> 250 blocks; L1.5 = 20000 tokens
	// -> ceil(20000/128) = 157 blocks; 100 groups.
	alloc, err := newPrefixAllocator(prefixLayerConfig{
		Layer1Tokens:  32000,
		Layer15Tokens: 20000,
		Layer15Groups: 100,
	}, 128)
	if err != nil {
		t.Fatalf("newPrefixAllocator: %v", err)
	}

	// Layer sizes and offset arithmetic.
	if got := alloc.L1Blocks(); got != 250 {
		t.Errorf("L1Blocks = %d, want 250", got)
	}
	if got := alloc.L15Blocks(); got != 157 {
		t.Errorf("L15Blocks = %d, want 157", got)
	}
	if got := alloc.PrefixBlocks(); got != 407 {
		t.Errorf("PrefixBlocks = %d, want 407", got)
	}
	// Whole-block reservation: 407*128 = 52096 > 32000+20000 = 52000.
	if got := alloc.PrefixTokens(); got != 52096 {
		t.Errorf("PrefixTokens = %d, want 52096", got)
	}
	if got := alloc.BlockSize(); got != 128 {
		t.Errorf("BlockSize = %d, want 128", got)
	}
	if got := alloc.GroupBase(3); got != 850 {
		t.Errorf("GroupBase(3) = %d, want 850", got)
	}
	// Session region starts after all group regions: 250 + 1000*200.
	if got := alloc.SessionBase(0); got != 200250 {
		t.Errorf("SessionBase(0) = %d, want 200250", got)
	}
	if got := alloc.SessionBase(7); got != 228250 {
		t.Errorf("SessionBase(7) = %d, want 228250", got)
	}

	// Turn 0: 60000 tokens -> 469 blocks = 250 L1 + 157 L1.5 + 62 session.
	turn0, err := alloc.TurnHashIDs(7, 3, 60000, nil)
	if err != nil {
		t.Fatalf("TurnHashIDs turn 0: %v", err)
	}
	want0 := append(append(wantRange(0, 250), wantRange(850, 157)...), wantRange(228250, 62)...)
	if !reflect.DeepEqual(turn0, want0) {
		t.Fatalf("turn 0 hash IDs = %v (len %d), want %v (len %d)",
			turn0, len(turn0), want0, len(want0))
	}
	if got := len(turn0); got != 469 {
		t.Errorf("turn 0 length = %d, want 469", got)
	}
	if got := turn0[250]; got != 850 {
		t.Errorf("turn 0 L1.5 start = %d, want 850", got)
	}
	if got := turn0[406]; got != 1006 {
		t.Errorf("turn 0 L1.5 end = %d, want 1006", got)
	}
	if got := turn0[407]; got != 228250 {
		t.Errorf("turn 0 session start = %d, want 228250", got)
	}

	// SessionBlocksNeeded agrees with the turn-0 split.
	if got := alloc.SessionBlocksNeeded(60000); got != 62 {
		t.Errorf("SessionBlocksNeeded(60000) = %d, want 62", got)
	}

	// Turn 1: input grows to 70000 tokens -> 547 blocks; session grows from
	// 62 to 140 blocks, continuing from the last carried ID.
	session0 := alloc.ExtractSessionIDs(turn0)
	if !reflect.DeepEqual(session0, wantRange(228250, 62)) {
		t.Fatalf("ExtractSessionIDs(turn 0) = %v, want %v", session0, wantRange(228250, 62))
	}
	turn1, err := alloc.TurnHashIDs(7, 3, 70000, session0)
	if err != nil {
		t.Fatalf("TurnHashIDs turn 1: %v", err)
	}
	want1 := append(append(wantRange(0, 250), wantRange(850, 157)...), wantRange(228250, 140)...)
	if !reflect.DeepEqual(turn1, want1) {
		t.Fatalf("turn 1 hash IDs = %v (len %d), want %v (len %d)",
			turn1, len(turn1), want1, len(want1))
	}
	if got := len(turn1); got != 547 {
		t.Errorf("turn 1 length = %d, want 547", got)
	}

	// Round-trip: extract the session IDs from turn 1 and rebuild the full
	// array with BuildHashIDs; the result must be identical.
	session1 := alloc.ExtractSessionIDs(turn1)
	if !reflect.DeepEqual(session1, wantRange(228250, 140)) {
		t.Fatalf("ExtractSessionIDs(turn 1) = %v, want %v", session1, wantRange(228250, 140))
	}
	rebuilt, err := alloc.BuildHashIDs(3, 70000, session1)
	if err != nil {
		t.Fatalf("BuildHashIDs: %v", err)
	}
	if !reflect.DeepEqual(rebuilt, turn1) {
		t.Fatalf("BuildHashIDs round-trip = %v (len %d), want %v (len %d)",
			rebuilt, len(rebuilt), turn1, len(turn1))
	}

	// Shrink: input drops to 55000 tokens -> 430 blocks; session truncates
	// from 140 carried IDs to 23.
	turn2, err := alloc.TurnHashIDs(7, 3, 55000, session1)
	if err != nil {
		t.Fatalf("TurnHashIDs shrink: %v", err)
	}
	want2 := append(append(wantRange(0, 250), wantRange(850, 157)...), wantRange(228250, 23)...)
	if !reflect.DeepEqual(turn2, want2) {
		t.Fatalf("shrink hash IDs = %v (len %d), want %v (len %d)",
			turn2, len(turn2), want2, len(want2))
	}

	// L1.5 truncation: 42000 tokens -> 329 blocks = 250 L1 + 79 L1.5, no session.
	turn3, err := alloc.TurnHashIDs(7, 3, 42000, nil)
	if err != nil {
		t.Fatalf("TurnHashIDs L1.5 truncation: %v", err)
	}
	want3 := append(wantRange(0, 250), wantRange(850, 79)...)
	if !reflect.DeepEqual(turn3, want3) {
		t.Fatalf("L1.5 truncation hash IDs = %v (len %d), want %v (len %d)",
			turn3, len(turn3), want3, len(want3))
	}
	if got := alloc.SessionBlocksNeeded(42000); got != 0 {
		t.Errorf("SessionBlocksNeeded(42000) = %d, want 0", got)
	}
	if got := alloc.ExtractSessionIDs(turn3); got != nil {
		t.Errorf("ExtractSessionIDs(L1.5-truncated) = %v, want nil", got)
	}

	// Tiny input: 100 tokens -> 1 block, all L1.
	turn4, err := alloc.TurnHashIDs(7, 3, 100, nil)
	if err != nil {
		t.Fatalf("TurnHashIDs tiny: %v", err)
	}
	if !reflect.DeepEqual(turn4, []int{0}) {
		t.Fatalf("tiny hash IDs = %v, want [0]", turn4)
	}
	if got := alloc.SessionBlocksNeeded(0); got != 0 {
		t.Errorf("SessionBlocksNeeded(0) = %d, want 0", got)
	}
}

// TestPrefixAllocatorErrors verifies the allocator's validation rejects the
// same inputs the source raises ValueError for.
func TestPrefixAllocatorErrors(t *testing.T) {
	cfg := prefixLayerConfig{Layer1Tokens: 32000, Layer15Tokens: 20000, Layer15Groups: 100}

	if _, err := newPrefixAllocator(cfg, 0); err == nil {
		t.Error("newPrefixAllocator(blockSize=0): expected error, got nil")
	}
	if _, err := newPrefixAllocator(prefixLayerConfig{Layer15Groups: 1001}, 128); err == nil {
		t.Error("newPrefixAllocator(num_groups=1001): expected error, got nil")
	}
	// ceil(25601/128) = 201 L1.5 blocks > MAX_GROUP_BLOCKS = 200.
	if _, err := newPrefixAllocator(prefixLayerConfig{Layer15Tokens: 25601, Layer15Groups: 1}, 128); err == nil {
		t.Error("newPrefixAllocator(layer1_5_tokens=25601): expected error, got nil")
	}

	alloc, err := newPrefixAllocator(cfg, 128)
	if err != nil {
		t.Fatalf("newPrefixAllocator: %v", err)
	}
	// 564224 tokens -> 4408 blocks; 4408 - 250 L1 - 157 L1.5 = 4001 session
	// blocks > MAX_SESSION_BLOCKS = 4000.
	if _, err := alloc.TurnHashIDs(0, 0, 564224, nil); err == nil {
		t.Error("TurnHashIDs(564224): expected session budget error, got nil")
	}
	// BuildHashIDs needs 62 session IDs for 60000 tokens; 3 is too few.
	if _, err := alloc.BuildHashIDs(3, 60000, []int{1, 2, 3}); err == nil {
		t.Error("BuildHashIDs with 3 session IDs: expected error, got nil")
	}
}
