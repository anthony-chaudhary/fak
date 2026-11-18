package main

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// TestSessionSynthesizerReset verifies the aiperf session-synthesis reset:
// Reset clears the synthesized state (turns, hashes, budget) and re-seeds from
// the base, with L1/L1.5 preserved and fresh L2+L3 allocated to the new
// session. The config is chosen so the re-seeded hash ids are hand-computable
// from the source's PrefixAllocator.turn_hash_ids formula.
//
// Hand-computed constants (source prefix_model.py + session_synthesizer.py):
//
//	blockSize=512, L1=32000 -> l1Blocks=ceil(32000/512)=63
//	L1.5=0 -> l15Blocks=0
//	numGroups=1 -> groupID=0 (deterministic)
//	L2 sigma=0, median=5000 -> L2=5000 (deterministic)
//	fixedPrefix=(63+0)*512=32256; initialCtx=32256+5000=37256
//	totalBlocks=ceil(37256/512)=73; l1Used=63; remaining=10; l15Used=0
//	sessionNeeded=10; sessionRegionBase=63+1000*200=200063
//	sessionBase(index 1)=200063+1*4000=204063
//	hashes = [0..62] ++ [204063..204072]
//	output sigma=0, median=1024 -> outputLength=1024 (deterministic)
func TestSessionSynthesizerReset(t *testing.T) {
	cfg := sessionSynthConfig{
		MaxPromptTokens: 167000,
		BlockSize:       512,
		Layer1Tokens:    32000,
		Layer15Tokens:   0,
		NumGroups:       1,
		ZipfAlpha:       0.6,
		L2Mu:            math.Log(5000), L2Sigma: 0,
		GenMu: math.Log(1024), GenSigma: 0,
		OutputMin:            30,
		ResetBaseProbability: 0.01,
		ResetContextScaling:  1.5,
		Seed:                 42,
	}
	s := newSessionSynthesizer(cfg)

	// Simulate a mid-session state: session 0 has grown to 3 turns.
	s.sessionCounter = 1
	s.sessionIndex = 0
	s.sessionID = "sess-old"
	s.groupID = 0
	s.turns = []synthTurn{
		{turnIndex: 0, inputLength: 37256, outputLength: 300, newTokens: 37256, timestampMS: 0, hashIDs: []int{0}},
		{turnIndex: 1, inputLength: 40000, outputLength: 300, newTokens: 2444, timestampMS: 100, hashIDs: []int{0, 1}},
		{turnIndex: 2, inputLength: 43000, outputLength: 300, newTokens: 2700, timestampMS: 200, hashIDs: []int{0, 1, 2}},
	}
	s.hashes = []int{0, 1, 2}
	s.budget = 167000 - 43000
	s.timestampMS = 200
	s.prevInput = 43000
	s.prevOutput = 300
	s.turnIdx = 3

	// Reset: clear the synthesized state and re-seed from the base.
	s.Reset()

	// 1. Session index advanced to the next monotonic index.
	if s.sessionIndex != 1 {
		t.Fatalf("sessionIndex = %d, want 1", s.sessionIndex)
	}
	// 2. Fresh session id, distinct from the old, with the sess- prefix.
	if s.sessionID == "sess-old" || !strings.HasPrefix(s.sessionID, "sess-") {
		t.Fatalf("sessionID = %q, want a fresh sess-* id", s.sessionID)
	}
	// 3. Turns cleared and re-seeded with a single turn 0.
	if len(s.turns) != 1 {
		t.Fatalf("len(turns) = %d, want 1", len(s.turns))
	}
	if s.turns[0].turnIndex != 0 {
		t.Fatalf("turns[0].turnIndex = %d, want 0", s.turns[0].turnIndex)
	}
	// 4. Timestamp and turn cursor reset.
	if s.timestampMS != 0 {
		t.Fatalf("timestampMS = %v, want 0", s.timestampMS)
	}
	if s.turnIdx != 1 {
		t.Fatalf("turnIdx = %d, want 1", s.turnIdx)
	}
	// 5. Initial context re-seeded from the base (L1+L1.5+L2 = 37256).
	if got := s.turns[0].inputLength; got != 37256 {
		t.Fatalf("initialCtx = %d, want 37256", got)
	}
	// 6. Budget re-seeded to the remaining context.
	if got := s.budget; got != 167000-37256 {
		t.Fatalf("budget = %d, want %d", got, 167000-37256)
	}
	// 7. Hashes regenerated fresh: hand-computed L1 ++ L1.5 ++ session blocks.
	wantHashes := make([]int, 0, 73)
	for i := 0; i < 63; i++ {
		wantHashes = append(wantHashes, i) // L1 preserved
	}
	for i := 0; i < 10; i++ {
		wantHashes = append(wantHashes, 204063+i) // fresh L2+L3 from session base
	}
	if !reflect.DeepEqual(s.hashes, wantHashes) {
		t.Fatalf("hashes = %v, want %v", s.hashes, wantHashes)
	}
	// 8. L1 preserved: first l1Blocks hashes are the global L1 blocks.
	for i := 0; i < 63; i++ {
		if s.hashes[i] != i {
			t.Fatalf("hashes[%d] = %d, want %d (L1 preserved)", i, s.hashes[i], i)
		}
	}
	// 9. Session blocks start at the new session base (fresh, no carry-over).
	if got := s.hashes[63]; got != 204063 {
		t.Fatalf("hashes[63] = %d, want 204063 (session base for index 1)", got)
	}
	// 10. prevInput/prevOutput re-seeded to the new turn 0.
	if s.prevInput != 37256 || s.prevOutput != 1024 {
		t.Fatalf("prevInput/prevOutput = %d/%d, want 37256/1024", s.prevInput, s.prevOutput)
	}
}
