package model

import (
	"bytes"
	"errors"
	"testing"
)

func argmaxTestLogits(logits []float32) int {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

// fak-test:runtime fast est=5ms
func TestPrefixSnapshotInkling2StateRoundTrip(t *testing.T) {
	hp, w := inkling2TinyModel()
	const modelID = "inkling2-tiny-test-v1"

	unparked, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{2, 5, 7, 3}
	for i, tok := range prompt {
		unparked.PrefillToken(tok, i == len(prompt)-1)
	}

	// Non-boundary capture (e.g. unaligned prefill_chunk or pos=0) must fail closed.
	if _, err := unparked.StateSnapshot(Inkling2SnapshotBoundary{
		Kind:      Inkling2SnapshotBoundaryPrefillChunk,
		ChunkSize: Inkling2PrefillSnapshotChunk,
		Pos:       unparked.Pos(),
	}, 8, modelID); !errors.Is(err, ErrInkling2SnapshotBoundary) {
		t.Fatalf("unaligned prefill_chunk snapshot err = %v, want ErrInkling2SnapshotBoundary", err)
	}

	// Capture at turn_end boundary.
	snap, err := unparked.StateSnapshot(Inkling2SnapshotBoundary{
		Kind: Inkling2SnapshotBoundaryTurnEnd,
		Pos:  unparked.Pos(),
	}, 8, modelID)
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.GlobalSlots) == 0 || len(snap.LocalRingSlots) == 0 || len(snap.Conv) == 0 {
		t.Fatalf("expected global (%d), local ring (%d), and conv (%d) state",
			len(snap.GlobalSlots), len(snap.LocalRingSlots), len(snap.Conv))
	}

	wireA, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	wireB, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wireA, wireB) {
		t.Fatal("MarshalBinary is not deterministic")
	}

	decoded, err := DecodeInkling2StateSnapshot(wireA, modelID, &hp)
	if err != nil {
		t.Fatal(err)
	}

	restored, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.RestoreStateSnapshot(decoded); err != nil {
		t.Fatal(err)
	}

	// Next greedy token (and one follow-up step) must match the unparked host.
	suffixTok := 4
	wantTok1 := argmaxTestLogits(unparked.Step(suffixTok))
	gotTok1 := argmaxTestLogits(restored.Step(suffixTok))
	if gotTok1 != wantTok1 {
		t.Fatalf("restored next greedy token = %d, want %d", gotTok1, wantTok1)
	}
	wantTok2 := argmaxTestLogits(unparked.Step(wantTok1))
	gotTok2 := argmaxTestLogits(restored.Step(gotTok1))
	if gotTok2 != wantTok2 {
		t.Fatalf("restored second greedy token = %d, want %d", gotTok2, wantTok2)
	}
}

// fak-test:runtime fast est=5ms
func TestPrefixSnapshotInkling2StateFailsClosed(t *testing.T) {
	hp, w := inkling2TinyModel()
	const modelID = "inkling2-tiny-test-v1"

	h, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	h.PrefillToken(2, false)
	h.PrefillToken(5, true)

	// Chunk boundary of 2 is valid at pos=2.
	snap, err := h.StateSnapshot(Inkling2SnapshotBoundary{
		Kind:      Inkling2SnapshotBoundaryPrefillChunk,
		ChunkSize: 2,
		Pos:       2,
	}, 4, modelID)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := snap.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	corrupt := append([]byte(nil), wire...)
	corrupt[len(corrupt)-1] ^= 0x80
	if _, err := DecodeInkling2StateSnapshot(corrupt, modelID, &hp); !errors.Is(err, ErrHostPrefixSnapshotIntegrity) {
		t.Fatalf("corrupt snapshot err = %v, want ErrHostPrefixSnapshotIntegrity", err)
	}

	wrongVersion := append([]byte(nil), wire...)
	wrongVersion[len(inkling2StateSnapshotMagic)+1]++
	if _, err := DecodeInkling2StateSnapshot(wrongVersion, modelID, &hp); !errors.Is(err, ErrHostPrefixSnapshotVersion) {
		t.Fatalf("wrong version err = %v, want ErrHostPrefixSnapshotVersion", err)
	}

	if _, err := DecodeInkling2StateSnapshot(wire, "other-model", &hp); !errors.Is(err, ErrInkling2StateSnapshotScope) {
		t.Fatalf("wrong model id err = %v, want ErrInkling2StateSnapshotScope", err)
	}

	wrongHP := hp
	wrongHP.SlidingWindow++
	if _, err := DecodeInkling2StateSnapshot(wire, modelID, &wrongHP); !errors.Is(err, ErrInkling2StateSnapshotScope) {
		t.Fatalf("wrong hparams err = %v, want ErrInkling2StateSnapshotScope", err)
	}
}
