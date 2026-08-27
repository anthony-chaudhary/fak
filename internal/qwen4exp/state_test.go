package qwen4exp

import (
	"errors"
	"testing"
	"time"
)

func TestStateSnapshotSplitRunIdentity(t *testing.T) {
	id := StateIdentity{"fak-native", "Qwen3.8-Flash-Next", "f5d08274", "sha256:manifest"}
	state := ManagedState{Position: 3, GDN: [][]float32{{1, 2, 3}}, QSAIndices: []uint32{7, 11}, QSASelection: []float32{.25, .75}, CacheMetadata: map[string]uint64{"prefix_tokens": 3}}
	snap, err := NewStateSnapshot(id, state, time.Unix(123, 0))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := snap.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseStateSnapshot(raw)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := parsed.RestoreState(id, 3)
	if err != nil {
		t.Fatal(err)
	}
	if restored.Position != 3 || restored.GDN[0][2] != 3 || restored.QSAIndices[1] != 11 || restored.CacheMetadata["prefix_tokens"] != 3 {
		t.Fatalf("restored = %+v", restored)
	}
	restored.GDN[0][0] = 99
	if parsed.State.GDN[0][0] != 1 {
		t.Fatal("restore aliased snapshot state")
	}
	receipt := parsed.Receipt(len(raw), 17*time.Microsecond)
	if receipt.Engine != "fak-native" || receipt.ContextPosition != 3 || receipt.StateBytes != len(raw) || receipt.RestoreNanoseconds != 17000 {
		t.Fatalf("receipt=%+v", receipt)
	}
}

func TestStateSnapshotFailsClosed(t *testing.T) {
	id := StateIdentity{"fak-native", "artifact", "revision", "sha256:manifest"}
	snap, _ := NewStateSnapshot(id, ManagedState{Position: 9, GDN: [][]float32{{1}}}, time.Unix(1, 0))
	raw, _ := snap.Marshal()
	raw[len(raw)-5] ^= 1
	if _, err := ParseStateSnapshot(raw); !errors.Is(err, ErrStateCorrupt) {
		t.Fatalf("corrupt error=%v", err)
	}
	if _, err := snap.RestoreState(StateIdentity{"other", "artifact", "revision", "sha256:manifest"}, 0); !errors.Is(err, ErrStateIdentity) {
		t.Fatalf("identity error=%v", err)
	}
	if _, err := snap.RestoreState(id, 10); !errors.Is(err, ErrStateStale) {
		t.Fatalf("stale error=%v", err)
	}
}

func TestStateSnapshotPreservesFP32Bits(t *testing.T) {
	id := StateIdentity{"fak-native", "artifact", "revision", "sha256:manifest"}
	state := ManagedState{Position: 4, GDN: [][]float32{{-0, 1.0 / 3.0}}, QSASelection: []float32{1.0 / 7.0}}
	snap, _ := NewStateSnapshot(id, state, time.Unix(2, 0))
	raw, _ := snap.Marshal()
	got, err := ParseStateSnapshot(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.State.GDN[0][1] != state.GDN[0][1] || got.State.QSASelection[0] != state.QSASelection[0] {
		t.Fatal("FP32 state changed across checkpoint")
	}
}
