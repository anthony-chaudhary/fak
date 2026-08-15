package model

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestHostPrefixSnapshotCodecRoundTripsCompleteQwenHybridState(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	be := newRecordingQwen35Backend(m)
	be.deviceMemory = true
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Prefill([]int{3, 7, 11, 5})

	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	host, err := snap.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()

	wireA, err := host.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	wireB, err := host.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wireA, wireB) {
		t.Fatal("same host prefix produced non-deterministic wire bytes")
	}

	decoded, err := DecodeHostPrefixSnapshot(wireA, be, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer decoded.Close()
	decodedWire, err := decoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(decodedWire, wireA) {
		t.Fatalf("decoded host prefix did not reproduce its complete wire image: header %s; body %s",
			firstWireDifference(wireA[:hostPrefixSnapshotHeader], decodedWire[:hostPrefixSnapshotHeader]),
			firstWireDifference(wireA[hostPrefixSnapshotHeader:], decodedWire[hostPrefixSnapshotHeader:]))
	}
	live, err := decoded.Restore()
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()
	roundTrip, err := live.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer roundTrip.Close()

	roundTripWire, err := roundTrip.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(roundTripWire, wireA) {
		t.Fatal("host/backend/host restore did not preserve the complete wire image")
	}
	if roundTrip.qwen35 == nil || !reflect.DeepEqual(roundTrip.qwen35.layers, host.qwen35.layers) {
		t.Fatal("Qwen3.5/3.6 convolution or recurrent state drifted across serialized host restore")
	}
}

func firstWireDifference(a, b []byte) string {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return fmt.Sprintf("offset=%d got=%02x want=%02x len=%d/%d", i, b[i], a[i], len(b), len(a))
		}
	}
	return fmt.Sprintf("length=%d/%d", len(b), len(a))
}

func TestHostPrefixSnapshotCodecFailsClosed(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	be := newRecordingQwen35Backend(m)
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Prefill([]int{3, 7, 11})
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	host, err := snap.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	wire, err := host.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	corrupt := append([]byte(nil), wire...)
	corrupt[len(corrupt)-1] ^= 0x80
	if _, err := DecodeHostPrefixSnapshot(corrupt, be, cfg); !errors.Is(err, ErrHostPrefixSnapshotIntegrity) {
		t.Fatalf("corruption error=%v, want integrity refusal", err)
	}

	wrongVersion := append([]byte(nil), wire...)
	wrongVersion[len(hostPrefixSnapshotMagic)+1]++
	if _, err := DecodeHostPrefixSnapshot(wrongVersion, be, cfg); !errors.Is(err, ErrHostPrefixSnapshotVersion) {
		t.Fatalf("version error=%v, want version refusal", err)
	}

	wrongScope := cfg
	wrongScope.VocabSize++
	if _, err := DecodeHostPrefixSnapshot(wire, be, wrongScope); !errors.Is(err, ErrHostPrefixSnapshotScope) {
		t.Fatalf("scope error=%v, want model-configuration refusal", err)
	}
}
