package model

import (
	"testing"
)

func TestGLM5NextKDALayerStateLifecycle(t *testing.T) {
	st := NewGLM5NextKDALayerState(64, 128, 4)
	if st.NumHeads != 64 || st.HeadDim != 128 || st.ConvWindow != 4 {
		t.Fatalf("unexpected dimensions: %+v", st)
	}

	// 64 * 128 * 128 = 1,048,576 floats
	if len(st.S) != 1048576 { //boundarylint:ignore CHANGE_DETECTOR_TEST a deliberate fixed-width invariant
		t.Fatalf("len(S) = %d, want 1048576", len(st.S))
	}
	// (4 - 1) * (64 * 128) = 3 * 8192 = 24576 floats
	if len(st.ConvQ) != 24576 || len(st.ConvK) != 24576 || len(st.ConvV) != 24576 { //boundarylint:ignore CHANGE_DETECTOR_TEST a deliberate fixed-width invariant
		t.Fatalf("unexpected conv lengths: Q=%d K=%d V=%d", len(st.ConvQ), len(st.ConvK), len(st.ConvV))
	}

	// Byte size: (1048576 + 3 * 24576) * 4 = (1048576 + 73728) * 4 = 1122304 * 4 = 4489216 bytes (~4.28 MB)
	expectedBytes := int64(len(st.S)+len(st.ConvQ)+len(st.ConvK)+len(st.ConvV)) * 4
	if got := st.ByteSize(); got != expectedBytes {
		t.Fatalf("ByteSize() = %d, want %d", got, expectedBytes)
	}

	// HeadMatrix indexing
	h0 := st.HeadMatrix(0)
	h63 := st.HeadMatrix(63)
	if len(h0) != 128*128 || len(h63) != 128*128 {
		t.Fatalf("head matrix slice lengths wrong: h0=%d h63=%d", len(h0), len(h63))
	}

	// Mutate state, clone, and verify isolation
	h0[42] = 3.14
	st.ConvQ[10] = 2.718

	dup := st.Clone()
	if dup.HeadMatrix(0)[42] != 3.14 || dup.ConvQ[10] != 2.718 {
		t.Fatal("clone did not copy values")
	}

	// Reset original and confirm clone unchanged
	st.Reset()
	if st.HeadMatrix(0)[42] != 0 || st.ConvQ[10] != 0 {
		t.Fatal("Reset did not zero state")
	}
	if dup.HeadMatrix(0)[42] != 3.14 || dup.ConvQ[10] != 2.718 {
		t.Fatal("clone was mutated by original Reset")
	}
}

func TestGLM5NextKDAStateCollective(t *testing.T) {
	cfg := DefaultGLM5NextConfig()
	collective := NewGLM5NextKDAState(cfg)

	if len(collective.Layers) != 34 { //boundarylint:ignore CHANGE_DETECTOR_TEST a deliberate fixed-width invariant
		t.Fatalf("expected 34 KDA layers, got %d", len(collective.Layers))
	}

	// Verify layers match 3:1 cadence
	for _, l := range cfg.KDALayers {
		if collective.Layers[l] == nil {
			t.Fatalf("missing layer state for KDA layer %d", l)
		}
	}

	// Total bytes across 34 layers: ~152.6 MB
	totalBytes := collective.TotalBytes()
	if totalBytes <= 0 || totalBytes != 34*collective.Layers[0].ByteSize() {
		t.Fatalf("unexpected TotalBytes: %d", totalBytes)
	}

	// Verify deep copy isolation
	collective.Layers[0].HeadMatrix(5)[100] = 99.9
	clone := collective.Clone()
	collective.Reset()

	if collective.Layers[0].HeadMatrix(5)[100] != 0 {
		t.Fatal("Reset failed to zero collective layer 0")
	}
	if clone.Layers[0].HeadMatrix(5)[100] != 99.9 {
		t.Fatal("clone collective lost value after original reset")
	}
}
