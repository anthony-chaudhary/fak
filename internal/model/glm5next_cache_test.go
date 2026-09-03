package model

import (
	"testing"
)

func TestGLM5NextHybridStateSnapshotAndRestore(t *testing.T) {
	cfg := DefaultGLM5NextConfig()
	state := NewGLM5NextHybridState(cfg)

	// Mutate KDA state in layer 0
	state.KDA.Layers[0].HeadMatrix(0)[10] = 42.0
	state.KDA.Layers[0].ConvQ[5] = 12.5

	// Mutate DSA state in layer 3
	dsa3 := state.DSABuffers[3]
	dsa3.K = []float32{1.0, 2.0, 3.0}
	dsa3.V = []float32{4.0, 5.0, 6.0}
	dsa3.IndexK = []float32{7.0, 8.0}
	state.NumTokens = 3

	// Take snapshot
	snap := state.Snapshot()

	// Mutate original further
	state.KDA.Layers[0].HeadMatrix(0)[10] = 999.0
	dsa3.K[0] = 999.0
	state.NumTokens = 100

	// Snapshot must be isolated
	if snap.KDA.Layers[0].HeadMatrix(0)[10] != 42.0 {
		t.Fatalf("snapshot KDA was mutated: got %g, want 42.0", snap.KDA.Layers[0].HeadMatrix(0)[10])
	}
	if snap.DSABuffers[3].K[0] != 1.0 {
		t.Fatalf("snapshot DSA was mutated: got %g, want 1.0", snap.DSABuffers[3].K[0])
	}
	if snap.NumTokens != 3 {
		t.Fatalf("snapshot NumTokens was mutated: got %d, want 3", snap.NumTokens)
	}

	// Restore snapshot into original
	state.Restore(snap)
	if state.KDA.Layers[0].HeadMatrix(0)[10] != 42.0 {
		t.Fatalf("restored KDA mismatch: got %g, want 42.0", state.KDA.Layers[0].HeadMatrix(0)[10])
	}
	if state.DSABuffers[3].K[0] != 1.0 {
		t.Fatalf("restored DSA mismatch: got %g, want 1.0", state.DSABuffers[3].K[0])
	}
	if state.NumTokens != 3 {
		t.Fatalf("restored NumTokens mismatch: got %d, want 3", state.NumTokens)
	}

	// Total bytes must be positive and non-zero
	if state.TotalByteSize() <= 0 {
		t.Fatalf("TotalByteSize = %d, expected > 0", state.TotalByteSize())
	}
}
